import cv2
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import sys
sys.path.append("../simulation")
from lysozyme import find_outer_points
from shapely.geometry import Polygon
from detectron2.utils.logger import setup_logger
setup_logger()

import os
import argparse
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode

def remove_biggest_convex(image, A_thres=90000):
    image_rev = deepcopy(image)

    mask = np.zeros_like(image_rev).astype(np.uint8)
    contours, _ = cv2.findContours(image_rev, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        if cv2.contourArea(contour) > A_thres:
            cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
            cv2.drawContours(mask, [contour], -1, (0), thickness=15)
    if np.sum(mask) == 0:
        return image_rev
    image_rev[(mask==0)] = 0
    return image_rev

def load_detectron2_model(train_data_path, model_weights_path):
    register_coco_instances("my_dataset_train", {}, os.path.join(train_data_path, "train_data.json"), os.path.join(train_data_path, "train"))
    register_coco_instances("my_dataset_val", {}, os.path.join(train_data_path, "val_data.json"), os.path.join(train_data_path, "val"))
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))

    cfg.DATASETS.TRAIN = ("my_dataset_train")
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 24  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 10000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.MODEL.WEIGHTS = model_weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1   # set a custom testing threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5

    predictor = DefaultPredictor(cfg)

    MetadataCatalog.get("my_dataset_train").thing_classes = ["lysozyme"]
    crystal_metadata = MetadataCatalog.get("my_dataset_train")

    return predictor, crystal_metadata



dilate_kernel_size = 3
kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
def preprocess_image(edge_img, predictor, crystal_metadata, detectron2_thres, max_area, min_area, A_thres=90000, print_progress=False, figure_save=False):
    edges_rev = ((255-edge_img)>=100).astype(np.uint8)*255
    if print_progress:
        fig, ax = plt.subplots(1, 7, figsize=(50, 10))
        ax[0].imshow(edges_rev, cmap='gray')
        ax[0].set_title('Original Edges')
        ax[0].invert_yaxis()
        if figure_save:
            if not os.path.exists("./_results"): os.makedirs("./_results")
            cv2.imwrite("./_results/1_edge_detection.png", edges_rev)

    edges_rev = remove_biggest_convex(edges_rev, A_thres=A_thres)

    if print_progress:
        ax[1].imshow(edges_rev, cmap='gray')
        ax[1].set_title('Remove Circle')
        ax[1].invert_yaxis()

    contours, hierchy = cv2.findContours(edges_rev, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # RETR_LIST
    
    if len(contours) == 0:
        return None, None, None, None
    
    if print_progress:
        # Draw only the outer contours
        mask = np.zeros_like(edges_rev)
        cv2.drawContours(mask, contours, -1, (255), thickness=3)
        ax[2].imshow(mask, cmap='gray')
        ax[2].invert_yaxis()
        ax[2].set_title('Contour Mask')

    # Find the valid contours using the area

    valid_contours = []
    for i, (cont, hier) in enumerate(zip(contours, hierchy[0])):

        cont_area = cv2.contourArea(cont)
        hull = cv2.convexHull(cont)
        hull_area = cv2.contourArea(hull)
        
        if not (min_area<hull_area<max_area): continue
        if not (min_area<cont_area<max_area): continue
        if hull_area/cont_area > 2: continue
        
        valid_contours.append(cont)

    if print_progress:
        mask = np.zeros_like(edges_rev)
        # Draw only the outer contours
        cv2.drawContours(mask, valid_contours, -1, (255), thickness=3)

        ax[3].imshow(mask, cmap='gray')
        ax[3].invert_yaxis()
        ax[3].set_title('Valid Contour using Area')


    hull_mask = np.zeros_like(edges_rev)

    mid_points = []
    hull_areas = []
    boundaries = []
    for cont in valid_contours:
        # Find the convex hull for the contour
        hull = cv2.convexHull(cont)
        hull_areas.append(cv2.contourArea(hull))
        
        # Find the convex hull boundary
        boundary = np.squeeze(cont)
        hull_boundary = find_outer_points(boundary)
        polygon = Polygon(hull_boundary)
        x_pixels, y_pixels = polygon.boundary.xy
        boundary = np.stack((np.array(y_pixels), np.array(x_pixels)), axis=1)
        boundaries.append(boundary)
        
        # Draw the convex hull on the mask, filling the area inside the hull
        x_mid = (np.max(boundary[:, 1]) + np.min(boundary[:, 1])) // 2
        y_mid = (np.max(boundary[:, 0]) + np.min(boundary[:, 0])) // 2
        mid_points.append([x_mid, y_mid])
        cv2.drawContours(hull_mask, [hull], -1, (255), thickness=cv2.FILLED)

    # Apply the mask to the original image (edges_rev)
    # This step keeps only the pixels that are inside the convex hulls
    edges_in_hulls = cv2.bitwise_and(edges_rev, hull_mask)
    if print_progress:
        # Display the result
        ax[4].imshow(edges_in_hulls, cmap='gray')
        ax[4].set_title('Pixels Inside Convex Hulls')
        ax[4].invert_yaxis()
        if figure_save:
            cv2.imwrite("./_results/2_noise_removed.png", edges_in_hulls)

    pre_mid_points, pre_hull_areas, pre_boundaries = mid_points, hull_areas, boundaries
    test_img = np.repeat(edges_in_hulls[:, :, np.newaxis], 3, axis=2)

    outputs = predictor(test_img)
    if print_progress:
        v = Visualizer(test_img[:, :, ::-1], metadata=crystal_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        ax[5].imshow(out.get_image()[:, :, ::-1])
        ax[5].invert_yaxis()
        ax[5].set_title('Detectron2')
        if figure_save:
            cv2.imwrite("./_results/3_instance_segmentation.png", out.get_image()[:, :, ::-1])
        
    results = outputs["instances"].to("cpu")
    scores = results.scores.numpy()
    pred_masks = results.pred_masks.numpy()
    
    mid_points = []
    hull_areas = []
    boundaries = []
    for crystal_i in range(len(scores)):
        if scores[crystal_i] < detectron2_thres:
            continue
        cryst_mask = pred_masks[crystal_i]
        dilated_mask = cv2.dilate(cryst_mask.astype(np.uint8), kernel, iterations=1)
        
        mask_indices = np.argwhere(dilated_mask==True)
        hull_area = mask_indices.shape[0]
        if not (min_area<hull_area<max_area): continue
        
        boundary = find_outer_points(mask_indices)

        x_mid = (np.max(mask_indices[:, 1]) + np.min(mask_indices[:, 1])) // 2
        y_mid = (np.max(mask_indices[:, 0]) + np.min(mask_indices[:, 0])) // 2
        mid_point = [x_mid, y_mid]

        mid_points.append(mid_point)
        hull_areas.append(hull_area)
        boundaries.append(boundary)
    
    detectron_mid_points = np.array(mid_points)
    for pre_mid_p, pre_hull_a, pre_bound in zip(pre_mid_points, pre_hull_areas, pre_boundaries):
        if np.min(np.sum((detectron_mid_points - pre_mid_p)**2, axis=1)**(1/2)) > 50:
            mid_points.append(pre_mid_p)
            hull_areas.append(pre_hull_a)
            boundaries.append(pre_bound)
    
    if print_progress:
        ax[6].imshow(edges_rev, cmap='gray')
        ax[6].scatter(np.array(mid_points)[:, 0], np.array(mid_points)[:, 1], c='r', s=5)
        for p in mid_points:
            ax[6].text(p[0], p[1], f'{p[0]}, {p[1]}', fontsize=5, color='r')
        ax[6].set_title('Detected Crystals')
        ax[6].invert_yaxis() 
        plt.tight_layout()
        plt.savefig("./_results/overall_process.png", dpi=300)
        plt.close()
        print("Overall process saved to ./_results/overall_process.png")

    if figure_save:
        edges_detected_boundaries = np.repeat(edges_rev[:, :, np.newaxis], 3, axis=2)
        edges_detected_boundaries_wb = 255 - np.repeat(edges_rev[:, :, np.newaxis], 3, axis=2)
        for boundary in boundaries:
            boundary = np.array(boundary)
            boundary[:,[0, 1]] = boundary[:,[1, 0]]
            cv2.drawContours(edges_detected_boundaries, [boundary], -1, (0,0,255), thickness=6)
            cv2.drawContours(edges_detected_boundaries_wb, [boundary], -1, (0,0,255), thickness=6)
        cv2.imwrite("./_results/4_detected_boundaries.png", edges_detected_boundaries)
        cv2.imwrite("./_results/5_detected_boundaries_wb.png", edges_detected_boundaries_wb)
        edges_detected_position = np.repeat(edges_rev[:, :, np.newaxis], 3, axis=2)
        edges_detected_position_wb = 255 - edges_detected_position
        for mid_p in mid_points:
            cv2.circle(edges_detected_position, (int(mid_p[0]), int(mid_p[1])), 10, (0,0,255), -1)
            cv2.circle(edges_detected_position_wb, (int(mid_p[0]), int(mid_p[1])), 10, (0,0,255), -1)
        cv2.imwrite("./_results/6_detected_positions.png", edges_detected_position)
        rgba_image = cv2.cvtColor(edges_detected_position_wb, cv2.COLOR_RGB2RGBA)
        white = np.all(rgba_image[:, :, :3] == [255, 255, 255], axis=2)
        rgba_image[white, 3] = 0
        cv2.imwrite("./_results/7_detected_positions_wb.png", rgba_image)
    
    if len(mid_points) == 0:
        return None, None, None, None
    return edges_in_hulls, mid_points, hull_areas, boundaries

def main(args):
    predictor, crystal_metadata = load_detectron2_model(args.train_data_path, args.model_weights_path)
    edge_img = cv2.imread(args.target_edge_image, cv2.IMREAD_GRAYSCALE)
    
    if edge_img is None:
        raise ValueError(f"Could not load image from {args.target_edge_image}")
    
    edges_in_hulls, mid_points, hull_areas, boundaries = preprocess_image(
        edge_img, 
        predictor, 
        crystal_metadata, 
        detectron2_thres=args.detectron2_thres,
        max_area=args.max_area,
        min_area=args.min_area,
        A_thres=args.max_area, 
        print_progress=args.print_progress, 
        figure_save=args.figure_save
    )
    
    return edges_in_hulls, mid_points, hull_areas, boundaries

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crystal detection inference using Detectron2')
    parser.add_argument('--target_edge_image', type=str, default="../data/edge_detected_vertical/fused/0.png",
                        help='Path to target edge detection image')
    parser.add_argument('--model_weights_path', type=str, default="../checkpoints/crystal_detection.pth",
                        help='Path to trained model weights')
    parser.add_argument('--train_data_path', type=str, default="../data/simulation_detection",
                        help='Path to training data directory containing train_data.json and val_data.json')
    parser.add_argument('--max_area', type=int, default=300*300,
                        help='Maximum area for crystal detection')
    parser.add_argument('--min_area', type=int, default=10,
                        help='Minimum area for crystal detection')
    parser.add_argument('--detectron2_thres', type=float, default=0.02,
                        help='Detectron2 score threshold for crystal detection')
    parser.add_argument('--print_progress', action='store_true', default=True,
                        help='Print progress and show intermediate results (default: True)')
    parser.add_argument('--no_print_progress', dest='print_progress', action='store_false',
                        help='Disable printing progress')
    parser.add_argument('--figure_save', action='store_true', default=True,
                        help='Save intermediate result figures (default: True)')
    parser.add_argument('--no_figure_save', dest='figure_save', action='store_false',
                        help='Disable saving figures')
    
    args = parser.parse_args()
    
    main(args)