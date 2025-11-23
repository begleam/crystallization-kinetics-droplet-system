import json
import numpy as np
import sys
sys.path.append("../simulation")
from lysozyme import get_lysozyme_3d_coords, get_augmented_image
from augmentation import aug_image
from collections import defaultdict
import cv2
from random import choice
from shapely.geometry import Polygon
import os
import parmap
import argparse

w_min, w_max, w_step = 2, 282, 1
l_min, l_max, l_step = 2, 282, 1
theta_min, theta_max = 0 / 180 * np.pi, 90 / 180 * np.pi
phi_min, phi_max = 0 / 180 * np.pi, 360 / 180 * np.pi
gamma_min, gamma_max = 0 / 180 * np.pi, 90 / 180 * np.pi

w_list = np.arange(w_min, w_max+1e-4, w_step)
w_max_list = np.roll(w_list, -1)
w_list = w_list[:-1]
w_max_list = w_max_list[:-1]

l_list = np.arange(l_min, l_max+1e-4, l_step)
l_max_list = np.roll(l_list, -1)
l_list = l_list[:-1]
l_max_list = l_max_list[:-1]

L2possibleWs = defaultdict(list)
for W_class_idx, (W_min, W_max) in enumerate(zip(w_list, w_max_list)):
    W = (W_min + W_max) / 2
    for L_class_idx, (L_min, L_max) in enumerate(zip(l_list, l_max_list)):
        L = (L_min + L_max) / 2
        cryst = get_lysozyme_3d_coords(W, L, 0, 0, 0)
        if cryst is not None:
            L2possibleWs[L_class_idx].append(W_class_idx)

category_id = 1
def process_image(image_id, save_dir, additive_pos_noise=False, split_save_dir=False):
# for image_id in tqdm(range(n_total_images)):

    seg_img_w = np.random.randint(1600, 2000)
    seg_img_h = np.random.randint(1600, 2000)
    input_seg_img = np.zeros((seg_img_w, seg_img_h)).astype(np.uint8)

    crystal_img_size = 380
    ALPHA = 0.326297

    x_step = crystal_img_size//np.random.randint(1, 4)
    x_pos_candidate = np.arange(np.random.randint(0, x_step), seg_img_w - crystal_img_size+x_step, x_step) # 이미 결정된 결정들의 위치를 저장 (겹치지 않게 하기 위함)
    y_step = crystal_img_size//np.random.randint(1, 4)
    y_pos_candidate = np.arange(np.random.randint(0, y_step), seg_img_h - crystal_img_size+y_step, y_step) # 이미 결정된 결정들의 위치를 저장 (겹치지 않게 하기 위함)
    xy_pos_candidate = np.array(np.meshgrid(x_pos_candidate, y_pos_candidate)).T.reshape(-1, 2).tolist()

    num_crystals = int(np.clip(np.abs(np.random.normal(30,10)), 1, min(len(xy_pos_candidate), 80)))
    
    already_drawn_image = np.zeros((seg_img_w, seg_img_h)).astype(np.uint8)

    local_annotations = []
    for cryst_idx in range(num_crystals):
        crystal_id = image_id*100 + cryst_idx
        crystal_annot = {}
        while True:
            while True:
                l_class_idx = int(np.random.normal(len(l_list)//5,len(l_list)//5))
                if 0 <= l_class_idx < len(l_list):
                    break
            w_class_idx = choice(L2possibleWs[l_class_idx])
            w_range_min, w_range_max = w_list[w_class_idx], w_max_list[w_class_idx]
            l_range_min, l_range_max = l_list[l_class_idx], l_max_list[l_class_idx]    
            
            W = np.random.uniform(w_range_min, w_range_max)
            L = np.random.uniform(l_range_min, l_range_max)
            theta = np.random.uniform(theta_min, theta_max)
            phi = np.random.uniform(phi_min, phi_max)
            gamma = np.random.uniform(gamma_min, gamma_max)
            image_original, image_aug, boundary = get_augmented_image(W, L, theta, phi, gamma, alpha=ALPHA, img_size=crystal_img_size, additive_pos_noise=additive_pos_noise, include_image_aug=False, return_boundary=True)
            if image_aug is not None:
                break
        
        x_pos, y_pos = xy_pos_candidate.pop(np.random.randint(0, len(xy_pos_candidate)))
        x_pos+=np.random.randint(-x_step//4, x_step//4)
        x_pos = np.clip(x_pos, 0, seg_img_w - crystal_img_size)
        y_pos+=np.random.randint(-y_step//4, y_step//4)
        y_pos = np.clip(y_pos, 0, seg_img_h - crystal_img_size)
        
        boundary_in_seg_img = boundary
        boundary_in_seg_img[:, 0] += y_pos
        boundary_in_seg_img[:, 1] += x_pos
        crystal_polygon = np.zeros((1, boundary.shape[0] * 2))
        crystal_polygon[0, 0::2] = boundary_in_seg_img[:, 0]
        crystal_polygon[0, 1::2] = boundary_in_seg_img[:, 1]
        area = Polygon(boundary_in_seg_img).area
        
        image_aug = image_aug.astype(np.uint8)
        image_aug[(already_drawn_image[x_pos:x_pos+crystal_img_size, y_pos:y_pos+crystal_img_size]!=0) & (image_aug!=0)] = 0
        cv2.fillConvexPoly(already_drawn_image, boundary_in_seg_img, 1)
    
        input_seg_img[x_pos:x_pos+crystal_img_size, y_pos:y_pos+crystal_img_size] = image_aug.astype(np.uint8) | input_seg_img[x_pos:x_pos+crystal_img_size, y_pos:y_pos+crystal_img_size].astype(np.uint8)
                
        crystal_annot['segmentation'] = crystal_polygon.tolist()
        crystal_annot['area'] = area
        crystal_annot['iscrowd'] = 0
        crystal_annot['image_id'] = image_id
        crystal_annot['bbox'] = [crystal_polygon[0, 0::2].min(), crystal_polygon[0, 1::2].min(), crystal_polygon[0, 0::2].max() - crystal_polygon[0, 0::2].min(), crystal_polygon[0, 1::2].max() - crystal_polygon[0, 1::2].min()]
        crystal_annot['category_id'] = category_id
        crystal_annot['id'] = crystal_id
        
        local_annotations.append(crystal_annot)        
    input_seg_img = aug_image(input_seg_img)
    file_name = f"{str(image_id).zfill(6)}.jpg"
    image_dict = {
        "file_name": file_name,
        "height": seg_img_w,
        "width": seg_img_h,
        "id": image_id
    }
    
    if split_save_dir:
        if not os.path.exists(f"{save_dir}/{image_id//10000}"): os.mkdir(f"{save_dir}/{image_id//10000}")
        cv2.imwrite(f"{save_dir}/{image_id//10000}/{file_name}", input_seg_img)
    else:
        cv2.imwrite(f"{save_dir}/{file_name}", input_seg_img)
    return image_dict, local_annotations

def main(args):
    custom_json_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    custom_json_data["categories"] = [
        {'supercategory': 'crystal', 'id': 1, 'name': 'lysozyme'}
    ]
    for d_type in ["train", "val"]:
        if not os.path.exists(f"{args.save_dir}/{d_type}"): os.makedirs(f"{args.save_dir}/{d_type}")  
        process_results = parmap.map(process_image, range(args.n_train_images if d_type == "train" else args.n_val_images), save_dir=f"{args.save_dir}/{d_type}", additive_pos_noise=args.additive_pos_noise, split_save_dir=args.split_save_dir,pm_pbar=True, pm_processes=args.n_processes)
        for image_dict, local_annotations in process_results:
            custom_json_data["images"].append(image_dict)
            custom_json_data["annotations"].extend(local_annotations)
        
        output_json_path = f"{args.save_dir}/{d_type}_data.json"
        with open(output_json_path, 'w') as json_file:
            json.dump(custom_json_data, json_file)

if __name__=="__main__":   
    parser = argparse.ArgumentParser(description='Prepare detection dataset for crystal kinetics')
    parser.add_argument('--save_dir', type=str, default="../data/simulation_detection",
                        help='Directory to save generated images')
    parser.add_argument('--n_train_images', type=int, default=500,
                        help='Total number of train images to generate')
    parser.add_argument('--n_val_images', type=int, default=100,
                        help='Total number of val images to generate')
    parser.add_argument('--n_processes', type=int, default=10,
                        help='Number of processes to use for parallel processing')
    parser.add_argument('--additive_pos_noise', action='store_true', default=False,
                        help='Add additive position noise to crystal positions')
    parser.add_argument('--split_save_dir', action='store_true', default=False,
                        help='Split save directory into subdirectories')
    args = parser.parse_args()
    
    main(args)
