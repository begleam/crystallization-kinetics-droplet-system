import cv2
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import argparse
import matplotlib.colors as mcolors
import pandas as pd

def get_ymin(image_path, image_threshold=150, distance_threshold=10):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image[image > image_threshold] = 255


    # Assuming 'image' is already defined and contains the image data
    x, y = np.where(image <= image_threshold)
    points = np.array(list(zip(x, y)))

    # Perform hierarchical clustering
    linked = linkage(points, 'single')

    # Form flat clusters
    clusters = fcluster(linked, distance_threshold, criterion='distance')

    unique_clusters, counts = np.unique(clusters, return_counts=True)
    max_idx = np.argmax(counts)

    droplet_points = points[clusters == unique_clusters[max_idx]]
    point1, point2 = droplet_points[np.argmax(droplet_points[:, 1])], droplet_points[np.argmin(droplet_points[:, 1])]
    y_min = np.min([point1[0], point2[0]])
    return y_min


# Example colormap dictionary: map specific values to colors
color_map = {
    0: 'blue',
    1: 'orange',
    2: 'purple',
    3: 'red',
    4: 'green'
}

# Creating a ListedColormap from the dictionary
cmap = mcolors.ListedColormap([color_map[key] for key in sorted(color_map.keys())])
def get_volume_from_image_path(image_path, y_min=None, image_threshold=150, distance_threshold=10, save_progress=False, save_dir="./_results"):
    image = cv2.imread(image_path)
    image_original = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if save_progress:
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        axes[0].imshow(image_original, cmap='gray')
        axes[0].set_title("Original Image")
        axes[0].axis('off')
    
    image[image > image_threshold] = 255


    if save_progress:
        axes[1].imshow(image, cmap='gray')
        axes[1].set_title("Thresholded Image")
        axes[1].axis('off')
    # Assuming 'image' is already defined and contains the image data
    x, y = np.where(image <= image_threshold)
    points = np.array(list(zip(x, y)))

    # Perform hierarchical clustering
    linked = linkage(points, 'single')

    # Form flat clusters
    clusters = fcluster(linked, distance_threshold, criterion='distance')

    if save_progress:
        axes[2].scatter(points[:, 1], points[:, 0], c=clusters, cmap=cmap, s=1)
        axes[2].set_xlim(0, image.shape[1])
        axes[2].set_ylim(0, image.shape[0])
        axes[2].invert_yaxis()
        axes[2].set_title("Hierarchical Clustering")
        axes[2].axis('off')

    unique_clusters, counts = np.unique(clusters, return_counts=True)
    max_idx = np.argmax(counts)

    droplet_points = points[clusters == unique_clusters[max_idx]]
    point1, point2 = droplet_points[np.argmax(droplet_points[:, 1])], droplet_points[np.argmin(droplet_points[:, 1])]
    if y_min is None:
        y_min = np.min([point1[0], point2[0]])

    droplet_points = droplet_points[droplet_points[:, 0] > y_min]

    mask = np.zeros_like(image)
    mask[droplet_points[:, 0], droplet_points[:, 1]] = 255

    if save_progress:
        axes[3].imshow(mask, cmap='gray')
        axes[3].set_title("Droplet Points")
        axes[3].axis('off')

    contour = cv2.findNonZero(mask)
    hull_points = cv2.convexHull(contour).squeeze()

    xs = hull_points[:, 0][np.argsort(hull_points[:, 0])]
    ys = hull_points[:, 1][np.argsort(hull_points[:, 0])]

    degree = 2
    
    new_points = np.zeros((len(xs), 2))
    new_points[:,0] = xs
    new_points[:,1] = ys

    original_points = new_points.copy()

    if point1[0] > y_min:
        coeff = np.polyfit([xs[-len(xs)//2], xs[-len(xs)//4], xs[-1]], [ys[-len(xs)//2], ys[-len(xs)//4], ys[-1]], degree)
        x_new = np.linspace(xs[-1], image.shape[1], image.shape[1] - xs[-1] + 1)
        y_new = np.polyval(coeff, x_new)
        x_new = x_new[y_new >=y_min]
        y_new = y_new[y_new >=y_min]
        new_points = np.concatenate([new_points, np.array([x_new, y_new]).T], axis=0)

    if point2[0] > y_min:
        coeff = np.polyfit([xs[0], xs[len(xs)//4], xs[len(xs)//2]], [ys[0], ys[len(xs)//4], ys[len(xs)//2]], degree)
        x_new = np.linspace(0, xs[0], xs[0] + 1)
        y_new = np.polyval(coeff, x_new)
        x_new = x_new[y_new >=y_min]
        y_new = y_new[y_new >=y_min]
        new_points = np.concatenate([np.array([x_new, y_new]).T, new_points], axis=0)    

    convex_image = np.zeros_like(image)
    cv2.drawContours(convex_image, [new_points.astype(np.int32)], 0, 255, thickness=cv2.FILLED)

    d_list = np.sum(convex_image/np.max(convex_image), axis=1)
    A_list = d_list ** 2 * np.pi / 4
    volume = np.sum(A_list)

    if save_progress:
        figure_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Draw original points in red
        for (x, y) in original_points:
            cv2.circle(figure_image, (int(x), int(y)), 6, (0,0,255), -1)
        
        # Draw y_min line in green
        for x_dashed in range(0, image.shape[1], 50):
            cv2.line(figure_image, (x_dashed+10, y_min), (x_dashed+40, y_min), (0, 255, 0), 5, lineType=cv2.LINE_AA)
        
        # Draw new points in blue
        for (x, y) in new_points:
            cv2.circle(figure_image, (int(x), int(y)), 6, (255,0,0), -1)
            
        # Find and draw the convex hull of the new points
        hull = cv2.convexHull(new_points.astype(np.float32))
        cv2.drawContours(figure_image, [hull.astype(np.int32)], 0, (0, 0, 255), thickness=5, lineType=cv2.LINE_AA)
        
        axes[4].imshow(cv2.cvtColor(figure_image, cv2.COLOR_BGR2RGB))
        axes[4].set_title("Detected Points")
        axes[4].axis('off')
        fig.tight_layout()
        fig.savefig(f"{save_dir}/detected_points_{image_path.split('/')[-1].split('.')[0]}.png", dpi=300)
        plt.close()
    return volume

def main(args):
    # Get image paths
    if not os.path.exists(args.horizontal_image_dir):
        raise ValueError(f"Image directory not found: {args.horizontal_image_dir}")
    
    img_path_list = [os.path.join(args.horizontal_image_dir, f) 
                     for f in os.listdir(args.horizontal_image_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if len(img_path_list) == 0:
        raise ValueError(f"No images found in {args.horizontal_image_dir}")
    
    # Calculate y_min for all images
    print("Calculating y_min for all images...")
    y_min_list = []
    for image_path in tqdm(img_path_list[::args.sample_step], desc='Finding y_min'):
        y_min = get_ymin(image_path, 
                        image_threshold=args.image_threshold, 
                        distance_threshold=args.distance_threshold)
        y_min_list.append(y_min)
    
    # Find most common y_min
    y_min_candidates, y_min_counts = np.unique(y_min_list, return_counts=True)
    y_min = y_min_candidates[np.argmax(y_min_counts)]
    y_min += -2
    
    # Calculate volumes
    print("Calculating volumes...")
    volume_data = []
    for img_path in tqdm(img_path_list, desc='Calculating pixel volume'):
        volume = get_volume_from_image_path(img_path, 
                                           y_min=y_min,
                                           image_threshold=args.image_threshold,
                                           distance_threshold=args.distance_threshold,
                                           save_progress=args.save_progress,
                                           save_dir=args.output_dir)
        volume_data.append(volume)
    
    V_list = np.array(volume_data)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save plot
    plt.figure(figsize=(30, 3))
    plt.scatter(np.arange(len(V_list)), V_list)
    plt.xlabel("Image Index")
    plt.ylabel("Volume")
    plt.title("Droplet Volume Prediction")
    plt.tight_layout()
    output_path = os.path.join(args.output_dir, 'droplet_volume_prediction.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Plot saved to {output_path}")
    
    # Save volume data as CSV
    df = pd.DataFrame({
        'img_path': img_path_list,
        'volume': volume_data
    })
    csv_path = os.path.join(args.output_dir, 'volume_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Volume data saved to {csv_path}")
    print(f"Total volumes calculated: {len(V_list)}")
    print(f"Volume statistics - Mean: {np.mean(V_list):.2f}, Std: {np.std(V_list):.2f}, Min: {np.min(V_list):.2f}, Max: {np.max(V_list):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Droplet volume prediction from edge-detected images')
    parser.add_argument('--horizontal_image_dir', type=str, 
                        default="../data/edge_detected_horizontal/fused",
                        help='Directory containing horizontal edge-detected images')
    parser.add_argument('--output_dir', type=str, default="./_results",
                        help='Directory to save output results')
    parser.add_argument('--image_threshold', type=int, default=150,
                        help='Image threshold for binarization (default: 150)')
    parser.add_argument('--distance_threshold', type=float, default=10.0,
                        help='Distance threshold for hierarchical clustering (default: 10.0)')
    parser.add_argument('--sample_step', type=int, default=1,
                        help='Step size of sampling images for calculating y_min(default: 1, i.e., every image)')
    parser.add_argument('--save_progress', action='store_true', default=True,
                        help='Save progress images showing detection steps for each image')
    
    args = parser.parse_args()
    
    main(args)