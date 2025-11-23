import numpy as np
from scipy.spatial import ConvexHull
import cv2

def rotate(vertices, theta, phi, gamma):   
    rotation_tilt = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    rotation_rot = np.array([
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi), np.cos(phi), 0],
        [0, 0, 1]
    ])
    
    rotation_matrix = np.dot(rotation_rot, rotation_tilt)
    tilted_vertices = np.dot(rotation_matrix, vertices.T).T
    rotated_vertices = rotate_around_line(tilted_vertices, tilted_vertices[0], tilted_vertices[9], gamma)    

    return rotated_vertices

def rotate_around_line(points, axis_point1, axis_point2, angle):
    axis = np.array(axis_point2) - np.array(axis_point1)
    
    # Normalize the rotation axis
    axis = axis / np.linalg.norm(axis)
    
    # Compute the rotation matrix using Rodrigues' rotation formula
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    
    I = np.eye(3)
    R = I + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    
    # Translate the points to the origin
    points = np.array(points) - np.array(axis_point1)
    
    # Apply the rotation matrix to each point
    rotated_points = np.dot(R, points.T).T
    
    # Translate the points back
    rotated_points = rotated_points + np.array(axis_point1)
    
    return rotated_points

def get_lysozyme_3d_coords(W, L, theta=0, phi=0, gamma=0, alpha=0.326297):
    vertices = np.array([
        [0, 0, (L+W*np.tan(alpha))/2], #v1 : 0 - 9
        [W/2, 0, L/2], #v2 : 1 - 12
        [0, W/2, L/2], #v3 : 2 - 13
        [-W/2, 0, L/2], #v4 : 3 - 10
        [0, -W/2, L/2], #v5 : 4 - 11
        [W/2, W/2, (L-W*np.tan(alpha))/2], #v6 : 5 - 16
        [-W/2, W/2, (L-W*np.tan(alpha))/2], #v7 : 6 - 17
        [-W/2, -W/2, (L-W*np.tan(alpha))/2], #v8 : 7 - 14
        [W/2, -W/2, (L-W*np.tan(alpha))/2], #v9 : 8 - 15
        [0, 0, - (L+W*np.tan(alpha))/2], #v-1 : 9 - 0
        [W/2, 0, - L/2], #v-2 : 10 - 3
        [0, W/2, -L/2], #v-3 : 11 - 4
        [-W/2, 0, -L/2], #v-4 : 12 - 1
        [0, -W/2, -L/2], #v-5 : 13 - 2
        [W/2, W/2, -(L-W*np.tan(alpha))/2], #v-6 : 14 - 7
        [-W/2, W/2, -(L-W*np.tan(alpha))/2], #v-7 : 15 - 8
        [-W/2, -W/2, -(L-W*np.tan(alpha))/2], #v-8 : 16 - 5
        [W/2, -W/2, -(L-W*np.tan(alpha))/2], #v-9 : 17 - 6
    ])
    if vertices[5][2] < vertices[14][2]:
        return None
    rotated_vertices = rotate(vertices, theta, phi, gamma)
    return rotated_vertices

opposite_pairs = {0:9, 1:12, 2:13, 3:10, 4:11, 5:16, 6:17, 7:14, 8:15, 9:0, 10:3, 11:4, 12:1, 13:2, 14:7, 15:8, 16:5, 17:6}
def get_lysozome_point_indices_seen_in_xy_plane(vertices_3d, return_hull=False):
    vertices_2d = vertices_3d[:, :2]
    hull_indices = ConvexHull(vertices_2d).vertices
    visible_indices = []
    for i, (x,y,z) in enumerate(vertices_3d):
        if i in hull_indices:
            visible_indices.append(i)
        else:
            if z >= vertices_3d[opposite_pairs[i]][2]:
                visible_indices.append(i)
    if return_hull:
        return visible_indices, hull_indices
    else:
        return visible_indices

def find_outer_points(nd_array):
    # Compute the Convex Hull of the points
    hull = ConvexHull(nd_array)

    # Return the points that form the Convex Hull
    return nd_array[hull.vertices]

line_indexes = [
        [0, 1], [0, 2], [0, 3], [0, 4], 
        [1, 5], [5, 2], [2, 6], [6, 3], [3, 7], [7, 4], [4, 8], [8, 1],
        [5, 14], [6, 15], [7, 16], [8, 17], 
        [9, 10], [9, 11], [9, 12], [9, 13], 
        [10, 14], [14, 11], [11, 15], [15, 12], [12, 16], [16, 13], [13, 17], [17, 10],
    ]

def get_node_pos(start_pos, end_pos, ratio):
    pos_added = start_pos * ratio + end_pos * (1 - ratio)
    return pos_added

def get_augmented_image(W, L, theta, phi, gamma, alpha=0.326297, img_size=380, additive_pos_noise=False,include_image_aug=True, return_boundary=False):
    coords_3d = get_lysozyme_3d_coords(W, L, theta, phi, gamma, alpha=alpha)
    if coords_3d is not None:
        coords_3d += img_size//2
    else:
        if return_boundary:
            return None, None, None
        else:
            return None, None    

    # 0.0. Alpha Augmentation
    alpha_noise_ratio = 0.5
    if np.random.uniform(0, 1) < alpha_noise_ratio:
        while True:
            alpha_aug = alpha + np.random.normal(0, alpha*0.5)
            coords_3d_aug = get_lysozyme_3d_coords(W, L, theta, phi, gamma, alpha=alpha_aug)
            if coords_3d_aug is not None:
                coords_3d_aug += img_size//2
                break
    else:
        coords_3d_aug = coords_3d.copy()

    indices_all = list(range((coords_3d.shape[0])))
    max_index = max(indices_all)
    indices_seen, indices_hull = get_lysozome_point_indices_seen_in_xy_plane(coords_3d, return_hull=True)

    # 0. Draw the original image
    image_original = np.zeros((img_size, img_size))
    for start, end in line_indexes:
        if (start not in indices_seen) or (end not in indices_seen):
            continue

        pt1 = (np.round(coords_3d[start][:2]).astype(int))
        pt2 = (np.round(coords_3d[end][:2]).astype(int))
        cv2.line(image_original, pt1, pt2, 255, 3)

    # 1. Move the points
    node_pos_noise_ratio = 0.3 + int(additive_pos_noise) * 0.3
    for i, pos in enumerate(coords_3d_aug):
        if np.random.uniform(0, 1) < node_pos_noise_ratio:
            if i in indices_hull:
                coords_3d_aug[i] += np.random.uniform(0, W * 0.15) #np.random.normal(0, min([W, L])/50+3+int(additive_pos_noise)*2, 3)
            else:
                coords_3d_aug[i] += np.random.uniform(0, W * 0.15) #np.random.normal(0, min([W, L])/50+1+int(additive_pos_noise)*2, 3)
    # 1-1. Move the center
    center_noise_ratio = 0.6
    if np.random.uniform(0, 1) < center_noise_ratio:
        coords_3d_aug += np.random.normal(0, 10, 3)

    # 2. Remove some nodes
    node_remove_ratio = 0.05
    is_special_case = (np.abs(theta)<1e-2) or (np.abs(theta-np.pi/2)<1e-2) or (np.abs(gamma)<1e-2) or (np.abs(gamma-np.pi/2)<1e-2)
    if is_special_case:
        node_remove_ratio-=0.3
    
    indices_seen_copy = indices_seen.copy()
    for i in indices_seen_copy:
        if i in indices_hull:
            continue
        if np.random.uniform(0, 1) < node_remove_ratio:
            indices_seen.remove(i)

    # 3. Define the edges drawn
    edge_remove_ratio = 0.1
    line_indexed_drawn = []
    for start, end in line_indexes:
        if is_special_case:
            if (start not in indices_seen) or (end not in indices_seen):
                continue
            elif (start in indices_hull) or (end in indices_hull):
                line_indexed_drawn.append([start, end])
            elif np.random.uniform(0, 1) > (edge_remove_ratio-0.3):
                line_indexed_drawn.append([start, end])
        else:
            if (start not in indices_seen) or (end not in indices_seen):
                continue
            elif (start in indices_hull) and (end in indices_hull):
                line_indexed_drawn.append([start, end])
            elif np.random.uniform(0, 1) > edge_remove_ratio:
                line_indexed_drawn.append([start, end])
        
        # line_indexed_drawn.append([start, end])

    # 3-1. Add some edges
    edge_add_ratio = 0.05
    for _ in range(3):
        if np.random.uniform(0, 1) < edge_add_ratio:
            start = np.random.randint(0, np.max(line_indexes)+1)
            end = np.random.randint(0, np.max(line_indexes)+1)
            if (start != end) and ([start, end] not in line_indexed_drawn) and ([end, start] not in line_indexed_drawn):
                line_indexed_drawn.append([start, end])
    
    # 4. Add nodes(=line) on the edges
    node_add_ratio = 1
    drawn_edge_remove_ratio = 0.05
    line_indexed_drawn_copy = line_indexed_drawn.copy()
    for start, end in line_indexed_drawn_copy:
        if np.random.uniform(0, 1) < node_add_ratio:
            n_added_points = np.random.randint(1, max(10, W//10)) + int(additive_pos_noise) * 5
            ratio_list = np.repeat(np.expand_dims(np.sort(np.random.uniform(0, 1, n_added_points))[::-1],axis=0), 3, axis=0).T
            coords_added_points = get_node_pos(coords_3d_aug[start], coords_3d_aug[end], ratio_list)
            coords_added_points += np.random.randint(-(W//50+1+int(additive_pos_noise)*2), W//50+1+int(additive_pos_noise)*2, coords_added_points.shape)
            
            line_indexed_drawn.remove([start, end])
            
            new_s = start
            for new_idx, new_pos in zip(range(max_index+1, max_index+1+n_added_points), coords_added_points):
                if np.random.uniform(0, 1) < drawn_edge_remove_ratio:
                    new_s = new_idx
                else:
                    line_indexed_drawn.append([new_s, new_idx])
                    new_s = new_idx
            line_indexed_drawn.append([new_s, end])    
            
            coords_3d_aug = np.append(coords_3d_aug, coords_added_points, axis=0)
            max_index += n_added_points
        

    # Draw the augmented image
    image_aug = np.zeros((img_size, img_size))
    for start, end in line_indexed_drawn:
        pt1 = (np.round(coords_3d_aug[start][:2]).astype(int))
        pt2 = (np.round(coords_3d_aug[end][:2]).astype(int))
        line_color = 255 #np.random.randint(100, 255)
        linewidth = np.random.randint(int((4 - 1)/(300 - 0) * (W - 0) + 1 + int(additive_pos_noise) * 2), int((15 - 4)/(300 - 0) * (W - 0) + 4 + int(additive_pos_noise) * 2))
        try:
            cv2.line(image_aug, pt1, pt2, line_color, linewidth, lineType=np.random.choice([-1, 4, 8, 16]))
        except:
            cv2.line(image_aug, pt1, pt2, line_color, linewidth, lineType=4)

    if return_boundary:
        boundary_points = []
        for start, end in line_indexed_drawn:
            pt1 = (np.round(coords_3d_aug[start][:2]).astype(int))
            boundary_points.append(np.array(pt1))
        boundary_points = find_outer_points(np.array(boundary_points))
    
    if include_image_aug:
        # 6. Image Augmentation
        # 6-1. gaussian blur
        gaussian_blur_ratio = 0.1
        if W <= 5:
            gaussian_blur_ratio = 0.01
        if np.random.uniform(0, 1) < gaussian_blur_ratio:
            image_aug = cv2.GaussianBlur(image_aug, (np.random.randint(1,4)*2-1, np.random.randint(1,4)*2-1), 0)
        # 6-2. blur
        blur_ratio = 0.1
        if np.random.uniform(0, 1) < blur_ratio:
            image_aug = cv2.blur(image_aug, (np.random.randint(1,4)*2-1, np.random.randint(1,4)*2-1), 0)
        # 6-3. brightness
        brigntness_ratio = 0.1
        if np.random.uniform(0, 1) < brigntness_ratio:
            value = np.random.randint(-50, 50)
            image_aug = np.clip(image_aug + value, 0, 255)
        # 6-4. make pixel random
        make_pixel_zero_ratio = 0.1
        if W <= 5:
            make_pixel_zero_ratio = 0.01
        if np.random.uniform(0, 1) < make_pixel_zero_ratio:

            total_pixels = image_aug.size
            num_pixels_to_zero = int((np.random.randint(0,20) / 100.0) * total_pixels)
            # Generate random indices
            indices_to_zero = np.random.choice(range(total_pixels), num_pixels_to_zero, replace=False)
            # Convert the 1D indices to 2D indices if necessary
            if image_aug.ndim > 1:
                rows, cols = image_aug.shape[:2]
                indices_to_zero = np.unravel_index(indices_to_zero, (rows, cols))
            # Set the selected pixels to zero
            image_aug[indices_to_zero] = np.random.randint(0, 255, size=indices_to_zero[0].shape)

        # 6-5. coarsely dropout
        coarsely_dropout_ratio = 0.1
        if W <= 5:
            coarsely_dropout_ratio = 0
        if np.random.uniform(0, 1) < coarsely_dropout_ratio:
            # Calculate the number of squares to drop
            dropout_ratio = np.random.uniform(0, 0.2)
            max_square_size = np.random.randint(2, W)
            num_squares = int(dropout_ratio * (image_aug.shape[0] * image_aug.shape[1]) / (max_square_size ** 2))
            for _ in range(num_squares):
                # Randomly select the top-left corner of the square
                square_size_x = np.random.randint(1, max_square_size)
                square_size_y = np.random.randint(1, max_square_size)
                x = np.random.randint(0, image_aug.shape[1] - square_size_x)
                y = np.random.randint(0, image_aug.shape[0] - square_size_y)
                
                # Set the pixels within the square to zero
                image_aug[y:y+square_size_y, x:x+square_size_x] = np.random.randint(0, 255, (square_size_y, square_size_x))

        # 6-6. additive gaussian noise
        additive_gaussian_noise_ratio = 0.1
        if W <= 5:
            additive_gaussian_noise_ratio = 0.01
        if np.random.uniform(0, 1) < additive_gaussian_noise_ratio:
            noise_std = np.random.randint(0, 80)
            noise = np.random.normal(0, noise_std, image_aug.shape)
            image_aug = np.clip(image_aug + noise, 0, 255)  # Ensure values stay within valid range    
        # 6-7. resize the image
        resize_ratio = 0.2
        if np.random.uniform(0, 1) < resize_ratio:
            resize_ratio = np.random.uniform(0.25, 4)
            new_size = (int(img_size * resize_ratio), int(img_size * resize_ratio))
            image_aug = cv2.resize(image_aug, new_size)
            image_aug = cv2.resize(image_aug, (img_size, img_size))
        # 6-8. only pixels >= 240
        pixel_thres_ratio = 0 
        pixel_thres_min = 50
        if np.random.uniform(0, 1) < pixel_thres_ratio:
            image_aug[image_aug <= pixel_thres_min] = 0
            image_aug[image_aug > pixel_thres_min] = 255
    
    if return_boundary:
        return image_original, image_aug, boundary_points
    else:
        return image_original, image_aug

def get_simulation_image(W, L, theta, phi, gamma, alpha=0.326297, img_size=380):
    coords_3d = get_lysozyme_3d_coords(W, L, theta, phi, gamma, alpha=alpha)
    if coords_3d is None:
        return np.zeros((img_size, img_size))
    
    coords_3d += img_size//2
    indices_seen, indices_hull = get_lysozome_point_indices_seen_in_xy_plane(coords_3d, return_hull=True)
    
    # 0. Draw the original image
    image = np.zeros((img_size, img_size))
    for start, end in line_indexes:
        if (start not in indices_seen) or (end not in indices_seen):
            continue

        pt1 = (np.round(coords_3d[start][:2]).astype(int))
        pt2 = (np.round(coords_3d[end][:2]).astype(int))
        cv2.line(image, pt1, pt2, 255, 1)
    return image