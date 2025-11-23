import numpy as np
import cv2

def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)
    total_pixels = image.size

    # Add salt noise
    num_salt = np.ceil(salt_prob * total_pixels)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = np.random.randint(100, 255, int(num_salt))

    # Add pepper noise
    num_pepper = np.ceil(pepper_prob * total_pixels)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0
    return noisy_image
    
def aug_image(image_aug):
    # 6. Image Augmentation
    # 6-1. gaussian blur
    gaussian_blur_ratio = 0.1
    if np.random.uniform(0, 1) < gaussian_blur_ratio:
        image_aug = cv2.GaussianBlur(image_aug, (np.random.randint(1,4)*2-1, np.random.randint(1,4)*2-1), 0)
    # 6-2. blur
    blur_ratio = 0.1
    if np.random.uniform(0, 1) < blur_ratio:
        image_aug = cv2.blur(image_aug, (np.random.randint(1,4)*2-1, np.random.randint(1,4)*2-1), 0)
    # 6-3. brightness
    brigntness_ratio = 0.1
    if np.random.uniform(0, 1) < brigntness_ratio:
        value = np.random.randint(-20, 20)
        image_aug = np.clip(image_aug + value, 0, 255)
    # 6-4. make pixel random
    make_pixel_zero_ratio = 0.1
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

    # 6-6. additive gaussian noise
    additive_gaussian_noise_ratio = 0.1
    if np.random.uniform(0, 1) < additive_gaussian_noise_ratio:
        noise_std = np.random.randint(0, 80)
        noise = np.random.normal(0, noise_std, image_aug.shape)
        image_aug = np.clip(image_aug + noise, 0, 255)  # Ensure values stay within valid range    

    # 6-8. only pixels >= 240
    pixel_thres_ratio = 0 #TODO: only pixels >=pixel_thres_min 얘는 resize와 합쳐지면 좀 이상해짐
    pixel_thres_min = 50
    if np.random.uniform(0, 1) < pixel_thres_ratio:
        image_aug[image_aug <= pixel_thres_min] = 0
        image_aug[image_aug > pixel_thres_min] = 255
    
    # 6-9. salt and pepper noise
    salt_pepper_ratio = 0.1
    if np.random.uniform(0, 1) < salt_pepper_ratio:
        salt_prob = np.random.uniform(0.01, 0.1)  # Probability of salt noise
        pepper_prob = np.random.uniform(0.01, 0.1)  # Probability of pepper noise
        image_aug = add_salt_and_pepper_noise(image_aug, salt_prob, pepper_prob)
    return image_aug