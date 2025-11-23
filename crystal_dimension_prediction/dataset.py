import numpy as np
from torch.utils.data import Dataset
from dimension_config import ALPHA, w_min, w_max, l_min, l_max, theta_min, theta_max, phi_min, phi_max, gamma_min, gamma_max
import sys
sys.path.append("../simulation")
from lysozyme import get_augmented_image
import torch
class RegressionDataset(Dataset):
    def __init__(self, dataset_size=10000, img_size=380, transform=None):
        self.w_min = w_min
        self.w_max = w_max
        self.l_min = l_min
        self.l_max = l_max
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.phi_min = phi_min
        self.phi_max = phi_max
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max        
        
        self.dataset_size = dataset_size
        self.transform = transform
        self.img_size = img_size
        self.ALPHA = ALPHA
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        while True:
            W = np.random.uniform(self.w_min, self.w_max)
            L = np.random.uniform(self.l_min, W+1)
            theta = np.random.uniform(self.theta_min, self.theta_max)
            phi = np.random.uniform(self.phi_min, self.phi_max)
            gamma = np.random.uniform(self.gamma_min, self.gamma_max)

            image_original1, image_aug1 = get_augmented_image(W, L, theta, phi, gamma, alpha=ALPHA, img_size=self.img_size, additive_pos_noise=False)
            image_original2, image_aug2 = get_augmented_image(W, L, theta, phi, gamma, alpha=ALPHA, img_size=self.img_size, additive_pos_noise=False)
            if image_aug1 is not None and image_aug2 is not None:
                break

        use_original_ratio = 0.01
        if np.random.uniform(0, 1) < use_original_ratio:
            image_aug1 = image_original1

        image_aug1 = np.expand_dims(image_aug1, axis=2).astype(np.uint8)
        image_aug1 = self.transform(image_aug1)
        return image_aug1, torch.FloatTensor([W / self.w_max]), torch.FloatTensor([L / self.l_max]), torch.FloatTensor([theta / self.theta_max]), torch.FloatTensor(np.array([np.sin(phi), np.cos(phi)])), torch.FloatTensor([gamma / self.gamma_max])

class RealTestDataset(Dataset):
    def __init__(self, image_list, img_size=380, transform=None):
        """_summary_

        Args:
            image_list (_type_): image should be a list of numpy array with shape of (H, W)
            transform (_type_, optional): _description_. Defaults to None.
        """
        self.image_list = image_list
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # Load and preprocess images
        img = self.image_list[idx]
        img = np.expand_dims(img, axis=2).astype(np.uint8)
        return self.transform(img)
