import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt
from dataset import RealTestDataset
from dimension_config import w_max, l_max, theta_max, gamma_max
from model import Regressor
from torchvision import transforms
import torch.nn as nn
import os
import argparse
import pandas as pd
import sys
sys.path.append("../simulation")
from lysozyme import get_simulation_image

def get_predictions_from_dataloader(model, dataloader, device='cuda', show_progress=True, desc='Predicting..'):
    w_predictions = []
    l_predictions = []
    theta_predictions = []
    phi_predictions = []
    gamma_predictions = []
    model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader, desc=desc, disable=not show_progress):
            img = data.to(device)
            preds0, preds1, preds2, preds3, preds4 = model(img)
            w_predictions.append(preds0.detach().cpu().numpy())
            l_predictions.append(preds1.detach().cpu().numpy())
            theta_predictions.append(preds2.detach().cpu().numpy())
            phi_predictions.append(preds3.detach().cpu().numpy())
            gamma_predictions.append(preds4.detach().cpu().numpy())
    phi_predictions = np.concatenate(phi_predictions, axis=0)
    phi_predictions = np.arctan2(phi_predictions[:, 0], phi_predictions[:, 1])
    return np.concatenate(w_predictions).reshape(-1), np.concatenate(l_predictions).reshape(-1), np.concatenate(theta_predictions).reshape(-1), phi_predictions, np.concatenate(gamma_predictions).reshape(-1)

def test(model, test_image_path_list, save_dir, img_size, transform, batch_size=64, show_progress=True, device='cuda'):

    image_list = [cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) for image_path in test_image_path_list]
    image_name_list = [image_path.split('/')[-1].split('.')[0] for image_path in test_image_path_list]
    real_dataset = RealTestDataset(image_list, img_size=img_size, transform=transform)
    real_dataloader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)

    w_predictions, l_predictions, theta_predictions, phi_predictions, gamma_predictions = get_predictions_from_dataloader(model, real_dataloader, device=device, show_progress=show_progress)

    # Store results for CSV
    results = []
    
    for row_idx, (real_img, w_predicted, l_predicted, theta_predicted, phi_predicted, gamma_predicted, image_name) in enumerate(zip(image_list, w_predictions, l_predictions, theta_predictions, phi_predictions, gamma_predictions, image_name_list)):
        n_img_in_fig = 2
        plt.figure(figsize=(5*n_img_in_fig, 5))
        plt.subplot(1, n_img_in_fig, 1)
        plt.imshow(real_img, cmap='gray')
        plt.title('Real')
        plt.axis('off')
        plt.subplot(1, n_img_in_fig, 2)
        
        W, L, theta, gamma = w_predicted * w_max, l_predicted * l_max, theta_predicted * theta_max, gamma_predicted * gamma_max
        phi = phi_predicted
        
        # Store results
        results.append({
            'img_path': test_image_path_list[row_idx],
            'W': W,
            'L': L,
            'theta': theta,
            'phi': phi,
            'gamma': gamma
        })
        
        plt.imshow(get_simulation_image(W, L, theta, phi, gamma, img_size=img_size), cmap='gray')
        plt.title(f'Predicted')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{image_name}.png')
        plt.close()
    
    # Save results to CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(save_dir, 'dimension_predictions.csv')
    df.to_csv(csv_path, index=False)
    print(f"Dimension predictions saved to {csv_path}")


def main(args):
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    
    # Create transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.img_size, args.img_size)),
    ])
    
    # Load model
    emb_dim = 1792
    model = Regressor(emb_dim=emb_dim)
    
    # Handle DataParallel if needed
    if args.use_parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    # Load weights
    if os.path.exists(args.model_weights_path):
        state_dict = torch.load(args.model_weights_path, map_location=device)
        # Handle DataParallel state dict
        if 'module.' in list(state_dict.keys())[0] and not isinstance(model, nn.DataParallel):
            # If saved with DataParallel but loading without
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k.replace('module.', '')] = v
            state_dict = new_state_dict
        elif 'module.' not in list(state_dict.keys())[0] and isinstance(model, nn.DataParallel):
            # If saved without DataParallel but loading with
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict['module.' + k] = v
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        print(f"Model loaded from {args.model_weights_path}")
    else:
        raise FileNotFoundError(f"Model weights not found at {args.model_weights_path}")
    
    model.eval()
    
    # Get test image paths
    if os.path.isdir(args.test_image_dir):
        test_image_path_list = [os.path.join(args.test_image_dir, f) for f in os.listdir(args.test_image_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    else:
        # Assume it's a single file
        test_image_path_list = [args.test_image_dir]
    
    if len(test_image_path_list) == 0:
        raise ValueError(f"No images found in {args.test_image_dir}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Run inference
    test(
        model,
        test_image_path_list,
        args.save_dir,
        args.img_size,
        transform,
        batch_size=args.batch_size,
        show_progress=args.show_progress,
        device=device
    )
    
    print(f"Results saved to {args.save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crystal dimension prediction inference')
    parser.add_argument('--model_weights_path', type=str, default="../checkpoints/crystal_dimension_prediction.pth",
                        help='Path to model weights file (.pth)')
    parser.add_argument('--test_image_dir', type=str, default="../data/crystal_images",
                        help='Directory containing test images or path to a single image')
    parser.add_argument('--save_dir', type=str, default="./_results",
                        help='Directory to save inference results')
    parser.add_argument('--img_size', type=int, default=380,
                        help='Image size for inference')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        choices=['cuda', 'cpu'],
                        help='Device to use for inference')
    parser.add_argument('--use_parallel', action='store_true', default=False,
                        help='Use DataParallel if multiple GPUs available')
    parser.add_argument('--show_progress', action='store_true', default=True,
                        help='Show progress bar during inference')
    
    args = parser.parse_args()
    
    main(args)

