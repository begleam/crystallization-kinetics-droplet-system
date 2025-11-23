import numpy as np
import sys
from model import Regressor
from dataset import RegressionDataset
from loss_fn import MultiTaskMSELoss, get_mae
from inference import test
from dimension_config import (
    w_max, l_max,
    theta_max,
    gamma_max
)
sys.path.append("../simulation")
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchvision import transforms
import os
import time
from tqdm import tqdm
import argparse

def train(model, dataloader, criterion, optimizer, scheduler, num_epochs=10, test_per_epoch=20, model_save_folder="../checkpoints/dimension_prediction", img_size=380, transform=None, batch_size=64, use_parallel=False, device='cuda', test_image_dir=None):
    for epoch in range(1, num_epochs+1):
        model.train()
        t0 = time.time()
        
        total_loss = 0
        n_total = 0
        total_w_loss = 0
        total_l_loss = 0
        total_theta_loss = 0
        total_phi_loss = 0
        total_gamma_loss = 0
        
        for data in tqdm(dataloader, desc=f'Epoch {epoch}/{num_epochs}', disable=False):
            img, label0, label1, label2, label3, label4 = data
            img, label0, label1, label2, label3, label4 = img.to(device), label0.to(device), label1.to(device), label2.to(device), label3.to(device), label4.to(device)
            
            optimizer.zero_grad()
            preds0, preds1, preds2, preds3, preds4 = model(img)
            
            loss, w_loss, l_loss, theta_loss, phi_loss, gamma_loss = criterion(preds0, preds1, preds2, preds3, preds4, label0, label1, label2, label3, label4)
            w_mae, l_mae, theta_mae, phi_mae, gamma_mae = get_mae(preds0 * w_max, label0 * w_max), get_mae(preds1 * l_max, label1 * l_max), get_mae(preds2 * theta_max, label2 * theta_max), get_mae(preds3, label3), get_mae(preds4 * gamma_max, label4 * gamma_max)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_total += img.size(0)
            total_w_loss += w_mae
            total_l_loss += l_mae
            total_theta_loss += theta_mae
            total_phi_loss += phi_mae
            total_gamma_loss += gamma_mae
            
        scheduler.step()
        print(f'Total MSE - {total_loss/n_total:.4f}, MAE - W: {total_w_loss/n_total:.4f}, L: {total_l_loss/n_total:.4f}, Theta: {total_theta_loss/n_total:.4f} pi, Phi: {total_phi_loss/n_total:.4f} pi, Gamma: {total_gamma_loss/n_total:.4f} pi, Time: {time.time()-t0:.2f}s')
        if (epoch) % test_per_epoch == 0:
            del img, label0, label1, label2, label3, label4, preds0, preds1, preds2, preds3, preds4
            if not os.path.exists(model_save_folder): os.makedirs(model_save_folder)
            if use_parallel: state_dict = model.module.state_dict()
            else: state_dict = model.state_dict()
            torch.save(state_dict, f'{model_save_folder}/shape_regressor_epoch_{epoch}.pth')
            torch.save(state_dict, f'{model_save_folder}/shape_regressor_last.pth')
            
            if test_image_dir is not None:
                results_figures_save_path = f'{model_save_folder}/dimension_regression_results'
                if not os.path.exists(results_figures_save_path): os.mkdir(results_figures_save_path)
                if not os.path.exists(f'{results_figures_save_path}/epoch_{epoch}'): os.mkdir(f'{results_figures_save_path}/epoch_{epoch}')
                test_image_path_list = [os.path.join(test_image_dir, f) for f in os.listdir(test_image_dir)]
                test(model, test_image_path_list, f'{results_figures_save_path}/epoch_{epoch}', img_size=img_size, transform=transform, batch_size=batch_size,show_progress=False, device=device)



def main(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.img_size, args.img_size)),
    ])

    dataset = RegressionDataset(dataset_size=args.dataset_size, img_size=args.img_size, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    use_parallel = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    emb_dim = 1792

    model = Regressor(emb_dim=emb_dim)

    if use_parallel:
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion = MultiTaskMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.t_0, T_mult=args.t_mult)

    train(
        model,
        dataloader,
        criterion,
        optimizer,
        scheduler,
        num_epochs=args.n_epochs,
        test_per_epoch=args.test_per_epoch,
        model_save_folder=args.model_save_folder,
        img_size=args.img_size,
        transform=transform,
        batch_size=args.batch_size,
        use_parallel=use_parallel,
        device=device,
        test_image_dir=args.test_image_dir
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train crystal dimension prediction model')
    parser.add_argument('--test_image_dir', type=str, default="../data/crystal_images",
                        help='Directory containing test images')
    parser.add_argument('--model_save_folder', type=str, default="../checkpoints/dimension_prediction",
                        help='Directory to save model checkpoints')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--dataset_size', type=int, default=10000,
                        help='Size of the training dataset')
    parser.add_argument('--img_size', type=int, default=380,
                        help='Image size for training')
    parser.add_argument('--n_epochs', type=int, default=120,
                        help='Number of training epochs')
    parser.add_argument('--test_per_epoch', type=int, default=10,
                        help='Test every N epochs')
    parser.add_argument('--t_0', type=int, default=40,
                        help='T_0 parameter for CosineAnnealingWarmRestarts scheduler')
    parser.add_argument('--t_mult', type=int, default=2,
                        help='T_mult parameter for CosineAnnealingWarmRestarts scheduler')
    
    args = parser.parse_args()
    
    main(args)