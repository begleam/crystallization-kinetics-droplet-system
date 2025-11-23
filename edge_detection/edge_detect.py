import torch
from utils import DexiNed, CrystalTestDataset, get_edges
from torch.utils.data import DataLoader
import os
import argparse


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    
    model = DexiNed().to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    
    img_path_list = [os.path.join(args.img_load_dir, f) for f in os.listdir(args.img_load_dir)]
    test_dataset = CrystalTestDataset(img_path_list=img_path_list,
                                      img_height=args.img_height,
                                      img_width=args.img_width,
                                      mean_bgr=[103.939, 116.779, 123.68],
                                      )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    get_edges(model, test_loader, device, args.output_dir, show_progress=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Edge detection using DexiNed model')
    parser.add_argument('--checkpoint_path', type=str, default="../checkpoints/DexiNed_10_model.pth",
                        help='Path to model checkpoint')
    parser.add_argument('--img_load_dir', type=str, default="../data/horizontal",
                        help='Directory containing input images, "../data/vertical" for vertical images')
    parser.add_argument('--output_dir', type=str, default="../data/edge_detected_horizontal",
                        help='Directory to save output edge detection results, "../data/edge_detected_vertical" for vertical images')
    parser.add_argument('--img_width', type=int, default=768,
                        help='Image width (768 for horizontal images, 3008 for vertical images)')
    parser.add_argument('--img_height', type=int, default=480,
                        help='Image height (480 for horizontal images, 2000 for vertical images)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for data loading')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    
    args = parser.parse_args()
    
    main(args)