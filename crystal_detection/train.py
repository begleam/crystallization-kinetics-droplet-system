import torch, detectron2
import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer


def main(args):
    # Register datasets
    train_json_path = os.path.join(args.load_dir, "train_data.json")
    train_image_dir = os.path.join(args.load_dir, "train")
    val_json_path = os.path.join(args.load_dir, "val_data.json")
    val_image_dir = os.path.join(args.load_dir, "val")
    
    register_coco_instances("my_dataset_train", {}, train_json_path, train_image_dir)
    register_coco_instances("my_dataset_val", {}, val_json_path, val_image_dir)
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    
    cfg.DATASETS.TRAIN = ("my_dataset_train")
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = args.ims_per_batch  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = args.base_lr  # pick a good LR
    cfg.SOLVER.MAX_ITER = args.max_iter  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.batch_size_per_image   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.OUTPUT_DIR = args.output_save_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train crystal detection model using Detectron2')
    parser.add_argument('--load_dir', type=str, default="../data/simulation_detection",
                        help='Directory containing train_data.json, val_data.json, train/, and val/ folders')
    parser.add_argument('--output_save_dir', type=str, default="../checkpoints/detection/",
                        help='Directory to save model checkpoints and outputs')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--ims_per_batch', type=int, default=24,
                        help='Images per batch (real batch size)')
    parser.add_argument('--base_lr', type=float, default=0.00025,
                        help='Base learning rate')
    parser.add_argument('--max_iter', type=int, default=30,
                        help='Maximum number of training iterations')
    parser.add_argument('--batch_size_per_image', type=int, default=512,
                        help='RoIHead batch size per image')
    
    args = parser.parse_args()
    
    main(args)