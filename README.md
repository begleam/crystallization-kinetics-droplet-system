# Crystal Kinetics Droplet System

A comprehensive system for crystal detection, dimension prediction, and droplet volume estimation from microscopy images.

## Overview

This project provides a complete pipeline for analyzing crystal kinetics in droplet systems:

1. **Edge Detection**: Detects edges in horizontal and vertical crystal images
2. **Crystal Detection**: Detects and segments crystals in images using Detectron2
3. **Dimension Prediction**: Predicts crystal dimensions (W, L, theta, phi, gamma) from images
4. **Droplet Volume Prediction**: Estimates droplet volume from edge-detected images

## Installation

### Requirements

```bash
pip install torch torchvision
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
pip install opencv-python numpy scipy matplotlib pandas tqdm scikit-learn shapely parmap
```

### Clone Repository

```bash
git clone https://github.com/begleam/crystallization-kinetics-droplet-system.git
cd crystal-kinetics-droplet-system
```

## Data and Checkpoints Download

The `data/` and `checkpoints/` folders are not included in the repository due to their large size. You need to download them separately.

Ensure the folder structure matches:
   ```
   crystal-kinetics-droplet-system/
   ├── data/
   │   ├── horizontal/
   │   ├── vertical/
   │   ├── edge_detected_horizontal/
   │   ├── edge_detected_vertical/
   │   └── ...
   └── checkpoints/
       ├── crystal_detection.pth
       ├── crystal_dimension_prediction.pth
       └── ...
   ```

### Download the data and checkpoints from the following sources:

- **data folder**: [Download Link](https://drive.google.com/file/d/1rR4NKN7yzDZHXTU1K2GQISrDNLWMGuvz/view?usp=drive_link)
- **checkpoints folder**: [Download Link](https://drive.google.com/file/d/1e4b8_c4FDiEKT3FE_cxElZf9VXxej-jh/view?usp=drive_link)

After downloading, extract them to the project root:

```bash
unzip data.zip
unzip checkpoints.zip
```

## Quick Start (Demo)

### Step 1: Edge Detection

First, run edge detection on both horizontal and vertical images:

**For horizontal images:**
```bash
python edge_detection/edge_detect.py \
    --checkpoint_path ../checkpoints/DexiNed_10_model.pth \
    --img_load_dir ../data/horizontal \
    --output_dir ../data/edge_detected_horizontal \
    --img_width 768 \
    --img_height 480 \
    --batch_size 16 \
    --num_workers 4
```

**For vertical images:**
```bash
python edge_detection/edge_detect.py \
    --checkpoint_path ../checkpoints/DexiNed_10_model.pth \
    --img_load_dir ../data/vertical \
    --output_dir ../data/edge_detected_vertical \
    --img_width 3008 \
    --img_height 2000 \
    --batch_size 16 \
    --num_workers 4
```

### Step 2: Droplet Volume Prediction

After edge detection, run volume prediction:

```bash
python droplet_volume_prediction/inference.py \
    --horizontal_image_dir ../data/edge_detected_horizontal/fused \
    --output_dir ./droplet_volume_prediction/_results \
    --image_threshold 150 \
    --distance_threshold 10.0 \
    --save_progress
```

Results will be saved in `./droplet_volume_prediction/_results/volume_data.csv`.

## Module Details

### 1. Edge Detection (`edge_detection/`)

Detects edges in crystal images using DexiNed model.

**Usage:**
```bash
python edge_detection/edge_detect.py \
    --checkpoint_path <path_to_model> \
    --img_load_dir <input_image_directory> \
    --output_dir <output_directory> \
    --img_width <image_width> \
    --img_height <image_height> \
    --batch_size <batch_size> \
    --num_workers <num_workers>
```

### 2. Crystal Detection (`crystal_detection/`)

Detects and segments crystals in images using Detectron2.

#### Training

1. **Prepare dataset:**
```bash
python crystal_detection/data_prepare.py \
    --save_dir ../data/simulation_detection \
    --n_train_images 500 \
    --n_val_images 100 \
    --n_processes 10
```

2. **Train model:**
```bash
python crystal_detection/train.py \
    --load_dir ../data/simulation_detection \
    --output_save_dir ../checkpoints/crystal_detection \
    --batch_size 128 \
    --dataset_size 10000 \
    --img_size 380 \
    --n_epochs 120 \
    --num_workers 8 \
    --ims_per_batch 24 \
    --base_lr 0.00025 \
    --max_iter 300000 \
    --batch_size_per_image 512
```

#### Inference

Using pretrained model:
```bash
python crystal_detection/inference.py \
    --target_edge_image ../data/edge_detected_vertical/fused/0.png \
    --model_weights_path ../checkpoints/crystal_detection.pth \
    --train_data_path ../data/simulation_detection \
    --max_area 90000 \
    --min_area 10 \
    --detectron2_thres 0.02 \
    --print_progress \
    --figure_save
```

### 3. Crystal Dimension Prediction (`crystal_dimension_prediction/`)

Predicts crystal dimensions (W, L, theta, phi, gamma) from images.

#### Training

```bash
python crystal_dimension_prediction/train.py \
    --test_image_dir ../data/crystal_images \
    --model_save_folder ../checkpoints/dimension_prediction \
    --batch_size 128 \
    --dataset_size 10000 \
    --img_size 380 \
    --n_epochs 120 \
    --test_per_epoch 10 \
    --t_0 40 \
    --t_mult 2
```

#### Inference

Using pretrained model:
```bash
python crystal_dimension_prediction/inference.py \
    --model_weights_path ../checkpoints/crystal_dimension_prediction.pth \
    --test_image_dir ../data/crystal_images \
    --save_dir ./crystal_dimension_prediction/_results \
    --img_size 380 \
    --batch_size 8 \
    --device cuda \
    --show_progress
```

Results will be saved in `./crystal_dimension_prediction/_results/dimension_predictions.csv` with columns: `img_path`, `W`, `L`, `theta`, `phi`, `gamma`.

### 4. Droplet Volume Prediction (`droplet_volume_prediction/`)

Estimates droplet volume from edge-detected horizontal images.

**Usage:**
```bash
python droplet_volume_prediction/inference.py \
    --horizontal_image_dir ../data/edge_detected_horizontal/fused \
    --output_dir ./droplet_volume_prediction/_results \
    --image_threshold 150 \
    --distance_threshold 10.0 \
    --sample_step 1 \
    --save_progress
```

Results will be saved in `./droplet_volume_prediction/_results/volume_data.csv` with columns: `img_path`, `volume`.

## Project Structure

```
crystal-kinetics-droplet-system/
├── data/                          # Data directory (download separately)
│   ├── horizontal/                # Horizontal crystal images
│   ├── vertical/                  # Vertical crystal images
│   ├── edge_detected_horizontal/  # Edge-detected horizontal images
│   ├── edge_detected_vertical/    # Edge-detected vertical images
│   └── simulation_detection/      # Simulated detection dataset
├── checkpoints/                   # Model checkpoints (download separately)
│   ├── crystal_detection.pth
│   ├── crystal_dimension_prediction.pth
│   └── DexiNed_10_model.pth
├── edge_detection/                # Edge detection module
├── crystal_detection/             # Crystal detection module
├── crystal_dimension_prediction/  # Dimension prediction module
├── droplet_volume_prediction/     # Volume prediction module
└── simulation/                    # Simulation utilities
```

## Output Files

- **Edge Detection**: Edge-detected images saved to specified output directory
- **Crystal Detection**: Detection results and visualization images saved to `_results/` directory
- **Dimension Prediction**: 
  - Visualization images: `_results/{image_name}.png`
  - CSV file: `_results/dimension_predictions.csv`
- **Volume Prediction**:
  - Plot: `_results/droplet_volume_prediction.png`
  - CSV file: `_results/volume_data.csv`
  - Progress images (if `--save_progress`): `_results/detected_points_{image_name}.png`

## Notes

- Make sure to run edge detection first before running droplet volume prediction
- For crystal detection inference, ensure the training data path is correct for dataset registration
- All paths in the commands are relative to the project root
- GPU is recommended for faster processing, but CPU is also supported

## License

[Add your license here]

## Citation

[Add citation information if applicable]

