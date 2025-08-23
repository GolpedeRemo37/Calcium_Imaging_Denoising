# 3D Denoising Calcium Submission

## Description

Submission for AI4Life Calcium Imaging Denoising Challenge 2025. Uses Noise2Void with NAFNet to denoise calcium imaging stacks with photon shot and read noise. Self-supervised, preserves dynamics. Docker-ready for Grand Challenge evaluation. (141 characters)

## Overview

This repository contains the submission for the [AI4Life Calcium Imaging Denoising Challenge 2025](https://ai4life-cidc25.grand-challenge.org/). It implements a self-supervised Noise2Void (N2V) denoising approach using the NAFNet architecture, optimized for 3D calcium imaging stacks with noise types like photon shot and read noise. The model preserves spatial-temporal dynamics and is packaged in a Docker container for Grand Challenge evaluation.

**Key Features**:
- Self-supervised training without ground truth using N2V
- Patch-based inference for efficient 3D stack denoising
- Handles three noise types (e.g., A1, B1, C1 stacks)
- Evaluates with PSNR/SSIM metrics on synthetic datasets
- Docker-compatible for Grand Challenge submission

## Requirements

- **Python**: 3.9 or higher
- **Packages** (see requirements.txt):

```text
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
tifffile>=2022.7.28
numpy>=1.21.0
tqdm>=4.64.0
scikit-image>=0.19.0
matplotlib>=3.5.0
torchmetrics>=0.11.0  # Optional for SSIM loss
```

**Hardware**: CPU (minimum, 4 cores, 8GB RAM); NVIDIA GPU recommended (8GB+ VRAM) for training.

**Docker**: Docker Desktop (Windows/Linux) or Docker Engine (Linux). NVIDIA Container Toolkit for GPU support.

**Storage**: 10GB+ free space for outputs.

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd 3D-denoising-calcium-submission
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

For Docker inference:

```bash
docker build -t denoising-calcium .
```

## Directory Structure

```
.
├── Dockerfile                  # Docker image for Grand Challenge inference
├── do_save.sh                  # Script to save model/outputs
├── do_test_run.sh              # Script to build and test Docker locally
├── evaluate.py                 # Custom evaluation script for metrics
├── inference.py                # Grand Challenge inference script
├── NafNet2Void.ipynb           # Jupyter Notebook to train the model
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── N2V_Vanilla/                # Core N2V + NAFNet package
│   ├── best_model_NafNet_Noise2Void.pth  # Trained model weights on the simplest configuration (look on models.py)
│   ├── config.py               # Training configuration
│   ├── dataset.py              # Training/validation datasets
│   ├── losses.py               # Loss functions
│   ├── models.py               # NAFNet model architecture
│   ├── trainer.py              # Training logic
│   ├── utils.py                # Utilities (denoising, metrics)
│   ├── .ipynb_checkpoints/     # Jupyter checkpoints (ignored)
│   └── __pycache__/            # Python cache (ignored)
└── test/                       # Test directories for local Docker runs
    ├── input/
    │   └── interface_0/
    │       ├── inputs.json     # Grand Challenge input metadata
    │       └── images/
    │           └── image-stack-unstructured-noise/
    │               ├── A1_stack_000_009.tif  # Test stack (noise type A)
    │               ├── B1_stack_000_009.tif  # Test stack (noise type B)
    │               └── C1_stack_000_009.tif  # Test stack (noise type C)
    └── output/
        └── interface_0/
            └── images/
                └── image-stack-denoised/
                    ├── A1_stack_000_009.tif  # Denoised output
                    ├── B1_stack_000_009.tif  # Denoised output
                    └── C1_stack_000_009.tif  # Denoised output
```

## Usage

### Training

Train the N2V model with NAFNet:

```bash
python main.py
```

- Configured for medium NAFNet, advanced loss, 50 epochs, batch size 32
- Outputs checkpoints, validation images, and curves to `./training_output`
- Expects training slices in `E:\PHD\phd_env\Proyectos\Denoising_challenge\Calcium\Data\Train\slices`

### Generating Test Stacks

Generate 3D test stacks from training slices (if `generate_stacks.py` is available):

```bash
python generate_stacks.py
```

- Creates stacks like `A1_stack_000_009.tif`, `B1_stack_000_009.tif`, and `C1_stack_000_009.tif`
- Outputs to `./training_output/stacks`

### Local Inference

Run inference locally:

```bash
python inference.py
```

- Expects input TIFF stacks in `/input/images/image-stack-unstructured-noise`
- Saves denoised stacks to `/output/images/image-stack-denoised`

### Docker Inference

Build and test the Docker container:

```bash
./do_test_run.sh
```

- Builds `denoising-calcium` image
- Processes test stacks in `./test/input/interface_0/images/image-stack-unstructured-noise`
- Outputs denoised stacks to `./test/output/interface_0/images/image-stack-denoised`

### Evaluation

Evaluate results with custom metrics:

```bash
python evaluate.py
```

- Computes PSNR/SSIM for denoised outputs against ground truth (if available)

### Saving Outputs

Use `do_save.sh` to save model weights or outputs (customize as needed):

```bash
./do_save.sh
```

## Notes

- **Test Stacks**: The test inputs (`A1_stack_000_009.tif`, etc.) correspond to three distinct noise types from the training data, as per the challenge requirements
- **Docker**: Ensure the model checkpoint (`best_model_NafNet_Noise2Void.pth`) is in `N2V_Vanilla/`. The Dockerfile copies it to `/opt/app/model/`
- **GPU Support**: For GPU inference, use an NVIDIA GPU with CUDA 11.8+ and NVIDIA Container Toolkit. Modify Dockerfile to use `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04` if needed
- **Challenge Info**: See [AI4Life CIDC25](https://ai4life-cidc25.grand-challenge.org/) for details on data format and evaluation

## License

MIT License. See LICENSE file for details.

## Acknowledgments

- Based on Noise2Void and NAFNet architectures
- Developed for the AI4Life Calcium Imaging Denoising Challenge 2025

## Instructions for GitHub

Create or update README.md:

```bash
echo "<paste the above content>" > README.md
```

Verify the file:

```bash
cat README.md
```

Commit and push to GitHub:

```bash
git add README.md
git commit -m "Add README for 3D Denoising Calcium Submission"
git push origin main
# Replace 'main' with your branch name if different
```

License setup (if not already created):

```bash
echo "MIT License" > LICENSE
git add LICENSE
git commit -m "Add MIT License"
git push origin main
```
