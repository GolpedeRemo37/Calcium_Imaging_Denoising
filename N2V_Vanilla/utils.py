"""
utils.py - Utility functions for training and inference
"""

import os
import numpy as np
import torch
import tifffile as tiff
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def denoise_full_image(model, image, patch_size=128, stride=None, device='cuda'):
    """
    Denoise a full image using patch-based inference with overlapping patches.
    
    Args:
        model: trained denoising model
        image: input image (H, W) numpy array
        patch_size: size of patches for processing
        stride: stride between patches (if None, uses patch_size for no overlap)
        device: torch device
    
    Returns:
        denoised image (H, W) numpy array
    """
    if stride is None:
        stride = patch_size // 2  # 50% overlap for better results
    
    H, W = image.shape
    
    # Normalize input
    imin, imax = image.min(), image.max()
    if imax > imin:
        img_norm = (image - imin) / (imax - imin)
    else:
        img_norm = image - imin
    
    # Initialize output arrays
    output = np.zeros_like(img_norm, dtype=np.float32)
    weight_map = np.zeros_like(img_norm, dtype=np.float32)
    
    # Process overlapping patches
    model.eval()
    with torch.no_grad():
        for i in range(0, H, stride):
            for j in range(0, W, stride):
                # Calculate patch boundaries
                i_end = min(i + patch_size, H)
                j_end = min(j + patch_size, W)
                
                # Adjust start positions if patch extends beyond image
                i_start = max(0, i_end - patch_size)
                j_start = max(0, j_end - patch_size)
                
                # Extract patch
                patch = img_norm[i_start:i_end, j_start:j_end]
                
                # Pad if necessary
                if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                    pad_h = patch_size - patch.shape[0]
                    pad_w = patch_size - patch.shape[1]
                    patch = np.pad(patch, ((0, pad_h), (0, pad_w)), mode='reflect')
                
                # Convert to tensor and process
                patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
                denoised_patch = model(patch_tensor).cpu().numpy()[0, 0]
                
                # Remove padding if added
                if patch.shape[0] > (i_end - i_start) or patch.shape[1] > (j_end - j_start):
                    denoised_patch = denoised_patch[:i_end-i_start, :j_end-j_start]
                
                # Add to output with weights
                output[i_start:i_end, j_start:j_end] += denoised_patch
                weight_map[i_start:i_end, j_start:j_end] += 1.0
    
    # Normalize by weights
    output = output / (weight_map + 1e-8)
    
    # Denormalize
    denoised = output * (imax - imin) + imin
    
    return denoised

def compute_metrics(gt, pred, data_range=None):
    """
    Compute PSNR and SSIM metrics between ground truth and prediction.
    
    Args:
        gt: ground truth image
        pred: predicted image
        data_range: data range for metrics (if None, computed from gt)
    
    Returns:
        dict with psnr and ssim values
    """
    if data_range is None:
        data_range = gt.max() - gt.min() if gt.max() > gt.min() else 1.0
    
    # Clamp prediction to GT range
    pred_clamped = np.clip(pred, gt.min(), gt.max())
    
    psnr = compare_psnr(gt, pred_clamped, data_range=data_range)
    ssim = compare_ssim(gt, pred_clamped, data_range=data_range)
    
    return {'psnr': psnr, 'ssim': ssim}

def save_comparison_image(noisy, denoised, gt, save_path, metrics=None, title=None):
    """
    Save a comparison image showing noisy input, denoised output, and ground truth.
    
    Args:
        noisy: noisy input image
        denoised: denoised output image  
        gt: ground truth image
        save_path: path to save the comparison
        metrics: optional dict with PSNR/SSIM values
        title: optional title for the figure
    """
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)
    
    def normalize_for_display(img):
        return (img - img.min()) / (img.max() - img.min()) if img.max() > img.min() else img - img.min()
    
    noisy_norm = normalize_for_display(noisy)
    denoised_norm = normalize_for_display(denoised)
    gt_norm = normalize_for_display(gt)
    
    # Noisy input
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(noisy_norm, cmap='gray')
    ax1.set_title('Noisy Input', fontsize=12)
    ax1.axis('off')
    
    # Denoised output
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(denoised_norm, cmap='gray')
    if metrics:
        metrics_text = f"PSNR: {metrics['psnr']:.2f}dB\nSSIM: {metrics['ssim']:.3f}"
        ax2.set_title(f'Denoised Output\n{metrics_text}', fontsize=12)
    else:
        ax2.set_title('Denoised Output', fontsize=12)
    ax2.axis('off')
    
    # Ground truth
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(gt_norm, cmap='gray')
    ax3.set_title('Ground Truth', fontsize=12)
    ax3.axis('off')
    
    if title:
        plt.suptitle(title, fontsize=14, y=0.95)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_training_curves(train_losses, val_psnrs, val_ssims, save_path):
    """
    Save training curves showing loss and validation metrics.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    val_epochs = range(1, len(val_psnrs) + 1)
    
    # Training loss
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Validation PSNR
    ax2.plot(val_epochs, val_psnrs, 'r-', linewidth=2, label='Validation PSNR')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('PSNR (dB)')
    ax2.set_title('Validation PSNR')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Validation SSIM
    ax3.plot(val_epochs, val_ssims, 'g-', linewidth=2, label='Validation SSIM')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('SSIM')
    ax3.set_title('Validation SSIM')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def setup_reproducibility(seed=42):
    """
    Setup reproducible training environment.
    """
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def print_model_info(model, config):
    """
    Print model information and configuration.
    """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=" * 60)
    print("MODEL INFORMATION")
    print("=" * 60)
    print(f"Architecture: NAFNet")
    print(f"Parameters: {num_params:,}")
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 60)

def create_directories(base_dir="./training_output"):
    """
    Create necessary directories for training outputs.
    """
    dirs = [
        base_dir,
        os.path.join(base_dir, "checkpoints"),
        os.path.join(base_dir, "validation_images"),
        os.path.join(base_dir, "training_curves"),
        os.path.join(base_dir, "logs")
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def save_checkpoint(model, optimizer, scheduler, epoch, loss, psnr, save_path, config=None):
    """
    Save training checkpoint with all necessary information.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'psnr': psnr,
        'config': config
    }
    
    torch.save(checkpoint, save_path)

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    Load training checkpoint and restore training state.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', float('inf')),
        'psnr': checkpoint.get('psnr', 0),
        'config': checkpoint.get('config', {})
    }

class AverageMeter:
    """
    Compute and store the average and current value of metrics.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count