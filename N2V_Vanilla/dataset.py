"""
dataset.py - Dataset classes and data utilities for Noise2Void training
"""

import os
import re
import glob
import random
from typing import List, Tuple

import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import Dataset

def list_images(folder: str, exts=("tif", "tiff", "png", "jpg", "jpeg")) -> List[str]:
    """List all image files in folder recursively."""
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, f"**/*.{ext}"), recursive=True))
    return sorted(files)

def find_val_pairs(gt_dir: str, noisy_dir: str) -> List[Tuple[str, List[str]]]:
    """
    Find GT files and their corresponding noisy frames.
    Assumes GT filenames contain 'F0_slice_{n}.tif' and noisy contain F1,F2,F3... with same slice index.
    """
    gt_files = list_images(gt_dir)
    noisy_files = list_images(noisy_dir)
    
    # Build mapping from slice index to noisy files
    noisy_map = {}
    patt = re.compile(r"_[sS]lice[_\-]?0*([0-9]+)\.")  # Capture slice number
    
    for nf in noisy_files:
        m = patt.search(os.path.basename(nf))
        if m:
            idx = int(m.group(1))
            noisy_map.setdefault(idx, []).append(nf)

    pairs = []
    for gf in gt_files:
        m = patt.search(os.path.basename(gf))
        if m:
            idx = int(m.group(1))
            if idx in noisy_map:
                pairs.append((gf, sorted(noisy_map[idx])))
    
    return pairs

class NoisyPatchDataset(Dataset):
    """
    Dataset for Noise2Void training that samples random patches from noisy images.
    This is the key dataset for self-supervised training without GT.
    """
    def __init__(self, root_dir: str, patch_size: int = 128, dataset_size: int = 20000):
        self.files = list_images(root_dir)
        assert len(self.files) > 0, f"No images found in {root_dir}"
        self.patch_size = patch_size
        self.dataset_size = dataset_size
        print(f"Found {len(self.files)} training images in {root_dir}")

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # Randomly pick an image file then a random crop
        f = random.choice(self.files)
        img = tiff.imread(f).astype(np.float32)
        
        # Normalize to 0..1 using local min/max
        imin, imax = img.min(), img.max()
        if imax > imin:
            img = (img - imin) / (imax - imin)
        else:
            img = img - imin
            
        h, w = img.shape
        
        # Pad if needed
        if h < self.patch_size or w < self.patch_size:
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
            h, w = img.shape
            
        # Random crop
        i = random.randint(0, h - self.patch_size)
        j = random.randint(0, w - self.patch_size)
        patch = img[i:i+self.patch_size, j:j+self.patch_size]
        
        # Convert to tensor CHW
        patch = torch.from_numpy(patch).float().unsqueeze(0)  # 1,H,W
        return patch

class ValidationDataset(Dataset):
    """
    Dataset for validation, loading GT and corresponding noisy images.
    Each item returns a GT image and a list of corresponding noisy images.
    Only used when GT is available for validation.
    """
    def __init__(self, gt_dir: str, noisy_dir: str, normalize: bool = False):
        self.val_pairs = find_val_pairs(gt_dir, noisy_dir)
        assert len(self.val_pairs) > 0, f"No valid GT-noisy pairs found in {gt_dir} and {noisy_dir}"
        self.normalize = normalize
        print(f"Found {len(self.val_pairs)} validation pairs")

    def __len__(self):
        return len(self.val_pairs)

    def __getitem__(self, idx):
        gt_path, noisy_paths = self.val_pairs[idx]
        
        # Load GT image
        gt = tiff.imread(gt_path).astype(np.float32)
        
        # Load noisy images
        noisy_images = [tiff.imread(path).astype(np.float32) for path in noisy_paths]
        
        if self.normalize:
            # Normalize GT and noisy images to [0,1] using GT's range for consistency
            imin, imax = gt.min(), gt.max()
            if imax > imin:
                gt = (gt - imin) / (imax - imin)
                noisy_images = [(img - imin) / (imax - imin) for img in noisy_images]
            else:
                gt = gt - imin
                noisy_images = [img - imin for img in noisy_images]
        
        # Convert to tensors (H,W) -> (1,H,W)
        gt_tensor = torch.from_numpy(gt).float().unsqueeze(0)
        noisy_tensors = [torch.from_numpy(img).float().unsqueeze(0) for img in noisy_images]
        
        return {
            'gt': gt_tensor,
            'noisy': noisy_tensors,
            'gt_path': gt_path,
            'noisy_paths': noisy_paths
        }

def make_masked_input(x: torch.Tensor, num_mask: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Noise2Void masking: replace masked pixels with random neighbor values.
    This is the core of N2V - we predict original pixel values from neighbors.
    
    Args:
        x: Input tensor (B, C, H, W)
        num_mask: Number of pixels to mask per patch
    
    Returns:
        x_masked: Input with masked pixels replaced by neighbors
        mask: Boolean mask indicating which pixels were masked
    """
    B, C, H, W = x.shape
    mask = torch.zeros_like(x, dtype=torch.bool)
    x_masked = x.clone()
    
    for b in range(B):
        # Sample num_mask unique pixel coordinates
        coords = set()
        while len(coords) < num_mask:
            yy = random.randrange(H)
            xx = random.randrange(W)
            coords.add((yy, xx))
            
        for (yy, xx) in coords:
            mask[b, 0, yy, xx] = True
            
            # Replace with neighbor pixel value: random offset within [-2,2], excluding (0,0)
            for tries in range(10):
                dy = random.randint(-2, 2)
                dx = random.randint(-2, 2)
                if dy == 0 and dx == 0:
                    continue
                ny, nx = yy + dy, xx + dx
                if 0 <= ny < H and 0 <= nx < W:
                    x_masked[b, 0, yy, xx] = x[b, 0, ny, nx]
                    break
            else:
                # Fallback: keep original value
                x_masked[b, 0, yy, xx] = x[b, 0, yy, xx]
                
    return x_masked, mask

def load_image(path: str, normalize=True) -> np.ndarray:
    """Load and optionally normalize image."""
    img = tiff.imread(path).astype(np.float32)
    if normalize:
        imin, imax = img.min(), img.max()
        if imax > imin:
            img = (img - imin) / (imax - imin)
        else:
            img = img - imin
    return img