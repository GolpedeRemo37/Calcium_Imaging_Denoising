"""
losses.py - Advanced loss functions for Noise2Void denoising
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from torchmetrics import StructuralSimilarityIndexMeasure
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    print("Warning: torchmetrics not available. Install with: pip install torchmetrics")
    TORCHMETRICS_AVAILABLE = False

class PerceptualLoss(nn.Module):
    """
    Simple perceptual loss using VGG features for better texture preservation.
    """
    def __init__(self, device, weight=0.1):
        super().__init__()
        self.weight = weight
        try:
            import torchvision.models as models
            vgg = models.vgg16(pretrained=True).features[:16].to(device)
            for param in vgg.parameters():
                param.requires_grad = False
            self.vgg = vgg
            self.available = True
        except ImportError:
            print("Warning: torchvision not available for perceptual loss")
            self.available = False
    
    def forward(self, pred, target):
        if not self.available or pred.shape[1] != 3:
            return 0.0
        
        # Convert grayscale to RGB if needed
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        return F.mse_loss(pred_features, target_features) * self.weight

class EdgePreservingLoss(nn.Module):
    """
    Edge-preserving loss to maintain image structure during denoising.
    """
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight
        
        # Sobel kernels for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def forward(self, pred, target):
        # Compute gradients
        pred_grad_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, self.sobel_y, padding=1)
        target_grad_x = F.conv2d(target, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(target, self.sobel_y, padding=1)
        
        # Edge magnitude
        pred_edges = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-6)
        target_edges = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-6)
        
        return F.l1_loss(pred_edges, target_edges) * self.weight

class NoiseAwareLoss(nn.Module):
    """
    Noise-aware loss that adapts based on local image statistics.
    Gives more weight to high-contrast regions and less to smooth areas.
    """
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight
    
    def forward(self, pred, target, mask=None):
        # Compute local variance to identify noisy vs smooth regions
        kernel = torch.ones(1, 1, 5, 5, device=pred.device) / 25
        local_mean = F.conv2d(target, kernel, padding=2)
        local_var = F.conv2d((target - local_mean)**2, kernel, padding=2)
        
        # Adaptive weights: higher weight for high-variance (detailed) regions
        weights = 1.0 + self.weight * torch.tanh(local_var * 10)
        
        if mask is not None:
            loss = weights * (pred - target)**2
            return loss[mask].mean()
        else:
            return (weights * (pred - target)**2).mean()

class CombinedLoss(nn.Module):
    """
    Combined loss function optimized for Noise2Void denoising.
    Includes multiple loss components for better denoising performance.
    """
    def __init__(self, device, data_range=1.0, l1_weight=1.0, mse_weight=1.0, 
                 ssim_weight=0.1, edge_weight=0.05, noise_weight=0.1):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        self.noise_weight = noise_weight
        
        # SSIM loss
        if TORCHMETRICS_AVAILABLE and ssim_weight > 0:
            self.ssim_loss = StructuralSimilarityIndexMeasure(data_range=data_range).to(device)
        else:
            self.ssim_loss = None
            if ssim_weight > 0:
                print("SSIM loss not available - using L1 + MSE only")
        
        # Edge preserving loss
        if edge_weight > 0:
            self.edge_loss = EdgePreservingLoss(weight=1.0)  # Weight handled in forward
        else:
            self.edge_loss = None
        
        # Noise-aware loss
        if noise_weight > 0:
            self.noise_loss = NoiseAwareLoss(weight=1.0)  # Weight handled in forward
        else:
            self.noise_loss = None
    
    def forward(self, output, target, mask=None):
        """
        Args:
            output: predicted values
            target: ground truth values  
            mask: optional mask to apply loss only to specific pixels (for N2V)
        """
        if mask is not None:
            # For N2V training - only compute loss on masked pixels
            output_masked = output[mask]
            target_masked = target[mask]
            
            # Basic losses on masked pixels only
            l1 = self.l1_loss(output_masked, target_masked)
            mse = self.mse_loss(output_masked, target_masked)
            total_loss = self.l1_weight * l1 + self.mse_weight * mse
            
            # Noise-aware loss on masked pixels
            if self.noise_loss is not None:
                noise_loss = self.noise_loss(output, target, mask)
                total_loss += self.noise_weight * noise_loss
        else:
            # For validation with full images
            l1 = self.l1_loss(output, target)
            mse = self.mse_loss(output, target)
            total_loss = self.l1_weight * l1 + self.mse_weight * mse
            
            # SSIM loss (works on full images)
            if self.ssim_loss is not None:
                ssim_val = self.ssim_loss(output, target)
                ssim_loss = 1 - ssim_val
                total_loss += self.ssim_weight * ssim_loss
            
            # Edge preserving loss
            if self.edge_loss is not None:
                edge_loss = self.edge_loss(output, target)
                total_loss += self.edge_weight * edge_loss
            
            # Noise-aware loss
            if self.noise_loss is not None:
                noise_loss = self.noise_loss(output, target)
                total_loss += self.noise_weight * noise_loss
        
        return total_loss

class AdaptiveCombinedLoss(nn.Module):
    """
    Adaptive loss that changes weights during training for optimal denoising.
    """
    def __init__(self, device, data_range=1.0, l1_weight=1.0, mse_weight=1.0, 
                 ssim_weight=0.1, edge_weight=0.05, noise_weight=0.1):
        super().__init__()
        self.base_loss = CombinedLoss(device, data_range, l1_weight, mse_weight, 
                                     ssim_weight, edge_weight, noise_weight)
        self.epoch = 0
    
    def update_epoch(self, epoch):
        """Update epoch for adaptive weighting."""
        self.epoch = epoch
        
        # Adaptive weighting: emphasize structure preservation later in training
        if self.epoch > 20:
            self.base_loss.edge_weight *= 1.2
            self.base_loss.ssim_weight *= 1.1
    
    def forward(self, output, target, mask=None):
        return self.base_loss(output, target, mask)

class TVLoss(nn.Module):
    """
    Total Variation loss for smoothness regularization.
    Helps reduce noise artifacts.
    """
    def __init__(self, weight=0.01):
        super().__init__()
        self.weight = weight
    
    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x-1]), 2).sum()
        return self.weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
    
    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

# Loss function factory
def create_loss_function(loss_type="adaptive", device="cuda", **kwargs):
    """
    Factory function to create loss functions optimized for denoising.
    
    Args:
        loss_type: "mse", "l1", "combined", "adaptive", "advanced"
        device: torch device
        **kwargs: additional arguments for loss functions
    
    Returns:
        loss function
    """
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "l1":
        return nn.L1Loss()
    elif loss_type == "combined":
        return CombinedLoss(device, **kwargs)
    elif loss_type == "adaptive":
        return AdaptiveCombinedLoss(device, **kwargs)
    elif loss_type == "advanced":
        # Most comprehensive loss for best denoising
        return CombinedLoss(
            device, 
            l1_weight=kwargs.get('l1_weight', 1.0),
            mse_weight=kwargs.get('mse_weight', 0.5),
            ssim_weight=kwargs.get('ssim_weight', 0.2),
            edge_weight=kwargs.get('edge_weight', 0.1),
            noise_weight=kwargs.get('noise_weight', 0.15),
            **{k: v for k, v in kwargs.items() if k not in ['l1_weight', 'mse_weight', 'ssim_weight', 'edge_weight', 'noise_weight']}
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

# Simple functional losses for backward compatibility
def mse_masked_loss(pred, target, mask):
    """MSE loss applied only to masked pixels."""
    return F.mse_loss(pred[mask], target[mask])

def l1_masked_loss(pred, target, mask):
    """L1 loss applied only to masked pixels."""
    return F.l1_loss(pred[mask], target[mask])

def combined_masked_loss(pred, target, mask, l1_weight=1.0, mse_weight=1.0):
    """Combined L1 + MSE loss applied only to masked pixels."""
    l1 = F.l1_loss(pred[mask], target[mask])
    mse = F.mse_loss(pred[mask], target[mask])
    return l1_weight * l1 + mse_weight * mse