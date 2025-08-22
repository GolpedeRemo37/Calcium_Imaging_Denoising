"""
config.py - Configuration settings for Noise2Void training
"""

import os
import torch  # Added to fix NameError
from .models import NAFNET_CONFIGS

class TrainingConfig:
    """
    Configuration class for Noise2Void training parameters.
    """
    
    def __init__(self):
        # === DATA PATHS ===
        self.train_dir = r"E:\PHD\phd_env\Proyectos\Denoising_challenge\Calcium\Data\Train\slices"
        self.val_gt_dir = r"E:\PHD\phd_env\Proyectos\Denoising_challenge\Calcium\Data\Val\GT\slices"
        self.val_noisy_dir = r"E:\PHD\phd_env\Proyectos\Denoising_challenge\Calcium\Data\Val\noisy\slices"
        self.output_dir = "./training_output"
        
        # === TRAINING HYPERPARAMETERS ===
        self.patch_size = 128
        self.batch_size = 32
        self.learning_rate = 1e-3
        self.epochs = 40
        self.num_masks_per_patch = 512  # Number of masked pixels per patch for N2V
        self.grad_clip = 1.0  # Gradient clipping for stability
        
        # === MODEL CONFIGURATION ===
        self.model_size = 'medium'  # Options: 'tiny', 'small', 'medium', 'large'
        self.nafnet_config = NAFNET_CONFIGS[self.model_size].copy()
        
        # === LOSS FUNCTION SETTINGS ===
        self.loss_type = "advanced"  # Options: "mse", "l1", "combined", "adaptive", "advanced"
        self.loss_weights = {
            'l1_weight': 1.0,
            'mse_weight': 0.5,
            'ssim_weight': 0.2,
            'edge_weight': 0.1,
            'noise_weight': 0.15
        }
        
        # === OPTIMIZER SETTINGS ===
        self.optimizer_type = "adam"  # Options: "adam", "adamw"
        self.weight_decay = 1e-4
        self.betas = (0.9, 0.999)
        
        # === SCHEDULER SETTINGS ===
        self.scheduler_type = "step"  # Options: "step", "cosine", "plateau"
        self.step_size = 15  # For StepLR
        self.gamma = 0.5     # For StepLR
        
        # === VALIDATION & LOGGING ===
        self.validate_every = 1       # Validate every N epochs
        self.save_images_every = 5    # Save validation images every N epochs
        self.max_val_images = 5       # Max images to save per validation
        self.max_val_batches = 20     # Max batches to process in validation
        self.plot_curves_every = 10   # Update training curves every N epochs
        self.checkpoint_every = 10    # Save checkpoint every N epochs
        
        # === DATASET SETTINGS ===
        self.dataset_size = 20000     # Size of training dataset (patches per epoch)
        self.num_workers = 4          # DataLoader workers (set to 0 if issues)
        self.pin_memory = True        # Use pinned memory for GPU
        
        # === REPRODUCIBILITY ===
        self.seed = 42
        self.deterministic = True
        
        # === DEVICE SETTINGS ===
        self.device = "auto"  # Options: "auto", "cuda", "cpu"
    
    def get_device(self):
        """Get the appropriate device for training."""
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device
    
    def to_dict(self):
        """Convert config to dictionary for serialization."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def update_paths(self, train_dir=None, val_gt_dir=None, val_noisy_dir=None, output_dir=None):
        """Update data paths."""
        if train_dir:
            self.train_dir = train_dir
        if val_gt_dir:
            self.val_gt_dir = val_gt_dir
        if val_noisy_dir:
            self.val_noisy_dir = val_noisy_dir
        if output_dir:
            self.output_dir = output_dir
    
    def update_model_config(self, model_size=None, custom_config=None):
        """Update model configuration."""
        if model_size and model_size in NAFNET_CONFIGS:
            self.model_size = model_size
            self.nafnet_config = NAFNET_CONFIGS[model_size].copy()
        elif custom_config:
            self.nafnet_config = custom_config.copy()
    
    def validate_config(self):
        """Validate configuration settings."""
        errors = []
        
        # Check paths exist
        if not os.path.exists(self.train_dir):
            errors.append(f"Training directory does not exist: {self.train_dir}")
        
        if self.val_gt_dir and not os.path.exists(self.val_gt_dir):
            errors.append(f"Validation GT directory does not exist: {self.val_gt_dir}")
        
        if self.val_noisy_dir and not os.path.exists(self.val_noisy_dir):
            errors.append(f"Validation noisy directory does not exist: {self.val_noisy_dir}")
        
        # Check hyperparameters
        if self.patch_size <= 0 or self.patch_size % 8 != 0:
            errors.append("Patch size must be positive and divisible by 8")
        
        if self.batch_size <= 0:
            errors.append("Batch size must be positive")
        
        if self.learning_rate <= 0:
            errors.append("Learning rate must be positive")
        
        if self.epochs <= 0:
            errors.append("Number of epochs must be positive")
        
        if self.num_masks_per_patch <= 0:
            errors.append("Number of masks per patch must be positive")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
        
        return True


# Predefined configurations for different scenarios
class FastTrainingConfig(TrainingConfig):
    """Configuration for fast training/experimentation."""
    
    def __init__(self):
        super().__init__()
        self.model_size = 'tiny'
        self.nafnet_config = NAFNET_CONFIGS['tiny'].copy()
        self.batch_size = 16
        self.epochs = 10
        self.dataset_size = 5000
        self.max_val_batches = 5
        self.save_images_every = 2
        self.plot_curves_every = 2


class HighQualityConfig(TrainingConfig):
    """Configuration for high-quality denoising."""
    
    def __init__(self):
        super().__init__()
        self.model_size = 'large'
        self.nafnet_config = NAFNET_CONFIGS['large'].copy()
        self.batch_size = 16  # Smaller batch due to larger model
        self.epochs = 80
        self.dataset_size = 40000
        self.learning_rate = 5e-4  # Lower learning rate for stability
        self.loss_type = "advanced"
        self.loss_weights = {
            'l1_weight': 1.0,
            'mse_weight': 0.3,
            'ssim_weight': 0.25,
            'edge_weight': 0.15,
            'noise_weight': 0.2
        }


class CPUTrainingConfig(TrainingConfig):
    """Configuration optimized for CPU training."""
    
    def __init__(self):
        super().__init__()
        self.device = "cpu"
        self.model_size = 'small'
        self.nafnet_config = NAFNET_CONFIGS['small'].copy()
        self.batch_size = 8
        self.num_workers = 0  # Avoid multiprocessing issues on some systems
        self.pin_memory = False
        self.dataset_size = 10000


def create_config(config_type="default"):
    """
    Factory function to create different configuration types.
    
    Args:
        config_type: Type of configuration
            - "default": Standard training configuration
            - "fast": Fast training for experimentation
            - "high_quality": High-quality denoising configuration
            - "cpu": CPU-optimized configuration
    
    Returns:
        Configuration instance
    """
    config_map = {
        "default": TrainingConfig,
        "fast": FastTrainingConfig,
        "high_quality": HighQualityConfig,
        "cpu": CPUTrainingConfig
    }
    
    if config_type not in config_map:
        raise ValueError(f"Unknown config type: {config_type}. Available: {list(config_map.keys())}")
    
    return config_map[config_type]()


# Example configurations for different use cases
EXAMPLE_CONFIGS = {
    "calcium_imaging": {
        "description": "Optimized for calcium imaging data",
        "model_size": "medium",
        "loss_type": "advanced",
        "loss_weights": {
            'l1_weight': 1.0,
            'mse_weight': 0.4,
            'ssim_weight': 0.3,
            'edge_weight': 0.2,
            'noise_weight': 0.25
        },
        "patch_size": 128,
        "batch_size": 24,
        "learning_rate": 8e-4,
        "epochs": 50
    },
    
    "fluorescence_microscopy": {
        "description": "Optimized for fluorescence microscopy",
        "model_size": "large",
        "loss_type": "advanced",
        "loss_weights": {
            'l1_weight': 1.0,
            'mse_weight': 0.3,
            'ssim_weight': 0.4,
            'edge_weight': 0.15,
            'noise_weight': 0.3
        },
        "patch_size": 256,
        "batch_size": 8,
        "learning_rate": 5e-4,
        "epochs": 100
    },
    
    "general_denoising": {
        "description": "General purpose denoising",
        "model_size": "medium",
        "loss_type": "combined",
        "loss_weights": {
            'l1_weight': 1.0,
            'mse_weight': 1.0,
            'ssim_weight': 0.1
        },
        "patch_size": 128,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "epochs": 40
    }
}