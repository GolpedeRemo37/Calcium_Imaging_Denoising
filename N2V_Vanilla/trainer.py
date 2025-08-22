"""
trainer.py - Training and validation logic for Noise2Void NAFNet
"""

import os
import time
from tqdm import tqdm
import torch
import numpy as np

from .dataset import make_masked_input
from .utils import denoise_full_image, compute_metrics, save_comparison_image, \
                  save_training_curves, save_checkpoint, AverageMeter

class Noise2VoidTrainer:
    """
    Trainer class for Noise2Void denoising with NAFNet.
    """
    
    def __init__(self, model, criterion, optimizer, scheduler, device, config):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        
        # Training metrics
        self.train_losses = []
        self.val_psnrs = []
        self.val_ssims = []
        self.best_psnr = -1.0
        self.start_epoch = 1
        
        # Create output directories
        self.output_dir = config.get('output_dir', './training_output')
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.validation_dir = os.path.join(self.output_dir, 'validation_images')
        
        for dir_path in [self.output_dir, self.checkpoint_dir, self.validation_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def train_epoch(self, train_loader, epoch):
        """
        Train for one epoch using Noise2Void masking strategy.
        """
        self.model.train()
        loss_meter = AverageMeter()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(self.device)
            
            # Apply Noise2Void masking
            x_masked, mask = make_masked_input(
                batch, 
                num_mask=self.config.get('num_masks_per_patch', 512)
            )
            x_masked = x_masked.to(self.device)
            mask = mask.to(self.device)
            
            # Forward pass
            pred = self.model(x_masked)
            
            # Compute loss only on masked pixels
            try:
                # Try with mask parameter (for advanced loss functions)
                loss = self.criterion(pred, batch, mask=mask)
            except TypeError:
                # Fallback to simple masked loss
                loss = self.criterion(pred[mask], batch[mask])
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            # Update metrics
            loss_meter.update(loss.item(), batch.size(0))
            pbar.set_postfix(loss=f"{loss_meter.avg:.6f}")
        
        # Step scheduler
        if self.scheduler:
            self.scheduler.step()
        
        return loss_meter.avg
    
    def validate_epoch(self, val_loader, epoch, save_images=False):
        """
        Validate the model on validation set with ground truth.
        """
        if len(val_loader) == 0:
            return 0.0, 0.0
        
        self.model.eval()
        per_gt_psnrs = []
        per_gt_ssims = []
        individual_results = []
        
        max_batches = min(len(val_loader), self.config.get('max_val_batches', 50))
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", total=max_batches)
            for batch_idx, batch in enumerate(pbar):
                if batch_idx >= max_batches:
                    break
                
                gt = batch['gt'].to(self.device)
                noisy_list = batch['noisy']
                gt_path = batch['gt_path'][0]
                noisy_paths = [p[0] for p in batch['noisy_paths']]
                
                for b in range(gt.shape[0]):
                    gt_np = gt[b, 0].cpu().numpy()
                    gt_psnrs = []
                    gt_ssims = []
                    
                    for noisy, noisy_path in zip(noisy_list, noisy_paths):
                        noisy_np = noisy[b, 0].cpu().numpy()
                        
                        # Denoise full image
                        denoised = denoise_full_image(
                            self.model, 
                            noisy_np, 
                            patch_size=self.config.get('patch_size', 128),
                            device=self.device
                        )
                        
                        # Compute metrics
                        metrics = compute_metrics(gt_np, denoised)
                        gt_psnrs.append(metrics['psnr'])
                        gt_ssims.append(metrics['ssim'])
                        
                        # Store results for visualization
                        individual_results.append({
                            'gt_path': gt_path,
                            'noisy_path': noisy_path,
                            'gt': gt_np,
                            'noisy': noisy_np,
                            'denoised': denoised,
                            'psnr': metrics['psnr'],
                            'ssim': metrics['ssim']
                        })
                    
                    # Per-GT averages
                    per_gt_psnrs.append(float(np.mean(gt_psnrs)))
                    per_gt_ssims.append(float(np.mean(gt_ssims)))
                    
                    pbar.set_postfix(
                        psnr=f"{per_gt_psnrs[-1]:.3f}",
                        ssim=f"{per_gt_ssims[-1]:.3f}"
                    )
        
        # Compute overall metrics
        mean_psnr = float(np.mean(per_gt_psnrs)) if per_gt_psnrs else 0.0
        mean_ssim = float(np.mean(per_gt_ssims)) if per_gt_ssims else 0.0
        
        # Save comparison images
        if save_images and individual_results:
            self._save_validation_images(individual_results, epoch)
        
        return mean_psnr, mean_ssim
    
    def _save_validation_images(self, results, epoch):
        """
        Save validation comparison images.
        """
        val_dir = os.path.join(self.validation_dir, f"epoch_{epoch}")
        os.makedirs(val_dir, exist_ok=True)
        
        max_images = self.config.get('max_val_images', 5)
        results_to_save = results[:max_images] if len(results) > max_images else results
        
        for i, result in enumerate(results_to_save):
            slice_name = os.path.basename(result['noisy_path']).replace('.tif', '')
            save_path = os.path.join(val_dir, f'comparison_{i+1}_{slice_name}.png')
            
            save_comparison_image(
                result['noisy'],
                result['denoised'],
                result['gt'],
                save_path,
                metrics={'psnr': result['psnr'], 'ssim': result['ssim']},
                title=f'Epoch {epoch} - {slice_name}'
            )
    
    def train(self, train_loader, val_loader, epochs):
        """
        Main training loop.
        """
        print(f"Starting training for {epochs} epochs...")
        print("=" * 60)
        
        for epoch in range(self.start_epoch, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            print("-" * 40)
            
            # Training
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            
            print(f"Train Loss: {train_loss:.6f}")
            if self.scheduler:
                print(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.2e}")
            
            # Validation
            save_images = (
                epoch % self.config.get('save_images_every', 5) == 0 or 
                epoch == 1 or 
                epoch == epochs
            )
            
            val_psnr, val_ssim = self.validate_epoch(val_loader, epoch, save_images)
            
            if val_psnr > 0:
                self.val_psnrs.append(val_psnr)
                self.val_ssims.append(val_ssim)
                print(f"Val PSNR: {val_psnr:.4f}, Val SSIM: {val_ssim:.4f}")
                
                # Save best model
                if val_psnr > self.best_psnr:
                    self.best_psnr = val_psnr
                    self._save_best_model(epoch, train_loss, val_psnr)
                    print(f"New best model saved! PSNR: {val_psnr:.4f}")
            
            # Save training curves periodically
            if epoch % self.config.get('plot_curves_every', 10) == 0:
                self._save_training_curves()
            
            # Save checkpoint
            if epoch % self.config.get('checkpoint_every', 10) == 0:
                self._save_checkpoint(epoch, train_loss, val_psnr)
        
        # Final training curves
        self._save_training_curves(final=True)
        
        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Best validation PSNR: {self.best_psnr:.4f}")
        print("=" * 60)
        
        return self.train_losses, self.val_psnrs, self.val_ssims
    
    def _save_best_model(self, epoch, loss, psnr):
        """Save the best model checkpoint."""
        save_path = os.path.join(self.checkpoint_dir, "best_model_NafNet_Noise2Void.pth")
        save_checkpoint(
            self.model, self.optimizer, self.scheduler,
            epoch, loss, psnr, save_path, self.config
        )
    
    def _save_checkpoint(self, epoch, loss, psnr):
        """Save regular training checkpoint."""
        save_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        save_checkpoint(
            self.model, self.optimizer, self.scheduler,
            epoch, loss, psnr, save_path, self.config
        )
    
    def _save_training_curves(self, final=False):
        """Save training curves plot."""
        suffix = "_final" if final else ""
        save_path = os.path.join(self.output_dir, f"training_curves{suffix}.png")
        save_training_curves(self.train_losses, self.val_psnrs, self.val_ssims, save_path)
    
    def resume_training(self, checkpoint_path):
        """Resume training from a checkpoint."""
        from .utils import load_checkpoint
        
        checkpoint_info = load_checkpoint(checkpoint_path, self.model, self.optimizer, self.scheduler)
        self.start_epoch = checkpoint_info['epoch'] + 1
        self.best_psnr = checkpoint_info['psnr']
        
        print(f"Resumed training from epoch {checkpoint_info['epoch']}")
        print(f"Previous best PSNR: {self.best_psnr:.4f}")

def create_trainer(model, criterion, optimizer, scheduler, device, config):
    """
    Factory function to create a Noise2Void trainer.
    """
    return Noise2VoidTrainer(model, criterion, optimizer, scheduler, device, config)