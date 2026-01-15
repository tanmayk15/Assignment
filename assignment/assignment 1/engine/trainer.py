"""
Training Engine
Handles model training with proper logging and checkpointing
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Dict
import yaml


class Trainer:
    """
    Training engine for Faster R-CNN
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        config: Dict,
        device: torch.device,
        output_dir: str,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # TensorBoard
        if config['training'].get('tensorboard', True):
            self.writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))
        else:
            self.writer = None
        
        # Training state
        self.epoch = 0
        self.iteration = 0
        self.best_map = 0.0
        
        # Training config
        self.num_epochs = config['training']['num_epochs']
        self.log_frequency = config['training'].get('log_frequency', 50)
        self.save_frequency = config['training'].get('save_frequency', 1)
        self.gradient_clip_norm = config['training'].get('gradient_clip_norm', 35.0)
        self.accumulation_steps = config['training'].get('accumulation_steps', 1)
        
        # Warmup
        self.warmup_iterations = config['training']['lr_schedule'].get('warmup_iterations', 500)
        self.warmup_method = config['training']['lr_schedule'].get('warmup_method', 'linear')
        self.base_lr = config['training']['optimizer']['lr']
    
    def train(self):
        """Main training loop"""
        print("=" * 60)
        print(f"Starting training for {self.num_epochs} epochs")
        print(f"Output directory: {self.output_dir}")
        print("=" * 60)
        
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            
            # Train one epoch
            train_metrics = self.train_epoch()
            
            # Validate
            if self.val_loader is not None:
                val_metrics = self.validate()
            else:
                val_metrics = {}
            
            # Update learning rate
            if self.iteration > self.warmup_iterations:
                self.scheduler.step()
            
            # Print epoch summary
            self._print_epoch_summary(train_metrics, val_metrics)
            
            # Save checkpoint
            if (epoch + 1) % self.save_frequency == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
            
            # Save best model
            if val_metrics and val_metrics.get('mAP', 0) > self.best_map:
                self.best_map = val_metrics['mAP']
                self.save_checkpoint('best_model.pth')
                print(f"✓ Saved best model (mAP: {self.best_map:.3f})")
        
        # Training complete
        print("\n" + "=" * 60)
        print(f"Training complete! Best mAP: {self.best_map:.3f}")
        print("=" * 60)
        
        if self.writer:
            self.writer.close()
    
    def train_epoch(self) -> Dict:
        """Train for one epoch"""
        self.model.train()
        
        losses = {
            'total': 0.0,
            'rpn_cls': 0.0,
            'rpn_reg': 0.0,
            'det_cls': 0.0,
            'det_reg': 0.0,
        }
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch+1}/{self.num_epochs}')
        
        self.optimizer.zero_grad()
        
        for batch_idx, (images, targets) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
            
            # Learning rate warmup
            if self.iteration < self.warmup_iterations:
                self._warmup_lr()
            
            # Forward pass
            _, batch_losses = self.model(images, targets)
            
            # Compute total loss
            loss = batch_losses.get('total_loss', 0)
            
            # Skip if loss is 0 or not a tensor
            if not isinstance(loss, torch.Tensor) or loss == 0:
                continue
            
            # Backward pass with gradient accumulation
            loss = loss / self.accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Gradient clipping
                if self.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_norm
                    )
                
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Accumulate losses
            for key in losses.keys():
                if key == 'total':
                    losses[key] += loss.item() * self.accumulation_steps
                else:
                    loss_key = f'loss_{key}' if not key.startswith('loss_') else key
                    losses[key] += batch_losses.get(loss_key, 0).item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item() * self.accumulation_steps,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # Logging
            if self.iteration % self.log_frequency == 0:
                self._log_metrics(batch_losses, prefix='train')
            
            self.iteration += 1
        
        # Average losses
        num_batches = len(self.train_loader)
        for key in losses.keys():
            losses[key] /= num_batches
        
        return losses
    
    def validate(self) -> Dict:
        """Validate the model"""
        self.model.eval()
        
        # This is a simplified validation - full implementation would compute mAP
        print("\nValidating...")
        
        losses = {
            'total': 0.0,
        }
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                           for k, v in t.items()} for t in targets]
                
                # Forward pass
                _, batch_losses = self.model(images, targets)
                
                losses['total'] += batch_losses.get('total_loss', 0).item()
        
        # Average losses
        num_batches = len(self.val_loader)
        for key in losses.keys():
            losses[key] /= num_batches
        
        # Log validation metrics
        if self.writer:
            self.writer.add_scalar('val/loss', losses['total'], self.epoch)
        
        # Placeholder mAP (would be computed by evaluator)
        losses['mAP'] = 0.0  # TODO: Integrate with evaluator
        
        return losses
    
    def _warmup_lr(self):
        """Learning rate warmup"""
        if self.warmup_method == 'linear':
            alpha = self.iteration / self.warmup_iterations
        elif self.warmup_method == 'constant':
            alpha = 1.0
        else:
            alpha = 1.0
        
        lr = self.base_lr * alpha
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _log_metrics(self, metrics: Dict, prefix: str = 'train'):
        """Log metrics to TensorBoard"""
        if not self.writer:
            return
        
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.writer.add_scalar(f'{prefix}/{key}', value, self.iteration)
        
        # Log learning rate
        self.writer.add_scalar(
            'train/lr',
            self.optimizer.param_groups[0]['lr'],
            self.iteration
        )
    
    def _print_epoch_summary(self, train_metrics: Dict, val_metrics: Dict):
        """Print summary after each epoch"""
        print(f"\nEpoch {self.epoch+1}/{self.num_epochs} Summary:")
        print(f"  Train Loss: {train_metrics['total']:.4f}")
        
        if val_metrics:
            print(f"  Val Loss: {val_metrics.get('total', 0):.4f}")
            print(f"  Val mAP: {val_metrics.get('mAP', 0):.4f}")
        
        print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
        print()
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'iteration': self.iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_map': self.best_map,
            'config': self.config,
        }
        
        filepath = os.path.join(self.output_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.iteration = checkpoint['iteration']
        self.best_map = checkpoint['best_map']
        
        print(f"Loaded checkpoint from {filepath}")
        print(f"  Epoch: {self.epoch}, Iteration: {self.iteration}, Best mAP: {self.best_map:.3f}")


if __name__ == "__main__":
    print("Trainer module - use via train.py script")
