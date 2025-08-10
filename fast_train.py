#!/usr/bin/env python3
"""
High-performance training script optimized for A100 GPUs.
This script includes all performance optimizations for fastest training.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
import time
import argparse
from collections import defaultdict

# Import our modules
from improved_models import SubjectPINN, AgeParameterModel
from enhanced_datasets import SubjectAwareDataset, create_subject_splits, create_filtered_dataset
from training_utils import SimplePhysicsLoss, EarlyStopping, MetricsTracker
from optimized_training_config import detect_optimal_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor and log training performance metrics."""
    
    def __init__(self):
        self.batch_times = []
        self.gpu_utils = []
        self.memory_uses = []
        self.start_time = None
        
    def start_batch(self):
        self.start_time = time.time()
        
    def end_batch(self, batch_size: int):
        if self.start_time is not None:
            batch_time = time.time() - self.start_time
            self.batch_times.append(batch_time)
            
            # Log performance every 100 batches
            if len(self.batch_times) % 100 == 0:
                avg_time = np.mean(self.batch_times[-100:])
                samples_per_sec = batch_size / avg_time
                
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() / 1024**3
                    memory_cached = torch.cuda.memory_reserved() / 1024**3
                    self.memory_uses.append(memory_used)
                    
                    logger.info(f"Performance: {samples_per_sec:.1f} samples/sec, "
                               f"GPU memory: {memory_used:.1f}GB used, {memory_cached:.1f}GB cached")
                else:
                    logger.info(f"Performance: {samples_per_sec:.1f} samples/sec")

class FastTrainer:
    """High-performance trainer with all optimizations enabled."""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Performance monitoring
        self.perf_monitor = PerformanceMonitor()
        
        # Mixed precision training
        self.use_amp = config.get('mixed_precision', True) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Mixed precision: {self.use_amp}")
        logger.info(f"Configuration: {config}")
        
        # Initialize components
        self.setup_data()
        self.setup_model()
        
    def setup_data(self):
        """Setup optimized data loading."""
        logger.info("Setting up optimized datasets...")
        
        # Load dataset
        self.dataset = SubjectAwareDataset(
            processed_data_folder=self.config['data_folder'],
            age_csv_path=self.config.get('age_csv_path'),
            min_points_per_subject=self.config.get('min_points_per_subject', 100)
        )
        
        # Create splits
        self.subject_splits = create_subject_splits(
            self.dataset,
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
            random_seed=42
        )
        
        # Create filtered datasets
        self.train_indices = create_filtered_dataset(self.dataset, self.subject_splits['train'])
        self.val_indices = create_filtered_dataset(self.dataset, self.subject_splits['val'])
        
        # Optimized data loaders
        self.train_loader = DataLoader(
            Subset(self.dataset, self.train_indices),
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 8),
            pin_memory=self.config.get('pin_memory', True),
            prefetch_factor=self.config.get('prefetch_factor', 4),
            persistent_workers=self.config.get('persistent_workers', True),
            drop_last=True  # For consistent batch sizes
        )
        
        self.val_loader = DataLoader(
            Subset(self.dataset, self.val_indices),
            batch_size=self.config.get('val_batch_size', self.config['batch_size']),
            shuffle=False,
            num_workers=self.config.get('num_workers', 8),
            pin_memory=self.config.get('pin_memory', True),
            persistent_workers=self.config.get('persistent_workers', True)
        )
        
        logger.info(f"Data loaded: {len(self.train_indices):,} train, {len(self.val_indices):,} val")
        logger.info(f"Batch size: {self.config['batch_size']}, Batches per epoch: {len(self.train_loader)}")
        
    def setup_model(self):
        """Setup optimized model."""
        logger.info("Setting up optimized model...")
        
        # Create model
        self.model = SubjectPINN(
            subject_ids=self.dataset.valid_subjects,
            hidden_dims=self.config.get('hidden_dims', [256, 256, 256, 256]),
            param_bounds=self.config.get('param_bounds')
        ).to(self.device)
        
        # Model compilation for faster execution (PyTorch 2.0+)
        if self.config.get('compile_model', True) and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model)
                logger.info("✅ Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
        
        # Optimized optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-5),
            eps=1e-6,  # More stable for mixed precision
            betas=(0.9, 0.95)  # Optimized for large batch training
        )
        
        # Learning rate scheduler
        scheduler_type = self.config.get('scheduler', 'cosine_annealing')
        if scheduler_type == 'cosine_annealing':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config.get('epochs', 50),
                eta_min=self.config['learning_rate'] * 0.01
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5
            )
        
        # Loss functions
        if self.config.get('use_simplified_physics', True):
            self.physics_loss = SimplePhysicsLoss(
                weight=self.config.get('physics_weight', 0.01)
            ).to(self.device)
        else:
            from training_utils import PhysicsLoss
            self.physics_loss = PhysicsLoss(
                weight=self.config.get('physics_weight', 0.01)
            ).to(self.device)
            
        self.data_loss = nn.MSELoss()
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.get('early_stopping_patience', 15),
            min_delta=1e-6
        )
        
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model created: {total_params:,} parameters")
        
    def train_epoch(self, epoch: int) -> dict:
        """Train one epoch with all optimizations."""
        self.model.train()
        
        epoch_losses = defaultdict(float)
        epoch_samples = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (t, age, xy_true, subject_idx) in enumerate(pbar):
            self.perf_monitor.start_batch()
            
            # Move to device
            t = t.to(self.device, non_blocking=True).requires_grad_(True)
            age = age.to(self.device, non_blocking=True) 
            xy_true = xy_true.to(self.device, non_blocking=True)
            subject_idx = subject_idx.to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                xy_pred, params = self.model(t, subject_idx)
                
                # Data loss
                data_loss = self.data_loss(xy_pred, xy_true)
                
                # Physics loss (computed less frequently for speed)
                physics_freq = self.config.get('physics_computation_frequency', 4)
                if batch_idx % physics_freq == 0:
                    physics_loss = self.physics_loss(t, xy_pred, params)
                else:
                    physics_loss = torch.tensor(0.0, device=self.device)
                
                # Total loss
                total_loss = data_loss + physics_loss
                
                # Gradient accumulation
                grad_accum_steps = self.config.get('gradient_accumulation_steps', 1)
                total_loss = total_loss / grad_accum_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                
                if (batch_idx + 1) % grad_accum_steps == 0:
                    # Gradient clipping in scaled space
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                total_loss.backward()
                
                if (batch_idx + 1) % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Track metrics
            batch_size = t.shape[0]
            epoch_losses['data'] += data_loss.item() * batch_size
            epoch_losses['physics'] += physics_loss.item() * batch_size  
            epoch_losses['total'] += (data_loss + physics_loss).item() * batch_size
            epoch_samples += batch_size
            
            # Update progress
            pbar.set_postfix({
                'Data': f"{data_loss.item():.4f}",
                'Physics': f"{physics_loss.item():.4f}",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Performance monitoring
            self.perf_monitor.end_batch(batch_size)
            
            # Memory management
            if batch_idx % self.config.get('empty_cache_frequency', 100) == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Return average losses
        return {k: v / epoch_samples for k, v in epoch_losses.items()}
    
    def validate(self) -> dict:
        """Fast validation."""
        self.model.eval()
        
        val_losses = defaultdict(float)
        val_samples = 0
        
        with torch.no_grad():
            for t, age, xy_true, subject_idx in self.val_loader:
                t = t.to(self.device, non_blocking=True).requires_grad_(True)
                age = age.to(self.device, non_blocking=True)
                xy_true = xy_true.to(self.device, non_blocking=True) 
                subject_idx = subject_idx.to(self.device, non_blocking=True)
                
                # Forward pass with mixed precision
                with autocast(enabled=self.use_amp):
                    xy_pred, params = self.model(t, subject_idx)
                    data_loss = self.data_loss(xy_pred, xy_true)
                    physics_loss = self.physics_loss(t, xy_pred, params)
                    total_loss = data_loss + physics_loss
                
                batch_size = t.shape[0]
                val_losses['data'] += data_loss.item() * batch_size
                val_losses['physics'] += physics_loss.item() * batch_size
                val_losses['total'] += total_loss.item() * batch_size
                val_samples += batch_size
        
        return {k: v / val_samples for k, v in val_losses.items()}
    
    def train(self):
        """Run optimized training."""
        logger.info("=== STARTING OPTIMIZED TRAINING ===")
        
        epochs = self.config.get('epochs', 50)
        val_frequency = self.config.get('val_frequency', 5)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Learning rate scheduling
            if hasattr(self.scheduler, 'step'):
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(train_metrics['total'])
                else:
                    self.scheduler.step()
            
            # Validation (less frequent for speed)
            if epoch % val_frequency == 0 or epoch == epochs - 1:
                val_metrics = self.validate()
                val_loss = val_metrics['total']
                
                logger.info(f"Epoch {epoch+1}/{epochs}")
                logger.info(f"  Train: Data={train_metrics['data']:.6f}, "
                           f"Physics={train_metrics['physics']:.6f}, "
                           f"Total={train_metrics['total']:.6f}")
                logger.info(f"  Val:   Data={val_metrics['data']:.6f}, "
                           f"Physics={val_metrics['physics']:.6f}, "
                           f"Total={val_loss:.6f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(epoch, val_loss)
                
                # Early stopping
                if self.early_stopping(val_loss, self.model):
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                logger.info(f"Epoch {epoch+1}/{epochs}: "
                           f"Train Loss={train_metrics['total']:.6f}")
        
        logger.info("Training completed!")
        
    def save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.get('checkpoint_dir', 'checkpoints_fast'))
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_dir / 'best_model.pth')
        logger.info(f"Checkpoint saved: epoch {epoch+1}, val_loss={val_loss:.6f}")

def main():
    parser = argparse.ArgumentParser(description='Fast Balance PINN Training')
    parser.add_argument('--data-folder', type=str, default='processed_data')
    parser.add_argument('--age-csv-path', type=str, default='user_ages.csv')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--auto-config', action='store_true', 
                       help='Automatically detect optimal config for hardware')
    args = parser.parse_args()
    
    # Get optimized configuration
    if args.auto_config:
        config = detect_optimal_config()
        logger.info("Using auto-detected optimal configuration")
    else:
        # Manual high-performance config
        config = {
            'batch_size': 16384,
            'num_workers': 8,
            'mixed_precision': True,
            'learning_rate': 2e-3,
            'use_simplified_physics': True,
            'physics_computation_frequency': 4,
            'val_frequency': 5,
        }
    
    # Override with command line args
    config.update({
        'data_folder': args.data_folder,
        'age_csv_path': args.age_csv_path,
        'epochs': args.epochs,
        'checkpoint_dir': 'checkpoints_fast',
        'param_bounds': {
            'K': (500.0, 3000.0),
            'B': (20.0, 150.0), 
            'tau': (0.05, 0.4)
        }
    })
    
    # Run training
    trainer = FastTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()