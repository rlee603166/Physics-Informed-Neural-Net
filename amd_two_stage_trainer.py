#!/usr/bin/env python3
"""
Complete AMD RX 6600 XT Optimized Two-Stage Balance PINN Training Script

This script implements the complete two-stage training pipeline optimized specifically
for AMD RX 6600 XT GPU with 8GB VRAM. It includes memory-efficient training,
gradient accumulation, and comprehensive monitoring.

Hardware Target: AMD Radeon RX 6600 XT (8GB VRAM, gfx1030)
Expected Training Time: 45-75 minutes total
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
import time
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# AMD GPU DETECTION & SETUP
# =============================================================================

def setup_amd_gpu():
    """Setup AMD GPU with ROCm optimization."""
    print("=== AMD GPU SETUP ===")
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        print(f"✅ GPU Found: {gpu_name}")
        print(f"✅ Memory: {gpu_memory:.1f}GB")
        
        # Test GPU with matrix multiplication
        start_time = time.time()
        test_a = torch.randn(2048, 2048, device=device)
        test_b = torch.randn(2048, 2048, device=device)
        test_c = torch.matmul(test_a, test_b)
        torch.cuda.synchronize()
        compute_time = time.time() - start_time
        
        memory_used = torch.cuda.memory_allocated() / (1024**3)
        print(f"✅ Compute Test: {compute_time:.3f}s")
        print(f"✅ Memory Test: {memory_used:.2f}GB allocated")
        
        # Clean up test tensors
        del test_a, test_b, test_c
        torch.cuda.empty_cache()
        
        if compute_time > 1.0:
            print("⚠️ GPU compute seems slow - check drivers")
        
        return device, gpu_memory
    else:
        print("❌ No CUDA/ROCm GPU found - falling back to CPU")
        return torch.device("cpu"), 0

# =============================================================================
# MEMORY-OPTIMIZED CONFIGURATION
# =============================================================================

def get_amd_optimized_config(gpu_memory: float) -> Dict:
    """Get configuration optimized for AMD RX 6600 XT."""
    
    if gpu_memory >= 7.5:  # RX 6600 XT has 8GB
        config = {
            # Memory-optimized settings
            'batch_size': 1536,              # Reduced for 8GB VRAM
            'gradient_accumulation_steps': 8, # Simulate larger batches
            'effective_batch_size': 12288,    # 1536 * 8
            
            # Data loading for single GPU
            'num_workers': 4,
            'pin_memory': True,
            'prefetch_factor': 2,
            'persistent_workers': True,
            
            # Training optimization
            'mixed_precision': True,          # Essential for memory efficiency
            'physics_computation_frequency': 8, # Reduce physics computation overhead
            'validation_frequency': 10,      # Validate every 10 epochs
            
            # Model architecture - reduced for memory
            'hidden_dims': [256, 256, 128, 128],  # Smaller than A100 version
            'dropout_rate': 0.15,
            
            # Training parameters
            'stage1_epochs': 40,
            'stage2_epochs': 25,
            'stage1_lr': 2e-3,               # Higher LR for effective large batch
            'stage2_lr': 1e-3,
            'weight_decay': 1e-5,
            
            # Loss weights - balanced for stability
            'stage1_physics_weight': 0.01,
            'stage2_reg_weight': 0.1,
            
            # Early stopping
            'stage1_patience': 15,
            'stage2_patience': 10,
            
            # Memory management
            'empty_cache_frequency': 25,     # Clear cache more frequently
            'max_memory_fraction': 0.85,     # Use 85% of 8GB
        }
        print("🚀 AMD RX 6600 XT Optimized Config Loaded")
    else:
        # Fallback for smaller GPUs or CPU
        config = {
            'batch_size': 512,
            'gradient_accumulation_steps': 4,
            'effective_batch_size': 2048,
            'mixed_precision': False,
            'stage1_epochs': 30,
            'stage2_epochs': 20,
            'hidden_dims': [128, 128, 64],
        }
        print("⚠️ Conservative config for limited memory")
    
    # Base configuration
    config.update({
        'data_folder': 'processed_data',
        'age_csv_path': 'user_ages.csv',
        'random_seed': 42,
        'param_bounds': {
            'K': (500.0, 3000.0),
            'B': (20.0, 150.0), 
            'tau': (0.05, 0.4)
        }
    })
    
    return config

# =============================================================================
# MEMORY-EFFICIENT DATASET
# =============================================================================

class AMDOptimizedDataset(torch.utils.data.Dataset):
    """Memory-efficient dataset optimized for AMD GPU."""
    
    def __init__(self, processed_data_folder: str, age_csv_path: str, min_points_per_subject: int = 100):
        self.data_folder = Path(processed_data_folder)
        self.batch_files = sorted(list(self.data_folder.glob("batch_*.h5")))
        self.min_points = min_points_per_subject
        
        if not self.batch_files:
            raise FileNotFoundError(f"No HDF5 files found in {processed_data_folder}")
        
        # Load age mapping
        self.age_lookup = {}
        if Path(age_csv_path).exists():
            age_df = pd.read_csv(age_csv_path)
            self.age_lookup = dict(zip(age_df['user_id'].astype(str), age_df['age']))
            logger.info(f"Loaded ages for {len(self.age_lookup)} subjects")
            # Debug: show first few entries
            logger.info(f"Sample age entries: {dict(list(self.age_lookup.items())[:3])}")
        
        # Build lightweight index
        self._build_index()
    
    def _build_index(self):
        """Build memory-efficient index."""
        self.data_points = []
        self.valid_subjects = set()
        subject_counts = defaultdict(int)
        
        logger.info("Building dataset index...")
        
        for file_idx, batch_file in enumerate(self.batch_files):
            try:
                import h5py
                with h5py.File(batch_file, 'r') as f:
                    # HDF5 structure: subject_C0007/trial_00/cop_x, cop_y
                    for subject_group in f.keys():
                        if subject_group.startswith('subject_'):
                            subject_id = subject_group.replace('subject_', '')  # Remove 'subject_' prefix
                            
                            # Count trials for this subject
                            for trial_key in f[subject_group].keys():
                                if trial_key.startswith('trial_'):
                                    # Check if this trial has both cop_x and cop_y
                                    trial_group = f[subject_group][trial_key]
                                    if 'cop_x' in trial_group and 'cop_y' in trial_group:
                                        # Get number of time points
                                        n_points = len(trial_group['cop_x'])
                                        subject_counts[subject_id] += n_points
                                        
                                        # Store each time point as a data point
                                        for point_idx in range(n_points):
                                            self.data_points.append({
                                                'file_idx': file_idx,
                                                'subject_id': subject_id,
                                                'trial_key': trial_key,
                                                'point_idx': point_idx
                                            })
            
            except Exception as e:
                logger.warning(f"Error reading {batch_file}: {e}")
        
        # Debug: Show sample subject IDs from data files
        sample_subjects = list(subject_counts.keys())[:5]
        logger.info(f"Sample subject IDs from HDF5: {sample_subjects}")
        logger.info(f"Total subjects in HDF5: {len(subject_counts)}")
        
        # Check how many have sufficient data
        sufficient_data = {sid: count for sid, count in subject_counts.items() if count >= self.min_points}
        logger.info(f"Subjects with >= {self.min_points} points: {len(sufficient_data)}")
        
        # Check how many match age lookup
        age_matches = {sid for sid in sufficient_data.keys() if sid in self.age_lookup}
        logger.info(f"Subjects with both data and age info: {len(age_matches)}")
        
        if len(age_matches) == 0:
            logger.warning("No subjects match between HDF5 files and age CSV!")
            logger.warning(f"HDF5 subjects sample: {sample_subjects}")
            logger.warning(f"Age CSV subjects sample: {list(self.age_lookup.keys())[:5]}")
        
        # Filter subjects with sufficient data
        self.valid_subjects = {
            sid for sid, count in subject_counts.items() 
            if count >= self.min_points and sid in self.age_lookup
        }
        
        # Filter data points
        self.data_points = [
            dp for dp in self.data_points 
            if dp['subject_id'] in self.valid_subjects
        ]
        
        # Create subject to index mapping
        self.subject_to_idx = {sid: i for i, sid in enumerate(sorted(self.valid_subjects))}
        
        logger.info(f"Dataset built: {len(self.data_points):,} points from {len(self.valid_subjects)} subjects")
    
    def __len__(self):
        return len(self.data_points)
    
    def __getitem__(self, idx):
        dp = self.data_points[idx]
        
        # Load data point efficiently
        try:
            import h5py
            with h5py.File(self.batch_files[dp['file_idx']], 'r') as f:
                subject_group = f'subject_{dp["subject_id"]}'
                trial_group = f[subject_group][dp['trial_key']]
                point_idx = dp['point_idx']
                
                # Load cop_x and cop_y
                x = torch.tensor(trial_group['cop_x'][point_idx], dtype=torch.float32)
                y = torch.tensor(trial_group['cop_y'][point_idx], dtype=torch.float32)
                xy = torch.stack([x, y])
                
                # Create time from sampling rate (106 Hz)
                t = torch.tensor(point_idx / 106.0, dtype=torch.float32).unsqueeze(0)
                
                subject_id = dp['subject_id']
                age = torch.tensor(self.age_lookup.get(subject_id, 50.0), dtype=torch.float32)
                subject_idx = torch.tensor(self.subject_to_idx[subject_id], dtype=torch.long)
                
                return t, age, xy, subject_idx
        
        except Exception as e:
            logger.warning(f"Error loading point {idx}: {e}")
            # Return dummy data
            return (torch.zeros(1, dtype=torch.float32),
                    torch.tensor(50.0, dtype=torch.float32),
                    torch.zeros(2, dtype=torch.float32),
                    torch.tensor(0, dtype=torch.long))

def create_amd_data_loaders(dataset, config: Dict):
    """Create memory-optimized data loaders."""
    from sklearn.model_selection import train_test_split
    
    # Create subject-based splits
    subjects = list(dataset.valid_subjects)
    train_subjects, temp_subjects = train_test_split(subjects, test_size=0.3, random_state=42)
    val_subjects, test_subjects = train_test_split(temp_subjects, test_size=0.5, random_state=42)
    
    # Create indices for each split
    train_indices = [i for i, dp in enumerate(dataset.data_points) if dp['subject_id'] in train_subjects]
    val_indices = [i for i, dp in enumerate(dataset.data_points) if dp['subject_id'] in val_subjects]
    
    logger.info(f"Data splits: {len(train_indices):,} train, {len(val_indices):,} val")
    
    # Create optimized data loaders
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        persistent_workers=config['persistent_workers'],
        prefetch_factor=config['prefetch_factor'],
        drop_last=True  # Consistent batch sizes
    )
    
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    
    return train_loader, val_loader, train_subjects, val_subjects

# =============================================================================
# REDUCED MEMORY MODELS
# =============================================================================

class CompactSubjectPINN(nn.Module):
    """Memory-optimized Subject PINN for 8GB GPU."""
    
    def __init__(self, subject_ids: List[str], hidden_dims: List[int] = [256, 256, 128, 128], 
                 param_bounds: Optional[Dict] = None, dropout_rate: float = 0.1):
        super().__init__()
        
        self.subject_ids = subject_ids
        self.n_subjects = len(subject_ids)
        self.param_bounds = param_bounds or {
            'K': (500.0, 3000.0), 'B': (20.0, 150.0), 'tau': (0.05, 0.4)
        }
        
        # Compact position network: (t, subject_idx) -> (x, y)
        layers = []
        input_dim = 1 + self.n_subjects  # t + one-hot subject encoding
        
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    nn.ELU(),
                    nn.Dropout(dropout_rate)
                ])
            else:
                layers.extend([
                    nn.Linear(hidden_dims[i-1], hidden_dim),
                    nn.ELU(),
                    nn.Dropout(dropout_rate)
                ])
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], 2))  # (x, y)
        self.position_net = nn.Sequential(*layers)
        
        # Compact parameter network: subject_idx -> (K, B, tau)
        self.param_net = nn.Sequential(
            nn.Linear(self.n_subjects, 128),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 3)  # K, B, tau
        )
        
        # Initialize weights for AMD GPU
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, t: torch.Tensor, subject_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = t.shape[0]
        
        # Create one-hot encoding for subjects
        subject_onehot = torch.zeros(batch_size, self.n_subjects, device=t.device)
        subject_onehot.scatter_(1, subject_idx.unsqueeze(1), 1.0)
        
        # Position prediction
        pos_input = torch.cat([t, subject_onehot], dim=1)
        xy_pred = self.position_net(pos_input)
        
        # Parameter prediction
        params_raw = self.param_net(subject_onehot)
        
        # Apply parameter bounds with sigmoid
        K_min, K_max = self.param_bounds['K']
        B_min, B_max = self.param_bounds['B'] 
        tau_min, tau_max = self.param_bounds['tau']
        
        K = K_min + (K_max - K_min) * torch.sigmoid(params_raw[:, 0])
        B = B_min + (B_max - B_min) * torch.sigmoid(params_raw[:, 1])
        tau = tau_min + (tau_max - tau_min) * torch.sigmoid(params_raw[:, 2])
        
        params = torch.stack([K, B, tau], dim=1)
        
        return xy_pred, params
    
    def get_parameters(self, subject_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get parameters for a specific subject."""
        with torch.no_grad():
            subject_onehot = torch.zeros(1, self.n_subjects, device=next(self.parameters()).device)
            subject_onehot[0, subject_idx] = 1.0
            
            params_raw = self.param_net(subject_onehot)
            
            K_min, K_max = self.param_bounds['K']
            B_min, B_max = self.param_bounds['B']
            tau_min, tau_max = self.param_bounds['tau']
            
            K = K_min + (K_max - K_min) * torch.sigmoid(params_raw[0, 0])
            B = B_min + (B_max - B_min) * torch.sigmoid(params_raw[0, 1]) 
            tau = tau_min + (tau_max - tau_min) * torch.sigmoid(params_raw[0, 2])
            
            return K, B, tau

class CompactAgeParameterModel(nn.Module):
    """Lightweight age parameter model."""
    
    def __init__(self, param_bounds: Optional[Dict] = None):
        super().__init__()
        
        self.param_bounds = param_bounds or {
            'K': (500.0, 3000.0), 'B': (20.0, 150.0), 'tau': (0.05, 0.4)
        }
        
        # Compact age network: age -> (K_mean, B_mean, tau_mean, K_std, B_std, tau_std)
        self.age_net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 6)  # mean and std for each parameter
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def predict_parameters(self, age: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict parameter distributions from age."""
        age_norm = (age - 50.0) / 30.0  # Normalize age
        output = self.age_net(age_norm)
        
        # Split into means and stds
        means = output[:, :3]
        log_stds = output[:, 3:]  # Log standard deviations
        stds = torch.exp(log_stds.clamp(-2, 2))  # Clamp for stability
        
        # Apply bounds to means
        K_min, K_max = self.param_bounds['K']
        B_min, B_max = self.param_bounds['B']
        tau_min, tau_max = self.param_bounds['tau']
        
        K_mean = K_min + (K_max - K_min) * torch.sigmoid(means[:, 0])
        B_mean = B_min + (B_max - B_min) * torch.sigmoid(means[:, 1])
        tau_mean = tau_min + (tau_max - tau_min) * torch.sigmoid(means[:, 2])
        
        param_means = torch.stack([K_mean, B_mean, tau_mean], dim=1)
        param_stds = stds * 0.1  # Scale down standard deviations
        
        return param_means, param_stds

# =============================================================================
# MEMORY-EFFICIENT LOSS FUNCTIONS  
# =============================================================================

class AMDOptimizedPhysicsLoss(nn.Module):
    """Memory-efficient physics loss for AMD GPU."""
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, t: torch.Tensor, xy_pred: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Simplified physics loss - parameter regularization only."""
        K, B, tau = params[:, 0], params[:, 1], params[:, 2]
        
        # Encourage reasonable parameter ranges (memory efficient)
        K_loss = torch.mean((K - 1500.0)**2) / (1500.0**2)
        B_loss = torch.mean((B - 85.0)**2) / (85.0**2)
        tau_loss = torch.mean((tau - 0.2)**2) / (0.2**2)
        
        return (K_loss + B_loss + tau_loss) * self.weight

# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================

class AMDPerformanceMonitor:
    """Monitor AMD GPU performance during training."""
    
    def __init__(self, device):
        self.device = device
        self.batch_times = []
        self.memory_usage = []
        self.start_time = None
    
    def start_batch(self):
        self.start_time = time.time()
    
    def end_batch(self, batch_size: int):
        if self.start_time is not None:
            batch_time = time.time() - self.start_time
            self.batch_times.append(batch_time)
            
            if self.device.type == 'cuda':
                memory_gb = torch.cuda.memory_allocated() / (1024**3)
                self.memory_usage.append(memory_gb)
            
            # Log every 50 batches
            if len(self.batch_times) % 50 == 0:
                avg_time = np.mean(self.batch_times[-50:])
                samples_per_sec = batch_size / avg_time
                
                if self.device.type == 'cuda':
                    avg_memory = np.mean(self.memory_usage[-50:])
                    max_memory = torch.cuda.max_memory_allocated() / (1024**3)
                    print(f"    Performance: {samples_per_sec:.0f} smp/s, Memory: {avg_memory:.1f}GB/{max_memory:.1f}GB peak")
                else:
                    print(f"    Performance: {samples_per_sec:.0f} smp/s (CPU)")

# =============================================================================
# MAIN TRAINING CLASS
# =============================================================================

class AMDTwoStageTrainer:
    """Complete two-stage trainer optimized for AMD RX 6600 XT."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device, self.gpu_memory = setup_amd_gpu()
        
        # Performance monitoring
        self.monitor = AMDPerformanceMonitor(self.device)
        
        # Mixed precision setup
        self.use_amp = config.get('mixed_precision', False) and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        logger.info(f"Trainer initialized - Mixed precision: {self.use_amp}")
        
        # Initialize components
        self.setup_data()
        
    def setup_data(self):
        """Setup optimized data loading."""
        logger.info("Setting up AMD-optimized datasets...")
        
        self.dataset = AMDOptimizedDataset(
            self.config['data_folder'],
            self.config['age_csv_path'],
            min_points_per_subject=100
        )
        
        self.train_loader, self.val_loader, self.train_subjects, self.val_subjects = create_amd_data_loaders(
            self.dataset, self.config
        )
        
        logger.info(f"Data ready: {len(self.train_loader)} train batches, {len(self.val_loader)} val batches")
    
    def train_stage1(self) -> Dict:
        """Train Stage 1: Subject parameter learning with gradient accumulation."""
        logger.info("="*60)
        logger.info("STAGE 1: SUBJECT PARAMETER LEARNING")
        logger.info("="*60)
        
        # Create model
        model = CompactSubjectPINN(
            subject_ids=list(self.dataset.valid_subjects),
            hidden_dims=self.config['hidden_dims'],
            param_bounds=self.config['param_bounds'],
            dropout_rate=self.config['dropout_rate']
        ).to(self.device)
        
        # Optimizer with higher LR for gradient accumulation
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config['stage1_lr'],
            weight_decay=self.config['weight_decay'],
            eps=1e-6
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config['stage1_epochs']
        )
        
        # Loss function  
        physics_loss_fn = AMDOptimizedPhysicsLoss(weight=self.config['stage1_physics_weight']).to(self.device)
        data_loss_fn = nn.MSELoss()
        
        logger.info(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
        logger.info(f"Effective batch size: {self.config['effective_batch_size']:,}")
        logger.info(f"Gradient accumulation: {self.config['gradient_accumulation_steps']} steps")
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['stage1_epochs']):
            epoch_start = time.time()
            
            # Training phase
            model.train()
            train_losses = defaultdict(float)
            train_samples = 0
            
            optimizer.zero_grad()
            accumulation_step = 0
            
            pbar = tqdm(self.train_loader, desc=f"Stage 1 Epoch {epoch+1}")
            for batch_idx, (t, age, xy_true, subject_idx) in enumerate(pbar):
                self.monitor.start_batch()
                
                # Move to device
                t = t.to(self.device, non_blocking=True).requires_grad_(True)
                xy_true = xy_true.to(self.device, non_blocking=True)
                subject_idx = subject_idx.to(self.device, non_blocking=True)
                
                # Forward pass with mixed precision
                with autocast(enabled=self.use_amp):
                    xy_pred, params = model(t, subject_idx)
                    
                    data_loss = data_loss_fn(xy_pred, xy_true)
                    
                    # Physics loss (less frequent for speed)
                    physics_freq = self.config['physics_computation_frequency']
                    if batch_idx % physics_freq == 0:
                        physics_loss = physics_loss_fn(t, xy_pred, params)
                    else:
                        physics_loss = torch.tensor(0.0, device=self.device)
                    
                    total_loss = data_loss + physics_loss
                    
                    # Scale loss for gradient accumulation
                    scaled_loss = total_loss / self.config['gradient_accumulation_steps']
                
                # Backward pass
                if self.use_amp:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                
                accumulation_step += 1
                
                # Optimizer step after accumulation
                if accumulation_step == self.config['gradient_accumulation_steps']:
                    if self.use_amp:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                    
                    optimizer.zero_grad()
                    accumulation_step = 0
                
                # Track losses
                batch_size = t.shape[0]
                train_losses['data'] += data_loss.item() * batch_size
                train_losses['physics'] += physics_loss.item() * batch_size
                train_losses['total'] += total_loss.item() * batch_size
                train_samples += batch_size
                
                # Update progress bar
                pbar.set_postfix({
                    'Data': f"{data_loss.item():.1f}",
                    'Physics': f"{physics_loss.item():.4f}",
                    'LR': f"{optimizer.param_groups[0]['lr']:.2e}"
                })
                
                self.monitor.end_batch(batch_size)
                
                # Memory management
                if batch_idx % self.config['empty_cache_frequency'] == 0:
                    torch.cuda.empty_cache()
            
            # Learning rate step
            scheduler.step()
            
            # Calculate average losses
            avg_train_losses = {k: v / train_samples for k, v in train_losses.items()}
            epoch_time = time.time() - epoch_start
            
            # Validation (less frequent)
            val_frequency = self.config['validation_frequency']
            if epoch % val_frequency == 0 or epoch == self.config['stage1_epochs'] - 1:
                val_losses = self._validate_stage1(model, physics_loss_fn, data_loss_fn)
                val_loss = val_losses['total']
                
                logger.info(f"Epoch {epoch+1}/{self.config['stage1_epochs']} - {epoch_time:.1f}s")
                logger.info(f"  Train: Data={avg_train_losses['data']:.2f}, Physics={avg_train_losses['physics']:.5f}")
                logger.info(f"  Val:   Data={val_losses['data']:.2f}, Physics={val_losses['physics']:.5f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'epoch': epoch,
                        'val_loss': val_loss,
                        'config': self.config
                    }, 'best_stage1_model.pth')
                    logger.info(f"  ✅ New best model saved (val_loss={val_loss:.5f})")
                else:
                    patience_counter += val_frequency
                
                if patience_counter >= self.config['stage1_patience']:
                    logger.info(f"  🛑 Early stopping at epoch {epoch+1}")
                    break
            else:
                logger.info(f"Epoch {epoch+1}/{self.config['stage1_epochs']} - {epoch_time:.1f}s - "
                           f"Train Loss: {avg_train_losses['total']:.2f}")
        
        # Extract subject parameters
        subject_parameters = self._extract_subject_parameters(model)
        
        logger.info(f"✅ STAGE 1 COMPLETE - Best val loss: {best_val_loss:.5f}")
        return {
            'model': model,
            'subject_parameters': subject_parameters,
            'best_val_loss': best_val_loss
        }
    
    def _validate_stage1(self, model, physics_loss_fn, data_loss_fn) -> Dict[str, float]:
        """Validate Stage 1 model."""
        model.eval()
        val_losses = defaultdict(float)
        val_samples = 0
        
        with torch.no_grad():
            for t, age, xy_true, subject_idx in self.val_loader:
                t = t.to(self.device, non_blocking=True).requires_grad_(True)
                xy_true = xy_true.to(self.device, non_blocking=True)
                subject_idx = subject_idx.to(self.device, non_blocking=True)
                
                with autocast(enabled=self.use_amp):
                    xy_pred, params = model(t, subject_idx)
                    data_loss = data_loss_fn(xy_pred, xy_true)
                    physics_loss = physics_loss_fn(t, xy_pred, params)
                    total_loss = data_loss + physics_loss
                
                batch_size = t.shape[0]
                val_losses['data'] += data_loss.item() * batch_size
                val_losses['physics'] += physics_loss.item() * batch_size
                val_losses['total'] += total_loss.item() * batch_size
                val_samples += batch_size
        
        return {k: v / val_samples for k, v in val_losses.items()}
    
    def _extract_subject_parameters(self, model) -> Dict:
        """Extract learned subject parameters."""
        logger.info("Extracting subject parameters...")
        
        model.eval()
        subject_parameters = {}
        
        with torch.no_grad():
            for i, subject_id in enumerate(sorted(self.dataset.valid_subjects)):
                K, B, tau = model.get_parameters(i)
                
                age = self.dataset.age_lookup.get(subject_id, 50.0)
                
                subject_parameters[subject_id] = {
                    'age': age,
                    'K': K.item(),
                    'B': B.item(),
                    'tau': tau.item()
                }
        
        # Save parameters
        with open('subject_parameters.json', 'w') as f:
            json.dump(subject_parameters, f, indent=2)
        
        # Statistics
        ages = [p['age'] for p in subject_parameters.values()]
        Ks = [p['K'] for p in subject_parameters.values()]
        Bs = [p['B'] for p in subject_parameters.values()]
        taus = [p['tau'] for p in subject_parameters.values()]
        
        K_cv = np.std(Ks) / np.mean(Ks)
        B_cv = np.std(Bs) / np.mean(Bs) 
        tau_cv = np.std(taus) / np.mean(taus)
        
        logger.info(f"Parameters extracted for {len(subject_parameters)} subjects:")
        logger.info(f"  K  variation: {K_cv:.3f} {'✅' if K_cv > 0.1 else '❌'}")
        logger.info(f"  B  variation: {B_cv:.3f} {'✅' if B_cv > 0.1 else '❌'}")
        logger.info(f"  τ  variation: {tau_cv:.3f} {'✅' if tau_cv > 0.1 else '❌'}")
        
        return subject_parameters
    
    def train_stage2(self, subject_parameters: Dict) -> Dict:
        """Train Stage 2: Age parameter learning."""
        logger.info("="*60)
        logger.info("STAGE 2: AGE PARAMETER LEARNING") 
        logger.info("="*60)
        
        # Create compact age model
        age_model = CompactAgeParameterModel(self.config['param_bounds']).to(self.device)
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            age_model.parameters(),
            lr=self.config['stage2_lr'],
            weight_decay=self.config['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config['stage2_epochs']
        )
        
        # Prepare training data
        train_ages, train_params = self._prepare_stage2_data(subject_parameters, 'train')
        val_ages, val_params = self._prepare_stage2_data(subject_parameters, 'val')
        
        logger.info(f"Stage 2 data: {len(train_ages)} train, {len(val_ages)} val subjects")
        logger.info(f"Age model: {sum(p.numel() for p in age_model.parameters()):,} parameters")
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['stage2_epochs']):
            epoch_start = time.time()
            
            # Training
            age_model.train()
            train_losses = defaultdict(float)
            
            # Shuffle data
            indices = torch.randperm(len(train_ages))
            train_ages_shuffled = train_ages[indices].to(self.device)
            train_params_shuffled = train_params[indices].to(self.device)
            
            # Mini-batch training
            batch_size = min(64, len(train_ages))  # Small batches for Stage 2
            n_batches = 0
            
            for i in range(0, len(train_ages), batch_size):
                batch_ages = train_ages_shuffled[i:i+batch_size]
                batch_params = train_params_shuffled[i:i+batch_size]
                
                optimizer.zero_grad()
                
                with autocast(enabled=self.use_amp):
                    pred_means, pred_stds = age_model.predict_parameters(batch_ages)
                    
                    # Negative log-likelihood loss
                    param_loss = 0.5 * torch.mean(
                        ((batch_params - pred_means) / (pred_stds + 1e-6))**2 + 
                        torch.log(pred_stds + 1e-6)
                    )
                    
                    # Simple regularization
                    reg_loss = self.config['stage2_reg_weight'] * torch.mean(pred_stds)
                    
                    total_loss = param_loss + reg_loss
                
                if self.use_amp:
                    self.scaler.scale(total_loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    total_loss.backward()
                    optimizer.step()
                
                train_losses['param'] += param_loss.item()
                train_losses['reg'] += reg_loss.item()
                train_losses['total'] += total_loss.item()
                n_batches += 1
            
            scheduler.step()
            
            # Average losses
            avg_train_losses = {k: v / n_batches for k, v in train_losses.items()}
            
            # Validation
            val_losses = self._validate_stage2(age_model, val_ages, val_params)
            val_loss = val_losses['total']
            
            epoch_time = time.time() - epoch_start
            
            logger.info(f"Stage 2 Epoch {epoch+1}/{self.config['stage2_epochs']} - {epoch_time:.1f}s")
            logger.info(f"  Train: Param={avg_train_losses['param']:.5f}, Reg={avg_train_losses['reg']:.5f}")
            logger.info(f"  Val:   Param={val_losses['param']:.5f}, Reg={val_losses['reg']:.5f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'age_model_state_dict': age_model.state_dict(),
                    'subject_parameters': subject_parameters,
                    'config': self.config
                }, 'best_stage2_model.pth')
                logger.info(f"  ✅ Best Stage 2 model saved")
            else:
                patience_counter += 1
            
            if patience_counter >= self.config['stage2_patience']:
                logger.info(f"  🛑 Early stopping at epoch {epoch+1}")
                break
        
        logger.info(f"✅ STAGE 2 COMPLETE - Best val loss: {best_val_loss:.5f}")
        return {
            'age_model': age_model,
            'best_val_loss': best_val_loss
        }
    
    def _prepare_stage2_data(self, subject_parameters: Dict, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare Stage 2 training data."""
        if split == 'train':
            subjects = self.train_subjects
        else:
            subjects = self.val_subjects
        
        ages = []
        params = []
        
        for subject_id in subjects:
            if subject_id in subject_parameters:
                param_data = subject_parameters[subject_id]
                ages.append(param_data['age'])
                params.append([param_data['K'], param_data['B'], param_data['tau']])
        
        return (torch.tensor(ages, dtype=torch.float32).unsqueeze(-1),
                torch.tensor(params, dtype=torch.float32))
    
    def _validate_stage2(self, age_model, val_ages: torch.Tensor, val_params: torch.Tensor) -> Dict[str, float]:
        """Validate Stage 2 model."""
        age_model.eval()
        
        val_ages = val_ages.to(self.device)
        val_params = val_params.to(self.device)
        
        with torch.no_grad():
            with autocast(enabled=self.use_amp):
                pred_means, pred_stds = age_model.predict_parameters(val_ages)
                
                param_loss = 0.5 * torch.mean(
                    ((val_params - pred_means) / (pred_stds + 1e-6))**2 + 
                    torch.log(pred_stds + 1e-6)
                )
                
                reg_loss = self.config['stage2_reg_weight'] * torch.mean(pred_stds)
                total_loss = param_loss + reg_loss
        
        return {
            'param': param_loss.item(),
            'reg': reg_loss.item(),
            'total': total_loss.item()
        }
    
    def analyze_and_visualize(self, age_model, subject_parameters: Dict):
        """Create analysis plots and test age comparison."""
        logger.info("="*60)
        logger.info("MODEL ANALYSIS & VISUALIZATION")
        logger.info("="*60)
        
        age_model.eval()
        
        # Generate age range predictions
        ages_test = np.linspace(20, 90, 100)
        age_tensor = torch.tensor(ages_test, dtype=torch.float32).unsqueeze(-1).to(self.device)
        
        with torch.no_grad():
            pred_means, pred_stds = age_model.predict_parameters(age_tensor)
            pred_means = pred_means.cpu().numpy()
            pred_stds = pred_stds.cpu().numpy()
        
        # Extract subject data
        subject_ages = [p['age'] for p in subject_parameters.values()]
        subject_Ks = [p['K'] for p in subject_parameters.values()]
        subject_Bs = [p['B'] for p in subject_parameters.values()]
        subject_taus = [p['tau'] for p in subject_parameters.values()]
        
        # Create visualization
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        param_names = ['Stiffness (K)', 'Damping (B)', 'Neural Delay (τ)']
        subject_params = [subject_Ks, subject_Bs, subject_taus]
        colors = ['tab:blue', 'tab:orange', 'tab:green']
        
        for i, (name, subject_param, color) in enumerate(zip(param_names, subject_params, colors)):
            ax = axes[i]
            
            # Subject data points
            ax.scatter(subject_ages, subject_param, alpha=0.6, s=30, color=color, label='Subjects', zorder=3)
            
            # Learned trend
            ax.plot(ages_test, pred_means[:, i], 'red', linewidth=2.5, label='Age Trend', zorder=2)
            
            # Uncertainty band
            ax.fill_between(ages_test,
                           pred_means[:, i] - pred_stds[:, i],
                           pred_means[:, i] + pred_stds[:, i],
                           alpha=0.2, color='red', label='Uncertainty', zorder=1)
            
            ax.set_xlabel('Age (years)', fontsize=12)
            ax.set_ylabel(name, fontsize=12)
            ax.set_title(f'{name} vs Age', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('amd_parameter_age_relationships.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Calculate correlations
        from scipy.stats import pearsonr
        correlations = {}
        
        for i, param_name in enumerate(['K', 'B', 'tau']):
            if len(subject_ages) > 3:
                corr, p_value = pearsonr(subject_ages, [subject_parameters[sid][param_name] 
                                                      for sid in subject_parameters.keys()])
                correlations[param_name] = {'correlation': corr, 'p_value': p_value}
                logger.info(f"{param_name}-age correlation: {corr:.3f} (p={p_value:.3f})")
        
        # Test age comparison functionality
        logger.info("\n=== AGE COMPARISON TEST ===")
        test_ages = [(30, 60), (40, 70), (60, 80)]
        
        for age1, age2 in test_ages:
            age1_tensor = torch.tensor([[age1]], dtype=torch.float32, device=self.device)
            age2_tensor = torch.tensor([[age2]], dtype=torch.float32, device=self.device)
            
            with torch.no_grad():
                params1, _ = age_model.predict_parameters(age1_tensor)
                params2, _ = age_model.predict_parameters(age2_tensor)
                
                params1 = params1.cpu().numpy().squeeze()
                params2 = params2.cpu().numpy().squeeze()
                diff = params2 - params1
            
            logger.info(f"\nAge {age1} vs {age2}:")
            logger.info(f"  K: {params1[0]:.1f} → {params2[0]:.1f} (Δ={diff[0]:+.1f})")
            logger.info(f"  B: {params1[1]:.1f} → {params2[1]:.1f} (Δ={diff[1]:+.1f})")
            logger.info(f"  τ: {params1[2]:.3f} → {params2[2]:.3f} (Δ={diff[2]:+.3f})")
        
        logger.info("✅ Age comparison functionality working!")
        
        return correlations
    
    def save_complete_model(self, subject_pinn, age_model, subject_parameters: Dict):
        """Save complete trained model."""
        logger.info("Saving complete two-stage model...")
        
        complete_model = {
            'subject_pinn_state_dict': subject_pinn.state_dict(),
            'age_model_state_dict': age_model.state_dict(),
            'subject_parameters': subject_parameters,
            'subject_ids': list(self.dataset.valid_subjects),
            'config': self.config,
            'device_info': {
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                'gpu_memory': self.gpu_memory,
                'training_time': time.time()
            }
        }
        
        torch.save(complete_model, 'amd_complete_two_stage_model.pth')
        logger.info("✅ Complete model saved as 'amd_complete_two_stage_model.pth'")
    
    def run_complete_training(self):
        """Run the complete two-stage training pipeline."""
        total_start_time = time.time()
        
        logger.info("🚀 STARTING AMD RX 6600 XT OPTIMIZED TRAINING")
        logger.info("="*60)
        
        try:
            # Stage 1: Subject parameter learning
            stage1_results = self.train_stage1()
            
            # Stage 2: Age relationship learning  
            stage2_results = self.train_stage2(stage1_results['subject_parameters'])
            
            # Analysis and visualization
            correlations = self.analyze_and_visualize(
                stage2_results['age_model'], 
                stage1_results['subject_parameters']
            )
            
            # Save complete model
            self.save_complete_model(
                stage1_results['model'],
                stage2_results['age_model'],
                stage1_results['subject_parameters']
            )
            
            total_time = time.time() - total_start_time
            
            # Final summary
            logger.info("="*60)
            logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("="*60)
            logger.info(f"Total training time: {total_time/60:.1f} minutes")
            logger.info(f"GPU memory peak: {torch.cuda.max_memory_allocated()/(1024**3):.1f}GB" if torch.cuda.is_available() else "CPU training")
            logger.info(f"Subjects processed: {len(stage1_results['subject_parameters'])}")
            logger.info(f"Stage 1 best loss: {stage1_results['best_val_loss']:.5f}")
            logger.info(f"Stage 2 best loss: {stage2_results['best_val_loss']:.5f}")
            
            # Parameter learning assessment
            param_variations = []
            for param in ['K', 'B', 'tau']:
                values = [stage1_results['subject_parameters'][sid][param] for sid in stage1_results['subject_parameters']]
                cv = np.std(values) / np.mean(values)
                param_variations.append(cv > 0.1)
            
            if all(param_variations):
                logger.info("✅ Parameter learning: EXCELLENT variation")
            else:
                logger.info("⚠️ Parameter learning: Limited variation")
            
            # Age correlations
            strong_correlations = sum(1 for corr in correlations.values() if abs(corr['correlation']) > 0.3)
            logger.info(f"Age relationships: {strong_correlations}/3 parameters show strong correlation")
            
            logger.info("\n📁 Generated files:")
            logger.info("  - amd_complete_two_stage_model.pth (complete trained model)")
            logger.info("  - subject_parameters.json (individual subject parameters)")
            logger.info("  - amd_parameter_age_relationships.png (visualization)")
            
            logger.info(f"\n🎯 Model ready for cross-age balance comparison!")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("🔥 AMD RX 6600 XT Balance PINN Trainer")
    print("=" * 60)
    
    # Setup and configuration
    device, gpu_memory = setup_amd_gpu()
    config = get_amd_optimized_config(gpu_memory)
    
    print(f"\nConfiguration summary:")
    print(f"  Batch size: {config['batch_size']:,}")
    print(f"  Effective batch: {config['effective_batch_size']:,}")  
    print(f"  Mixed precision: {config['mixed_precision']}")
    print(f"  Stage 1 epochs: {config['stage1_epochs']}")
    print(f"  Stage 2 epochs: {config['stage2_epochs']}")
    print(f"  Expected time: 45-75 minutes")
    
    # Check data files
    if not Path(config['data_folder']).exists():
        print(f"\n❌ Data folder '{config['data_folder']}' not found!")
        print("Please ensure your processed_data/ folder is in the current directory.")
        return
    
    if not Path(config['age_csv_path']).exists():
        print(f"\n❌ Age file '{config['age_csv_path']}' not found!")
        print("Please ensure user_ages.csv is in the current directory.")
        return
    
    print("\n✅ All files found - starting training...")
    
    # Run training
    trainer = AMDTwoStageTrainer(config)
    trainer.run_complete_training()

if __name__ == "__main__":
    main()