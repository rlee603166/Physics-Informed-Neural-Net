#!/usr/bin/env python3
"""
Revised AMD RX 6600 XT Optimized Two-Stage Balance PINN Training Script

This script implements the complete two-stage training pipeline with a true
physics-informed loss function to ensure meaningful parameter learning. It is
optimized for AMD RX 6600 XT GPU with 8GB VRAM and includes memory-efficient
training, gradient accumulation, and comprehensive monitoring.

Hardware Target: AMD Radeon RX 6600 XT (8GB VRAM, gfx1030)
Expected Training Time: 50-80 minutes total
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
            'batch_size': 1024,              # Reduced for more frequent physics computation
            'gradient_accumulation_steps': 8, 
            'effective_batch_size': 8192,
            
            # Data loading for single GPU
            'num_workers': 4,
            'pin_memory': True,
            'prefetch_factor': 2,
            'persistent_workers': True,
            
            # Training optimization
            'mixed_precision': True,
            
            # === FIX 1: COMPUTE PHYSICS EVERY BATCH ===
            'physics_computation_frequency': 1,  # Changed from 4 to 1 - critical fix!
            'validation_frequency': 5,
            
            # Model architecture - enhanced capacity
            'hidden_dims': [256, 256, 128, 128],
            'param_net_dims': [256, 128, 64],
            'dropout_rate': 0.15,
            
            # Training parameters (will be overridden by three-stage settings)
            'total_stage1_epochs': 60,  # Total epochs for stage 1 (sum of 3 stages)
            'stage2_epochs': 25,
            'stage1_lr': 1e-3,
            'stage2_lr': 1e-3,
            'weight_decay': 1e-5,
            
            # === NEW: THREE-STAGE TRAINING CURRICULUM ===
            'three_stage_training': True,
            'stage1_epochs': 5,    # Stage 1: Pure data fitting, no physics
            'stage2_epochs': 10,   # Stage 2: Gradual physics introduction
            'stage3_epochs': 5,    # REDUCED: was 45, now 5 for debugging (20 total epochs)
            
            # === PHYSICS WEIGHT SCHEDULE (DISABLED FOR DEBUGGING) ===
            'stage1_physics_weight': 0.0,    # No physics in stage 1
            'stage2_physics_weight': 0.0,    # DISABLED: was 50.0
            'stage3_physics_weight': 0.0,    # DISABLED: was 100.0
            
            # === PARAMETER NOISE SCHEDULE ===
            'stage1_noise_std': 0.1,   # High noise in stage 1
            'stage2_noise_std': 0.05,  # Medium noise in stage 2
            'stage3_noise_std': 0.02,  # Low noise in stage 3
            
            # === NEW LOSS COMPONENTS ===
            'centering_weight': 10.0,        # Force parameters toward center
            'variance_penalty_weight': 5.0,   # Penalize low parameter variance
            'smooth_boundary_weight': 3.0,    # Smooth boundary penalty
            'strong_correlation_weight': 50.0, # Strong anti-correlation penalty
            'btau_extra_correlation_weight': 2.0, # Extra B-tau penalty
            
            # === CENTERING TARGETS (physiologically reasonable centers) ===
            'centering_targets': {
                'K': 1250.0,   # Center of K range (500-2000)
                'B': 80.0,     # Center of B range (40-120)
                'tau': 0.2     # Center of tau range (0.1-0.3)
            },
            'centering_weights': {
                'K': 1.0,      # Standard weight for K centering
                'B': 0.5,      # Lower weight for B centering
                'tau': 0.5     # Lower weight for tau centering
            },
            
            # === LEGACY SETTINGS (now superseded by three-stage) ===
            'param_diversity_weight': 2.0,   # Kept for compatibility
            'param_bound_penalty': 5.0,      # Replaced by smooth boundary
            'stage1_correlation_weight': 15.0, # Replaced by strong correlation
            'use_curriculum_learning': False,  # Replaced by three-stage training
            
            'stage2_reg_weight': 0.1,
            
            # Early stopping
            'stage1_patience': 20,  # Increased patience
            'stage2_patience': 10,
            
            # Memory management
            'empty_cache_frequency': 25,
            'max_memory_fraction': 0.90,     # Use 90% of 8GB
        }
        print("🚀 AMD RX 6600 XT Optimized Config Loaded (Fixed Physics + Diversity)")
    else:
        # Fallback for smaller GPUs or CPU
        config = {
            'batch_size': 512,
            'gradient_accumulation_steps': 4,
            'effective_batch_size': 2048,
            'mixed_precision': False,
            'physics_computation_frequency': 1,
            'stage1_epochs': 40,
            'stage2_epochs': 20,
            'hidden_dims': [128, 128, 64],
            'param_net_dims': [128, 64],
            'stage1_physics_weight': 10.0,  # Reduced for normalized data
            'param_diversity_weight': 1.0,
            'param_bound_penalty': 2.0,
            'stage1_correlation_weight': 10.0,  # Conservative correlation penalty
            'use_curriculum_learning': False,
        }
        print("⚠️ Conservative config for limited memory")
    
    # Base configuration
    config.update({
        'data_folder': 'processed_data',
        # 'age_csv_path': 'user_ages_synthetic.csv',
        'age_csv_path': 'user_ages.csv',
        'random_seed': 42,
        'param_bounds': {
            'K': (500.0, 2000.0),    # Tightened from 3000 for more realistic stiffness
            'B': (40.0, 120.0),      # Tightened from 20-150 for physiological damping
            'tau': (0.1, 0.3)        # Tightened from 0.05-0.4 for realistic neural delays
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
        
        # === CRITICAL FIX: DATA NORMALIZATION PARAMETERS ===
        # COP data is in absolute force plate coordinates - need to center around equilibrium
        self.x_mean = 349.0  # Center X position (mm)
        self.x_scale = 10.0  # Scale for ~[-1,1] range (6mm variation / 0.6)
        self.y_mean = 925.0  # Center Y position (mm) 
        self.y_scale = 20.0  # Scale for ~[-1,1] range (37mm variation / 1.8)
        
        if not self.batch_files:
            raise FileNotFoundError(f"No HDF5 files found in {processed_data_folder}")
        
        self.age_lookup = {}
        if Path(age_csv_path).exists():
            age_df = pd.read_csv(age_csv_path)
            self.age_lookup = dict(zip(age_df['user_id'].astype(str), age_df['age']))
        
        self._build_index()
        logger.info(f"Data normalization: X=({self.x_mean}±{self.x_scale}), Y=({self.y_mean}±{self.y_scale})")
    
    def _build_index(self):
        self.data_points = []
        self.valid_subjects = set()
        subject_counts = defaultdict(int)
        
        logger.info("Building dataset index...")
        
        for file_idx, batch_file in enumerate(self.batch_files):
            try:
                import h5py
                with h5py.File(batch_file, 'r') as f:
                    for subject_group in f.keys():
                        if subject_group.startswith('subject_'):
                            subject_id = subject_group.replace('subject_', '')
                            for trial_key in f[subject_group].keys():
                                if trial_key.startswith('trial_'):
                                    trial_group = f[subject_group][trial_key]
                                    if 'cop_x' in trial_group:
                                        n_points = len(trial_group['cop_x'])
                                        subject_counts[subject_id] += n_points
                                        for point_idx in range(n_points):
                                            self.data_points.append({
                                                'file_idx': file_idx,
                                                'subject_id': subject_id,
                                                'trial_key': trial_key,
                                                'point_idx': point_idx
                                            })
            except Exception as e:
                logger.warning(f"Error reading {batch_file}: {e}")
        
        self.valid_subjects = {
            sid for sid, count in subject_counts.items() 
            if count >= self.min_points and sid in self.age_lookup
        }
        
        self.data_points = [dp for dp in self.data_points if dp['subject_id'] in self.valid_subjects]
        self.subject_to_idx = {sid: i for i, sid in enumerate(sorted(self.valid_subjects))}
        
        logger.info(f"Dataset built: {len(self.data_points):,} points from {len(self.valid_subjects)} subjects")
    
    def denormalize_positions(self, xy_normalized: torch.Tensor) -> torch.Tensor:
        """Convert normalized positions back to mm coordinates for analysis/visualization."""
        x_mm = xy_normalized[..., 0] * self.x_scale + self.x_mean
        y_mm = xy_normalized[..., 1] * self.y_scale + self.y_mean
        return torch.stack([x_mm, y_mm], dim=-1)
    
    def __len__(self):
        return len(self.data_points)
    
    def __getitem__(self, idx):
        dp = self.data_points[idx]
        try:
            import h5py
            with h5py.File(self.batch_files[dp['file_idx']], 'r') as f:
                subject_group = f'subject_{dp["subject_id"]}'
                trial_group = f[subject_group][dp['trial_key']]
                point_idx = dp['point_idx']
                
                # Load raw COP data and normalize to center around equilibrium
                x_raw = torch.tensor(trial_group['cop_x'][point_idx], dtype=torch.float32)
                y_raw = torch.tensor(trial_group['cop_y'][point_idx], dtype=torch.float32)
                
                # Normalize: center around 0 and scale to roughly [-1,1]
                x_normalized = (x_raw - self.x_mean) / self.x_scale
                y_normalized = (y_raw - self.y_mean) / self.y_scale
                
                xy = torch.stack([x_normalized, y_normalized])
                
                t = torch.tensor(point_idx / 106.0, dtype=torch.float32).unsqueeze(0)
                
                subject_id = dp['subject_id']
                age = torch.tensor(self.age_lookup.get(subject_id, 50.0), dtype=torch.float32)
                subject_idx = torch.tensor(self.subject_to_idx[subject_id], dtype=torch.long)
                
                return t, age, xy, subject_idx
        except Exception as e:
            logger.warning(f"Error loading point {idx}: {e}")
            return (torch.zeros(1), torch.tensor(50.0), torch.zeros(2), torch.tensor(0))

def create_amd_data_loaders(dataset, config: Dict):
    from sklearn.model_selection import train_test_split
    
    subjects = list(dataset.valid_subjects)
    train_subjects, temp_subjects = train_test_split(subjects, test_size=0.3, random_state=42)
    val_subjects, _ = train_test_split(temp_subjects, test_size=0.5, random_state=42)
    
    train_indices = [i for i, dp in enumerate(dataset.data_points) if dp['subject_id'] in train_subjects]
    val_indices = [i for i, dp in enumerate(dataset.data_points) if dp['subject_id'] in val_subjects]
    
    logger.info(f"Data splits: {len(train_indices):,} train, {len(val_indices):,} val")
    
    train_loader = DataLoader(
        Subset(dataset, train_indices), batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers'], pin_memory=config['pin_memory'],
        persistent_workers=config['persistent_workers'], prefetch_factor=config['prefetch_factor'],
        drop_last=True
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices), batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=config['pin_memory']
    )
    
    return train_loader, val_loader, train_subjects, val_subjects

# =============================================================================
# REVISED MEMORY MODELS
# =============================================================================

class CompactSubjectPINN(nn.Module):
    """Memory-optimized Subject PINN with enhanced capacity and better initialization."""
    
    def __init__(self, subject_ids: List[str], hidden_dims: List[int], 
                 param_net_dims: List[int], param_bounds: Dict, dropout_rate: float, dataset=None):
        super().__init__()
        
        self.subject_ids = subject_ids
        self.n_subjects = len(subject_ids)
        self.param_bounds = param_bounds
        self.dataset = dataset  # Store dataset for denormalization
        
        # Position network: (t, subject_idx) -> (x, y)
        pos_layers = []
        input_dim = 1 + self.n_subjects
        for i, hidden_dim in enumerate(hidden_dims):
            in_dim = input_dim if i == 0 else hidden_dims[i-1]
            pos_layers.extend([nn.Linear(in_dim, hidden_dim), nn.ELU(), nn.Dropout(dropout_rate)])
        pos_layers.append(nn.Linear(hidden_dims[-1], 2))
        self.position_net = nn.Sequential(*pos_layers)
        
        # Enhanced parameter network for greater capacity
        param_layers = []
        param_input_dim = self.n_subjects
        for i, hidden_dim in enumerate(param_net_dims):
            in_dim = param_input_dim if i == 0 else param_net_dims[i-1]
            param_layers.extend([nn.Linear(in_dim, hidden_dim), nn.ELU(), nn.Dropout(dropout_rate)])
        self.param_output = nn.Linear(param_net_dims[-1], 3)
        param_layers.append(self.param_output)
        self.param_net = nn.Sequential(*param_layers)
        
        self.apply(self._init_weights)
        
        # === FIX 5: BETTER INITIALIZATION FOR PARAMETERS ===
        # Initialize the final layer to output values in the middle of the parameter ranges
        with torch.no_grad():
            # Set biases to center sigmoid outputs around 0.5
            self.param_output.bias.data = torch.zeros(3)  # This gives sigmoid(0) = 0.5
            # Scale weights to prevent extreme initial values
            self.param_output.weight.data *= 0.1
            
            # === NEW: INITIALIZE POSITION NETWORK FOR NORMALIZED DATA ===
            # Position network should start with small outputs near 0 for centered data
            final_pos_layer = self.position_net[-1]  # Last Linear layer (positions)
            final_pos_layer.weight.data *= 0.01  # Small initial predictions near equilibrium
            final_pos_layer.bias.data.zero_()     # Start at origin (0,0)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, t: torch.Tensor, subject_idx: torch.Tensor, 
                noise_std: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        subject_onehot = F.one_hot(subject_idx, num_classes=self.n_subjects).float()
        
        pos_input = torch.cat([t, subject_onehot], dim=1)
        xy_pred = self.position_net(pos_input)
        
        params_raw = self.param_net(subject_onehot)
        
        K_min, K_max = self.param_bounds['K']
        B_min, B_max = self.param_bounds['B'] 
        tau_min, tau_max = self.param_bounds['tau']
        
        K = K_min + (K_max - K_min) * torch.sigmoid(params_raw[:, 0])
        B = B_min + (B_max - B_min) * torch.sigmoid(params_raw[:, 1])
        tau = tau_min + (tau_max - tau_min) * torch.sigmoid(params_raw[:, 2])
        
        params = torch.stack([K, B, tau], dim=1)
        
        # Add noise during training to prevent deterministic parameter collapse
        if self.training and noise_std > 0.0:
            noise = noise_std * torch.randn_like(params)
            params = params + noise
            
            # Clamp to bounds after adding noise (avoid in-place operations)
            K_clamped = torch.clamp(params[:, 0], K_min, K_max)
            B_clamped = torch.clamp(params[:, 1], B_min, B_max)
            tau_clamped = torch.clamp(params[:, 2], tau_min, tau_max)
            params = torch.stack([K_clamped, B_clamped, tau_clamped], dim=1)
        
        return xy_pred, params
    
    def get_parameters(self, subject_idx: int) -> Tuple[float, float, float]:
        with torch.no_grad():
            subject_onehot = F.one_hot(torch.tensor([subject_idx], device=next(self.parameters()).device), num_classes=self.n_subjects).float()
            params_raw = self.param_net(subject_onehot).squeeze()
            
            K_min, K_max = self.param_bounds['K']
            B_min, B_max = self.param_bounds['B']
            tau_min, tau_max = self.param_bounds['tau']
            
            K = K_min + (K_max - K_min) * torch.sigmoid(params_raw[0])
            B = B_min + (B_max - B_min) * torch.sigmoid(params_raw[1]) 
            tau = tau_min + (tau_max - tau_min) * torch.sigmoid(params_raw[2])
            
            return K.item(), B.item(), tau.item()
    
    def denormalize_positions(self, xy_normalized: torch.Tensor) -> torch.Tensor:
        """Convert normalized position predictions back to mm coordinates."""
        if self.dataset is None:
            raise ValueError("Dataset not provided - cannot denormalize positions")
        return self.dataset.denormalize_positions(xy_normalized)

class CompactAgeParameterModel(nn.Module):
    """Lightweight age parameter model."""
    def __init__(self, param_bounds: Dict):
        super().__init__()
        self.param_bounds = param_bounds
        self.age_net = nn.Sequential(
            nn.Linear(1, 64), nn.ELU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ELU(), nn.Linear(32, 6)
        )
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
    
    def predict_parameters(self, age: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        age_norm = (age - 50.0) / 30.0
        output = self.age_net(age_norm)
        means, log_stds = output[:, :3], output[:, 3:]
        stds = torch.exp(log_stds.clamp(-2, 2))
        
        K_min, K_max = self.param_bounds['K']
        B_min, B_max = self.param_bounds['B']
        tau_min, tau_max = self.param_bounds['tau']
        
        K_mean = K_min + (K_max - K_min) * torch.sigmoid(means[:, 0])
        B_mean = B_min + (B_max - B_min) * torch.sigmoid(means[:, 1])
        tau_mean = tau_min + (tau_max - tau_min) * torch.sigmoid(means[:, 2])
        
        param_means = torch.stack([K_mean, B_mean, tau_mean], dim=1)
        param_stds = stds * 0.1
        return param_means, param_stds

# =============================================================================
# PHYSICS-INFORMED LOSS FUNCTIONS
# =============================================================================

class BalancePhysicsLoss(nn.Module):
    """
    Enforces the physics of a damped harmonic oscillator with neural delay.
    Calculates the residual of the governing ODE: x'' + (B/m)x' + (K/m)x(t-τ) = 0
    """
    def __init__(self, weight: float = 1.0, mass: float = 70.0):
        super().__init__()
        self.weight = weight
        self.mass = mass  # Assumed average mass in kg

    def forward(self, t: torch.Tensor, xy_pred: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Calculates the physics residual loss."""
        K, B, tau = params[:, 0], params[:, 1], params[:, 2]

        # 1. Calculate first derivative (velocity) of xy_pred w.r.t. t
        vel = torch.autograd.grad(outputs=xy_pred, inputs=t, 
                                  grad_outputs=torch.ones_like(xy_pred), 
                                  create_graph=True)[0]

        # 2. Calculate second derivative (acceleration)
        accel = torch.autograd.grad(outputs=vel, inputs=t, 
                                    grad_outputs=torch.ones_like(vel), 
                                    create_graph=True)[0]

        # 3. Apply Taylor approximation for the delayed term: x(t-τ) ≈ x(t) - τ*x'(t)
        # Unsqueeze is needed for broadcasting: [B] -> [B, 1]
        tau_exp = tau.unsqueeze(1)
        xy_delayed = xy_pred - tau_exp * vel

        # 4. Calculate the physics residual for both x and y directions
        # The equation is: accel + (B/m)*vel + (K/m)*pos_delayed = 0
        B_exp = B.unsqueeze(1)
        K_exp = K.unsqueeze(1)
        
        residual = accel + (B_exp / self.mass) * vel + (K_exp / self.mass) * xy_delayed

        # 5. The loss is the mean squared error of the residuals
        physics_loss = F.mse_loss(residual, torch.zeros_like(residual))
        
        return physics_loss * self.weight

# === NEW: PARAMETER DIVERSITY AND BOUNDARY REGULARIZATION ===
class ParameterDiversityLoss(nn.Module):
    """Encourages diverse parameters across subjects and prevents boundary collapse."""
    
    def __init__(self, diversity_weight: float = 1.0, bound_penalty: float = 5.0, param_bounds: Dict = None):
        super().__init__()
        self.diversity_weight = diversity_weight
        self.bound_penalty = bound_penalty
        self.param_bounds = param_bounds
    
    def forward(self, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        params: [batch_size, 3] containing K, B, tau
        Returns: (diversity_loss, boundary_loss) as separate terms for monitoring
        """
        # 1. Diversity loss: Penalize low variance across batch
        # We want high variance, so we penalize the inverse
        param_std = torch.std(params, dim=0)
        # Add small epsilon to avoid division by zero and scale
        diversity_loss = torch.mean(1.0 / (param_std + 0.1))
        
        # 2. Boundary penalty: Penalize parameters stuck at bounds
        boundary_loss = torch.tensor(0.0, device=params.device)
        
        if self.param_bounds:
            K, B, tau = params[:, 0], params[:, 1], params[:, 2]
            
            K_min, K_max = self.param_bounds['K']
            B_min, B_max = self.param_bounds['B']
            tau_min, tau_max = self.param_bounds['tau']
            
            # Normalize parameters to [0, 1] range
            K_norm = (K - K_min) / (K_max - K_min)
            B_norm = (B - B_min) / (B_max - B_min)
            tau_norm = (tau - tau_min) / (tau_max - tau_min)
            
            # Penalize values very close to 0 or 1 (the bounds)
            # Using a smooth exponential penalty that increases near boundaries
            epsilon = 0.05  # Within 5% of bounds is penalized
            
            K_bound_penalty = torch.mean(torch.exp(-10 * torch.min(K_norm, 1 - K_norm) / epsilon))
            B_bound_penalty = torch.mean(torch.exp(-10 * torch.min(B_norm, 1 - B_norm) / epsilon))
            tau_bound_penalty = torch.mean(torch.exp(-10 * torch.min(tau_norm, 1 - tau_norm) / epsilon))
            
            boundary_loss = K_bound_penalty + B_bound_penalty + tau_bound_penalty
        
        return self.diversity_weight * diversity_loss, self.bound_penalty * boundary_loss

class ParameterCorrelationLoss(nn.Module):
    """
    Penalizes correlation between B (damping) and tau (neural delay) parameters.
    
    Addresses the biomechanical unrealism of correlated B and τ, which represent
    independent physiological processes:
    - B: Rate of energy dissipation in balance recovery
    - τ: Time lag in neural control responses
    """
    def __init__(self, weight: float = 15.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        params: [batch_size, 3] containing K, B, tau
        Returns: correlation loss penalizing B-tau correlation
        """
        B = params[:, 1]    # Damping parameters
        tau = params[:, 2]  # Neural delay parameters
        
        # Normalize parameters to zero mean for correlation calculation
        B_norm = (B - B.mean()) / (B.std() + 1e-8)
        tau_norm = (tau - tau.mean()) / (tau.std() + 1e-8)
        
        # Calculate Pearson correlation coefficient
        correlation = torch.mean(B_norm * tau_norm)
        
        # Penalize ANY strong correlation (positive OR negative)
        # Use abs() not relu() since current problem is negative correlation (-1.0)
        correlation_loss = torch.abs(correlation)
        
        return correlation_loss * self.weight

class MultiParameterCenteringLoss(nn.Module):
    """
    Forces parameters toward center values and penalizes low variance to prevent mode collapse.
    
    This loss addresses catastrophic parameter collapse by:
    1. Pulling parameters toward physiologically reasonable center values
    2. Penalizing low variance across subjects to encourage diversity
    3. Using weighted importance for different parameters
    """
    def __init__(self, centering_targets: Dict[str, float], centering_weights: Dict[str, float], 
                 param_bounds: Dict[str, Tuple[float, float]], variance_weight: float = 2.0):
        super().__init__()
        self.centering_targets = centering_targets
        self.centering_weights = centering_weights
        self.param_bounds = param_bounds
        self.variance_weight = variance_weight
    
    def forward(self, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        params: [batch_size, 3] containing K, B, tau
        Returns: (centering_loss, variance_loss) as separate components
        """
        K, B, tau = params[:, 0], params[:, 1], params[:, 2]
        
        # Normalize to [0, 1] range for consistent centering
        K_min, K_max = self.param_bounds['K']
        B_min, B_max = self.param_bounds['B']
        tau_min, tau_max = self.param_bounds['tau']
        
        K_norm = (K - K_min) / (K_max - K_min)
        B_norm = (B - B_min) / (B_max - B_min)
        tau_norm = (tau - tau_min) / (tau_max - tau_min)
        
        # Calculate normalized target positions (should be 0.5 for center)
        K_target_norm = (self.centering_targets['K'] - K_min) / (K_max - K_min)
        B_target_norm = (self.centering_targets['B'] - B_min) / (B_max - B_min)
        tau_target_norm = (self.centering_targets['tau'] - tau_min) / (tau_max - tau_min)
        
        # Centering losses - pull toward center values
        K_center_loss = self.centering_weights['K'] * F.mse_loss(K_norm, torch.full_like(K_norm, K_target_norm))
        B_center_loss = self.centering_weights['B'] * F.mse_loss(B_norm, torch.full_like(B_norm, B_target_norm))
        tau_center_loss = self.centering_weights['tau'] * F.mse_loss(tau_norm, torch.full_like(tau_norm, tau_target_norm))
        
        centering_loss = K_center_loss + B_center_loss + tau_center_loss
        
        # Variance penalty - penalize low diversity
        # We want HIGH variance, so we penalize 1/variance
        param_stds = torch.stack([K.std(), B.std(), tau.std()])
        variance_penalty = torch.mean(1.0 / (param_stds + 0.01))  # Prevent division by zero
        variance_loss = self.variance_weight * variance_penalty
        
        return centering_loss, variance_loss

class SmoothBoundaryPenalty(nn.Module):
    """
    Smooth tanh-based boundary penalty to replace harsh exponential penalties.
    
    Uses a gradual tanh-based penalty that smoothly increases as parameters
    approach bounds, avoiding the harsh cliff-like behavior of exponential penalties.
    """
    def __init__(self, weight: float = 5.0, margin: float = 0.1):
        super().__init__()
        self.weight = weight
        self.margin = margin  # How close to bounds before penalty kicks in
    
    def forward(self, params: torch.Tensor, param_bounds: Dict[str, Tuple[float, float]]) -> torch.Tensor:
        """
        params: [batch_size, 3] containing K, B, tau
        param_bounds: Dictionary with min/max bounds for each parameter
        Returns: smooth boundary penalty
        """
        K, B, tau = params[:, 0], params[:, 1], params[:, 2]
        
        # Normalize parameters to [0, 1] range
        K_min, K_max = param_bounds['K']
        B_min, B_max = param_bounds['B']
        tau_min, tau_max = param_bounds['tau']
        
        K_norm = (K - K_min) / (K_max - K_min)
        B_norm = (B - B_min) / (B_max - B_min)
        tau_norm = (tau - tau_min) / (tau_max - tau_min)
        
        boundary_penalty = 0.0
        
        # For each parameter, calculate distance from nearest bound
        for param_norm in [K_norm, B_norm, tau_norm]:
            # Distance from lower bound (0) and upper bound (1)
            dist_from_bounds = torch.min(param_norm, 1.0 - param_norm)
            
            # Apply smooth tanh penalty when within margin of bounds
            # tanh grows smoothly from 0 to 1 as distance approaches 0
            penalty = torch.tanh(self.margin / (dist_from_bounds + 1e-6))
            boundary_penalty += torch.mean(penalty)
        
        return self.weight * boundary_penalty

class StrongCorrelationLoss(nn.Module):
    """
    Enhanced correlation loss that penalizes correlation between ALL parameter pairs.
    
    Prevents mode collapse by ensuring K, B, and tau remain uncorrelated,
    which is biologically realistic since they represent independent processes.
    """
    def __init__(self, weight: float = 50.0, btau_extra_weight: float = 2.0):
        super().__init__()
        self.weight = weight
        self.btau_extra_weight = btau_extra_weight  # Extra penalty for B-tau correlation
    
    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        params: [batch_size, 3] containing K, B, tau
        Returns: total correlation penalty across all parameter pairs
        """
        K, B, tau = params[:, 0], params[:, 1], params[:, 2]
        
        # Normalize all parameters for correlation calculation
        K_norm = (K - K.mean()) / (K.std() + 1e-8)
        B_norm = (B - B.mean()) / (B.std() + 1e-8)
        tau_norm = (tau - tau.mean()) / (tau.std() + 1e-8)
        
        # Calculate all pairwise correlations
        K_B_corr = torch.abs(torch.mean(K_norm * B_norm))
        K_tau_corr = torch.abs(torch.mean(K_norm * tau_norm))
        B_tau_corr = torch.abs(torch.mean(B_norm * tau_norm))
        
        # Total correlation loss with extra weight on B-tau (the worst offender)
        total_correlation = K_B_corr + K_tau_corr + (self.btau_extra_weight * B_tau_corr)
        
        return self.weight * total_correlation

# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================

class AMDPerformanceMonitor:
    def __init__(self, device):
        self.device = device; self.batch_times = []; self.memory_usage = []; self.start_time = None
    def start_batch(self): self.start_time = time.time()
    def end_batch(self, batch_size: int):
        if self.start_time:
            self.batch_times.append(time.time() - self.start_time)
            if self.device.type == 'cuda': self.memory_usage.append(torch.cuda.memory_allocated() / (1024**3))
            if len(self.batch_times) % 50 == 0:
                avg_time = np.mean(self.batch_times[-50:]); samples_per_sec = batch_size / avg_time
                if self.device.type == 'cuda':
                    avg_mem = np.mean(self.memory_usage[-50:]); max_mem = torch.cuda.max_memory_allocated() / (1024**3)
                    print(f"    Perf: {samples_per_sec:.0f} smp/s, Mem: {avg_mem:.1f}/{max_mem:.1f}GB")
                else: print(f"    Perf: {samples_per_sec:.0f} smp/s (CPU)")

# =============================================================================
# MAIN TRAINING CLASS
# =============================================================================

class AMDTwoStageTrainer:
    def __init__(self, config: Dict):
        self.config = config
        self.device, self.gpu_memory = setup_amd_gpu()
        self.monitor = AMDPerformanceMonitor(self.device)
        self.use_amp = config.get('mixed_precision', False) and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        logger.info(f"Trainer initialized - Mixed precision: {self.use_amp}")
        self.setup_data()
    
    def setup_data(self):
        logger.info("Setting up AMD-optimized datasets...")
        self.dataset = AMDOptimizedDataset(self.config['data_folder'], self.config['age_csv_path'])
        self.train_loader, self.val_loader, self.train_subjects, self.val_subjects = create_amd_data_loaders(self.dataset, self.config)
        logger.info(f"Data ready: {len(self.train_loader)} train batches, {len(self.val_loader)} val batches")
    
    def train_stage1(self) -> Dict:
        logger.info("="*60 + "\nSTAGE 1: THREE-STAGE SUBJECT PARAMETER LEARNING\n" + "="*60)
        
        model = CompactSubjectPINN(
            subject_ids=list(self.dataset.valid_subjects),
            hidden_dims=self.config['hidden_dims'],
            param_net_dims=self.config['param_net_dims'],
            param_bounds=self.config['param_bounds'],
            dropout_rate=self.config['dropout_rate'],
            dataset=self.dataset  # Pass dataset for denormalization
        ).to(self.device)
        
        # Calculate total epochs for three-stage training
        if self.config.get('three_stage_training', False):
            stage1_epochs = self.config['stage1_epochs']  # 5 epochs
            stage2_epochs = self.config['stage2_epochs']  # 10 epochs  
            stage3_epochs = self.config['stage3_epochs']  # 45 epochs
            total_epochs = stage1_epochs + stage2_epochs + stage3_epochs
        else:
            total_epochs = self.config['total_stage1_epochs']
            stage1_epochs = stage2_epochs = stage3_epochs = 0
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config['stage1_lr'], weight_decay=self.config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
        
        # Initialize ALL loss functions
        data_loss_fn = nn.MSELoss()
        physics_loss_fn = BalancePhysicsLoss(weight=1.0).to(self.device)  # Weight will be set dynamically
        
        # NEW: Advanced loss functions for mode collapse prevention
        centering_loss_fn = MultiParameterCenteringLoss(
            centering_targets=self.config['centering_targets'],
            centering_weights=self.config['centering_weights'],
            param_bounds=self.config['param_bounds'],
            variance_weight=self.config['variance_penalty_weight']
        ).to(self.device)
        
        smooth_boundary_fn = SmoothBoundaryPenalty(
            weight=self.config['smooth_boundary_weight']
        ).to(self.device)
        
        strong_correlation_fn = StrongCorrelationLoss(
            weight=self.config['strong_correlation_weight'],
            btau_extra_weight=self.config['btau_extra_correlation_weight']
        ).to(self.device)
        
        logger.info(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
        logger.info(f"Effective batch size: {self.config['effective_batch_size']:,}")
        logger.info(f"Three-stage training: {stage1_epochs}+{stage2_epochs}+{stage3_epochs} = {total_epochs} epochs")
        
        best_val_loss, patience_counter = float('inf'), 0
        
        for epoch in range(total_epochs):
            epoch_start = time.time()
            
            # === THREE-STAGE CURRICULUM: DETERMINE CURRENT STAGE ===
            if self.config.get('three_stage_training', False):
                if epoch < stage1_epochs:
                    # Stage 1: Pure data fitting, no physics, high noise
                    current_stage = 1
                    current_physics_weight = self.config['stage1_physics_weight']  # 0.0
                    current_noise_std = self.config['stage1_noise_std']  # 0.1
                    stage_name = "PURE DATA FITTING"
                elif epoch < stage1_epochs + stage2_epochs:
                    # Stage 2: Gradual physics introduction, medium noise
                    current_stage = 2
                    current_physics_weight = self.config['stage2_physics_weight']  # 50.0
                    current_noise_std = self.config['stage2_noise_std']  # 0.05
                    stage_name = "GRADUAL PHYSICS"
                else:
                    # Stage 3: Full physics, low noise
                    current_stage = 3
                    current_physics_weight = self.config['stage3_physics_weight']  # 100.0
                    current_noise_std = self.config['stage3_noise_std']  # 0.02
                    stage_name = "FULL PHYSICS"
                
                physics_loss_fn.weight = current_physics_weight
                
                # Log stage transitions
                if epoch == 0 or epoch == stage1_epochs or epoch == stage1_epochs + stage2_epochs:
                    logger.info(f"🔄 ENTERING STAGE {current_stage}: {stage_name}")
                    logger.info(f"   Physics weight: {current_physics_weight}, Noise std: {current_noise_std}")
            else:
                # Legacy single-stage training
                current_noise_std = 0.0
                physics_loss_fn.weight = self.config.get('stage1_physics_weight', 20.0)
            
            model.train()
            train_losses = defaultdict(float)
            train_samples = 0
            optimizer.zero_grad()
            
            pbar = tqdm(self.train_loader, desc=f"Stage 1 Epoch {epoch+1}")
            for batch_idx, (t, _, xy_true, subject_idx) in enumerate(pbar):
                self.monitor.start_batch()
                t = t.to(self.device, non_blocking=True)
                t.requires_grad_(True)  # Critical for physics gradients
                xy_true = xy_true.to(self.device, non_blocking=True)
                subject_idx = subject_idx.to(self.device, non_blocking=True)
                
                with autocast(enabled=self.use_amp):
                    # Forward pass with parameter noise injection
                    xy_pred, params = model(t, subject_idx, noise_std=current_noise_std)
                    
                    # Data loss (always computed)
                    data_loss = data_loss_fn(xy_pred, xy_true)
                    
                    # Physics loss (varies by stage)
                    if current_physics_weight > 0:
                        physics_loss = physics_loss_fn(t, xy_pred, params)
                    else:
                        physics_loss = torch.tensor(0.0, device=self.device)
                    
                    # === NEW: ADVANCED REGULARIZATION FOR MODE COLLAPSE PREVENTION ===
                    # Centering loss - force parameters toward physiological centers
                    centering_loss, variance_loss = centering_loss_fn(params)
                    
                    # Smooth boundary penalty - prevent parameter collapse to bounds
                    boundary_loss = smooth_boundary_fn(params, self.config['param_bounds'])
                    
                    # Strong correlation penalty - prevent parameter correlation
                    correlation_loss = strong_correlation_fn(params)
                    
                    # Total loss combines all components with appropriate weighting
                    total_loss = (data_loss + physics_loss + 
                                 self.config['centering_weight'] * centering_loss +
                                 variance_loss +  # Already weighted in centering_loss_fn
                                 boundary_loss +  # Already weighted in smooth_boundary_fn
                                 correlation_loss)  # Already weighted in strong_correlation_fn
                    
                    scaled_loss = total_loss / self.config['gradient_accumulation_steps']
                
                if self.use_amp: 
                    self.scaler.scale(scaled_loss).backward()
                else: 
                    scaled_loss.backward()
                
                if (batch_idx + 1) % self.config['gradient_accumulation_steps'] == 0:
                    if self.use_amp:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                    optimizer.zero_grad()
                
                batch_size = t.shape[0]
                train_losses['data'] += data_loss.item() * batch_size
                train_losses['physics'] += physics_loss.item() * batch_size
                train_losses['centering'] += centering_loss.item() * batch_size
                train_losses['variance'] += variance_loss.item() * batch_size
                train_losses['boundary'] += boundary_loss.item() * batch_size
                train_losses['correlation'] += correlation_loss.item() * batch_size
                train_samples += batch_size
                
                pbar.set_postfix({
                    'Stage': current_stage if self.config.get('three_stage_training', False) else 'Legacy',
                    'Data': f"{data_loss.item():.4f}", 
                    'Phys': f"{physics_loss.item():.4f}",
                    'Cent': f"{centering_loss.item():.4f}",
                    'Corr': f"{correlation_loss.item():.4f}"
                })
                
                self.monitor.end_batch(batch_size)
                
                if batch_idx % self.config['empty_cache_frequency'] == 0: 
                    torch.cuda.empty_cache()

            scheduler.step()
            avg_train_loss = {k: v / train_samples for k, v in train_losses.items()}
            
            if (epoch + 1) % self.config['validation_frequency'] == 0 or epoch == 0 or epoch == total_epochs - 1:
                val_losses = self._validate_stage1(model, physics_loss_fn, data_loss_fn, 
                                                  centering_loss_fn, smooth_boundary_fn, 
                                                  strong_correlation_fn, current_physics_weight)
                
                stage_info = f"Stage {current_stage}" if self.config.get('three_stage_training', False) else "Legacy"
                logger.info(f"Epoch {epoch+1}/{total_epochs} ({time.time()-epoch_start:.1f}s) - {stage_info}")
                logger.info(f"  Train - Data: {avg_train_loss['data']:.4f}, Phys: {avg_train_loss['physics']:.4f}, Cent: {avg_train_loss['centering']:.4f}, Var: {avg_train_loss['variance']:.4f}, Bound: {avg_train_loss['boundary']:.4f}, Corr: {avg_train_loss['correlation']:.4f}")
                logger.info(f"  Val   - Data: {val_losses['data']:.4f}, Phys: {val_losses['physics']:.4f}, Corr: {val_losses['correlation']:.4f}, Total: {val_losses['total']:.4f}")
                
                # Check parameter diversity
                self._log_parameter_diversity(model, epoch)
                
                if val_losses['total'] < best_val_loss:
                    best_val_loss = val_losses['total']
                    patience_counter = 0
                    torch.save(model.state_dict(), 'best_stage1_model.pth')
                    logger.info(f"  ✅ New best model saved (val_loss={best_val_loss:.5f})")
                else:
                    patience_counter += self.config['validation_frequency']
                
                if patience_counter >= self.config['stage1_patience']:
                    logger.info(f"🛑 Early stopping at epoch {epoch+1}")
                    break
        
        model.load_state_dict(torch.load('best_stage1_model.pth'))
        subject_parameters = self._extract_subject_parameters(model)
        logger.info(f"✅ STAGE 1 COMPLETE - Best val loss: {best_val_loss:.5f}")
        return {'model': model, 'subject_parameters': subject_parameters, 'best_val_loss': best_val_loss}
    
    def _validate_stage1(self, model, physics_loss_fn, data_loss_fn, 
                        centering_loss_fn, smooth_boundary_fn, strong_correlation_fn,
                        current_physics_weight) -> Dict[str, float]:
        model.eval()
        val_losses = defaultdict(float)
        val_samples = 0
        
        # Don't use torch.no_grad() because physics loss needs gradients for derivatives
        for t, _, xy_true, subject_idx in self.val_loader:
            t = t.to(self.device)
            t.requires_grad_(True)  # Critical: physics loss needs gradients
            xy_true = xy_true.to(self.device)
            subject_idx = subject_idx.to(self.device)

            with autocast(enabled=self.use_amp):
                # No noise during validation
                xy_pred, params = model(t, subject_idx, noise_std=0.0)
                
                data_loss = data_loss_fn(xy_pred, xy_true)
                
                # Physics loss (only if weight > 0)
                if current_physics_weight > 0:
                    physics_loss = physics_loss_fn(t, xy_pred, params)
                else:
                    physics_loss = torch.tensor(0.0, device=self.device)
                
                centering_loss, variance_loss = centering_loss_fn(params)
                boundary_loss = smooth_boundary_fn(params, self.config['param_bounds'])
                correlation_loss = strong_correlation_fn(params)
                
                batch_size = t.shape[0]
                val_losses['data'] += data_loss.item() * batch_size
                val_losses['physics'] += physics_loss.item() * batch_size
                val_losses['centering'] += centering_loss.item() * batch_size
                val_losses['variance'] += variance_loss.item() * batch_size
                val_losses['boundary'] += boundary_loss.item() * batch_size
                val_losses['correlation'] += correlation_loss.item() * batch_size
                
                # Total validation loss
                total_loss = (data_loss.item() + physics_loss.item() + 
                             self.config['centering_weight'] * centering_loss.item() +
                             variance_loss.item() + boundary_loss.item() + correlation_loss.item())
                val_losses['total'] += total_loss * batch_size
                val_samples += batch_size
            
        return {k: v / val_samples for k, v in val_losses.items()} if val_samples > 0 else defaultdict(float)
    
    def _log_parameter_diversity(self, model, epoch):
        """Log statistics about learned parameter diversity."""
        model.eval()
        all_params = []
        
        with torch.no_grad():
            # Sample a subset of subjects to check diversity
            n_sample = min(100, len(self.dataset.valid_subjects))
            sample_indices = torch.randperm(len(self.dataset.valid_subjects))[:n_sample]
            
            for idx in sample_indices:
                K, B, tau = model.get_parameters(idx.item())
                all_params.append([K, B, tau])
        
        all_params = np.array(all_params)
        
        # Calculate coefficient of variation for each parameter
        K_cv = np.std(all_params[:, 0]) / (np.mean(all_params[:, 0]) + 1e-8)
        B_cv = np.std(all_params[:, 1]) / (np.mean(all_params[:, 1]) + 1e-8)
        tau_cv = np.std(all_params[:, 2]) / (np.mean(all_params[:, 2]) + 1e-8)
        
        logger.info(f"  Parameter Diversity (CV) - K: {K_cv:.3f}, B: {B_cv:.3f}, τ: {tau_cv:.3f}")
        
        # Check B-τ correlation specifically
        B_values = all_params[:, 1]
        tau_values = all_params[:, 2]
        B_tau_correlation = np.corrcoef(B_values, tau_values)[0, 1]
        correlation_status = '⚠️ HIGH' if abs(B_tau_correlation) > 0.7 else '✅ OK'
        logger.info(f"  B-τ correlation: {B_tau_correlation:.3f} {correlation_status}")
        
        # Check if parameters are stuck at bounds
        K_min, K_max = self.config['param_bounds']['K']
        B_min, B_max = self.config['param_bounds']['B']
        tau_min, tau_max = self.config['param_bounds']['tau']
        
        K_at_min = np.mean(np.abs(all_params[:, 0] - K_min) < 50)
        K_at_max = np.mean(np.abs(all_params[:, 0] - K_max) < 50)
        B_at_min = np.mean(np.abs(all_params[:, 1] - B_min) < 5)
        B_at_max = np.mean(np.abs(all_params[:, 1] - B_max) < 5)
        tau_at_min = np.mean(np.abs(all_params[:, 2] - tau_min) < 0.01)
        tau_at_max = np.mean(np.abs(all_params[:, 2] - tau_max) < 0.01)
        
        if K_at_min > 0.5 or K_at_max > 0.5:
            logger.warning(f"  ⚠️ {K_at_min*100:.0f}% of K at min, {K_at_max*100:.0f}% at max")
        if B_at_min > 0.5 or B_at_max > 0.5:
            logger.warning(f"  ⚠️ {B_at_min*100:.0f}% of B at min, {B_at_max*100:.0f}% at max")
        if tau_at_min > 0.5 or tau_at_max > 0.5:
            logger.warning(f"  ⚠️ {tau_at_min*100:.0f}% of τ at min, {tau_at_max*100:.0f}% at max")

    def _extract_subject_parameters(self, model) -> Dict:
        logger.info("Extracting subject parameters...")
        model.eval()
        subject_parameters = {}
        sorted_subjects = sorted(self.dataset.valid_subjects)
        subject_map = {sid: i for i, sid in enumerate(sorted_subjects)}

        with torch.no_grad():
            for subject_id in sorted_subjects:
                idx = subject_map[subject_id]
                K, B, tau = model.get_parameters(idx)
                age = self.dataset.age_lookup.get(subject_id, 50.0)
                subject_parameters[subject_id] = {'age': age, 'K': K, 'B': B, 'tau': tau}
        
        with open('subject_parameters.json', 'w') as f: 
            json.dump(subject_parameters, f, indent=2)
        
        Ks = [p['K'] for p in subject_parameters.values()]
        Bs = [p['B'] for p in subject_parameters.values()]
        taus = [p['tau'] for p in subject_parameters.values()]
        
        K_cv = np.std(Ks) / np.mean(Ks)
        B_cv = np.std(Bs) / np.mean(Bs)
        tau_cv = np.std(taus) / np.mean(taus)
        
        logger.info(f"Parameters extracted for {len(subject_parameters)} subjects:")
        logger.info(f"  K  variation (CV): {K_cv:.3f} {'✅ Good diversity' if K_cv > 0.15 else '⚠️ Low variation'}")
        logger.info(f"  B  variation (CV): {B_cv:.3f} {'✅ Good diversity' if B_cv > 0.15 else '⚠️ Low variation'}")
        logger.info(f"  τ  variation (CV): {tau_cv:.3f} {'✅ Good diversity' if tau_cv > 0.15 else '⚠️ Low variation'}")
        logger.info(f"  K  range: [{min(Ks):.1f}, {max(Ks):.1f}]")
        logger.info(f"  B  range: [{min(Bs):.1f}, {max(Bs):.1f}]")
        logger.info(f"  τ  range: [{min(taus):.3f}, {max(taus):.3f}]")
        
        return subject_parameters
    
    def train_stage2(self, subject_parameters: Dict) -> Dict:
        logger.info("="*60 + "\nSTAGE 2: AGE PARAMETER LEARNING\n" + "="*60)
        age_model = CompactAgeParameterModel(self.config['param_bounds']).to(self.device)
        optimizer = torch.optim.AdamW(age_model.parameters(), lr=self.config['stage2_lr'], weight_decay=self.config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['stage2_epochs'])
        train_ages, train_params = self._prepare_stage2_data(subject_parameters, 'train')
        val_ages, val_params = self._prepare_stage2_data(subject_parameters, 'val')
        logger.info(f"Stage 2 data: {len(train_ages)} train, {len(val_ages)} val subjects")
        best_val_loss, patience_counter = float('inf'), 0
        
        for epoch in range(self.config['stage2_epochs']):
            age_model.train()
            indices = torch.randperm(len(train_ages))
            train_ages_shuffled, train_params_shuffled = train_ages[indices].to(self.device), train_params[indices].to(self.device)
            batch_size = min(64, len(train_ages))
            
            epoch_losses = []
            for i in range(0, len(train_ages), batch_size):
                batch_ages, batch_params = train_ages_shuffled[i:i+batch_size], train_params_shuffled[i:i+batch_size]
                optimizer.zero_grad()
                
                with autocast(enabled=self.use_amp):
                    pred_means, pred_stds = age_model.predict_parameters(batch_ages)
                    param_loss = 0.5 * torch.mean(((batch_params - pred_means) / (pred_stds + 1e-6))**2 + torch.log(pred_stds + 1e-6))
                    reg_loss = self.config['stage2_reg_weight'] * torch.mean(pred_stds)
                    total_loss = param_loss + reg_loss
                
                if self.use_amp: 
                    self.scaler.scale(total_loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else: 
                    total_loss.backward()
                    optimizer.step()
                
                epoch_losses.append(total_loss.item())
            
            scheduler.step()
            val_losses = self._validate_stage2(age_model, val_ages, val_params)
            
            logger.info(f"Stage 2 Epoch {epoch+1} | Train Loss: {np.mean(epoch_losses):.4f} | Val Loss: {val_losses['total']:.4f}")
            
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                patience_counter = 0
                torch.save(age_model.state_dict(), 'best_stage2_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= self.config['stage2_patience']: 
                logger.info("🛑 Early stopping")
                break
        
        age_model.load_state_dict(torch.load('best_stage2_model.pth'))
        logger.info(f"✅ STAGE 2 COMPLETE - Best val loss: {best_val_loss:.5f}")
        return {'age_model': age_model, 'best_val_loss': best_val_loss}

    def _prepare_stage2_data(self, subject_parameters: Dict, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        subjects = self.train_subjects if split == 'train' else self.val_subjects
        ages, params = [], []
        for subject_id in subjects:
            if subject_id in subject_parameters:
                p = subject_parameters[subject_id]
                ages.append(p['age'])
                params.append([p['K'], p['B'], p['tau']])
        return torch.tensor(ages, dtype=torch.float32).unsqueeze(-1), torch.tensor(params, dtype=torch.float32)

    def _validate_stage2(self, age_model, val_ages, val_params) -> Dict[str, float]:
        age_model.eval()
        val_ages, val_params = val_ages.to(self.device), val_params.to(self.device)
        with torch.no_grad():
            with autocast(enabled=self.use_amp):
                pred_means, pred_stds = age_model.predict_parameters(val_ages)
                param_loss = 0.5 * torch.mean(((val_params - pred_means) / (pred_stds + 1e-6))**2 + torch.log(pred_stds + 1e-6))
                reg_loss = self.config['stage2_reg_weight'] * torch.mean(pred_stds)
        return {'total': (param_loss + reg_loss).item()}

    def analyze_and_visualize(self, age_model, subject_parameters: Dict):
        logger.info("="*60 + "\nMODEL ANALYSIS & VISUALIZATION\n" + "="*60)
        age_model.eval().to('cpu')
        ages_test = torch.linspace(20, 90, 100).unsqueeze(-1)
        with torch.no_grad():
            pred_means, pred_stds = age_model.predict_parameters(ages_test)
        
        df = pd.DataFrame(subject_parameters.values())
        
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharex=True)
        param_map = {'K': 'Stiffness (K)', 'B': 'Damping (B)', 'tau': 'Neural Delay (τ)'}
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, 3))
        
        for i, (p_key, p_name) in enumerate(param_map.items()):
            ax = axes[i]
            # Plot individual subject data
            ax.scatter(df['age'], df[p_key], alpha=0.6, s=35, color=colors[i], 
                      label='Learned Subjects', zorder=3, edgecolors='w', linewidth=0.5)
            # Plot learned age trend
            ax.plot(ages_test.numpy(), pred_means[:, i], 'r-', linewidth=3, 
                   label='Age Trend', zorder=2)
            # Plot uncertainty
            ax.fill_between(ages_test.numpy().flatten(), 
                            (pred_means[:, i] - 2*pred_stds[:, i]).numpy(), 
                            (pred_means[:, i] + 2*pred_stds[:, i]).numpy(),
                            alpha=0.2, color='red', label='95% Confidence Interval', zorder=1)
            ax.set_title(f'{p_name} vs. Age', fontsize=14, fontweight='bold')
            ax.set_ylabel(p_name, fontsize=12)
            ax.set_xlabel('Age (years)', fontsize=12)
            ax.legend(fontsize=10)
            
            # Add statistics
            correlation = np.corrcoef(df['age'], df[p_key])[0, 1]
            ax.text(0.05, 0.95, f'r = {correlation:.3f}', transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top')

        plt.suptitle('Learned Biomechanical Parameters vs. Age (Fixed Physics)', fontsize=18, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('amd_parameter_age_relationships_fixed.png', dpi=150)
        plt.show()
        
        # Print summary statistics
        logger.info("\nParameter-Age Correlations:")
        for p_key in ['K', 'B', 'tau']:
            correlation = np.corrcoef(df['age'], df[p_key])[0, 1]
            logger.info(f"  {p_key}: r = {correlation:.3f}")

    def run_complete_training(self):
        total_start_time = time.time()
        logger.info("🚀 STARTING THREE-STAGE ANTI-COLLAPSE AMD RX 6600 XT TRAINING")
        logger.info("Mode collapse fixes: Three-stage curriculum, parameter centering, noise injection, strong correlation penalty")
        
        try:
            stage1_results = self.train_stage1()
            stage2_results = self.train_stage2(stage1_results['subject_parameters'])
            self.analyze_and_visualize(stage2_results['age_model'], stage1_results['subject_parameters'])
            
            logger.info("="*60 + "\n🎉 TRAINING COMPLETED SUCCESSFULLY!\n" + "="*60)
            logger.info(f"Total training time: {(time.time() - total_start_time)/60:.1f} minutes")
            if torch.cuda.is_available(): 
                logger.info(f"GPU memory peak: {torch.cuda.max_memory_allocated()/(1024**3):.1f}GB")
            
        except Exception as e: 
            logger.error(f"Training failed: {e}", exc_info=True)
            raise

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("🔥 AMD RX 6600 XT Balance PINN Trainer (NORMALIZED DATA) 🔥")
    print("=" * 60)
    
    device, gpu_memory = setup_amd_gpu()
    config = get_amd_optimized_config(gpu_memory)
    
    data_folder = Path(config['data_folder'])
    age_csv = Path(config['age_csv_path'])
    if not data_folder.exists() or not age_csv.exists():
        print(f"\n❌ Data files not found! Ensure '{data_folder}' and '{age_csv}' exist.")
        return
    
    print("\n✅ All files found - starting training...")
    print("🔧 MODE COLLAPSE FIXES APPLIED:")
    print("  - THREE-STAGE CURRICULUM: Data fitting → Gradual physics → Full physics")
    print("  - PARAMETER CENTERING: Force K→1250, B→80, τ→0.2 centers")
    print("  - NOISE INJECTION: 0.1 → 0.05 → 0.02 parameter noise schedule")
    print("  - SMOOTH BOUNDARIES: Tanh-based penalty (replaces harsh exponential)")
    print("  - STRONG CORRELATION PENALTY: 50x weight, all parameter pairs")
    print("  - VARIANCE PENALTY: Explicitly penalize low parameter diversity")
    print("  - PHYSICS SCHEDULE: 0 → 50 → 100 weight progression")
    
    trainer = AMDTwoStageTrainer(config)
    trainer.run_complete_training()

if __name__ == "__main__":
    main()
