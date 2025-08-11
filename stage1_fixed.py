#!/usr/bin/env python3
"""
Fixed Stage 1: Encouraging Parameter Diversity for Better Age Learning

This script fixes the Stage 1 model to encourage meaningful parameter
diversity across subjects, enabling better age-parameter relationships.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
# IMPORT DATASET FROM ORIGINAL
# =============================================================================

# Import dataset classes from original script
import sys
sys.path.append('.')

try:
    from amd_two_stage_trainer import AMDOptimizedDataset, create_amd_data_loaders, setup_amd_gpu
    logger.info("Imported dataset components from original script")
except ImportError as e:
    logger.error(f"Could not import from amd_two_stage_trainer.py: {e}")
    logger.error("Make sure amd_two_stage_trainer.py is in the current directory")
    exit(1)

# =============================================================================
# FIXED SUBJECT PINN WITH DIVERSITY
# =============================================================================

class DiverseSubjectPINN(nn.Module):
    """Fixed Subject PINN that encourages parameter diversity."""
    
    def __init__(self, subject_ids: List[str], hidden_dims: List[int] = [256, 256, 128, 128], 
                 param_bounds: Optional[Dict] = None, dropout_rate: float = 0.1):
        super().__init__()
        
        self.subject_ids = subject_ids
        self.n_subjects = len(subject_ids)
        self.param_bounds = param_bounds or {
            'K': (500.0, 3000.0), 'B': (20.0, 150.0), 'tau': (0.05, 0.4)
        }
        
        # Position network: (t, subject_idx) -> (x, y)
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
        
        layers.append(nn.Linear(hidden_dims[-1], 2))  # (x, y)
        self.position_net = nn.Sequential(*layers)
        
        # FIXED: Larger parameter network with more capacity
        self.param_net = nn.Sequential(
            nn.Linear(self.n_subjects, 512),  # Much larger
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 3)  # K, B, tau
        )
        
        # Initialize for diversity
        self.apply(self._init_weights_for_diversity)
        
        logger.info(f"Created diverse model for {self.n_subjects} subjects")
        logger.info(f"Parameter network: {self.n_subjects} -> 512 -> 256 -> 128 -> 3")
    
    def _init_weights_for_diversity(self, module):
        """Initialize weights to encourage parameter diversity."""
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                # Add random bias to encourage diversity
                if hasattr(module, 'out_features') and module.out_features == 3:  # Parameter output layer
                    torch.nn.init.uniform_(module.bias, -0.5, 0.5)  # More spread
                else:
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
        
        # Apply parameter bounds with sigmoid (no change here)
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

# =============================================================================
# FIXED PHYSICS LOSS WITH REAL EQUATIONS
# =============================================================================

class RealPhysicsLoss(nn.Module):
    """Real physics loss using balance equations."""
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        self.mse = nn.MSELoss()
    
    def forward(self, t: torch.Tensor, xy_pred: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Real physics loss using balance dynamics."""
        batch_size = t.shape[0]
        
        # Ensure t requires grad for derivatives
        if not t.requires_grad:
            t.requires_grad_(True)
            
        x_pred, y_pred = xy_pred[:, 0], xy_pred[:, 1]
        
        # Compute derivatives (expensive but necessary)
        dx_dt = torch.autograd.grad(x_pred.sum(), t, create_graph=True, retain_graph=True)[0].squeeze(-1)
        dy_dt = torch.autograd.grad(y_pred.sum(), t, create_graph=True, retain_graph=True)[0].squeeze(-1)
        
        # Second derivatives
        d2x_dt2 = torch.autograd.grad(dx_dt.sum(), t, create_graph=True, retain_graph=True)[0].squeeze(-1)
        d2y_dt2 = torch.autograd.grad(dy_dt.sum(), t, create_graph=True, retain_graph=True)[0].squeeze(-1)
        
        # Balance model parameters
        g, L, m = 9.81, 1.0, 70.0
        K, B, tau = params[:, 0], params[:, 1], params[:, 2]
        
        # Balance dynamics equations (from original train.py)
        residual_x = d2x_dt2 - (g/L)*x_pred + (K/(m*L**2))*x_pred + (B/(m*L**2))*dx_dt
        residual_y = d2y_dt2 - (g/L)*y_pred + (K/(m*L**2))*y_pred + (B/(m*L**2))*dy_dt
        
        # Physics loss - encourage residuals to be zero
        physics_loss = self.mse(residual_x, torch.zeros_like(residual_x)) + \
                      self.mse(residual_y, torch.zeros_like(residual_y))
        
        return physics_loss * self.weight

# =============================================================================
# PARAMETER DIVERSITY LOSS
# =============================================================================

class ParameterDiversityLoss(nn.Module):
    """Loss to encourage parameter diversity across subjects."""
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, params: torch.Tensor, subject_idx: torch.Tensor) -> torch.Tensor:
        """Encourage parameter diversity."""
        # Only compute if we have multiple subjects in batch
        unique_subjects = torch.unique(subject_idx)
        if len(unique_subjects) < 2:
            return torch.tensor(0.0, device=params.device)
        
        # Compute parameter statistics per subject in batch
        subject_params = {}
        for subj in unique_subjects:
            mask = subject_idx == subj
            if mask.sum() > 0:
                subject_params[subj.item()] = params[mask].mean(dim=0)
        
        if len(subject_params) < 2:
            return torch.tensor(0.0, device=params.device)
        
        # Stack subject parameters
        param_matrix = torch.stack(list(subject_params.values()))  # [n_subjects_in_batch, 3]
        
        # Encourage diversity by penalizing low variance
        param_stds = torch.std(param_matrix, dim=0)  # [3]
        
        # Target standard deviations (reasonable diversity)
        target_stds = torch.tensor([400.0, 25.0, 0.05], device=params.device)  # K, B, tau targets
        
        # Loss = penalty for being too uniform
        diversity_loss = torch.sum(torch.relu(target_stds - param_stds))
        
        return diversity_loss * self.weight

# =============================================================================
# FIXED TRAINER
# =============================================================================

class FixedStage1Trainer:
    """Fixed Stage 1 trainer encouraging parameter diversity."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device, self.gpu_memory = setup_amd_gpu()
        
        # Mixed precision setup
        self.use_amp = config.get('mixed_precision', False) and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        logger.info(f"Fixed trainer initialized - Mixed precision: {self.use_amp}")
        
        # Setup data
        self.setup_data()
        
    def setup_data(self):
        """Setup data loading."""
        logger.info("Setting up datasets...")
        
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
        """Train Stage 1 with diversity encouragement."""
        logger.info("="*60)
        logger.info("FIXED STAGE 1: DIVERSE PARAMETER LEARNING")
        logger.info("="*60)
        
        # Create diverse model
        model = DiverseSubjectPINN(
            subject_ids=list(self.dataset.valid_subjects),
            hidden_dims=self.config['hidden_dims'],
            param_bounds=self.config['param_bounds'],
            dropout_rate=self.config['dropout_rate']
        ).to(self.device)
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config['stage1_lr'],
            weight_decay=self.config['weight_decay'],
            eps=1e-6
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config['stage1_epochs']
        )
        
        # FIXED: Loss functions
        data_loss_fn = nn.MSELoss()
        physics_loss_fn = RealPhysicsLoss(weight=self.config['physics_weight'])
        diversity_loss_fn = ParameterDiversityLoss(weight=self.config['diversity_weight'])
        
        logger.info(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
        logger.info(f"Loss weights: Physics={self.config['physics_weight']}, Diversity={self.config['diversity_weight']}")
        
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
            
            pbar = tqdm(self.train_loader, desc=f"Fixed Stage 1 Epoch {epoch+1}")
            for batch_idx, (t, age, xy_true, subject_idx) in enumerate(pbar):
                
                # Move to device
                t = t.to(self.device, non_blocking=True).requires_grad_(True)
                xy_true = xy_true.to(self.device, non_blocking=True)
                subject_idx = subject_idx.to(self.device, non_blocking=True)
                
                # Forward pass
                with autocast(enabled=self.use_amp):
                    xy_pred, params = model(t, subject_idx)
                    
                    data_loss = data_loss_fn(xy_pred, xy_true)
                    
                    # Real physics loss (less frequent for speed)
                    if batch_idx % self.config['physics_frequency'] == 0:
                        physics_loss = physics_loss_fn(t, xy_pred, params)
                    else:
                        physics_loss = torch.tensor(0.0, device=self.device)
                    
                    # Diversity loss (encourage parameter variation)
                    diversity_loss = diversity_loss_fn(params, subject_idx)
                    
                    total_loss = data_loss + physics_loss + diversity_loss
                    
                    # Scale for gradient accumulation
                    scaled_loss = total_loss / self.config['gradient_accumulation_steps']
                
                # Backward pass
                if self.use_amp:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                
                accumulation_step += 1
                
                # Optimizer step
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
                train_losses['diversity'] += diversity_loss.item() * batch_size
                train_losses['total'] += total_loss.item() * batch_size
                train_samples += batch_size
                
                # Update progress bar
                pbar.set_postfix({
                    'Data': f"{data_loss.item():.1f}",
                    'Physics': f"{physics_loss.item():.4f}",
                    'Diversity': f"{diversity_loss.item():.2f}",
                    'LR': f"{optimizer.param_groups[0]['lr']:.2e}"
                })
                
                # Memory management
                if batch_idx % 25 == 0:
                    torch.cuda.empty_cache()
            
            scheduler.step()
            
            # Calculate average losses
            avg_train_losses = {k: v / train_samples for k, v in train_losses.items()}
            epoch_time = time.time() - epoch_start
            
            # Validation
            if epoch % 5 == 0 or epoch == self.config['stage1_epochs'] - 1:
                val_losses = self._validate_stage1(model, physics_loss_fn, data_loss_fn, diversity_loss_fn)
                val_loss = val_losses['total']
                
                logger.info(f"Epoch {epoch+1}/{self.config['stage1_epochs']} - {epoch_time:.1f}s")
                logger.info(f"  Train: Data={avg_train_losses['data']:.2f}, Physics={avg_train_losses['physics']:.5f}, Diversity={avg_train_losses['diversity']:.3f}")
                logger.info(f"  Val:   Data={val_losses['data']:.2f}, Physics={val_losses['physics']:.5f}, Diversity={val_losses['diversity']:.3f}")
                
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
                    }, 'fixed_stage1_model.pth')
                    logger.info(f"  ✅ New best model saved (val_loss={val_loss:.5f})")
                else:
                    patience_counter += 5
                
                if patience_counter >= self.config['stage1_patience']:
                    logger.info(f"  🛑 Early stopping at epoch {epoch+1}")
                    break
            else:
                logger.info(f"Epoch {epoch+1}/{self.config['stage1_epochs']} - {epoch_time:.1f}s - "
                           f"Train Loss: {avg_train_losses['total']:.2f}")
        
        # Extract subject parameters
        subject_parameters = self._extract_subject_parameters(model)
        
        logger.info(f"✅ FIXED STAGE 1 COMPLETE - Best val loss: {best_val_loss:.5f}")
        return {
            'model': model,
            'subject_parameters': subject_parameters,
            'best_val_loss': best_val_loss
        }
    
    def _validate_stage1(self, model, physics_loss_fn, data_loss_fn, diversity_loss_fn) -> Dict[str, float]:
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
                    diversity_loss = diversity_loss_fn(params, subject_idx)
                    total_loss = data_loss + physics_loss + diversity_loss
                
                batch_size = t.shape[0]
                val_losses['data'] += data_loss.item() * batch_size
                val_losses['physics'] += physics_loss.item() * batch_size
                val_losses['diversity'] += diversity_loss.item() * batch_size
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
        with open('fixed_subject_parameters.json', 'w') as f:
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
        logger.info(f"  K  variation: {K_cv:.3f} {'✅' if K_cv > 0.15 else '❌'} (target: >0.15)")
        logger.info(f"  B  variation: {B_cv:.3f} {'✅' if B_cv > 0.15 else '❌'} (target: >0.15)")
        logger.info(f"  τ  variation: {tau_cv:.3f} {'✅' if tau_cv > 0.15 else '❌'} (target: >0.15)")
        
        # Show parameter ranges
        logger.info(f"Parameter ranges:")
        logger.info(f"  K:  {min(Ks):.1f} - {max(Ks):.1f}")
        logger.info(f"  B:  {min(Bs):.1f} - {max(Bs):.1f}")
        logger.info(f"  τ:  {min(taus):.3f} - {max(taus):.3f}")
        
        return subject_parameters

# =============================================================================
# CONFIGURATION
# =============================================================================

def get_fixed_config() -> Dict:
    """Get fixed configuration encouraging diversity."""
    return {
        # Data settings
        'data_folder': 'processed_data',
        'age_csv_path': 'user_ages.csv',
        'param_bounds': {
            'K': (500.0, 3000.0),
            'B': (20.0, 150.0), 
            'tau': (0.05, 0.4)
        },
        
        # Model architecture
        'hidden_dims': [256, 256, 128, 128],
        'dropout_rate': 0.15,
        
        # Training parameters
        'stage1_epochs': 40,
        'stage1_lr': 2e-3,
        'weight_decay': 1e-5,
        'stage1_patience': 20,
        
        # Loss weights (FIXED)
        'physics_weight': 0.005,    # Reduced from 0.01 - less rigid
        'diversity_weight': 0.02,   # NEW - encourage parameter diversity
        
        # Memory optimization
        'batch_size': 1536,
        'gradient_accumulation_steps': 8,
        'mixed_precision': True,
        'physics_frequency': 8,     # Compute physics loss less frequently
    }

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("🔧 Fixed Stage 1: Encouraging Parameter Diversity")
    print("=" * 60)
    
    config = get_fixed_config()
    
    print(f"\nFixes implemented:")
    print(f"  ✅ Removed rigid parameter regularization") 
    print(f"  ✅ Added real physics loss equations")
    print(f"  ✅ Larger parameter network (231->512->256->128->3)")
    print(f"  ✅ Parameter diversity loss (weight={config['diversity_weight']})")
    print(f"  ✅ Reduced physics weight ({config['physics_weight']})")
    
    print(f"\nTraining configuration:")
    print(f"  Stage 1 epochs: {config['stage1_epochs']}")
    print(f"  Expected time: 30-45 minutes")
    
    # Check data files
    required_files = ['processed_data', 'user_ages.csv']
    for f in required_files:
        if not Path(f).exists():
            print(f"\n❌ Missing: {f}")
            return
    
    print("\n✅ All files found - starting fixed Stage 1...")
    
    try:
        # Run fixed Stage 1 training
        trainer = FixedStage1Trainer(config)
        results = trainer.train_stage1()
        
        # Success summary
        print("\n" + "="*60)
        print("🎉 FIXED STAGE 1 COMPLETED!")
        print("="*60)
        print(f"Best validation loss: {results['best_val_loss']:.5f}")
        print(f"Subjects with parameters: {len(results['subject_parameters'])}")
        
        print("\n📁 Generated files:")
        print("  - fixed_stage1_model.pth (diverse model)")
        print("  - fixed_subject_parameters.json (diverse parameters)")
        
        # Check if diversity was achieved
        params = results['subject_parameters']
        Ks = [p['K'] for p in params.values()]
        K_cv = np.std(Ks) / np.mean(Ks)
        
        if K_cv > 0.15:
            print(f"\n🎯 SUCCESS: Parameter diversity achieved (K CV: {K_cv:.3f})")
            print("  ➡️ Now run Stage 2 with: python stage2_fix.py")
            print("     (but update stage2_fix.py to load 'fixed_subject_parameters.json')")
        else:
            print(f"\n⚠️ Limited diversity achieved (K CV: {K_cv:.3f})")
            print("  Consider increasing diversity_weight or reducing physics_weight")
        
    except Exception as e:
        logger.error(f"Fixed Stage 1 training failed: {e}")
        raise

if __name__ == "__main__":
    main()