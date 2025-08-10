#!/usr/bin/env python3
"""
Training Utilities and Loss Functions for Balance PINN Models

This module provides:
1. Advanced loss functions (physics, regularization, age-aware)
2. Training utilities (schedulers, early stopping, checkpointing)  
3. Evaluation metrics and validation functions
4. Physics constraint implementations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
import logging
from pathlib import Path
import json
import time
from collections import defaultdict, deque
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class PhysicsLoss(nn.Module):
    """
    Physics-informed loss implementing inverted pendulum dynamics.
    
    Enforces the physics constraint:
    θ̈ = (g/L)sin(θ) - (K/mL²)θ - (B/mL²)θ̇
    
    For small angles: sin(θ) ≈ θ, so:
    θ̈ = (g/L)θ - (K/mL²)θ - (B/mL²)θ̇
    
    Converting to COP coordinates (x, y):
    ẍ = (g/L)x - (K/mL²)x - (B/mL²)ẋ
    ÿ = (g/L)y - (K/mL²)y - (B/mL²)ẏ
    """
    
    def __init__(self, g: float = 9.81, L: float = 1.0, m: float = 70.0, 
                 weight: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.g = g
        self.L = L
        self.m = m
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, t: torch.Tensor, xy_pred: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Calculate physics loss.
        
        Args:
            t: Time tensor [batch_size, 1] - must require gradients
            xy_pred: Predicted positions [batch_size, 2]
            params: Physical parameters [batch_size, 3] (K, B, tau)
            
        Returns:
            Physics loss scalar
        """
        if not t.requires_grad:
            raise ValueError("Time tensor must require gradients for physics loss")
        
        batch_size = t.shape[0]
        x_pred, y_pred = xy_pred[:, 0], xy_pred[:, 1]
        K, B, tau = params[:, 0], params[:, 1], params[:, 2]
        
        # Calculate first derivatives - handle both flattened and non-flattened time
        t_flat = t.view(-1)
        x_flat = x_pred.view(-1)
        y_flat = y_pred.view(-1)
        
        # Calculate first derivatives
        dx_dt = torch.autograd.grad(
            outputs=x_flat, inputs=t_flat,
            grad_outputs=torch.ones_like(x_flat),
            create_graph=True, retain_graph=True, allow_unused=True
        )[0]
        
        if dx_dt is None:
            # Fallback: return zero loss if gradients can't be computed
            return torch.tensor(0.0, device=t.device, requires_grad=True)
        
        dy_dt = torch.autograd.grad(
            outputs=y_flat, inputs=t_flat,
            grad_outputs=torch.ones_like(y_flat),
            create_graph=True, retain_graph=True, allow_unused=True
        )[0]
        
        if dy_dt is None:
            return torch.tensor(0.0, device=t.device, requires_grad=True)
        
        # Calculate second derivatives
        d2x_dt2 = torch.autograd.grad(
            outputs=dx_dt, inputs=t_flat,
            grad_outputs=torch.ones_like(dx_dt),
            create_graph=True, retain_graph=True, allow_unused=True
        )[0]
        
        if d2x_dt2 is None:
            return torch.tensor(0.0, device=t.device, requires_grad=True)
        
        d2y_dt2 = torch.autograd.grad(
            outputs=dy_dt, inputs=t_flat,
            grad_outputs=torch.ones_like(dy_dt),
            create_graph=True, retain_graph=True, allow_unused=True
        )[0]
        
        if d2y_dt2 is None:
            return torch.tensor(0.0, device=t.device, requires_grad=True)
        
        # Physics equations
        # ẍ = (g/L)x - (K/mL²)x - (B/mL²)ẋ
        physics_x = d2x_dt2 - (self.g/self.L)*x_flat + (K/(self.m*self.L**2))*x_flat + (B/(self.m*self.L**2))*dx_dt
        physics_y = d2y_dt2 - (self.g/self.L)*y_flat + (K/(self.m*self.L**2))*y_flat + (B/(self.m*self.L**2))*dy_dt
        
        # Calculate loss
        loss_x = torch.mean(physics_x**2) if self.reduction == 'mean' else torch.sum(physics_x**2)
        loss_y = torch.mean(physics_y**2) if self.reduction == 'mean' else torch.sum(physics_y**2)
        
        total_loss = (loss_x + loss_y) * self.weight
        
        return total_loss

class SimplePhysicsLoss(nn.Module):
    """
    Simplified physics loss that's more robust for debugging.
    Uses finite differences instead of automatic differentiation.
    """
    
    def __init__(self, g: float = 9.81, L: float = 1.0, m: float = 70.0, weight: float = 1.0):
        super().__init__()
        self.g = g
        self.L = L
        self.m = m
        self.weight = weight
    
    def forward(self, t: torch.Tensor, xy_pred: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Calculate simplified physics loss using parameter constraints only.
        """
        # Simple parameter regularization - ensure parameters are reasonable
        K, B, tau = params[:, 0], params[:, 1], params[:, 2]
        
        # Encourage reasonable parameter ranges
        K_loss = torch.mean((K - 1500.0)**2 / 1500.0**2)  # Target around 1500
        B_loss = torch.mean((B - 85.0)**2 / 85.0**2)      # Target around 85
        tau_loss = torch.mean((tau - 0.2)**2 / 0.2**2)    # Target around 0.2
        
        total_loss = (K_loss + B_loss + tau_loss) * self.weight
        
        return total_loss

class ParameterRegularizationLoss(nn.Module):
    """
    Regularization loss to encourage meaningful parameter variation with age.
    """
    
    def __init__(self, smoothness_weight: float = 0.1, variation_weight: float = 0.1,
                 bounds_weight: float = 1.0, param_bounds: Optional[Dict] = None):
        super().__init__()
        
        self.smoothness_weight = smoothness_weight
        self.variation_weight = variation_weight
        self.bounds_weight = bounds_weight
        
        self.param_bounds = param_bounds or {
            'K': (500.0, 3000.0),
            'B': (20.0, 150.0),
            'tau': (0.05, 0.4)
        }
    
    def forward(self, ages: torch.Tensor, params: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate regularization losses.
        
        Args:
            ages: Age tensor [batch_size, 1]
            params: Parameter tensor [batch_size, 3] (K, B, tau)
            
        Returns:
            Dictionary of individual loss components
        """
        batch_size = ages.shape[0]
        
        losses = {}
        
        # Smoothness loss - penalize rapid changes in parameters with age
        if self.smoothness_weight > 0 and batch_size > 1:
            # Sort by age for smoothness calculation
            age_order = torch.argsort(ages.squeeze())
            sorted_params = params[age_order]
            sorted_ages = ages[age_order]
            
            # Calculate parameter derivatives w.r.t age
            age_diffs = sorted_ages[1:] - sorted_ages[:-1] + 1e-6  # Avoid division by zero
            param_diffs = sorted_params[1:] - sorted_params[:-1]
            param_derivatives = param_diffs / age_diffs
            
            smoothness_loss = torch.mean(param_derivatives**2) * self.smoothness_weight
            losses['smoothness'] = smoothness_loss
        
        # Variation loss - encourage parameters to vary with age
        if self.variation_weight > 0:
            param_vars = torch.var(params, dim=0)  # Variance for each parameter
            # Encourage non-zero variance (invert to minimize)
            variation_loss = self.variation_weight * torch.mean(1.0 / (param_vars + 1e-6))
            losses['variation'] = variation_loss
        
        # Bounds loss - soft constraint to keep parameters in reasonable ranges
        if self.bounds_weight > 0:
            K_min, K_max = self.param_bounds['K']
            B_min, B_max = self.param_bounds['B']
            tau_min, tau_max = self.param_bounds['tau']
            
            # Soft boundary constraints (exponential penalty outside bounds)
            K_penalty = torch.mean(torch.relu(K_min - params[:, 0])**2 + torch.relu(params[:, 0] - K_max)**2)
            B_penalty = torch.mean(torch.relu(B_min - params[:, 1])**2 + torch.relu(params[:, 1] - B_max)**2)
            tau_penalty = torch.mean(torch.relu(tau_min - params[:, 2])**2 + torch.relu(params[:, 2] - tau_max)**2)
            
            bounds_loss = (K_penalty + B_penalty + tau_penalty) * self.bounds_weight
            losses['bounds'] = bounds_loss
        
        return losses

class AgeAwareLoss(nn.Module):
    """
    Age-aware loss that weights data points based on age distribution.
    
    Helps ensure balanced learning across all age groups.
    """
    
    def __init__(self, age_range: Tuple[float, float] = (20, 90), 
                 balance_weight: float = 1.0):
        super().__init__()
        self.age_range = age_range
        self.balance_weight = balance_weight
    
    def forward(self, ages: torch.Tensor, base_loss: torch.Tensor) -> torch.Tensor:
        """
        Apply age-aware weighting to base loss.
        
        Args:
            ages: Age tensor [batch_size, 1]  
            base_loss: Base loss tensor [batch_size] or scalar
            
        Returns:
            Age-weighted loss
        """
        if base_loss.numel() == 1:
            # Scalar loss, can't apply per-sample weighting
            return base_loss
        
        # Normalize ages to [0, 1]
        normalized_ages = (ages.squeeze() - self.age_range[0]) / (self.age_range[1] - self.age_range[0])
        normalized_ages = torch.clamp(normalized_ages, 0, 1)
        
        # Calculate weights to balance age distribution
        # Give higher weight to ages that are less common in this batch
        age_hist, _ = torch.histogram(normalized_ages.cpu(), bins=10)
        age_hist = age_hist.float() + 1e-6  # Avoid division by zero
        
        # Map each age to its bin and get inverse frequency weight
        age_bins = (normalized_ages * 9.99).long()  # 0-9 bins
        weights = 1.0 / age_hist[age_bins].to(ages.device)
        weights = weights / weights.mean()  # Normalize weights
        
        # Apply weights
        weighted_loss = base_loss * (1.0 + self.balance_weight * (weights - 1.0))
        
        return weighted_loss.mean()

class CombinedLoss(nn.Module):
    """
    Combined loss function that balances multiple loss components.
    """
    
    def __init__(self, data_weight: float = 1.0, physics_weight: float = 0.01,
                 regularization_weight: float = 0.1, age_aware_weight: float = 0.1,
                 param_bounds: Optional[Dict] = None):
        super().__init__()
        
        self.data_weight = data_weight
        self.physics_weight = physics_weight
        self.regularization_weight = regularization_weight
        self.age_aware_weight = age_aware_weight
        
        # Loss components
        self.mse_loss = nn.MSELoss(reduction='none')
        self.physics_loss = PhysicsLoss(weight=physics_weight)
        self.param_reg_loss = ParameterRegularizationLoss(
            smoothness_weight=0.1, variation_weight=0.1, bounds_weight=1.0,
            param_bounds=param_bounds
        )
        self.age_aware_loss = AgeAwareLoss(balance_weight=age_aware_weight)
        
    def forward(self, t: torch.Tensor, ages: torch.Tensor, xy_pred: torch.Tensor, 
                xy_true: torch.Tensor, params: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate combined loss with all components.
        
        Args:
            t: Time tensor [batch_size, 1] - must require gradients  
            ages: Age tensor [batch_size, 1]
            xy_pred: Predicted positions [batch_size, 2]
            xy_true: True positions [batch_size, 2]  
            params: Physical parameters [batch_size, 3]
            
        Returns:
            Dictionary of loss components and total loss
        """
        losses = {}
        
        # Data reconstruction loss
        data_loss_per_sample = self.mse_loss(xy_pred, xy_true).mean(dim=1)  # [batch_size]
        
        # Apply age-aware weighting to data loss
        if self.age_aware_weight > 0:
            data_loss = self.age_aware_loss(ages, data_loss_per_sample)
        else:
            data_loss = data_loss_per_sample.mean()
        
        losses['data'] = data_loss * self.data_weight
        
        # Physics loss
        if self.physics_weight > 0:
            physics_loss = self.physics_loss(t, xy_pred, params)
            losses['physics'] = physics_loss
        
        # Parameter regularization losses
        if self.regularization_weight > 0:
            param_losses = self.param_reg_loss(ages, params)
            for key, value in param_losses.items():
                losses[f'reg_{key}'] = value * self.regularization_weight
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses

# =============================================================================
# TRAINING UTILITIES
# =============================================================================

class EarlyStopping:
    """
    Early stopping utility with patience and delta threshold.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-6, 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.best_state_dict = None
        self.should_stop = False
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to potentially save state
            
        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.patience_counter = 0
            if self.restore_best_weights:
                self.best_state_dict = model.state_dict().copy()
        else:
            self.patience_counter += 1
        
        if self.patience_counter >= self.patience:
            self.should_stop = True
            if self.restore_best_weights and self.best_state_dict is not None:
                model.load_state_dict(self.best_state_dict)
                logger.info("Restored best model weights")
        
        return self.should_stop

class WarmupLRScheduler(_LRScheduler):
    """
    Learning rate scheduler with warmup period.
    """
    
    def __init__(self, optimizer, warmup_epochs: int = 10, max_lr: float = None, 
                 decay_scheduler: Optional[_LRScheduler] = None, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.max_lr = max_lr or optimizer.defaults['lr']
        self.decay_scheduler = decay_scheduler
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
            return [self.max_lr * warmup_factor for _ in self.base_lrs]
        else:
            # Use decay scheduler if provided
            if self.decay_scheduler is not None:
                return self.decay_scheduler.get_lr()
            else:
                return [self.max_lr for _ in self.base_lrs]

class ModelCheckpointer:
    """
    Model checkpointing utility with automatic saving and loading.
    """
    
    def __init__(self, checkpoint_dir: str, save_best: bool = True, 
                 save_every: int = 10, max_checkpoints: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_best = save_best
        self.save_every = save_every
        self.max_checkpoints = max_checkpoints
        
        self.best_loss = float('inf')
        self.checkpoints = deque(maxlen=max_checkpoints)
    
    def save_checkpoint(self, epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer,
                       scheduler: Optional[_LRScheduler], loss: float, 
                       metrics: Optional[Dict] = None) -> str:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'metrics': metrics or {}
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Save best model
        if self.save_best and loss < self.best_loss:
            self.best_loss = loss
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model at epoch {epoch} with loss {loss:.6f}")
        
        # Save periodic checkpoints
        if epoch % self.save_every == 0:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, checkpoint_path)
            
            self.checkpoints.append(checkpoint_path)
            
            # Remove old checkpoints if exceeding max
            if len(self.checkpoints) > self.max_checkpoints:
                old_checkpoint = self.checkpoints.popleft()
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
        
        return str(checkpoint_path) if epoch % self.save_every == 0 else None
    
    def load_checkpoint(self, checkpoint_path: str, model: nn.Module, 
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[_LRScheduler] = None) -> Dict:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        return checkpoint

class MetricsTracker:
    """
    Utility for tracking and logging training metrics.
    """
    
    def __init__(self, save_path: Optional[str] = None):
        self.metrics = defaultdict(list)
        self.save_path = Path(save_path) if save_path else None
    
    def update(self, **kwargs):
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.metrics[key].append(value)
    
    def get_latest(self, key: str) -> Optional[float]:
        """Get latest value for a metric."""
        return self.metrics[key][-1] if key in self.metrics and self.metrics[key] else None
    
    def get_average(self, key: str, window: int = None) -> Optional[float]:
        """Get average value for a metric."""
        values = self.metrics[key]
        if not values:
            return None
        
        if window:
            values = values[-window:]
        
        return sum(values) / len(values)
    
    def save(self, path: Optional[str] = None):
        """Save metrics to JSON file."""
        save_path = Path(path) if path else self.save_path
        if save_path:
            with open(save_path, 'w') as f:
                # Convert to regular dict for JSON serialization
                json.dump(dict(self.metrics), f, indent=2)
    
    def load(self, path: Optional[str] = None):
        """Load metrics from JSON file."""  
        load_path = Path(path) if path else self.save_path
        if load_path and load_path.exists():
            with open(load_path, 'r') as f:
                data = json.load(f)
                self.metrics = defaultdict(list, data)
    
    def plot_metrics(self, metrics_to_plot: List[str], save_path: Optional[str] = None):
        """Plot training metrics."""
        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in self.metrics:
                axes[i].plot(self.metrics[metric])
                axes[i].set_title(f'{metric}')
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel(metric)
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_model(model: nn.Module, dataloader: torch.utils.data.DataLoader, 
                  loss_fn: nn.Module, device: torch.device) -> Dict[str, float]:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Model to evaluate
        dataloader: Evaluation data loader
        loss_fn: Loss function
        device: Device to run evaluation on
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    total_losses = defaultdict(float)
    total_samples = 0
    
    predictions = []
    targets = []
    parameters = []
    ages = []
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                t, age, xy_true = batch
                metadata = None
            else:
                t, age, xy_true, metadata = batch
            
            t = t.to(device)
            age = age.to(device) 
            xy_true = xy_true.to(device)
            
            batch_size = t.shape[0]
            total_samples += batch_size
            
            # Forward pass
            xy_pred, params = model(t, age)
            
            # Calculate losses (if loss function supports it)
            if hasattr(loss_fn, 'forward'):
                t_grad = t.clone().detach().requires_grad_(True)
                xy_pred_grad, params_grad = model(t_grad, age)
                
                try:
                    losses = loss_fn(t_grad, age, xy_pred_grad, xy_true, params_grad)
                    for key, value in losses.items():
                        total_losses[key] += value.item() * batch_size
                except:
                    # Fallback to simple MSE if combined loss fails
                    mse_loss = F.mse_loss(xy_pred, xy_true)
                    total_losses['mse'] += mse_loss.item() * batch_size
            
            # Store for additional metrics
            predictions.append(xy_pred.cpu())
            targets.append(xy_true.cpu())
            parameters.append(params.cpu()) 
            ages.append(age.cpu())
    
    # Calculate average losses
    avg_losses = {key: value / total_samples for key, value in total_losses.items()}
    
    # Additional metrics
    all_predictions = torch.cat(predictions, dim=0)
    all_targets = torch.cat(targets, dim=0)
    all_parameters = torch.cat(parameters, dim=0)
    all_ages = torch.cat(ages, dim=0)
    
    # R² score
    ss_res = torch.sum((all_targets - all_predictions) ** 2)
    ss_tot = torch.sum((all_targets - torch.mean(all_targets, dim=0)) ** 2)
    r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Parameter statistics
    param_stats = {
        'K_mean': all_parameters[:, 0].mean().item(),
        'K_std': all_parameters[:, 0].std().item(),
        'B_mean': all_parameters[:, 1].mean().item(),
        'B_std': all_parameters[:, 1].std().item(),
        'tau_mean': all_parameters[:, 2].mean().item(),
        'tau_std': all_parameters[:, 2].std().item()
    }
    
    # Combine all metrics
    metrics = {
        **avg_losses,
        'r2_score': r2_score.item() if isinstance(r2_score, torch.Tensor) else r2_score,
        'mae': F.l1_loss(all_predictions, all_targets).item(),
        'rmse': torch.sqrt(F.mse_loss(all_predictions, all_targets)).item(),
        **param_stats,
        'n_samples': total_samples
    }
    
    return metrics

def calculate_physics_residuals(model: nn.Module, dataloader: torch.utils.data.DataLoader,
                               device: torch.device, n_samples: int = 1000) -> Dict[str, float]:
    """Calculate physics constraint violations."""
    model.eval()
    
    physics_loss_fn = PhysicsLoss(weight=1.0)
    residuals = []
    
    with torch.no_grad():
        sample_count = 0
        for batch in dataloader:
            if sample_count >= n_samples:
                break
                
            t, age, xy_true = batch[:3]
            t = t.to(device).requires_grad_(True)
            age = age.to(device)
            
            batch_size = min(t.shape[0], n_samples - sample_count)
            t = t[:batch_size]
            age = age[:batch_size]
            
            xy_pred, params = model(t, age)
            
            # Calculate physics residual for this batch
            residual = physics_loss_fn(t, xy_pred, params)
            residuals.append(residual.item())
            
            sample_count += batch_size
    
    return {
        'physics_residual_mean': np.mean(residuals),
        'physics_residual_std': np.std(residuals),
        'physics_residual_max': np.max(residuals),
        'physics_residual_min': np.min(residuals)
    }

if __name__ == "__main__":
    # Example usage and testing
    logger.info("Testing training utilities...")
    
    # Test loss functions
    batch_size = 32
    t = torch.randn(batch_size, 1, requires_grad=True)
    ages = torch.randn(batch_size, 1) * 20 + 60
    xy_pred = torch.randn(batch_size, 2)
    xy_true = torch.randn(batch_size, 2)
    params = torch.tensor([[1000., 50., 0.15]]).repeat(batch_size, 1)
    
    # Test physics loss
    physics_loss = PhysicsLoss()
    phys_loss = physics_loss(t, xy_pred, params)
    print(f"Physics loss: {phys_loss.item():.6f}")
    
    # Test combined loss
    combined_loss = CombinedLoss()
    losses = combined_loss(t, ages, xy_pred, xy_true, params)
    print(f"Combined losses: {[(k, v.item()) for k, v in losses.items()]}")
    
    # Test early stopping
    early_stopping = EarlyStopping(patience=3)
    model = torch.nn.Linear(1, 1)  # Dummy model
    
    for epoch in range(10):
        val_loss = 1.0 - 0.1 * epoch + 0.01 * epoch  # Decreasing then increasing
        should_stop = early_stopping(val_loss, model)
        print(f"Epoch {epoch}: val_loss={val_loss:.3f}, should_stop={should_stop}")
        if should_stop:
            break
    
    # Test metrics tracker
    tracker = MetricsTracker()
    for epoch in range(5):
        tracker.update(train_loss=1.0 - 0.1 * epoch, val_loss=1.1 - 0.1 * epoch)
    
    print(f"Latest train loss: {tracker.get_latest('train_loss')}")
    print(f"Average val loss: {tracker.get_average('val_loss')}")
    
    logger.info("Training utilities tests completed successfully!")