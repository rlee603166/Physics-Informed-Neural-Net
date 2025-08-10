#!/usr/bin/env python3
"""
Improved Balance PINN Model Architectures

This module implements two enhanced architectures for balance assessment:
1. Two-Stage Architecture: Separates individual parameter learning from age trends
2. Improved Single-Stage Architecture: Enhanced parameter network with regularization

Both models address the issues found in the original architecture and are designed
for effective cross-age balance comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# BASE COMPONENTS
# =============================================================================

class AgeNet(nn.Module):
    """Enhanced neural network for learning age-dependent parameter relationships."""
    
    def __init__(self, input_dim: int = 1, output_dim: int = 1, hidden_dims: List[int] = [128, 128, 64],
                 dropout_rate: float = 0.1, activation: str = 'elu'):
        super().__init__()
        
        self.activation = getattr(F, activation.lower()) if hasattr(F, activation.lower()) else F.elu
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, age: torch.Tensor) -> torch.Tensor:
        x = age
        for i, layer in enumerate(self.network):
            if isinstance(layer, nn.Linear):
                x = self.activation(layer(x)) if i < len(self.network) - 1 else layer(x)
            else:
                x = layer(x)
        return x

class PhysicsNet(nn.Module):
    """Physics-based trajectory prediction network."""
    
    def __init__(self, input_dim: int = 4, output_dim: int = 2, hidden_dims: List[int] = [256, 256, 256, 256],
                 activation: str = 'tanh'):
        super().__init__()
        
        self.activation = getattr(F, activation.lower()) if hasattr(F, activation.lower()) else F.tanh
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
        # Initialize weights for physics-informed learning
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.network):
            if i < len(self.network) - 1:
                x = self.activation(layer(x))
            else:
                x = layer(x)
        return x

# =============================================================================
# MODEL 1: TWO-STAGE ARCHITECTURE
# =============================================================================

class SubjectPINN(nn.Module):
    """
    Stage 1: Learn individual parameters for each subject.
    
    This model learns subject-specific physical parameters (K, B, τ) by directly
    optimizing them for each subject's data, ensuring physics constraints are satisfied.
    """
    
    def __init__(self, subject_ids: List[str], hidden_dims: List[int] = [256, 256, 256, 256],
                 param_bounds: Optional[Dict[str, Tuple[float, float]]] = None):
        super().__init__()
        
        self.subject_ids = subject_ids
        self.num_subjects = len(subject_ids)
        self.subject_to_idx = {sid: i for i, sid in enumerate(subject_ids)}
        
        # Default parameter bounds (physiologically reasonable)
        self.param_bounds = param_bounds or {
            'K': (500.0, 3000.0),    # Stiffness: 500-3000 N⋅m/rad
            'B': (20.0, 150.0),      # Damping: 20-150 N⋅m⋅s/rad  
            'tau': (0.05, 0.4)       # Neural delay: 50-400 ms
        }
        
        # Learnable parameters for each subject
        # Using log-space for better optimization
        self.log_K = nn.Parameter(torch.randn(self.num_subjects, 1) * 0.1)
        self.log_B = nn.Parameter(torch.randn(self.num_subjects, 1) * 0.1)
        self.logit_tau = nn.Parameter(torch.randn(self.num_subjects, 1) * 0.1)
        
        # Physics solver network
        self.physics_net = PhysicsNet(
            input_dim=4,  # [t, K, B, tau]
            output_dim=2,  # [x, y]
            hidden_dims=hidden_dims
        )
        
        # Initialize parameters to reasonable values
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize subject parameters to reasonable values."""
        K_min, K_max = self.param_bounds['K']
        B_min, B_max = self.param_bounds['B']
        tau_min, tau_max = self.param_bounds['tau']
        
        # Initialize to middle of ranges
        K_init = np.log((K_min + K_max) / 2)
        B_init = np.log((B_min + B_max) / 2)
        tau_init = np.log((tau_min + tau_max) / 2 / (1 - (tau_min + tau_max) / 2))
        
        nn.init.constant_(self.log_K, K_init)
        nn.init.constant_(self.log_B, B_init)
        nn.init.constant_(self.logit_tau, tau_init)
    
    def get_parameters(self, subject_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get physical parameters for given subjects."""
        K_min, K_max = self.param_bounds['K']
        B_min, B_max = self.param_bounds['B']
        tau_min, tau_max = self.param_bounds['tau']
        
        # Transform parameters to valid ranges
        K = torch.clamp(torch.exp(self.log_K[subject_indices]), K_min, K_max)
        B = torch.clamp(torch.exp(self.log_B[subject_indices]), B_min, B_max)
        tau = torch.clamp(tau_min + (tau_max - tau_min) * torch.sigmoid(self.logit_tau[subject_indices]), tau_min, tau_max)
        
        return K, B, tau
    
    def forward(self, t: torch.Tensor, subject_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            t: Time tensor [batch_size, 1]
            subject_indices: Subject index tensor [batch_size, 1]
            
        Returns:
            xy_pred: Predicted positions [batch_size, 2]
            params: Physical parameters [batch_size, 3]
        """
        K, B, tau = self.get_parameters(subject_indices.long().squeeze(-1))
        
        # Prepare input for physics network
        physics_input = torch.cat([t, K, B, tau], dim=1)
        xy_pred = self.physics_net(physics_input)
        
        params = torch.cat([K, B, tau], dim=1)
        return xy_pred, params

class AgeParameterModel(nn.Module):
    """
    Stage 2: Learn how parameters vary with age.
    
    Takes learned subject parameters from Stage 1 and learns smooth functions
    mapping age to physical parameters for population-level trends.
    """
    
    def __init__(self, param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                 use_probabilistic: bool = True):
        super().__init__()
        
        self.param_bounds = param_bounds or {
            'K': (500.0, 3000.0),
            'B': (20.0, 150.0), 
            'tau': (0.05, 0.4)
        }
        
        self.use_probabilistic = use_probabilistic
        
        # Age-to-parameter networks
        output_dim = 2 if use_probabilistic else 1
        
        self.K_net = AgeNet(input_dim=1, output_dim=output_dim, hidden_dims=[128, 128, 64])
        self.B_net = AgeNet(input_dim=1, output_dim=output_dim, hidden_dims=[128, 128, 64])
        self.tau_net = AgeNet(input_dim=1, output_dim=output_dim, hidden_dims=[128, 128, 64])
        
        # Physics solver network (same as Stage 1)
        self.physics_net = PhysicsNet(
            input_dim=4,  # [t, K, B, tau]
            output_dim=2,  # [x, y]
            hidden_dims=[256, 256, 256, 256]
        )
    
    def predict_parameters(self, age: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Predict parameters from age."""
        K_out = self.K_net(age)
        B_out = self.B_net(age)
        tau_out = self.tau_net(age)
        
        if self.use_probabilistic:
            # Split into mean and log_std
            K_mean, K_log_std = K_out[:, 0:1], K_out[:, 1:2]
            B_mean, B_log_std = B_out[:, 0:1], B_out[:, 1:2]
            tau_mean, tau_log_std = tau_out[:, 0:1], tau_out[:, 1:2]
            
            # Apply parameter bounds
            K_min, K_max = self.param_bounds['K']
            B_min, B_max = self.param_bounds['B']
            tau_min, tau_max = self.param_bounds['tau']
            
            K_mean = K_min + (K_max - K_min) * torch.sigmoid(K_mean)
            B_mean = B_min + (B_max - B_min) * torch.sigmoid(B_mean)
            tau_mean = tau_min + (tau_max - tau_min) * torch.sigmoid(tau_mean)
            
            K_std = torch.exp(K_log_std) * (K_max - K_min) * 0.1  # 10% of range
            B_std = torch.exp(B_log_std) * (B_max - B_min) * 0.1
            tau_std = torch.exp(tau_log_std) * (tau_max - tau_min) * 0.1
            
            means = torch.cat([K_mean, B_mean, tau_mean], dim=1)
            stds = torch.cat([K_std, B_std, tau_std], dim=1)
            
            return means, stds
        else:
            # Deterministic output
            K_min, K_max = self.param_bounds['K']
            B_min, B_max = self.param_bounds['B']
            tau_min, tau_max = self.param_bounds['tau']
            
            K = K_min + (K_max - K_min) * torch.sigmoid(K_out)
            B = B_min + (B_max - B_min) * torch.sigmoid(B_out)
            tau = tau_min + (tau_max - tau_min) * torch.sigmoid(tau_out)
            
            return torch.cat([K, B, tau], dim=1)
    
    def sample_parameters(self, age: torch.Tensor) -> torch.Tensor:
        """Sample parameters from learned distributions."""
        if self.use_probabilistic:
            means, stds = self.predict_parameters(age)
            # Sample using reparameterization trick
            epsilon = torch.randn_like(means)
            params = means + stds * epsilon
            
            # Clamp to bounds
            K_min, K_max = self.param_bounds['K']
            B_min, B_max = self.param_bounds['B']
            tau_min, tau_max = self.param_bounds['tau']
            
            params[:, 0] = torch.clamp(params[:, 0], K_min, K_max)
            params[:, 1] = torch.clamp(params[:, 1], B_min, B_max)
            params[:, 2] = torch.clamp(params[:, 2], tau_min, tau_max)
            
            return params
        else:
            return self.predict_parameters(age)
    
    def forward(self, t: torch.Tensor, age: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            t: Time tensor [batch_size, 1]
            age: Age tensor [batch_size, 1]
            deterministic: If True, use mean parameters instead of sampling
            
        Returns:
            xy_pred: Predicted positions [batch_size, 2]
            params: Physical parameters [batch_size, 3]
        """
        if deterministic and self.use_probabilistic:
            params, _ = self.predict_parameters(age)
        else:
            params = self.sample_parameters(age)
        
        # Prepare input for physics network
        physics_input = torch.cat([t, params], dim=1)
        xy_pred = self.physics_net(physics_input)
        
        return xy_pred, params

class TwoStagePINN(nn.Module):
    """
    Complete two-stage PINN system.
    
    Combines both stages for inference and provides utilities for training
    and cross-age comparison.
    """
    
    def __init__(self, subject_ids: List[str], param_bounds: Optional[Dict[str, Tuple[float, float]]] = None):
        super().__init__()
        
        self.subject_pinn = SubjectPINN(subject_ids, param_bounds=param_bounds)
        self.age_model = AgeParameterModel(param_bounds=param_bounds)
        
        self.subject_ids = subject_ids
        self.param_bounds = param_bounds
    
    def compare_ages(self, subject_data: Dict, target_age: float, n_samples: int = 100) -> Dict:
        """
        Compare a subject's balance to a target age group.
        
        Args:
            subject_data: Dict with subject's parameters
            target_age: Age to compare against
            n_samples: Number of samples for comparison
            
        Returns:
            Comparison metrics and similarity scores
        """
        self.eval()
        
        with torch.no_grad():
            age_tensor = torch.tensor([[target_age]], dtype=torch.float32)
            
            # Get target age parameter distribution
            if self.age_model.use_probabilistic:
                age_means, age_stds = self.age_model.predict_parameters(age_tensor)
                age_dist = Normal(age_means, age_stds)
                age_samples = age_dist.sample((n_samples,))
            else:
                age_params = self.age_model.predict_parameters(age_tensor)
                age_samples = age_params.repeat(n_samples, 1, 1)
            
            # Compare subject parameters to age distribution
            subject_params = torch.tensor([
                subject_data['K'], subject_data['B'], subject_data['tau']
            ]).unsqueeze(0)
            
            # Calculate similarity scores
            similarities = []
            for i in range(n_samples):
                param_diff = torch.abs(subject_params - age_samples[i])
                similarity = torch.exp(-param_diff.sum()).item()
                similarities.append(similarity)
            
            return {
                'mean_similarity': np.mean(similarities),
                'std_similarity': np.std(similarities),
                'age_param_mean': age_means.squeeze().numpy() if self.age_model.use_probabilistic else age_params.squeeze().numpy(),
                'age_param_std': age_stds.squeeze().numpy() if self.age_model.use_probabilistic else None,
                'subject_params': subject_params.squeeze().numpy()
            }
    
    def find_balance_age(self, subject_data: Dict, age_range: Tuple[float, float] = (20, 90), 
                        resolution: int = 100) -> Dict:
        """
        Find the age that best matches a subject's balance parameters.
        
        Args:
            subject_data: Dict with subject's parameters
            age_range: Age range to search
            resolution: Number of ages to test
            
        Returns:
            Best matching age and confidence scores
        """
        self.eval()
        
        ages = np.linspace(age_range[0], age_range[1], resolution)
        similarities = []
        
        with torch.no_grad():
            subject_params = torch.tensor([
                subject_data['K'], subject_data['B'], subject_data['tau']
            ]).unsqueeze(0)
            
            for age in ages:
                comparison = self.compare_ages(subject_data, age, n_samples=50)
                similarities.append(comparison['mean_similarity'])
        
        best_idx = np.argmax(similarities)
        best_age = ages[best_idx]
        confidence = similarities[best_idx]
        
        return {
            'balance_age': best_age,
            'confidence': confidence,
            'age_similarities': list(zip(ages, similarities)),
            'actual_age': subject_data.get('age', None)
        }

# =============================================================================
# MODEL 2: IMPROVED SINGLE-STAGE ARCHITECTURE  
# =============================================================================

class ImprovedBalancePINN(nn.Module):
    """
    Enhanced single-stage PINN with improved parameter learning.
    
    Key improvements:
    - Larger, more capable parameter network
    - Probabilistic parameter outputs
    - Age-aware regularization
    - Better loss balancing
    """
    
    def __init__(self, hidden_dim: int = 256, num_layers: int = 8, 
                 param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                 use_probabilistic: bool = True, dropout_rate: float = 0.1):
        super().__init__()
        
        self.param_bounds = param_bounds or {
            'K': (500.0, 3000.0),
            'B': (20.0, 150.0),
            'tau': (0.05, 0.4)
        }
        
        self.use_probabilistic = use_probabilistic
        
        # Enhanced parameter network
        output_dim = 6 if use_probabilistic else 3  # (mean, std) for each param or just mean
        
        param_layers = []
        param_layers.extend([
            nn.Linear(1, 128),
            nn.ELU(),
            nn.Dropout(dropout_rate)
        ])
        
        for _ in range(2):  # Additional layers
            param_layers.extend([
                nn.Linear(128, 128),
                nn.ELU(),
                nn.Dropout(dropout_rate)
            ])
        
        param_layers.extend([
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, output_dim)
        ])
        
        self.parameter_net = nn.Sequential(*param_layers)
        
        # Enhanced solver network
        solver_layers = []
        input_dim = 1 + 3  # time + parameters
        
        solver_layers.extend([nn.Linear(input_dim, hidden_dim), nn.Tanh()])
        
        for _ in range(num_layers - 1):
            solver_layers.extend([
                nn.Linear(hidden_dim, hidden_dim), 
                nn.Tanh()
            ])
        
        solver_layers.append(nn.Linear(hidden_dim, 2))  # x, y output
        
        self.solver_net = nn.Sequential(*solver_layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0)
    
    def predict_parameters(self, age: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Predict parameters from age."""
        param_output = self.parameter_net(age)
        
        if self.use_probabilistic:
            # Split into mean and log_std
            param_means = param_output[:, :3]
            param_log_stds = param_output[:, 3:]
            
            # Apply parameter bounds to means
            K_min, K_max = self.param_bounds['K']
            B_min, B_max = self.param_bounds['B']
            tau_min, tau_max = self.param_bounds['tau']
            
            K_mean = K_min + (K_max - K_min) * torch.sigmoid(param_means[:, 0:1])
            B_mean = B_min + (B_max - B_min) * torch.sigmoid(param_means[:, 1:2])
            tau_mean = tau_min + (tau_max - tau_min) * torch.sigmoid(param_means[:, 2:3])
            
            # Standard deviations (as fraction of parameter range)
            K_std = torch.exp(param_log_stds[:, 0:1]) * (K_max - K_min) * 0.1
            B_std = torch.exp(param_log_stds[:, 1:2]) * (B_max - B_min) * 0.1  
            tau_std = torch.exp(param_log_stds[:, 2:3]) * (tau_max - tau_min) * 0.1
            
            means = torch.cat([K_mean, B_mean, tau_mean], dim=1)
            stds = torch.cat([K_std, B_std, tau_std], dim=1)
            
            return means, stds
        else:
            # Deterministic output with parameter bounds
            K_min, K_max = self.param_bounds['K']
            B_min, B_max = self.param_bounds['B']
            tau_min, tau_max = self.param_bounds['tau']
            
            K = K_min + (K_max - K_min) * torch.sigmoid(param_output[:, 0:1])
            B = B_min + (B_max - B_min) * torch.sigmoid(param_output[:, 1:2])
            tau = tau_min + (tau_max - tau_min) * torch.sigmoid(param_output[:, 2:3])
            
            return torch.cat([K, B, tau], dim=1)
    
    def sample_parameters(self, age: torch.Tensor) -> torch.Tensor:
        """Sample parameters from learned distributions."""
        if self.use_probabilistic:
            means, stds = self.predict_parameters(age)
            # Sample using reparameterization trick
            epsilon = torch.randn_like(means)
            params = means + stds * epsilon
            
            # Clamp to bounds
            K_min, K_max = self.param_bounds['K']
            B_min, B_max = self.param_bounds['B']
            tau_min, tau_max = self.param_bounds['tau']
            
            params[:, 0] = torch.clamp(params[:, 0], K_min, K_max)
            params[:, 1] = torch.clamp(params[:, 1], B_min, B_max)  
            params[:, 2] = torch.clamp(params[:, 2], tau_min, tau_max)
            
            return params
        else:
            return self.predict_parameters(age)
    
    def forward(self, t: torch.Tensor, age: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            t: Time tensor [batch_size, 1]
            age: Age tensor [batch_size, 1]
            deterministic: If True, use mean parameters instead of sampling
            
        Returns:
            xy_pred: Predicted positions [batch_size, 2]
            params: Physical parameters [batch_size, 3]
        """
        if deterministic and self.use_probabilistic:
            params, _ = self.predict_parameters(age)
        else:
            params = self.sample_parameters(age)
        
        # Solver network input: [time, K, B, tau]
        solver_input = torch.cat([t, params], dim=1)
        xy_pred = self.solver_net(solver_input)
        
        return xy_pred, params
    
    def compare_ages(self, subject_age: float, target_age: float, n_samples: int = 100) -> Dict:
        """Compare balance at two different ages."""
        self.eval()
        
        with torch.no_grad():
            subject_age_tensor = torch.tensor([[subject_age]], dtype=torch.float32)
            target_age_tensor = torch.tensor([[target_age]], dtype=torch.float32)
            
            # Get parameter distributions for both ages
            if self.use_probabilistic:
                subject_means, subject_stds = self.predict_parameters(subject_age_tensor)
                target_means, target_stds = self.predict_parameters(target_age_tensor)
                
                # Sample parameters
                subject_samples = []
                target_samples = []
                
                for _ in range(n_samples):
                    subject_params = self.sample_parameters(subject_age_tensor)
                    target_params = self.sample_parameters(target_age_tensor)
                    subject_samples.append(subject_params)
                    target_samples.append(target_params)
                
                subject_samples = torch.cat(subject_samples, dim=0)
                target_samples = torch.cat(target_samples, dim=0)
                
            else:
                subject_params = self.predict_parameters(subject_age_tensor)
                target_params = self.predict_parameters(target_age_tensor)
                subject_samples = subject_params.repeat(n_samples, 1)
                target_samples = target_params.repeat(n_samples, 1)
            
            # Calculate similarity metrics
            param_diffs = torch.abs(subject_samples - target_samples)
            similarities = torch.exp(-param_diffs.sum(dim=1))
            
            return {
                'mean_similarity': similarities.mean().item(),
                'std_similarity': similarities.std().item(),
                'parameter_difference': param_diffs.mean(dim=0).numpy(),
                'subject_params': subject_samples.mean(dim=0).numpy(),
                'target_params': target_samples.mean(dim=0).numpy()
            }
    
    def find_balance_age(self, subject_age: float, age_range: Tuple[float, float] = (20, 90), 
                        resolution: int = 100) -> Dict:
        """Find the age that produces most similar balance parameters."""
        self.eval()
        
        ages = np.linspace(age_range[0], age_range[1], resolution)
        similarities = []
        
        for age in ages:
            if age == subject_age:
                similarities.append(1.0)  # Perfect similarity to self
            else:
                comparison = self.compare_ages(subject_age, age, n_samples=50)
                similarities.append(comparison['mean_similarity'])
        
        # Find peak (excluding self if present)
        if subject_age >= age_range[0] and subject_age <= age_range[1]:
            # Remove self-similarity for analysis
            filtered_similarities = similarities.copy()
            self_idx = np.argmin(np.abs(ages - subject_age))
            filtered_similarities[self_idx] = 0
            best_idx = np.argmax(filtered_similarities)
        else:
            best_idx = np.argmax(similarities)
        
        best_age = ages[best_idx]
        confidence = similarities[best_idx]
        
        return {
            'balance_age': best_age,
            'confidence': confidence,
            'age_similarities': list(zip(ages, similarities)),
            'actual_age': subject_age
        }

def create_model(model_type: str, **kwargs) -> nn.Module:
    """Factory function to create models."""
    if model_type == 'two_stage':
        if 'subject_ids' not in kwargs:
            raise ValueError("subject_ids required for two-stage model")
        return TwoStagePINN(**kwargs)
    elif model_type == 'improved_single':
        return ImprovedBalancePINN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

if __name__ == "__main__":
    # Example usage and testing
    logger.info("Testing model architectures...")
    
    # Test improved single-stage model
    model = ImprovedBalancePINN(use_probabilistic=True)
    
    # Test forward pass
    t = torch.randn(32, 1) * 10  # Random times
    age = torch.randn(32, 1) * 20 + 60  # Ages around 60
    
    xy_pred, params = model(t, age)
    print(f"Improved Single-Stage - Output shapes: xy_pred={xy_pred.shape}, params={params.shape}")
    print(f"Parameter ranges: K={params[:, 0].min().item():.1f}-{params[:, 0].max().item():.1f}")
    
    # Test age comparison
    comparison = model.compare_ages(65.0, 75.0, n_samples=50)
    print(f"Age comparison similarity: {comparison['mean_similarity']:.4f}")
    
    # Test two-stage model
    subject_ids = ['C0001', 'C0002', 'C0003']
    two_stage = TwoStagePINN(subject_ids)
    
    # Test subject PINN
    subject_indices = torch.tensor([[0], [1], [2]])
    t_test = torch.randn(3, 1) * 10
    xy_pred2, params2 = two_stage.subject_pinn(t_test, subject_indices)
    print(f"Two-Stage Subject PINN - Output shapes: xy_pred={xy_pred2.shape}, params={params2.shape}")
    
    logger.info("Model architecture tests completed successfully!")