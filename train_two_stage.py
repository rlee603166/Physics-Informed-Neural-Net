#!/usr/bin/env python3
"""
Two-Stage Training Pipeline for Balance PINN

Stage 1: Train SubjectPINN to learn individual subject parameters
Stage 2: Train AgeParameterModel to learn population-level age trends

This approach separates the learning of individual physics constraints from
population-level age relationships, providing better interpretability and
more reliable parameter estimation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import json
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict

# Import our modules
from improved_models import SubjectPINN, AgeParameterModel, TwoStagePINN
from enhanced_datasets import SubjectAwareDataset, create_subject_splits, create_filtered_dataset
from training_utils import (
    PhysicsLoss, ParameterRegularizationLoss, EarlyStopping, 
    ModelCheckpointer, MetricsTracker, evaluate_model
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TwoStageTrainer:
    """
    Two-stage training pipeline for Balance PINN models.
    
    Stage 1: Learn subject-specific parameters
    Stage 2: Learn age-dependent parameter trends
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.dataset = None
        self.subject_pinn = None
        self.age_model = None
        self.full_model = None
        
        # Training utilities
        self.checkpointer = ModelCheckpointer(
            checkpoint_dir=config['checkpoint_dir'],
            save_best=True,
            save_every=config.get('save_every', 10)
        )
        self.metrics_tracker = MetricsTracker(
            save_path=Path(config['checkpoint_dir']) / 'metrics.json'
        )
        
        # Results storage
        self.stage1_results = {}
        self.stage2_results = {}
    
    def setup_data(self):
        """Setup datasets and data loaders."""
        logger.info("Setting up datasets...")
        
        # Load subject-aware dataset
        self.dataset = SubjectAwareDataset(
            processed_data_folder=self.config['data_folder'],
            age_csv_path=self.config.get('age_csv_path'),
            min_points_per_subject=self.config.get('min_points_per_subject', 100)
        )
        
        # Create train/val/test splits by subjects
        self.subject_splits = create_subject_splits(
            self.dataset,
            train_ratio=self.config['train_ratio'],
            val_ratio=self.config['val_ratio'], 
            test_ratio=self.config['test_ratio'],
            random_seed=self.config.get('random_seed', 42)
        )
        
        # Create filtered datasets for each split
        self.train_indices = create_filtered_dataset(self.dataset, self.subject_splits['train'])
        self.val_indices = create_filtered_dataset(self.dataset, self.subject_splits['val'])
        self.test_indices = create_filtered_dataset(self.dataset, self.subject_splits['test'])
        
        # Create data loaders
        self.train_loader = DataLoader(
            Subset(self.dataset, self.train_indices),
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            Subset(self.dataset, self.val_indices),
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            Subset(self.dataset, self.test_indices),
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 4)
        )
        
        logger.info(f"Data setup complete:")
        logger.info(f"  Train subjects: {len(self.subject_splits['train'])}, points: {len(self.train_indices):,}")
        logger.info(f"  Val subjects: {len(self.subject_splits['val'])}, points: {len(self.val_indices):,}")
        logger.info(f"  Test subjects: {len(self.subject_splits['test'])}, points: {len(self.test_indices):,}")
    
    def setup_stage1_model(self):
        """Setup Stage 1: Subject PINN model."""
        logger.info("Setting up Stage 1 model (Subject PINN)...")
        
        # Get all valid subject IDs
        subject_ids = self.dataset.valid_subjects
        
        # Create subject PINN model
        self.subject_pinn = SubjectPINN(
            subject_ids=subject_ids,
            hidden_dims=self.config.get('stage1_hidden_dims', [256, 256, 256, 256]),
            param_bounds=self.config.get('param_bounds')
        ).to(self.device)
        
        # Optimizer and scheduler
        self.stage1_optimizer = torch.optim.Adam(
            self.subject_pinn.parameters(),
            lr=self.config['stage1_lr'],
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        
        self.stage1_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.stage1_optimizer,
            mode='min',
            factor=0.5,
            patience=self.config.get('scheduler_patience', 10),
            verbose=True
        )
        
        # Loss function
        self.stage1_loss_fn = PhysicsLoss(
            weight=self.config.get('stage1_physics_weight', 0.1),
            g=9.81, L=1.0, m=70.0
        )
        
        # Early stopping
        self.stage1_early_stopping = EarlyStopping(
            patience=self.config.get('stage1_patience', 20),
            min_delta=1e-6
        )
        
        total_params = sum(p.numel() for p in self.subject_pinn.parameters())
        logger.info(f"Stage 1 model created with {total_params:,} parameters")
    
    def train_stage1(self):
        """Train Stage 1: Subject-specific parameter learning."""
        logger.info("=" * 50)
        logger.info("STAGE 1: Training Subject PINN")
        logger.info("=" * 50)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config['stage1_epochs']):
            # Training phase
            self.subject_pinn.train()
            train_losses = defaultdict(float)
            train_samples = 0
            
            pbar = tqdm(self.train_loader, desc=f"Stage 1 Epoch {epoch+1}")
            for batch_idx, (t, age, xy_true, subject_idx) in enumerate(pbar):
                t = t.to(self.device).requires_grad_(True)
                age = age.to(self.device)
                xy_true = xy_true.to(self.device)
                subject_idx = subject_idx.to(self.device)
                
                # Forward pass
                xy_pred, params = self.subject_pinn(t, subject_idx)
                
                # Data loss
                data_loss = nn.functional.mse_loss(xy_pred, xy_true)
                
                # Physics loss
                physics_loss = self.stage1_loss_fn(t, xy_pred, params)
                
                # Total loss
                total_loss = data_loss + physics_loss
                
                # Backward pass
                self.stage1_optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.subject_pinn.parameters(), max_norm=1.0)
                
                self.stage1_optimizer.step()
                
                # Track losses
                batch_size = t.shape[0]
                train_losses['data'] += data_loss.item() * batch_size
                train_losses['physics'] += physics_loss.item() * batch_size
                train_losses['total'] += total_loss.item() * batch_size
                train_samples += batch_size
                
                # Update progress bar
                pbar.set_postfix({
                    'Data': f"{data_loss.item():.6f}",
                    'Physics': f"{physics_loss.item():.6f}",
                    'Total': f"{total_loss.item():.6f}"
                })
            
            # Calculate average training losses
            avg_train_losses = {k: v / train_samples for k, v in train_losses.items()}
            
            # Validation phase
            val_metrics = self._evaluate_stage1(self.val_loader)
            val_loss = val_metrics['total_loss']
            
            # Learning rate scheduling
            self.stage1_scheduler.step(val_loss)
            
            # Logging
            logger.info(f"Stage 1 Epoch {epoch+1}/{self.config['stage1_epochs']}")
            logger.info(f"  Train - Data: {avg_train_losses['data']:.6f}, Physics: {avg_train_losses['physics']:.6f}, Total: {avg_train_losses['total']:.6f}")
            logger.info(f"  Val   - Data: {val_metrics['data_loss']:.6f}, Physics: {val_metrics['physics_loss']:.6f}, Total: {val_loss:.6f}")
            logger.info(f"  LR: {self.stage1_optimizer.param_groups[0]['lr']:.2e}")
            
            # Track metrics
            self.metrics_tracker.update(
                stage1_train_data_loss=avg_train_losses['data'],
                stage1_train_physics_loss=avg_train_losses['physics'],
                stage1_train_total_loss=avg_train_losses['total'],
                stage1_val_data_loss=val_metrics['data_loss'],
                stage1_val_physics_loss=val_metrics['physics_loss'],
                stage1_val_total_loss=val_loss,
                stage1_lr=self.stage1_optimizer.param_groups[0]['lr']
            )
            
            # Save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.checkpointer.save_checkpoint(
                    epoch, self.subject_pinn, self.stage1_optimizer, 
                    self.stage1_scheduler, val_loss, val_metrics
                )
            
            # Early stopping
            if self.stage1_early_stopping(val_loss, self.subject_pinn):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Extract learned parameters for Stage 2
        self._extract_subject_parameters()
        
        # Final evaluation
        test_metrics = self._evaluate_stage1(self.test_loader)
        self.stage1_results = {
            'best_val_loss': best_val_loss,
            'test_metrics': test_metrics,
            'epochs_trained': epoch + 1
        }
        
        logger.info("Stage 1 training completed!")
        logger.info(f"Best validation loss: {best_val_loss:.6f}")
        logger.info(f"Test metrics: {test_metrics}")
    
    def _evaluate_stage1(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate Stage 1 model."""
        self.subject_pinn.eval()
        
        total_losses = defaultdict(float)
        total_samples = 0
        
        with torch.no_grad():
            for t, age, xy_true, subject_idx in dataloader:
                t = t.to(self.device).requires_grad_(True)
                age = age.to(self.device)
                xy_true = xy_true.to(self.device)
                subject_idx = subject_idx.to(self.device)
                
                # Forward pass
                xy_pred, params = self.subject_pinn(t, subject_idx)
                
                # Losses
                data_loss = nn.functional.mse_loss(xy_pred, xy_true)
                physics_loss = self.stage1_loss_fn(t, xy_pred, params)
                total_loss = data_loss + physics_loss
                
                batch_size = t.shape[0]
                total_losses['data_loss'] += data_loss.item() * batch_size
                total_losses['physics_loss'] += physics_loss.item() * batch_size
                total_losses['total_loss'] += total_loss.item() * batch_size
                total_samples += batch_size
        
        return {k: v / total_samples for k, v in total_losses.items()}
    
    def _extract_subject_parameters(self):
        """Extract learned subject parameters for Stage 2."""
        logger.info("Extracting subject parameters...")
        
        self.subject_pinn.eval()
        
        subject_parameters = {}
        
        with torch.no_grad():
            for i, subject_id in enumerate(self.dataset.valid_subjects):
                subject_idx = torch.tensor([[i]], dtype=torch.long).to(self.device)
                K, B, tau = self.subject_pinn.get_parameters(subject_idx.squeeze())
                
                subject_info = self.dataset.get_subject_info(subject_id)
                age = subject_info.get('age', 0)
                
                subject_parameters[subject_id] = {
                    'age': age,
                    'K': K.item(),
                    'B': B.item(),
                    'tau': tau.item(),
                    'n_points': subject_info.get('n_points', 0)
                }
        
        self.subject_parameters = subject_parameters
        
        # Save parameters
        params_path = Path(self.config['checkpoint_dir']) / 'subject_parameters.json'
        with open(params_path, 'w') as f:
            json.dump(subject_parameters, f, indent=2)
        
        # Log parameter statistics
        ages = [p['age'] for p in subject_parameters.values()]
        Ks = [p['K'] for p in subject_parameters.values()]
        Bs = [p['B'] for p in subject_parameters.values()]
        taus = [p['tau'] for p in subject_parameters.values()]
        
        logger.info(f"Extracted parameters for {len(subject_parameters)} subjects:")
        logger.info(f"  Age range: {min(ages):.1f} - {max(ages):.1f}")
        logger.info(f"  K range: {min(Ks):.1f} - {max(Ks):.1f}")
        logger.info(f"  B range: {min(Bs):.1f} - {max(Bs):.1f}")
        logger.info(f"  τ range: {min(taus):.3f} - {max(taus):.3f}")
    
    def setup_stage2_model(self):
        """Setup Stage 2: Age parameter model."""
        logger.info("Setting up Stage 2 model (Age Parameter Model)...")
        
        self.age_model = AgeParameterModel(
            param_bounds=self.config.get('param_bounds'),
            use_probabilistic=self.config.get('stage2_probabilistic', True)
        ).to(self.device)
        
        # Optimizer and scheduler
        self.stage2_optimizer = torch.optim.Adam(
            self.age_model.parameters(),
            lr=self.config['stage2_lr'],
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        
        self.stage2_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.stage2_optimizer,
            mode='min',
            factor=0.5,
            patience=self.config.get('scheduler_patience', 10)
        )
        
        # Loss components
        self.stage2_param_loss = nn.MSELoss()
        self.stage2_reg_loss = ParameterRegularizationLoss(
            smoothness_weight=0.1,
            variation_weight=0.1,
            param_bounds=self.config.get('param_bounds')
        )
        
        # Early stopping
        self.stage2_early_stopping = EarlyStopping(
            patience=self.config.get('stage2_patience', 15),
            min_delta=1e-6
        )
        
        total_params = sum(p.numel() for p in self.age_model.parameters())
        logger.info(f"Stage 2 model created with {total_params:,} parameters")
    
    def _create_stage2_dataset(self) -> Tuple[List, List]:
        """Create dataset for Stage 2 training (age -> parameters)."""
        ages = []
        parameters = []
        
        for subject_id in self.subject_splits['train']:
            if subject_id in self.subject_parameters:
                param_data = self.subject_parameters[subject_id]
                ages.append(param_data['age'])
                parameters.append([param_data['K'], param_data['B'], param_data['tau']])
        
        return ages, parameters
    
    def train_stage2(self):
        """Train Stage 2: Age-dependent parameter learning."""
        logger.info("=" * 50)
        logger.info("STAGE 2: Training Age Parameter Model")
        logger.info("=" * 50)
        
        # Create Stage 2 dataset
        train_ages, train_params = self._create_stage2_dataset()
        
        # Validation data
        val_ages = []
        val_params = []
        for subject_id in self.subject_splits['val']:
            if subject_id in self.subject_parameters:
                param_data = self.subject_parameters[subject_id]
                val_ages.append(param_data['age'])
                val_params.append([param_data['K'], param_data['B'], param_data['tau']])
        
        # Convert to tensors
        train_ages_tensor = torch.tensor(train_ages, dtype=torch.float32).unsqueeze(-1)
        train_params_tensor = torch.tensor(train_params, dtype=torch.float32)
        val_ages_tensor = torch.tensor(val_ages, dtype=torch.float32).unsqueeze(-1)
        val_params_tensor = torch.tensor(val_params, dtype=torch.float32)
        
        logger.info(f"Stage 2 data: {len(train_ages)} train, {len(val_ages)} val subjects")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config['stage2_epochs']):
            # Training phase
            self.age_model.train()
            
            # Shuffle training data
            indices = torch.randperm(len(train_ages))
            train_ages_shuffled = train_ages_tensor[indices].to(self.device)
            train_params_shuffled = train_params_tensor[indices].to(self.device)
            
            train_losses = defaultdict(float)
            n_batches = 0
            
            # Mini-batch training
            batch_size = min(self.config.get('stage2_batch_size', 32), len(train_ages))
            
            for i in range(0, len(train_ages), batch_size):
                batch_ages = train_ages_shuffled[i:i+batch_size]
                batch_params = train_params_shuffled[i:i+batch_size]
                
                # Forward pass
                if self.age_model.use_probabilistic:
                    pred_means, pred_stds = self.age_model.predict_parameters(batch_ages)
                    
                    # Negative log-likelihood loss
                    param_loss = 0.5 * torch.mean(
                        ((batch_params - pred_means) / (pred_stds + 1e-6))**2 + 
                        torch.log(pred_stds + 1e-6)
                    )
                else:
                    pred_params = self.age_model.predict_parameters(batch_ages)
                    param_loss = self.stage2_param_loss(pred_params, batch_params)
                
                # Regularization losses
                reg_losses = self.stage2_reg_loss(batch_ages, batch_params)
                total_reg_loss = sum(reg_losses.values())
                
                # Total loss
                total_loss = param_loss + self.config.get('stage2_reg_weight', 0.1) * total_reg_loss
                
                # Backward pass
                self.stage2_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.age_model.parameters(), max_norm=1.0)
                self.stage2_optimizer.step()
                
                # Track losses
                train_losses['param'] += param_loss.item()
                train_losses['reg'] += total_reg_loss.item()
                train_losses['total'] += total_loss.item()
                n_batches += 1
            
            # Average training losses
            avg_train_losses = {k: v / n_batches for k, v in train_losses.items()}
            
            # Validation
            val_metrics = self._evaluate_stage2(val_ages_tensor, val_params_tensor)
            val_loss = val_metrics['total_loss']
            
            # Learning rate scheduling
            self.stage2_scheduler.step(val_loss)
            
            # Logging
            logger.info(f"Stage 2 Epoch {epoch+1}/{self.config['stage2_epochs']}")
            logger.info(f"  Train - Param: {avg_train_losses['param']:.6f}, Reg: {avg_train_losses['reg']:.6f}, Total: {avg_train_losses['total']:.6f}")
            logger.info(f"  Val   - Param: {val_metrics['param_loss']:.6f}, Reg: {val_metrics['reg_loss']:.6f}, Total: {val_loss:.6f}")
            logger.info(f"  LR: {self.stage2_optimizer.param_groups[0]['lr']:.2e}")
            
            # Track metrics
            self.metrics_tracker.update(
                stage2_train_param_loss=avg_train_losses['param'],
                stage2_train_reg_loss=avg_train_losses['reg'],
                stage2_train_total_loss=avg_train_losses['total'],
                stage2_val_param_loss=val_metrics['param_loss'],
                stage2_val_reg_loss=val_metrics['reg_loss'],
                stage2_val_total_loss=val_loss,
                stage2_lr=self.stage2_optimizer.param_groups[0]['lr']
            )
            
            # Save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save both models together
                checkpoint = {
                    'epoch': epoch,
                    'subject_pinn_state_dict': self.subject_pinn.state_dict(),
                    'age_model_state_dict': self.age_model.state_dict(),
                    'stage2_optimizer_state_dict': self.stage2_optimizer.state_dict(),
                    'val_loss': val_loss,
                    'subject_parameters': self.subject_parameters
                }
                torch.save(checkpoint, Path(self.config['checkpoint_dir']) / 'best_two_stage_model.pth')
            
            # Early stopping
            if self.stage2_early_stopping(val_loss, self.age_model):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Test evaluation
        test_ages = []
        test_params = []
        for subject_id in self.subject_splits['test']:
            if subject_id in self.subject_parameters:
                param_data = self.subject_parameters[subject_id]
                test_ages.append(param_data['age'])
                test_params.append([param_data['K'], param_data['B'], param_data['tau']])
        
        if test_ages:
            test_ages_tensor = torch.tensor(test_ages, dtype=torch.float32).unsqueeze(-1)
            test_params_tensor = torch.tensor(test_params, dtype=torch.float32)
            test_metrics = self._evaluate_stage2(test_ages_tensor, test_params_tensor)
        else:
            test_metrics = {}
        
        self.stage2_results = {
            'best_val_loss': best_val_loss,
            'test_metrics': test_metrics,
            'epochs_trained': epoch + 1
        }
        
        logger.info("Stage 2 training completed!")
        logger.info(f"Best validation loss: {best_val_loss:.6f}")
        logger.info(f"Test metrics: {test_metrics}")
    
    def _evaluate_stage2(self, ages_tensor: torch.Tensor, params_tensor: torch.Tensor) -> Dict[str, float]:
        """Evaluate Stage 2 model."""
        self.age_model.eval()
        
        ages_tensor = ages_tensor.to(self.device)
        params_tensor = params_tensor.to(self.device)
        
        with torch.no_grad():
            if self.age_model.use_probabilistic:
                pred_means, pred_stds = self.age_model.predict_parameters(ages_tensor)
                param_loss = 0.5 * torch.mean(
                    ((params_tensor - pred_means) / (pred_stds + 1e-6))**2 + 
                    torch.log(pred_stds + 1e-6)
                )
            else:
                pred_params = self.age_model.predict_parameters(ages_tensor)
                param_loss = self.stage2_param_loss(pred_params, params_tensor)
            
            # Regularization
            reg_losses = self.stage2_reg_loss(ages_tensor, params_tensor)
            total_reg_loss = sum(reg_losses.values())
            
            total_loss = param_loss + self.config.get('stage2_reg_weight', 0.1) * total_reg_loss
        
        return {
            'param_loss': param_loss.item(),
            'reg_loss': total_reg_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def create_full_model(self):
        """Create the full two-stage model for inference."""
        logger.info("Creating full two-stage model...")
        
        self.full_model = TwoStagePINN(
            subject_ids=self.dataset.valid_subjects,
            param_bounds=self.config.get('param_bounds')
        )
        
        # Transfer weights
        self.full_model.subject_pinn.load_state_dict(self.subject_pinn.state_dict())
        self.full_model.age_model.load_state_dict(self.age_model.state_dict())
        
        # Save complete model
        full_model_path = Path(self.config['checkpoint_dir']) / 'complete_two_stage_model.pth'
        torch.save({
            'model_state_dict': self.full_model.state_dict(),
            'subject_ids': self.dataset.valid_subjects,
            'param_bounds': self.config.get('param_bounds'),
            'subject_parameters': self.subject_parameters,
            'config': self.config
        }, full_model_path)
        
        logger.info(f"Complete model saved to: {full_model_path}")
    
    def generate_analysis_plots(self):
        """Generate analysis plots and visualizations."""
        logger.info("Generating analysis plots...")
        
        output_dir = Path(self.config['checkpoint_dir']) / 'plots'
        output_dir.mkdir(exist_ok=True)
        
        # Plot 1: Training curves
        if hasattr(self.metrics_tracker, 'metrics') and self.metrics_tracker.metrics:
            self.metrics_tracker.plot_metrics(
                ['stage1_train_total_loss', 'stage1_val_total_loss', 'stage2_train_total_loss', 'stage2_val_total_loss'],
                save_path=output_dir / 'training_curves.png'
            )
        
        # Plot 2: Parameter vs Age relationships
        self._plot_parameter_relationships(output_dir)
        
        # Plot 3: Subject parameter distributions
        self._plot_parameter_distributions(output_dir)
        
        logger.info(f"Analysis plots saved to: {output_dir}")
    
    def _plot_parameter_relationships(self, output_dir: Path):
        """Plot learned parameter-age relationships."""
        if not hasattr(self, 'age_model') or self.age_model is None:
            return
        
        self.age_model.eval()
        
        # Generate age range
        ages = np.linspace(20, 90, 100)
        age_tensor = torch.tensor(ages, dtype=torch.float32).unsqueeze(-1).to(self.device)
        
        with torch.no_grad():
            if self.age_model.use_probabilistic:
                param_means, param_stds = self.age_model.predict_parameters(age_tensor)
                param_means = param_means.cpu().numpy()
                param_stds = param_stds.cpu().numpy()
            else:
                param_means = self.age_model.predict_parameters(age_tensor).cpu().numpy()
                param_stds = None
        
        # Extract subject data for plotting
        subject_ages = [p['age'] for p in self.subject_parameters.values()]
        subject_Ks = [p['K'] for p in self.subject_parameters.values()]
        subject_Bs = [p['B'] for p in self.subject_parameters.values()]
        subject_taus = [p['tau'] for p in self.subject_parameters.values()]
        
        # Create plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        param_names = ['Stiffness (K)', 'Damping (B)', 'Neural Delay (τ)']
        subject_params = [subject_Ks, subject_Bs, subject_taus]
        
        for i, (name, subject_param) in enumerate(zip(param_names, subject_params)):
            ax = axes[i]
            
            # Plot subject data points
            ax.scatter(subject_ages, subject_param, alpha=0.6, s=20, color='blue', label='Subjects')
            
            # Plot learned curve
            ax.plot(ages, param_means[:, i], 'red', linewidth=2, label='Learned Trend')
            
            # Plot uncertainty if available
            if param_stds is not None:
                ax.fill_between(ages, 
                               param_means[:, i] - param_stds[:, i],
                               param_means[:, i] + param_stds[:, i],
                               alpha=0.2, color='red')
            
            ax.set_xlabel('Age (years)')
            ax.set_ylabel(name)
            ax.set_title(f'{name} vs Age')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'parameter_age_relationships.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_parameter_distributions(self, output_dir: Path):
        """Plot parameter distributions."""
        subject_Ks = [p['K'] for p in self.subject_parameters.values()]
        subject_Bs = [p['B'] for p in self.subject_parameters.values()]
        subject_taus = [p['tau'] for p in self.subject_parameters.values()]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        params = [subject_Ks, subject_Bs, subject_taus]
        names = ['Stiffness (K)', 'Damping (B)', 'Neural Delay (τ)']
        
        for i, (param, name) in enumerate(zip(params, names)):
            ax = axes[i]
            ax.hist(param, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel(name)
            ax.set_ylabel('Frequency')
            ax.set_title(f'{name} Distribution')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'parameter_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_complete_training(self):
        """Run the complete two-stage training pipeline."""
        logger.info("Starting complete two-stage training pipeline...")
        
        try:
            # Setup
            self.setup_data()
            
            # Stage 1: Subject parameter learning
            self.setup_stage1_model()
            self.train_stage1()
            
            # Stage 2: Age relationship learning
            self.setup_stage2_model()
            self.train_stage2()
            
            # Create full model
            self.create_full_model()
            
            # Generate analysis
            self.generate_analysis_plots()
            
            # Save final results
            self.save_results()
            
            logger.info("Two-stage training completed successfully!")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def save_results(self):
        """Save training results and model information."""
        results = {
            'config': self.config,
            'stage1_results': self.stage1_results,
            'stage2_results': self.stage2_results,
            'data_info': {
                'n_subjects_total': len(self.dataset.valid_subjects),
                'n_train_subjects': len(self.subject_splits['train']),
                'n_val_subjects': len(self.subject_splits['val']),
                'n_test_subjects': len(self.subject_splits['test']),
                'n_train_points': len(self.train_indices),
                'n_val_points': len(self.val_indices),
                'n_test_points': len(self.test_indices)
            }
        }
        
        results_path = Path(self.config['checkpoint_dir']) / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save metrics
        self.metrics_tracker.save()
        
        logger.info(f"Training results saved to: {results_path}")

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Two-Stage Balance PINN Training')
    
    # Data arguments
    parser.add_argument('--data-folder', type=str, default='processed_data',
                       help='Path to processed data folder')
    parser.add_argument('--age-csv-path', type=str, default='user_ages.csv',
                       help='Path to age CSV file')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_two_stage',
                       help='Directory to save checkpoints')
    
    # Training arguments
    parser.add_argument('--stage1-epochs', type=int, default=100,
                       help='Number of epochs for Stage 1')
    parser.add_argument('--stage2-epochs', type=int, default=50,
                       help='Number of epochs for Stage 2')
    parser.add_argument('--batch-size', type=int, default=4096,
                       help='Batch size for training')
    parser.add_argument('--stage1-lr', type=float, default=1e-3,
                       help='Learning rate for Stage 1')
    parser.add_argument('--stage2-lr', type=float, default=1e-3,
                       help='Learning rate for Stage 2')
    
    # Model arguments
    parser.add_argument('--stage1-physics-weight', type=float, default=0.1,
                       help='Physics loss weight for Stage 1')
    parser.add_argument('--stage2-reg-weight', type=float, default=0.1,
                       help='Regularization weight for Stage 2')
    
    args = parser.parse_args()
    
    # Create config
    config = {
        'data_folder': args.data_folder,
        'age_csv_path': args.age_csv_path,
        'checkpoint_dir': args.checkpoint_dir,
        'stage1_epochs': args.stage1_epochs,
        'stage2_epochs': args.stage2_epochs,
        'batch_size': args.batch_size,
        'stage1_lr': args.stage1_lr,
        'stage2_lr': args.stage2_lr,
        'stage1_physics_weight': args.stage1_physics_weight,
        'stage2_reg_weight': args.stage2_reg_weight,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'random_seed': 42,
        'num_workers': 4,
        'weight_decay': 1e-5,
        'stage1_patience': 20,
        'stage2_patience': 15,
        'scheduler_patience': 10,
        'min_points_per_subject': 100,
        'stage2_batch_size': 32,
        'stage2_probabilistic': True,
        'save_every': 10,
        'param_bounds': {
            'K': (500.0, 3000.0),
            'B': (20.0, 150.0),
            'tau': (0.05, 0.4)
        }
    }
    
    # Create trainer and run
    trainer = TwoStageTrainer(config)
    trainer.run_complete_training()

if __name__ == "__main__":
    main()