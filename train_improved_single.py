#!/usr/bin/env python3
"""
Improved Single-Stage Training Pipeline for Balance PINN

This training pipeline implements the enhanced single-stage architecture with:
- Improved parameter network with probabilistic outputs
- Better loss balancing and age-aware regularization  
- Comprehensive evaluation and age-comparison capabilities
- Advanced training utilities and monitoring
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
from improved_models import ImprovedBalancePINN
from enhanced_datasets import EnhancedBalanceDataset, create_subject_splits, create_filtered_dataset
from training_utils import (
    CombinedLoss, EarlyStopping, ModelCheckpointer, MetricsTracker, 
    evaluate_model, calculate_physics_residuals, WarmupLRScheduler
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedSingleStageTrainer:
    """
    Improved single-stage training pipeline with enhanced loss functions
    and age-aware training strategies.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.dataset = None
        self.model = None
        
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
        self.training_results = {}
    
    def setup_data(self):
        """Setup datasets and data loaders."""
        logger.info("Setting up datasets...")
        
        # Load enhanced dataset
        self.dataset = EnhancedBalanceDataset(
            processed_data_folder=self.config['data_folder'],
            age_csv_path=self.config.get('age_csv_path'),
            normalize=self.config.get('normalize_data', False),
            augment=self.config.get('augment_data', False)
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
            pin_memory=True,
            drop_last=True  # For consistent batch sizes
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
    
    def setup_model(self):
        """Setup improved single-stage model."""
        logger.info("Setting up improved single-stage model...")
        
        self.model = ImprovedBalancePINN(
            hidden_dim=self.config.get('hidden_dim', 256),
            num_layers=self.config.get('num_layers', 8),
            param_bounds=self.config.get('param_bounds'),
            use_probabilistic=self.config.get('use_probabilistic', True),
            dropout_rate=self.config.get('dropout_rate', 0.1)
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-5),
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler with warmup
        base_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=self.config.get('scheduler_patience', 10),
            verbose=True
        )
        
        self.scheduler = WarmupLRScheduler(
            self.optimizer,
            warmup_epochs=self.config.get('warmup_epochs', 5),
            max_lr=self.config['learning_rate'],
            decay_scheduler=base_scheduler
        )
        
        # Combined loss function
        self.loss_fn = CombinedLoss(
            data_weight=self.config.get('data_weight', 1.0),
            physics_weight=self.config.get('physics_weight', 0.01),
            regularization_weight=self.config.get('regularization_weight', 0.1),
            age_aware_weight=self.config.get('age_aware_weight', 0.1),
            param_bounds=self.config.get('param_bounds')
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.get('patience', 20),
            min_delta=1e-6,
            restore_best_weights=True
        )
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Model created:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Using probabilistic outputs: {self.model.use_probabilistic}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = defaultdict(float)
        epoch_samples = 0
        
        # Enable gradient calculation for time (needed for physics loss)
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            if len(batch) == 4:
                t, age, xy_true, metadata = batch
            else:
                t, age, xy_true = batch[:3]
            
            # Move to device and enable gradients for time
            t = t.to(self.device).requires_grad_(True)
            age = age.to(self.device)
            xy_true = xy_true.to(self.device)
            
            # Forward pass
            xy_pred, params = self.model(t, age, deterministic=False)
            
            # Calculate combined loss
            losses = self.loss_fn(t, age, xy_pred, xy_true, params)
            total_loss = losses['total']
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track losses
            batch_size = t.shape[0]
            for key, value in losses.items():
                epoch_losses[key] += value.item() * batch_size
            epoch_samples += batch_size
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{total_loss.item():.6f}",
                'Data': f"{losses['data'].item():.6f}",
                'Physics': f"{losses.get('physics', torch.tensor(0)).item():.6f}",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        # Calculate average losses
        avg_losses = {key: value / epoch_samples for key, value in epoch_losses.items()}
        
        return avg_losses
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        epoch_losses = defaultdict(float)
        epoch_samples = 0
        
        predictions = []
        targets = []
        parameters = []
        ages_list = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Validation {epoch+1}", leave=False):
                if len(batch) == 4:
                    t, age, xy_true, metadata = batch
                else:
                    t, age, xy_true = batch[:3]
                
                # For validation, we need gradients for physics loss calculation
                t = t.to(self.device).requires_grad_(True)
                age = age.to(self.device)
                xy_true = xy_true.to(self.device)
                
                # Forward pass (deterministic for validation)
                xy_pred, params = self.model(t, age, deterministic=True)
                
                # Calculate losses
                losses = self.loss_fn(t, age, xy_pred, xy_true, params)
                
                # Track losses
                batch_size = t.shape[0]
                for key, value in losses.items():
                    epoch_losses[key] += value.item() * batch_size
                epoch_samples += batch_size
                
                # Store for additional metrics
                predictions.append(xy_pred.detach().cpu())
                targets.append(xy_true.detach().cpu())
                parameters.append(params.detach().cpu())
                ages_list.append(age.detach().cpu())
        
        # Calculate average losses
        avg_losses = {key: value / epoch_samples for key, value in epoch_losses.items()}
        
        # Additional validation metrics
        all_predictions = torch.cat(predictions, dim=0)
        all_targets = torch.cat(targets, dim=0)
        all_parameters = torch.cat(parameters, dim=0)
        all_ages = torch.cat(ages_list, dim=0)
        
        # R² score
        ss_res = torch.sum((all_targets - all_predictions) ** 2)
        ss_tot = torch.sum((all_targets - torch.mean(all_targets, dim=0)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Parameter statistics
        param_stats = {
            'K_mean': all_parameters[:, 0].mean().item(),
            'K_std': all_parameters[:, 0].std().item(),
            'K_min': all_parameters[:, 0].min().item(),
            'K_max': all_parameters[:, 0].max().item(),
            'B_mean': all_parameters[:, 1].mean().item(),
            'B_std': all_parameters[:, 1].std().item(),
            'B_min': all_parameters[:, 1].min().item(),
            'B_max': all_parameters[:, 1].max().item(),
            'tau_mean': all_parameters[:, 2].mean().item(),
            'tau_std': all_parameters[:, 2].std().item(),
            'tau_min': all_parameters[:, 2].min().item(),
            'tau_max': all_parameters[:, 2].max().item()
        }
        
        # Check parameter variation
        param_variation = {
            'K_cv': param_stats['K_std'] / (param_stats['K_mean'] + 1e-6),
            'B_cv': param_stats['B_std'] / (param_stats['B_mean'] + 1e-6),
            'tau_cv': param_stats['tau_std'] / (param_stats['tau_mean'] + 1e-6)
        }
        
        # Combine metrics
        validation_metrics = {
            **avg_losses,
            'r2_score': r2_score.item() if isinstance(r2_score, torch.Tensor) else r2_score,
            'mae': torch.nn.functional.l1_loss(all_predictions, all_targets).item(),
            'rmse': torch.sqrt(torch.nn.functional.mse_loss(all_predictions, all_targets)).item(),
            **param_stats,
            **param_variation
        }
        
        return validation_metrics
    
    def train(self):
        """Main training loop."""
        logger.info("=" * 60)
        logger.info("STARTING IMPROVED SINGLE-STAGE TRAINING")
        logger.info("=" * 60)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config['epochs']):
            # Training phase
            train_metrics = self.train_epoch(epoch)
            
            # Validation phase  
            val_metrics = self.validate_epoch(epoch)
            
            # Learning rate scheduling
            if hasattr(self.scheduler, 'step'):
                if hasattr(self.scheduler, 'decay_scheduler') and epoch >= self.config.get('warmup_epochs', 5):
                    self.scheduler.decay_scheduler.step(val_metrics['total'])
                else:
                    self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            val_loss = val_metrics['total']
            
            # Logging
            logger.info(f"Epoch {epoch+1}/{self.config['epochs']}")
            logger.info(f"  Train - Total: {train_metrics['total']:.6f}, Data: {train_metrics['data']:.6f}, "
                       f"Physics: {train_metrics.get('physics', 0):.6f}")
            logger.info(f"  Val   - Total: {val_loss:.6f}, Data: {val_metrics['data']:.6f}, "
                       f"R²: {val_metrics['r2_score']:.4f}, MAE: {val_metrics['mae']:.4f}")
            logger.info(f"  Params - K: {val_metrics['K_mean']:.1f}±{val_metrics['K_std']:.1f} (CV: {val_metrics['K_cv']:.3f}), "
                       f"B: {val_metrics['B_mean']:.1f}±{val_metrics['B_std']:.1f} (CV: {val_metrics['B_cv']:.3f}), "
                       f"τ: {val_metrics['tau_mean']:.3f}±{val_metrics['tau_std']:.3f} (CV: {val_metrics['tau_cv']:.3f})")
            logger.info(f"  LR: {current_lr:.2e}")
            
            # Track metrics
            metrics_update = {
                'epoch': epoch + 1,
                'learning_rate': current_lr,
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()}
            }
            self.metrics_tracker.update(**metrics_update)
            
            # Save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = self.checkpointer.save_checkpoint(
                    epoch, self.model, self.optimizer, self.scheduler, 
                    val_loss, val_metrics
                )
                logger.info(f"✅ New best validation loss: {val_loss:.6f}")
            
            # Early stopping
            if self.early_stopping(val_loss, self.model):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Final evaluation
        logger.info("Running final evaluation...")
        
        # Test set evaluation
        test_metrics = self.evaluate_test_set()
        
        # Physics compliance check
        physics_metrics = calculate_physics_residuals(self.model, self.test_loader, self.device)
        
        # Age comparison analysis
        age_analysis = self.analyze_age_relationships()
        
        # Store results
        self.training_results = {
            'best_val_loss': best_val_loss,
            'total_epochs': epoch + 1,
            'test_metrics': test_metrics,
            'physics_metrics': physics_metrics,
            'age_analysis': age_analysis,
            'config': self.config
        }
        
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {best_val_loss:.6f}")
        logger.info(f"Test R² score: {test_metrics.get('r2_score', 'N/A')}")
        logger.info(f"Physics residual: {physics_metrics.get('physics_residual_mean', 'N/A'):.6f}")
    
    def evaluate_test_set(self) -> Dict[str, float]:
        """Comprehensive test set evaluation."""
        logger.info("Evaluating test set...")
        
        # Use the evaluate_model function from training_utils
        test_metrics = evaluate_model(self.model, self.test_loader, self.loss_fn, self.device)
        
        return test_metrics
    
    def analyze_age_relationships(self) -> Dict:
        """Analyze how parameters vary with age."""
        logger.info("Analyzing age-parameter relationships...")
        
        self.model.eval()
        
        # Generate age range for analysis
        age_range = np.linspace(20, 90, 100)
        age_analysis = {
            'age_range': age_range.tolist(),
            'K_values': [],
            'B_values': [],
            'tau_values': []
        }
        
        with torch.no_grad():
            for age in age_range:
                age_tensor = torch.tensor([[age]], dtype=torch.float32).to(self.device)
                
                if self.model.use_probabilistic:
                    param_means, param_stds = self.model.predict_parameters(age_tensor)
                    params = param_means.cpu().numpy().squeeze()
                    
                    # Also store uncertainties
                    if 'K_uncertainties' not in age_analysis:
                        age_analysis.update({
                            'K_uncertainties': [],
                            'B_uncertainties': [],
                            'tau_uncertainties': []
                        })
                    
                    stds = param_stds.cpu().numpy().squeeze()
                    age_analysis['K_uncertainties'].append(stds[0])
                    age_analysis['B_uncertainties'].append(stds[1])
                    age_analysis['tau_uncertainties'].append(stds[2])
                else:
                    params = self.model.predict_parameters(age_tensor).cpu().numpy().squeeze()
                
                age_analysis['K_values'].append(params[0])
                age_analysis['B_values'].append(params[1])
                age_analysis['tau_values'].append(params[2])
        
        # Calculate parameter variation statistics
        age_analysis['statistics'] = {
            'K_variation_coeff': np.std(age_analysis['K_values']) / np.mean(age_analysis['K_values']),
            'B_variation_coeff': np.std(age_analysis['B_values']) / np.mean(age_analysis['B_values']),
            'tau_variation_coeff': np.std(age_analysis['tau_values']) / np.mean(age_analysis['tau_values']),
            'age_correlation_K': np.corrcoef(age_range, age_analysis['K_values'])[0, 1],
            'age_correlation_B': np.corrcoef(age_range, age_analysis['B_values'])[0, 1],
            'age_correlation_tau': np.corrcoef(age_range, age_analysis['tau_values'])[0, 1]
        }
        
        return age_analysis
    
    def test_age_comparisons(self) -> Dict:
        """Test age comparison functionality."""
        logger.info("Testing age comparison capabilities...")
        
        # Test comparing different ages
        test_ages = [25, 35, 45, 55, 65, 75, 85]
        comparisons = {}
        
        for i, age1 in enumerate(test_ages[:-1]):
            for age2 in test_ages[i+1:]:
                comparison_key = f"age_{age1}_vs_{age2}"
                comparison = self.model.compare_ages(age1, age2, n_samples=100)
                comparisons[comparison_key] = comparison
        
        # Find balance age for some example ages
        balance_age_tests = {}
        for test_age in [30, 50, 70]:
            balance_result = self.model.find_balance_age(test_age)
            balance_age_tests[f"test_age_{test_age}"] = balance_result
        
        return {
            'age_comparisons': comparisons,
            'balance_age_tests': balance_age_tests
        }
    
    def generate_analysis_plots(self):
        """Generate comprehensive analysis plots."""
        logger.info("Generating analysis plots...")
        
        output_dir = Path(self.config['checkpoint_dir']) / 'plots'
        output_dir.mkdir(exist_ok=True)
        
        # Training curves
        if hasattr(self.metrics_tracker, 'metrics') and self.metrics_tracker.metrics:
            self._plot_training_curves(output_dir)
        
        # Parameter-age relationships
        if 'age_analysis' in self.training_results:
            self._plot_parameter_age_relationships(output_dir)
        
        # Parameter distributions over validation set
        self._plot_validation_parameter_distributions(output_dir)
        
        # Age comparison matrix
        self._plot_age_comparison_matrix(output_dir)
        
        logger.info(f"Analysis plots saved to: {output_dir}")
    
    def _plot_training_curves(self, output_dir: Path):
        """Plot training and validation curves."""
        metrics_to_plot = [
            ['train_total', 'val_total'],
            ['train_data', 'val_data'],
            ['val_r2_score', 'val_mae'],
            ['val_K_cv', 'val_B_cv', 'val_tau_cv']
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metrics in enumerate(metrics_to_plot):
            ax = axes[i]
            for metric in metrics:
                if metric in self.metrics_tracker.metrics:
                    ax.plot(self.metrics_tracker.metrics[metric], label=metric)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.set_title(f'Training Metrics - {i+1}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_parameter_age_relationships(self, output_dir: Path):
        """Plot learned parameter-age relationships."""
        age_analysis = self.training_results['age_analysis']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        ages = age_analysis['age_range']
        param_names = ['K_values', 'B_values', 'tau_values']
        param_labels = ['Stiffness (K)', 'Damping (B)', 'Neural Delay (τ)']
        
        for i, (param_name, label) in enumerate(zip(param_names, param_labels)):
            ax = axes[i]
            values = age_analysis[param_name]
            
            ax.plot(ages, values, 'b-', linewidth=2, label='Predicted')
            
            # Plot uncertainty bands if available
            if f'{param_name.split("_")[0]}_uncertainties' in age_analysis:
                uncertainties = age_analysis[f'{param_name.split("_")[0]}_uncertainties']
                ax.fill_between(ages, 
                               np.array(values) - np.array(uncertainties),
                               np.array(values) + np.array(uncertainties),
                               alpha=0.2, color='blue')
            
            ax.set_xlabel('Age (years)')
            ax.set_ylabel(label)
            ax.set_title(f'{label} vs Age')
            ax.grid(True, alpha=0.3)
            
            # Add correlation info
            corr = age_analysis['statistics'][f'age_correlation_{param_name.split("_")[0]}']
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'parameter_age_relationships.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_validation_parameter_distributions(self, output_dir: Path):
        """Plot parameter distributions from validation set."""
        self.model.eval()
        
        ages_list = []
        parameters_list = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                if len(batch) == 4:
                    t, age, xy_true, metadata = batch
                else:
                    t, age, xy_true = batch[:3]
                
                t = t.to(self.device)
                age = age.to(self.device)
                
                xy_pred, params = self.model(t, age, deterministic=True)
                
                ages_list.append(age.cpu())
                parameters_list.append(params.cpu())
        
        all_ages = torch.cat(ages_list, dim=0).numpy().squeeze()
        all_params = torch.cat(parameters_list, dim=0).numpy()
        
        # Create scatter plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        param_names = ['Stiffness (K)', 'Damping (B)', 'Neural Delay (τ)']
        
        for i, name in enumerate(param_names):
            ax = axes[i]
            scatter = ax.scatter(all_ages, all_params[:, i], alpha=0.5, s=1, c=all_ages, cmap='viridis')
            ax.set_xlabel('Age (years)')
            ax.set_ylabel(name)
            ax.set_title(f'{name} Distribution')
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label='Age')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'parameter_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_age_comparison_matrix(self, output_dir: Path):
        """Plot age comparison similarity matrix."""
        test_ages = range(30, 81, 10)  # 30, 40, 50, 60, 70, 80
        n_ages = len(test_ages)
        
        similarity_matrix = np.zeros((n_ages, n_ages))
        
        for i, age1 in enumerate(test_ages):
            for j, age2 in enumerate(test_ages):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    comparison = self.model.compare_ages(float(age1), float(age2), n_samples=50)
                    similarity_matrix[i, j] = comparison['mean_similarity']
        
        # Plot matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(similarity_matrix, cmap='viridis', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(n_ages))
        ax.set_yticks(range(n_ages))
        ax.set_xticklabels(test_ages)
        ax.set_yticklabels(test_ages)
        
        ax.set_xlabel('Target Age')
        ax.set_ylabel('Subject Age')
        ax.set_title('Age Comparison Similarity Matrix')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Similarity Score')
        
        # Add text annotations
        for i in range(n_ages):
            for j in range(n_ages):
                text = ax.text(j, i, f'{similarity_matrix[i, j]:.3f}',
                             ha="center", va="center", color="white" if similarity_matrix[i, j] < 0.5 else "black")
        
        plt.tight_layout()
        plt.savefig(output_dir / 'age_comparison_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self):
        """Save all training results and model."""
        logger.info("Saving training results...")
        
        # Save training results
        results_path = Path(self.config['checkpoint_dir']) / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.training_results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        
        # Save metrics
        self.metrics_tracker.save()
        
        # Save final model with full config
        model_path = Path(self.config['checkpoint_dir']) / 'final_model.pth'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'training_results': self.training_results,
            'model_architecture': 'ImprovedBalancePINN'
        }, model_path)
        
        logger.info(f"Results saved to: {self.config['checkpoint_dir']}")
        logger.info(f"  - Training results: {results_path}")
        logger.info(f"  - Final model: {model_path}")
        logger.info(f"  - Metrics: {self.metrics_tracker.save_path}")
    
    def run_complete_training(self):
        """Run the complete training pipeline."""
        logger.info("Starting complete improved single-stage training pipeline...")
        
        try:
            # Setup
            self.setup_data()
            self.setup_model()
            
            # Training
            self.train()
            
            # Additional analysis
            age_comparison_results = self.test_age_comparisons()
            self.training_results['age_comparisons'] = age_comparison_results
            
            # Generate plots
            self.generate_analysis_plots()
            
            # Save everything
            self.save_results()
            
            logger.info("Improved single-stage training completed successfully!")
            
            # Print summary
            self._print_training_summary()
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def _print_training_summary(self):
        """Print a summary of training results."""
        logger.info("=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 60)
        
        results = self.training_results
        
        logger.info(f"Training completed in {results['total_epochs']} epochs")
        logger.info(f"Best validation loss: {results['best_val_loss']:.6f}")
        
        if 'test_metrics' in results:
            test = results['test_metrics']
            logger.info(f"Test Performance:")
            logger.info(f"  - R² Score: {test.get('r2_score', 'N/A'):.4f}")
            logger.info(f"  - MAE: {test.get('mae', 'N/A'):.4f}")
            logger.info(f"  - RMSE: {test.get('rmse', 'N/A'):.4f}")
        
        if 'physics_metrics' in results:
            phys = results['physics_metrics']
            logger.info(f"Physics Compliance:")
            logger.info(f"  - Mean Residual: {phys.get('physics_residual_mean', 'N/A'):.6f}")
            logger.info(f"  - Max Residual: {phys.get('physics_residual_max', 'N/A'):.6f}")
        
        if 'age_analysis' in results:
            age = results['age_analysis']['statistics']
            logger.info(f"Age-Parameter Relationships:")
            logger.info(f"  - K Variation Coeff: {age.get('K_variation_coeff', 'N/A'):.4f}")
            logger.info(f"  - B Variation Coeff: {age.get('B_variation_coeff', 'N/A'):.4f}")
            logger.info(f"  - τ Variation Coeff: {age.get('tau_variation_coeff', 'N/A'):.4f}")
            logger.info(f"  - Age-K Correlation: {age.get('age_correlation_K', 'N/A'):.4f}")
            logger.info(f"  - Age-B Correlation: {age.get('age_correlation_B', 'N/A'):.4f}")
            logger.info(f"  - Age-τ Correlation: {age.get('age_correlation_tau', 'N/A'):.4f}")

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Improved Single-Stage Balance PINN Training')
    
    # Data arguments
    parser.add_argument('--data-folder', type=str, default='processed_data',
                       help='Path to processed data folder')
    parser.add_argument('--age-csv-path', type=str, default='user_ages.csv',
                       help='Path to age CSV file')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_improved_single',
                       help='Directory to save checkpoints')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4096,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay for regularization')
    
    # Model arguments
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='Hidden dimension for networks')
    parser.add_argument('--num-layers', type=int, default=8,
                       help='Number of layers in solver network')
    parser.add_argument('--dropout-rate', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--use-probabilistic', action='store_true',
                       help='Use probabilistic parameter outputs')
    
    # Loss weights
    parser.add_argument('--data-weight', type=float, default=1.0,
                       help='Data loss weight')
    parser.add_argument('--physics-weight', type=float, default=0.01,
                       help='Physics loss weight')
    parser.add_argument('--regularization-weight', type=float, default=0.1,
                       help='Regularization loss weight')
    parser.add_argument('--age-aware-weight', type=float, default=0.1,
                       help='Age-aware loss weight')
    
    # Data processing
    parser.add_argument('--normalize-data', action='store_true',
                       help='Normalize input data')
    parser.add_argument('--augment-data', action='store_true',
                       help='Apply data augmentation')
    
    args = parser.parse_args()
    
    # Create config
    config = {
        # Data
        'data_folder': args.data_folder,
        'age_csv_path': args.age_csv_path,
        'normalize_data': args.normalize_data,
        'augment_data': args.augment_data,
        
        # Training
        'checkpoint_dir': args.checkpoint_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'random_seed': 42,
        'num_workers': 4,
        
        # Model
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'dropout_rate': args.dropout_rate,
        'use_probabilistic': args.use_probabilistic,
        
        # Loss weights
        'data_weight': args.data_weight,
        'physics_weight': args.physics_weight,
        'regularization_weight': args.regularization_weight,
        'age_aware_weight': args.age_aware_weight,
        
        # Training utilities
        'patience': 20,
        'scheduler_patience': 10,
        'warmup_epochs': 5,
        'save_every': 10,
        
        # Parameter bounds
        'param_bounds': {
            'K': (500.0, 3000.0),
            'B': (20.0, 150.0),
            'tau': (0.05, 0.4)
        }
    }
    
    # Create trainer and run
    trainer = ImprovedSingleStageTrainer(config)
    trainer.run_complete_training()

if __name__ == "__main__":
    main()