#!/usr/bin/env python3
"""
Comprehensive Balance PINN Model Performance Tester

This module provides extensive testing and evaluation capabilities for trained PINN models,
including data reconstruction, physics compliance, parameter analysis, and cross-validation.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import h5py
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings
import logging
from datetime import datetime

# Import the model architecture from train.py
from train import BalancePINN, BalanceAgeDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BalancePINNTester:
    """Comprehensive testing framework for Balance PINN models."""
    
    def __init__(self, model_paths: List[str], data_folder: str, output_dir: str = "test_results"):
        """
        Initialize the tester with model paths and data.
        
        Args:
            model_paths: List of paths to trained model files
            data_folder: Path to processed data folder
            output_dir: Directory to save test results
        """
        self.model_paths = model_paths
        self.data_folder = Path(data_folder)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize models and data
        self.models = {}
        self.dataset = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Test results storage
        self.results = {
            'model_info': {},
            'data_performance': {},
            'physics_compliance': {},
            'parameter_analysis': {},
            'cross_validation': {},
            'visualizations': {}
        }
        
    def load_models(self):
        """Load all trained models."""
        logger.info(f"Loading models on device: {self.device}")
        
        for model_path in self.model_paths:
            model_path = Path(model_path)
            if not model_path.exists():
                logger.warning(f"Model file not found: {model_path}")
                continue
                
            try:
                # Create model instance with same architecture as training
                model = BalancePINN(hidden_dim=256, num_layers=8)
                
                # Load state dict
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                
                model_name = model_path.stem
                self.models[model_name] = model
                
                # Store model info
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                self.results['model_info'][model_name] = {
                    'path': str(model_path),
                    'total_parameters': total_params,
                    'trainable_parameters': trainable_params,
                    'device': str(self.device)
                }
                
                logger.info(f"Loaded {model_name}: {total_params:,} parameters")
                
            except Exception as e:
                logger.error(f"Failed to load model {model_path}: {e}")
                
        if not self.models:
            raise ValueError("No models were successfully loaded!")
            
    def load_dataset(self):
        """Load the processed dataset."""
        logger.info("Loading processed dataset...")
        
        try:
            self.dataset = BalanceAgeDataset(str(self.data_folder))
            logger.info(f"Loaded dataset with {len(self.dataset):,} data points")
            
            # Get dataset statistics
            ages = []
            sampling_rate = self.dataset.sampling_rate
            
            # Sample some data points to get age distribution
            sample_indices = np.random.choice(len(self.dataset), min(1000, len(self.dataset)), replace=False)
            for idx in sample_indices:
                _, age_tensor, _ = self.dataset[idx]
                ages.append(age_tensor.item())
                
            self.results['model_info']['dataset'] = {
                'total_points': len(self.dataset),
                'sampling_rate': sampling_rate,
                'age_range': [min(ages), max(ages)],
                'age_mean': np.mean(ages),
                'age_std': np.std(ages)
            }
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
            
    def test_data_reconstruction(self, n_samples: int = 1000):
        """Test model's ability to reconstruct training data."""
        logger.info("Testing data reconstruction performance...")
        
        # Sample test data
        test_indices = np.random.choice(len(self.dataset), n_samples, replace=False)
        
        for model_name, model in self.models.items():
            logger.info(f"Testing reconstruction for {model_name}...")
            
            predictions = []
            targets = []
            times = []
            ages = []
            
            with torch.no_grad():
                for idx in tqdm(test_indices, desc=f"Testing {model_name}"):
                    t, age, xy_true = self.dataset[idx]
                    
                    t = t.unsqueeze(0).to(self.device)
                    age = age.unsqueeze(0).to(self.device)
                    
                    xy_pred, params = model(t, age)
                    
                    predictions.append(xy_pred.cpu().numpy())
                    targets.append(xy_true.numpy())
                    times.append(t.cpu().item())
                    ages.append(age.cpu().item())
            
            predictions = np.array(predictions).squeeze()
            targets = np.array(targets).squeeze()
            
            # Calculate metrics
            mse = np.mean((predictions - targets) ** 2)
            mae = np.mean(np.abs(predictions - targets))
            rmse = np.sqrt(mse)
            
            # Per-coordinate metrics
            mse_x = np.mean((predictions[:, 0] - targets[:, 0]) ** 2)
            mse_y = np.mean((predictions[:, 1] - targets[:, 1]) ** 2)
            mae_x = np.mean(np.abs(predictions[:, 0] - targets[:, 0]))
            mae_y = np.mean(np.abs(predictions[:, 1] - targets[:, 1]))
            
            # R² score
            ss_res = np.sum((targets - predictions) ** 2)
            ss_tot = np.sum((targets - np.mean(targets, axis=0)) ** 2)
            r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            self.results['data_performance'][model_name] = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'mse_x': mse_x,
                'mse_y': mse_y,
                'mae_x': mae_x,
                'mae_y': mae_y,
                'r2_score': r2_score,
                'n_samples': n_samples,
                'predictions': predictions,
                'targets': targets,
                'times': times,
                'ages': ages
            }
            
            logger.info(f"{model_name} - MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2_score:.4f}")
    
    def test_physics_compliance(self, n_samples: int = 500):
        """Test physics constraint compliance."""
        logger.info("Testing physics compliance...")
        
        test_indices = np.random.choice(len(self.dataset), n_samples, replace=False)
        
        for model_name, model in self.models.items():
            logger.info(f"Testing physics compliance for {model_name}...")
            
            physics_residuals = []
            learned_params = []
            
            for idx in tqdm(test_indices, desc=f"Physics test {model_name}"):
                t, age, _ = self.dataset[idx]
                
                t = t.unsqueeze(0).to(self.device).requires_grad_(True)
                age = age.unsqueeze(0).to(self.device)
                
                # Forward pass
                xy_pred, params = model(t, age)
                x_pred, y_pred = xy_pred[:, 0], xy_pred[:, 1]
                
                # Calculate derivatives
                dx_dt = torch.autograd.grad(x_pred, t, create_graph=True)[0]
                dy_dt = torch.autograd.grad(y_pred, t, create_graph=True)[0]
                d2x_dt2 = torch.autograd.grad(dx_dt, t, create_graph=True)[0]
                d2y_dt2 = torch.autograd.grad(dy_dt, t, create_graph=True)[0]
                
                # Physics parameters
                g, L, m = 9.81, 1.0, 70.0
                K, B, tau = params[:, 0], params[:, 1], params[:, 2]
                
                # Physics residuals (inverted pendulum equation)
                residual_x = d2x_dt2 - (g/L)*x_pred + (K/(m*L**2))*x_pred + (B/(m*L**2))*dx_dt
                residual_y = d2y_dt2 - (g/L)*y_pred + (K/(m*L**2))*y_pred + (B/(m*L**2))*dy_dt
                
                physics_residuals.append([
                    residual_x.item(),
                    residual_y.item()
                ])
                
                learned_params.append([
                    K.item(),
                    B.item(),
                    tau.item(),
                    age.item()
                ])
            
            physics_residuals = np.array(physics_residuals)
            learned_params = np.array(learned_params)
            
            # Calculate physics compliance metrics
            residual_rms_x = np.sqrt(np.mean(physics_residuals[:, 0] ** 2))
            residual_rms_y = np.sqrt(np.mean(physics_residuals[:, 1] ** 2))
            residual_rms_total = np.sqrt(np.mean(physics_residuals ** 2))
            
            self.results['physics_compliance'][model_name] = {
                'residual_rms_x': residual_rms_x,
                'residual_rms_y': residual_rms_y,
                'residual_rms_total': residual_rms_total,
                'residuals': physics_residuals,
                'learned_params': learned_params
            }
            
            logger.info(f"{model_name} - Physics RMS residual: {residual_rms_total:.6f}")
    
    def analyze_parameters(self):
        """Analyze learned age-dependent parameters."""
        logger.info("Analyzing learned parameters...")
        
        for model_name, model in self.models.items():
            logger.info(f"Parameter analysis for {model_name}...")
            
            # Test age range
            age_range = np.linspace(50, 90, 100)
            
            K_values = []
            B_values = []
            tau_values = []
            
            with torch.no_grad():
                for age in age_range:
                    age_tensor = torch.tensor([[age]], dtype=torch.float32).to(self.device)
                    
                    # Get parameters from ParameterNet
                    params_raw = model.parameter_net(age_tensor)
                    K = 2000 * torch.sigmoid(params_raw[:, 0:1])
                    B = 100 * torch.sigmoid(params_raw[:, 1:2])
                    tau = 0.3 * torch.sigmoid(params_raw[:, 2:3])
                    
                    K_values.append(K.item())
                    B_values.append(B.item())
                    tau_values.append(tau.item())
            
            # Check if parameters change with age (non-constant)
            K_variation = np.std(K_values) / np.mean(K_values)
            B_variation = np.std(B_values) / np.mean(B_values)
            tau_variation = np.std(tau_values) / np.mean(tau_values)
            
            self.results['parameter_analysis'][model_name] = {
                'age_range': age_range.tolist(),
                'K_values': K_values,
                'B_values': B_values,
                'tau_values': tau_values,
                'K_stats': {
                    'mean': np.mean(K_values),
                    'std': np.std(K_values),
                    'min': np.min(K_values),
                    'max': np.max(K_values),
                    'variation_coeff': K_variation
                },
                'B_stats': {
                    'mean': np.mean(B_values),
                    'std': np.std(B_values),
                    'min': np.min(B_values),
                    'max': np.max(B_values),
                    'variation_coeff': B_variation
                },
                'tau_stats': {
                    'mean': np.mean(tau_values),
                    'std': np.std(tau_values),
                    'min': np.min(tau_values),
                    'max': np.max(tau_values),
                    'variation_coeff': tau_variation
                }
            }
            
            logger.info(f"{model_name} - K: {np.mean(K_values):.1f}±{np.std(K_values):.1f}, "
                       f"B: {np.mean(B_values):.1f}±{np.std(B_values):.1f}, "
                       f"τ: {np.mean(tau_values):.3f}±{np.std(tau_values):.3f}")
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations."""
        logger.info("Generating visualizations...")
        
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # 1. Model comparison plot
        self._plot_model_comparison()
        
        # 2. Parameter vs age plots
        self._plot_parameters_vs_age()
        
        # 3. Physics residuals
        self._plot_physics_residuals()
        
        # 4. Prediction scatter plots
        self._plot_prediction_scatter()
        
        # 5. Sample trajectory reconstructions
        self._plot_sample_trajectories()
    
    def _plot_model_comparison(self):
        """Plot model performance comparison."""
        if len(self.models) < 2:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        model_names = list(self.models.keys())
        metrics = ['mse', 'mae', 'r2_score']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            values = [self.results['data_performance'][name][metric] for name in model_names]
            
            bars = ax.bar(model_names, values)
            ax.set_title(f'{metric.upper()} Comparison')
            ax.set_ylabel(metric.upper())
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.4f}', ha='center', va='bottom')
        
        # Physics compliance comparison
        ax = axes[1, 1]
        physics_values = [self.results['physics_compliance'][name]['residual_rms_total'] 
                         for name in model_names]
        bars = ax.bar(model_names, physics_values)
        ax.set_title('Physics RMS Residual Comparison')
        ax.set_ylabel('RMS Residual')
        
        for bar, value in zip(bars, physics_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.6f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_parameters_vs_age(self):
        """Plot learned parameters vs age."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for model_name, model in self.models.items():
            data = self.results['parameter_analysis'][model_name]
            ages = data['age_range']
            
            axes[0].plot(ages, data['K_values'], label=model_name, marker='o', markersize=2)
            axes[0].set_title('Stiffness (K) vs Age')
            axes[0].set_xlabel('Age (years)')
            axes[0].set_ylabel('Stiffness (N·m/rad)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(ages, data['B_values'], label=model_name, marker='o', markersize=2)
            axes[1].set_title('Damping (B) vs Age')
            axes[1].set_xlabel('Age (years)')
            axes[1].set_ylabel('Damping (N·m·s/rad)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            axes[2].plot(ages, data['tau_values'], label=model_name, marker='o', markersize=2)
            axes[2].set_title('Neural Delay (τ) vs Age')
            axes[2].set_xlabel('Age (years)')
            axes[2].set_ylabel('Delay (s)')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'parameters_vs_age.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_physics_residuals(self):
        """Plot physics residuals distribution."""
        fig, axes = plt.subplots(1, len(self.models), figsize=(6*len(self.models), 5))
        if len(self.models) == 1:
            axes = [axes]
        
        for i, (model_name, model) in enumerate(self.models.items()):
            residuals = self.results['physics_compliance'][model_name]['residuals']
            
            axes[i].hist2d(residuals[:, 0], residuals[:, 1], bins=50, alpha=0.7)
            axes[i].set_title(f'Physics Residuals - {model_name}')
            axes[i].set_xlabel('X Residual')
            axes[i].set_ylabel('Y Residual')
            
            # Add zero lines
            axes[i].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[i].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'physics_residuals.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_prediction_scatter(self):
        """Plot predicted vs actual values."""
        fig, axes = plt.subplots(len(self.models), 2, figsize=(10, 5*len(self.models)))
        if len(self.models) == 1:
            axes = axes.reshape(1, -1)
        
        for i, (model_name, model) in enumerate(self.models.items()):
            data = self.results['data_performance'][model_name]
            predictions = data['predictions']
            targets = data['targets']
            
            # X coordinate
            axes[i, 0].scatter(targets[:, 0], predictions[:, 0], alpha=0.5, s=1)
            axes[i, 0].plot([targets[:, 0].min(), targets[:, 0].max()], 
                           [targets[:, 0].min(), targets[:, 0].max()], 'r--')
            axes[i, 0].set_title(f'{model_name} - X Coordinate')
            axes[i, 0].set_xlabel('Actual X')
            axes[i, 0].set_ylabel('Predicted X')
            axes[i, 0].grid(True, alpha=0.3)
            
            # Y coordinate
            axes[i, 1].scatter(targets[:, 1], predictions[:, 1], alpha=0.5, s=1)
            axes[i, 1].plot([targets[:, 1].min(), targets[:, 1].max()], 
                           [targets[:, 1].min(), targets[:, 1].max()], 'r--')
            axes[i, 1].set_title(f'{model_name} - Y Coordinate')
            axes[i, 1].set_xlabel('Actual Y')
            axes[i, 1].set_ylabel('Predicted Y')
            axes[i, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'prediction_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_sample_trajectories(self, n_samples: int = 6):
        """Plot sample trajectory reconstructions."""
        # Sample some trajectories from the same subject
        sample_indices = np.random.choice(len(self.dataset), n_samples * 10, replace=False)
        
        # Get subject data points (multiple time points from same subjects)
        subject_trajectories = {}
        
        for idx in sample_indices:
            # Try to get subject info from HDF5 file directly
            file_path, subject_key, trial_key, point_idx = self.dataset.index_map[idx]
            subject_trial = f"{subject_key}_{trial_key}"
            
            if subject_trial not in subject_trajectories:
                subject_trajectories[subject_trial] = []
            
            t, age, xy = self.dataset[idx]
            subject_trajectories[subject_trial].append({
                'time': t.item(),
                'age': age.item(),
                'xy': xy.numpy(),
                'idx': idx
            })
        
        # Select subjects with enough data points
        good_subjects = {k: v for k, v in subject_trajectories.items() 
                        if len(v) >= 10}
        
        if len(good_subjects) == 0:
            logger.warning("No subjects with enough data points for trajectory plotting")
            return
            
        selected_subjects = list(good_subjects.keys())[:min(n_samples, len(good_subjects))]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, subject_key in enumerate(selected_subjects):
            if i >= len(axes):
                break
                
            trajectory = good_subjects[subject_key]
            trajectory.sort(key=lambda x: x['time'])  # Sort by time
            
            times = [p['time'] for p in trajectory]
            actual_x = [p['xy'][0] for p in trajectory]
            actual_y = [p['xy'][1] for p in trajectory]
            age = trajectory[0]['age']
            
            # Get predictions from the first model
            model_name = list(self.models.keys())[0]
            model = self.models[model_name]
            
            predicted_x = []
            predicted_y = []
            
            with torch.no_grad():
                for point in trajectory:
                    t_tensor = torch.tensor([[point['time']]], dtype=torch.float32).to(self.device)
                    age_tensor = torch.tensor([[point['age']]], dtype=torch.float32).to(self.device)
                    
                    xy_pred, _ = model(t_tensor, age_tensor)
                    predicted_x.append(xy_pred[0, 0].cpu().item())
                    predicted_y.append(xy_pred[0, 1].cpu().item())
            
            axes[i].plot(actual_x, actual_y, 'b-', label='Actual', linewidth=2)
            axes[i].plot(predicted_x, predicted_y, 'r--', label='Predicted', linewidth=2)
            axes[i].set_title(f'Subject {subject_key.split("_")[0]} (Age: {age:.0f})')
            axes[i].set_xlabel('X Position')
            axes[i].set_ylabel('Y Position')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            axes[i].axis('equal')
        
        # Remove empty subplots
        for i in range(len(selected_subjects), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sample_trajectories.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self):
        """Generate comprehensive performance report."""
        logger.info("Generating performance report...")
        
        report_path = self.output_dir / 'performance_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("BALANCE PINN MODEL PERFORMANCE REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Dataset info
            f.write("DATASET INFORMATION\n")
            f.write("-" * 30 + "\n")
            dataset_info = self.results['model_info']['dataset']
            f.write(f"Total data points: {dataset_info['total_points']:,}\n")
            f.write(f"Sampling rate: {dataset_info['sampling_rate']} Hz\n")
            f.write(f"Age range: {dataset_info['age_range'][0]:.1f} - {dataset_info['age_range'][1]:.1f} years\n")
            f.write(f"Age mean ± std: {dataset_info['age_mean']:.1f} ± {dataset_info['age_std']:.1f} years\n\n")
            
            # Model information
            f.write("MODEL INFORMATION\n")
            f.write("-" * 30 + "\n")
            for model_name, info in self.results['model_info'].items():
                if model_name == 'dataset':
                    continue
                f.write(f"Model: {model_name}\n")
                f.write(f"  Path: {info['path']}\n")
                f.write(f"  Parameters: {info['total_parameters']:,}\n")
                f.write(f"  Device: {info['device']}\n\n")
            
            # Data reconstruction performance
            f.write("DATA RECONSTRUCTION PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            for model_name, perf in self.results['data_performance'].items():
                f.write(f"Model: {model_name}\n")
                f.write(f"  MSE: {perf['mse']:.6f}\n")
                f.write(f"  MAE: {perf['mae']:.6f}\n")
                f.write(f"  RMSE: {perf['rmse']:.6f}\n")
                f.write(f"  R² Score: {perf['r2_score']:.4f}\n")
                f.write(f"  MSE X: {perf['mse_x']:.6f}\n")
                f.write(f"  MSE Y: {perf['mse_y']:.6f}\n")
                f.write(f"  MAE X: {perf['mae_x']:.6f}\n")
                f.write(f"  MAE Y: {perf['mae_y']:.6f}\n\n")
            
            # Physics compliance
            f.write("PHYSICS COMPLIANCE\n")
            f.write("-" * 25 + "\n")
            for model_name, phys in self.results['physics_compliance'].items():
                f.write(f"Model: {model_name}\n")
                f.write(f"  RMS Residual X: {phys['residual_rms_x']:.6f}\n")
                f.write(f"  RMS Residual Y: {phys['residual_rms_y']:.6f}\n")
                f.write(f"  RMS Residual Total: {phys['residual_rms_total']:.6f}\n\n")
            
            # Parameter analysis
            f.write("LEARNED PARAMETERS ANALYSIS\n")
            f.write("-" * 35 + "\n")
            for model_name, params in self.results['parameter_analysis'].items():
                f.write(f"Model: {model_name}\n")
                f.write(f"  Stiffness K (N·m/rad):\n")
                f.write(f"    Mean: {params['K_stats']['mean']:.1f}\n")
                f.write(f"    Std: {params['K_stats']['std']:.1f}\n")
                f.write(f"    Range: {params['K_stats']['min']:.1f} - {params['K_stats']['max']:.1f}\n")
                f.write(f"    Variation Coeff: {params['K_stats']['variation_coeff']:.4f}\n")
                
                f.write(f"  Damping B (N·m·s/rad):\n")
                f.write(f"    Mean: {params['B_stats']['mean']:.1f}\n")
                f.write(f"    Std: {params['B_stats']['std']:.1f}\n")
                f.write(f"    Range: {params['B_stats']['min']:.1f} - {params['B_stats']['max']:.1f}\n")
                f.write(f"    Variation Coeff: {params['B_stats']['variation_coeff']:.4f}\n")
                
                f.write(f"  Neural Delay τ (s):\n")
                f.write(f"    Mean: {params['tau_stats']['mean']:.3f}\n")
                f.write(f"    Std: {params['tau_stats']['std']:.3f}\n")
                f.write(f"    Range: {params['tau_stats']['min']:.3f} - {params['tau_stats']['max']:.3f}\n")
                f.write(f"    Variation Coeff: {params['tau_stats']['variation_coeff']:.4f}\n\n")
            
            # Model recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 20 + "\n")
            
            # Find best performing model
            best_model = min(self.results['data_performance'].keys(), 
                           key=lambda x: self.results['data_performance'][x]['mse'])
            
            f.write(f"Best performing model (lowest MSE): {best_model}\n\n")
            
            # Analysis
            for model_name in self.results['data_performance'].keys():
                perf = self.results['data_performance'][model_name]
                phys = self.results['physics_compliance'][model_name]
                params = self.results['parameter_analysis'][model_name]
                
                f.write(f"Analysis for {model_name}:\n")
                
                # Data fitting quality
                if perf['r2_score'] > 0.8:
                    f.write("  ✓ Good data reconstruction (R² > 0.8)\n")
                else:
                    f.write(f"  ⚠ Poor data reconstruction (R² = {perf['r2_score']:.3f})\n")
                
                # Physics compliance
                if phys['residual_rms_total'] < 1e-3:
                    f.write("  ✓ Good physics compliance (residual < 1e-3)\n")
                else:
                    f.write(f"  ⚠ Physics compliance issues (residual = {phys['residual_rms_total']:.6f})\n")
                
                # Parameter learning
                if params['K_stats']['variation_coeff'] > 0.1:
                    f.write("  ✓ Stiffness varies with age\n")
                else:
                    f.write("  ⚠ Stiffness may not be learning age dependence\n")
                
                if params['tau_stats']['variation_coeff'] > 0.1:
                    f.write("  ✓ Neural delay varies with age\n")
                else:
                    f.write("  ⚠ Neural delay may not be learning age dependence\n")
                
                f.write("\n")
        
        logger.info(f"Report saved to {report_path}")
    
    def run_all_tests(self):
        """Run all tests and generate complete analysis."""
        logger.info("Starting comprehensive model testing...")
        
        try:
            # Load models and data
            self.load_models()
            self.load_dataset()
            
            # Run all tests
            self.test_data_reconstruction()
            self.test_physics_compliance()
            self.analyze_parameters()
            self.generate_visualizations()
            self.generate_report()
            
            logger.info(f"Testing complete! Results saved to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Testing failed: {e}")
            raise

def main():
    """Main function to run comprehensive model testing."""
    
    # Configuration
    model_paths = [
        "trained_models/best_balance_pinn.pth",
        "trained_models/simple_pinn_weights.pth"
    ]
    data_folder = "processed_data"
    output_dir = "test_results"
    
    # Filter existing models
    existing_models = [p for p in model_paths if Path(p).exists()]
    
    if not existing_models:
        logger.error("No trained models found!")
        return
    
    # Create tester and run all tests
    tester = BalancePINNTester(existing_models, data_folder, output_dir)
    tester.run_all_tests()

if __name__ == "__main__":
    main()