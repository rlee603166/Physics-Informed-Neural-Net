#!/usr/bin/env python3
"""
FINAL SCRIPT (Corrected): Two-Stage Inverted Double Pendulum PINN Trainer

This script implements the complete two-stage training pipeline using a more
realistic inverted double pendulum physics model to capture both ankle and hip
balance strategies. It includes the three-stage training curriculum and advanced
regularization to prevent model collapse and learn meaningful parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
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
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')
import h5py

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# CONFIGURATION FOR DOUBLE PENDULUM
# =============================================================================

def get_config() -> Dict:
    """Provides the master configuration for the training run."""
    config = {
        # Data and Hardware
        'data_folder': 'processed_data',
        'age_csv_path': 'user_ages.csv',
        'batch_size': 768,
        'num_workers': 4,
        'mixed_precision': True,
        
        # Data Normalization to [-1, 1]
        'data_normalization_bounds': {
            'x': (320.0, 380.0), # Min/Max for COP_x in mm
            'y': (880.0, 980.0)  # Min/Max for COP_y in mm
        },
        
        # Training Schedule
        'stage1_epochs_total': 60,
        'stage2_epochs': 50,
        'stage1_lr': 2e-4,
        'stage2_lr': 1e-3,
        'weight_decay': 1e-6,
        'early_stopping_patience': 10,
        'min_delta': 1e-5,
        'best_model_path_stage1': 'best_stage1_model.pth',
        'best_model_path_stage2': 'best_stage2_model.pth',
        'stage1_results_path': 'stage1_parameter_results.csv',

        # Three-Stage Curriculum for Stage 1
        'stage1_warmup_epochs': 5,
        'stage2_gradual_epochs': 15,
        'physics_schedule': {'warmup': 0.0, 'gradual': 50.0, 'full': 100.0},
        'noise_schedule': {'warmup': 0.1, 'gradual': 0.05, 'full': 0.02},
        
        # Advanced Loss Weights
        'centering_weight': 1.5,
        'variance_penalty_weight': 10.0,
        'smooth_boundary_weight': 5.0,
        'correlation_weight': 50.0,
        
        # Double Pendulum Parameters (5 total)
        'param_bounds': {
            'K_ankle': (500.0, 3000.0), 'B_ankle': (20.0, 150.0),
            'K_hip': (500.0, 2500.0), 'B_hip': (20.0, 150.0),
            'tau': (0.1, 0.3)
        },
        'centering_targets': {
            'K_ankle': 1600.0, 'B_ankle': 80.0,
            'K_hip': 1200.0, 'B_hip': 70.0,
            'tau': 0.18
        },
    }
    return config

# =============================================================================
# DATASET
# =============================================================================

class BalanceDataset(Dataset):
    """Loads, normalizes, and stores all COP data points in memory."""
    def __init__(self, config):
        self.config = config
        self.data_folder = Path(config['data_folder'])
        self.batch_files = sorted(list(self.data_folder.glob("*.h5")))
        self.age_lookup = pd.read_csv(config['age_csv_path'], index_col='user_id')['age'].to_dict()
        
        self.data_points, self.subject_map = self._load_all_data()
        self.n_subjects = len(self.subject_map)
        logger.info(f"Dataset ready: {len(self.data_points):,} points from {self.n_subjects} subjects loaded into memory.")

    def _load_all_data(self):
        valid_subjects = set()
        for h5_file in self.batch_files:
            with h5py.File(h5_file, 'r') as f:
                for sid_full in f.keys():
                    sid = sid_full.replace('subject_', '')
                    if sid in self.age_lookup:
                        valid_subjects.add(sid)
        subject_map = {sid: i for i, sid in enumerate(sorted(list(valid_subjects)))}
        
        logger.info("Loading all data points into memory (this may take a moment)...")
        
        all_data = []
        x_min, x_max = self.config['data_normalization_bounds']['x']
        y_min, y_max = self.config['data_normalization_bounds']['y']

        for h5_file in tqdm(self.batch_files, desc="Loading data files"):
            with h5py.File(h5_file, 'r') as f:
                for sid_full in f.keys():
                    sid = sid_full.replace('subject_', '')
                    if sid in subject_map:
                        for trial_key in f[sid_full].keys():
                            if 'cop_x' in f[sid_full][trial_key]:
                                x_raw = f[sid_full][trial_key]['cop_x'][:]
                                y_raw = f[sid_full][trial_key]['cop_y'][:]
                                
                                # Normalization is done here, during loading
                                x_norm = 2 * (x_raw - x_min) / (x_max - x_min) - 1
                                y_norm = 2 * (y_raw - y_min) / (y_max - y_min) - 1
                                
                                subject_idx_val = subject_map[sid]

                                for i in range(len(x_raw)):
                                    all_data.append({
                                        't': torch.tensor(i / 106.0, dtype=torch.float32).unsqueeze(0),
                                        'xy': torch.tensor([x_norm[i], y_norm[i]], dtype=torch.float32),
                                        'subject_idx': torch.tensor(subject_idx_val, dtype=torch.long),
                                        'subject_id': sid # Keep for splitting
                                    })
        return all_data, subject_map

    def __len__(self): 
        return len(self.data_points)
    
    def __getitem__(self, idx):
        # __getitem__ is now very fast
        dp = self.data_points[idx]
        return dp['t'], dp['xy'], dp['subject_idx']

# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class DoublePendulumPhysicsLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        self.m1, self.m2, self.l1, self.l2, self.g = 60.0, 10.0, 0.5, 0.4, 9.81

    def forward(self, t, xy_pred, params):
        K_a, B_a = params['K_ankle'], params['B_ankle']
        K_h, B_h = params['K_hip'], params['B_hip']
        tau = params['tau']

        # === FIX: Calculate derivatives for x and y components separately to avoid IndexError ===
        vel_x = torch.autograd.grad(xy_pred[:, 0].sum(), t, create_graph=True)[0]
        vel_y = torch.autograd.grad(xy_pred[:, 1].sum(), t, create_graph=True)[0]
        accel_x = torch.autograd.grad(vel_x.sum(), t, retain_graph=True)[0]
        accel_y = torch.autograd.grad(vel_y.sum(), t, retain_graph=True)[0]
        
        theta1 = xy_pred[:, 0] / self.l1
        theta2 = xy_pred[:, 1] / self.l2
        theta1_delayed = theta1 - tau * vel_x.squeeze(-1)
        theta2_delayed = theta2 - tau * vel_y.squeeze(-1)

        res_ankle = (self.m1*self.l1 + self.m2*self.l2)*accel_x.squeeze(-1) + B_a*vel_x.squeeze(-1) + K_a*theta1_delayed - self.m1*self.g*theta1
        res_hip = self.m2*self.l2*accel_y.squeeze(-1) + B_h*vel_y.squeeze(-1) + K_h*theta2_delayed - self.m2*self.g*theta2
        
        loss = F.mse_loss(res_ankle, torch.zeros_like(res_ankle)) + \
               F.mse_loss(res_hip, torch.zeros_like(res_hip))
               
        return loss * self.weight

class AdvancedRegularizationLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    def forward(self, params, params_norm):
        center_loss, var_loss = 0, 0
        param_tensors = torch.stack(list(params.values()), dim=1)
        for i, name in enumerate(self.config['param_bounds'].keys()):
            p = param_tensors[:, i]
            target = self.config['centering_targets'][name]
            center_loss += ((p.mean() - target) / target)**2
            var_loss += 1.0 / (p.std() + 1e-6)
        
        boundary_loss = F.relu(0.05 - params_norm).mean() + F.relu(0.05 - (1.0 - params_norm)).mean()
        
        b_a_norm = (params['B_ankle'] - params['B_ankle'].mean()) / (params['B_ankle'].std() + 1e-8)
        b_h_norm = (params['B_hip'] - params['B_hip'].mean()) / (params['B_hip'].std() + 1e-8)
        tau_norm = (params['tau'] - params['tau'].mean()) / (params['tau'].std() + 1e-8)
        corr_loss = torch.abs(torch.mean(b_a_norm * tau_norm)) + torch.abs(torch.mean(b_h_norm * tau_norm))
        
        return self.config['centering_weight'] * center_loss, \
               self.config['variance_penalty_weight'] * var_loss, \
               self.config['smooth_boundary_weight'] * boundary_loss, \
               self.config['correlation_weight'] * corr_loss

# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

class DoublePendulumPINN(nn.Module):
    def __init__(self, n_subjects, bounds):
        super().__init__()
        self.n_subjects, self.bounds = n_subjects, bounds
        self.position_net = nn.Sequential(nn.Linear(1 + n_subjects, 256), nn.Tanh(), nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 2))
        self.param_net = nn.Sequential(nn.Linear(n_subjects, 256), nn.ReLU(), nn.Linear(256, 5))
        # Initialize final layers for better starting point
        with torch.no_grad():
            self.param_net[-1].bias.data.zero_() # Center sigmoid outputs
            self.position_net[-1].weight.data *= 0.01 # Start with small sway
            self.position_net[-1].bias.data.zero_()

    def forward(self, t, subject_idx, noise_std=0.0):
        subject_onehot = F.one_hot(subject_idx, num_classes=self.n_subjects).float()
        xy_pred = self.position_net(torch.cat([t, subject_onehot], dim=1))
        
        params_norm_raw = self.param_net(subject_onehot)
        if self.training and noise_std > 0:
            params_norm_raw += noise_std * torch.randn_like(params_norm_raw)
        params_norm = torch.sigmoid(params_norm_raw)
        
        params = {}
        param_names = list(self.bounds.keys())
        for i, name in enumerate(param_names):
            min_b, max_b = self.bounds[name]
            # Clamp to prevent out-of-bounds due to noise
            params[name] = torch.clamp(min_b + (max_b - min_b) * params_norm[:, i], min=min_b, max=max_b)
        return xy_pred, params, params_norm

class AgeParameterModelDP(nn.Module):
    def __init__(self, bounds):
        super().__init__()
        self.bounds = bounds
        self.age_net = nn.Sequential(nn.Linear(1, 128), nn.ReLU(), nn.Linear(128, 10)) # 5 means + 5 stds
    def forward(self, age):
        output = self.age_net((age - 50.0) / 30.0)
        means_raw, log_stds = output[:, :5], output[:, 5:]
        stds = torch.exp(log_stds)
        means = {}
        param_names = list(self.bounds.keys())
        for i, name in enumerate(param_names):
            min_b, max_b = self.bounds[name]
            means[name] = min_b + (max_b - min_b) * torch.sigmoid(means_raw[:, i])
        return means, stds

# =============================================================================
# MAIN TRAINER CLASS
# =============================================================================

class TwoStageTrainer:
    def __init__(self, config):
        self.config = config
        self.dataset = BalanceDataset(config)
        self.n_subjects = self.dataset.n_subjects
        
        from sklearn.model_selection import train_test_split
        
        all_subject_ids = sorted(list(self.dataset.subject_map.keys()))
        train_sids, val_sids = train_test_split(all_subject_ids, test_size=0.3, random_state=42)
        train_sids_set = set(train_sids)
        
        train_indices, val_indices = [], []
        # This can be slow for large datasets, so we show a progress bar
        for i, dp in enumerate(tqdm(self.dataset.data_points, desc="Building subject-wise split")):
            if dp['subject_id'] in train_sids_set:
                train_indices.append(i)
            else:
                val_indices.append(i)
        
        logger.info(f"Subject-wise split: {len(train_sids)} train subjects, {len(val_sids)} validation subjects.")
        logger.info(f"Data points: {len(train_indices):,} train, {len(val_indices):,} validation.")

        self.train_loader = DataLoader(Subset(self.dataset, train_indices), batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=True)
        self.val_loader = DataLoader(Subset(self.dataset, val_indices), batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True)
        logging.info(f"Initialized trainer for {self.n_subjects} subjects on {DEVICE}.")

    def train_stage1(self):
        logging.info("\n--- Starting Stage 1: Double Pendulum Parameter Identification ---")
        model = DoublePendulumPINN(self.n_subjects, self.config['param_bounds']).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config['stage1_lr'], weight_decay=self.config['weight_decay'])
        scaler = GradScaler(enabled=self.config['mixed_precision'])

        data_loss_fn = nn.MSELoss()
        physics_loss_fn = DoublePendulumPhysicsLoss()
        regularization_loss_fn = AdvancedRegularizationLoss(self.config)

        total_epochs = self.config['stage1_epochs_total']
        
        # Early stopping variables
        best_val_loss = float('inf')
        epochs_no_improve = 0
        patience = self.config.get('early_stopping_patience', 10)
        min_delta = self.config.get('min_delta', 1e-5)

        pbar = tqdm(range(total_epochs), desc="Stage 1")
        for epoch in pbar:
            if epoch < self.config['stage1_warmup_epochs']:
                physics_weight = self.config['physics_schedule']['warmup']
                noise_std = self.config['noise_schedule']['warmup']
            elif epoch < self.config['stage1_warmup_epochs'] + self.config['stage2_gradual_epochs']:
                physics_weight = self.config['physics_schedule']['gradual']
                noise_std = self.config['noise_schedule']['gradual']
            else:
                physics_weight = self.config['physics_schedule']['full']
                noise_std = self.config['noise_schedule']['full']
            physics_loss_fn.weight = physics_weight
            
            # --- Training Loop ---
            model.train()
            train_loss = 0.0
            for t, xy_true, subject_idx in self.train_loader:
                t, xy_true, subject_idx = t.to(DEVICE), xy_true.to(DEVICE), subject_idx.to(DEVICE)
                t.requires_grad_(True)
                
                optimizer.zero_grad()
                with autocast(enabled=self.config['mixed_precision']):
                    xy_pred, params, params_norm = model(t, subject_idx, noise_std=noise_std)
                    l_data = data_loss_fn(xy_pred, xy_true)
                    l_physics = physics_loss_fn(t, xy_pred, params)
                    l_center, l_var, l_bound, l_corr = regularization_loss_fn(params, params_norm)
                    total_loss = l_data + l_physics + l_center + l_var + l_bound + l_corr
                
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += total_loss.item()
            
            avg_train_loss = train_loss / len(self.train_loader)

            # --- Validation Loop ---
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for t, xy_true, subject_idx in self.val_loader:
                    t, xy_true, subject_idx = t.to(DEVICE), xy_true.to(DEVICE), subject_idx.to(DEVICE)
                    t.requires_grad_(True) # Still need for physics loss
                    with autocast(enabled=self.config['mixed_precision']):
                        xy_pred, params, params_norm = model(t, subject_idx, noise_std=0.0) # No noise for validation
                        l_data = data_loss_fn(xy_pred, xy_true)
                        l_physics = physics_loss_fn(t, xy_pred, params)
                        l_center, l_var, l_bound, l_corr = regularization_loss_fn(params, params_norm)
                        total_loss = l_data + l_physics + l_center + l_var + l_bound + l_corr
                    val_loss += total_loss.item()
            
            avg_val_loss = val_loss / len(self.val_loader)
            pbar.set_postfix({'train_loss': f'{avg_train_loss:.4f}', 'val_loss': f'{avg_val_loss:.4f}'})

            # --- Early Stopping Check & Logging ---
            if avg_val_loss < best_val_loss - min_delta:
                logging.info(f"Epoch {epoch+1}/{total_epochs} - Val Loss improved to {avg_val_loss:.4f}. Saving model.")
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), self.config['best_model_path_stage1'])
            else:
                epochs_no_improve += 1
                logging.info(f"Epoch {epoch+1}/{total_epochs} - Val Loss did not improve. Patience: {epochs_no_improve}/{patience}.")

            if epochs_no_improve >= patience:
                logging.info(f"Early stopping triggered at epoch {epoch+1}. Best val_loss: {best_val_loss:.4f}")
                break
        
        logging.info(f"Loading best model from {self.config['best_model_path_stage1']} with val_loss: {best_val_loss:.4f}")
        model.load_state_dict(torch.load(self.config['best_model_path_stage1']))
        model.eval()

        logging.info("Extracting final parameters from Stage 1...")
        all_indices = torch.arange(self.n_subjects).to(DEVICE)
        dummy_t = torch.zeros(self.n_subjects, 1).to(DEVICE)
        with torch.no_grad():
            _, final_params, _ = model(dummy_t, all_indices)
        
        results = {name: p.cpu().numpy() for name, p in final_params.items()}
        results_df = pd.DataFrame(results, index=self.dataset.subject_map.keys())
        full_age_df = pd.read_csv(self.config['age_csv_path']).rename(columns={'user_id':'subject_id'}).set_index('subject_id')
        results_df = results_df.join(full_age_df)
        results_df.to_csv(self.config['stage1_results_path'])
        logging.info(f"Stage 1 results saved to {self.config['stage1_results_path']}")
        
        logging.info("✅ Stage 1 Complete.")
        return results_df

    def train_stage2(self, stage1_results_df):
        logging.info("\n--- Starting Stage 2: Learning Age Relationships ---")
        param_names = list(self.config['param_bounds'].keys())
        
        stage1_results_df = stage1_results_df.dropna(subset=['age'])
        
        # --- Train/Val split for stage 2 ---
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(stage1_results_df, test_size=0.3, random_state=42)

        train_ages = torch.tensor(train_df['age'].values, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
        train_params_true = torch.tensor(train_df[param_names].values, dtype=torch.float32).to(DEVICE)
        
        val_ages = torch.tensor(val_df['age'].values, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
        val_params_true = torch.tensor(val_df[param_names].values, dtype=torch.float32).to(DEVICE)

        model = AgeParameterModelDP(self.config['param_bounds']).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config['stage2_lr'])
        loss_fn = nn.MSELoss()
        
        # Early stopping variables
        best_val_loss = float('inf')
        epochs_no_improve = 0
        patience = self.config.get('early_stopping_patience', 10)
        min_delta = self.config.get('min_delta', 1e-5)

        pbar = tqdm(range(self.config['stage2_epochs']), desc="Stage 2")
        for epoch in pbar:
            # --- Training ---
            model.train()
            optimizer.zero_grad()
            params_pred_mean, _ = model(train_ages)
            train_loss = loss_fn(torch.stack(list(params_pred_mean.values()), dim=1), train_params_true)
            train_loss.backward()
            optimizer.step()
            
            # --- Validation ---
            model.eval()
            with torch.no_grad():
                val_params_pred_mean, _ = model(val_ages)
                val_loss = loss_fn(torch.stack(list(val_params_pred_mean.values()), dim=1), val_params_true)

            pbar.set_postfix({'train_loss': f'{train_loss.item():.4f}', 'val_loss': f'{val_loss.item():.4f}'})

            # --- Early Stopping Check & Logging ---
            if val_loss < best_val_loss - min_delta:
                logging.info(f"Epoch {epoch+1}/{self.config['stage2_epochs']} - Val Loss improved to {val_loss.item():.4f}. Saving model.")
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), self.config['best_model_path_stage2'])
            else:
                epochs_no_improve += 1
                logging.info(f"Epoch {epoch+1}/{self.config['stage2_epochs']} - Val Loss did not improve. Patience: {epochs_no_improve}/{patience}.")

            if epochs_no_improve >= patience:
                logging.info(f"Early stopping triggered at epoch {epoch+1}. Best val_loss: {best_val_loss:.4f}")
                break
        
        logging.info(f"Loading best stage 2 model from {self.config['best_model_path_stage2']} with val_loss: {best_val_loss:.4f}")
        model.load_state_dict(torch.load(self.config['best_model_path_stage2']))
        
        logging.info("✅ Stage 2 Complete.")
        return model

    def analyze_and_visualize(self, stage1_df, stage2_model):
        logging.info("\n--- Final Analysis & Visualization ---")
        stage2_model.eval().cpu()
        param_names = list(self.config['param_bounds'].keys())
        fig, axes = plt.subplots(2, 3, figsize=(24, 14))
        fig.suptitle('Two-Stage Double Pendulum: Final Results', fontsize=22, fontweight='bold')
        axes = axes.flatten()

        age_range = torch.linspace(20, 90, 100).unsqueeze(-1)
        with torch.no_grad():
            pred_means_dict, _ = stage2_model(age_range)

        for i, name in enumerate(param_names):
            sns.scatterplot(data=stage1_df, x='age', y=name, ax=axes[i], alpha=0.7, label='Stage 1 Subjects')
            axes[i].plot(age_range.numpy(), pred_means_dict[name].numpy(), color='red', linewidth=3, label='Stage 2 Age Trend')
            axes[i].set_title(f'{name} vs. Age', fontweight='bold')
            axes[i].legend()
        
        corr = stage1_df[param_names].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='vlag', center=0, ax=axes[5])
        axes[5].set_title('Parameter Correlation', fontweight='bold')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('two_stage_double_pendulum_results.png', dpi=150)
        logging.info("✅ Final analysis plot saved to 'two_stage_double_pendulum_results.png'")
        plt.show()

    def run_pipeline(self):
        stage1_results = self.train_stage1()
        stage2_model = self.train_stage2(stage1_results)
        self.analyze_and_visualize(stage1_results, stage2_model)

def main():
    logging.info(f"🔥 Initializing Two-Stage Double Pendulum Trainer on {DEVICE}")
    config = get_config()
    trainer = TwoStageTrainer(config)
    trainer.run_pipeline()

if __name__ == "__main__":
    main()
