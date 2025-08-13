#!/usr/bin/env python3
"""
Analysis Script for the Double Pendulum PINN Model

This script loads a trained DoublePendulumPINN and performs a detailed
analysis of its learned parameters to check for:
1. Boundary Collapse: How many subjects are stuck at min/max parameter values.
2. Parameter Diversity: The distribution and variance of each parameter.
3. Age Correlation: Whether any parameters show a trend with age.
4. Parameter Redundancy: Correlation between the learned parameters themselves.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import seaborn as sns
import matplotlib.pyplot as plt

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG = { # We need the bounds from the config to interpret the parameters
    'param_bounds': {
        'K_ankle': (500., 3000.), 'B_ankle': (10., 150.),
        'K_hip': (500., 3000.), 'B_hip': (10., 150.),
        'tau': (0.05, 0.4)
    },
    'age_csv_path': 'user_ages.csv',
}

# --- Copy Model Definition from Training Script ---
class DoublePendulumPINN(nn.Module):
    def __init__(self, n_subjects, bounds):
        super().__init__()
        self.n_subjects = n_subjects
        self.bounds = bounds
        self.trajectory_net = nn.Sequential(nn.Linear(1 + n_subjects, 128), nn.Tanh(), nn.Linear(128, 2))
        self.param_net = nn.Sequential(nn.Linear(n_subjects, 128), nn.ReLU(), nn.Linear(128, 5))

    def get_subject_params(self, subject_idx_tensor):
        """Gets the physical parameters for a given subject index tensor."""
        with torch.no_grad():
            subject_onehot = F.one_hot(subject_idx_tensor, num_classes=self.n_subjects).float()
            params_norm = torch.sigmoid(self.param_net(subject_onehot))
            
            params = {}
            param_names = ['K_ankle', 'B_ankle', 'K_hip', 'B_hip', 'tau']
            for i, name in enumerate(param_names):
                min_b, max_b = self.bounds[name]
                params[name] = min_b + (max_b - min_b) * params_norm[:, i]
            return params

# --- Main Analysis Logic ---
def main():
    logging.info("🔥 Starting Double Pendulum PINN Analysis Script")
    
    # --- Load Model ---
    try:
        # First try to load the model to determine the number of subjects it was trained with
        checkpoint = torch.load('double_pendulum_pinn.pth', map_location=DEVICE)
        
        # Infer number of subjects from the saved model architecture
        # trajectory_net expects: 1 (time) + n_subjects
        # param_net expects: n_subjects
        trajectory_input_size = checkpoint['trajectory_net.0.weight'].shape[1]
        param_input_size = checkpoint['param_net.0.weight'].shape[1]
        
        n_subjects_trained = trajectory_input_size - 1  # subtract 1 for time dimension
        
        logging.info(f"Model was trained with {n_subjects_trained} subjects")
        
        # Get current subject info from age CSV
        age_df = pd.read_csv(CONFIG['age_csv_path'])
        n_subjects_current = age_df['user_id'].nunique()
        logging.info(f"Current CSV has {n_subjects_current} subjects")
        
        if n_subjects_trained != n_subjects_current:
            logging.warning(f"⚠️ Subject count mismatch! Using trained count ({n_subjects_trained}) for analysis")
        
        model = DoublePendulumPINN(n_subjects_trained, CONFIG['param_bounds']).to(DEVICE)
        model.load_state_dict(checkpoint)
        model.eval()
        logging.info("✅ Model loaded successfully.")
    except FileNotFoundError:
        logging.error("❌ Model file not found! Make sure 'double_pendulum_pinn.pth' exists.")
        return

    # --- Extract Parameters for All Subjects ---
    logging.info("Extracting parameters for all subjects...")
    subject_indices = torch.arange(n_subjects_trained).to(DEVICE)
    params_dict = model.get_subject_params(subject_indices)
    
    # Convert to a Pandas DataFrame for analysis
    df = pd.DataFrame({name: tensor.cpu().numpy() for name, tensor in params_dict.items()})
    df['age'] = age_df['age']
    logging.info(f"Extracted parameters for {len(df)} subjects.")

    # --- Perform Boundary Analysis ---
    param_names = list(CONFIG['param_bounds'].keys())
    boundary_thresholds = {'K_ankle': 100, 'B_ankle': 5, 'K_hip': 100, 'B_hip': 5, 'tau': 0.02}
    total_at_boundaries = 0
    
    print("\n" + "="*50 + "\nSUBJECT PARAMETER BOUNDARY ANALYSIS\n" + "="*50)
    for name in param_names:
        min_b, max_b = CONFIG['param_bounds'][name]
        thresh = boundary_thresholds[name]
        at_min = (df[name] <= min_b + thresh).sum()
        at_max = (df[name] >= max_b - thresh).sum()
        total = at_min + at_max
        percent = 100 * total / len(df)
        print(f"Parameter: {name} | At boundaries: {total} subjects ({percent:.1f}%)")
        total_at_boundaries += total
    
    # --- Plotting ---
    logging.info("Generating analysis plots...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Double Pendulum PINN: Parameter Analysis', fontsize=20, fontweight='bold')
    axes = axes.flatten()

    # Histograms
    for i, name in enumerate(param_names):
        sns.histplot(df, x=name, ax=axes[i], bins=25, kde=True)
        axes[i].set_title(f'{name} Distribution', fontweight='bold')
    # Remove empty subplot
    axes[5].set_visible(False)

    # Correlation Matrix Heatmap
    corr = df[param_names].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='vlag', center=0, ax=axes[5])
    axes[5].set_title('Parameter Correlation', fontweight='bold')
    axes[5].set_visible(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('double_pendulum_analysis.png', dpi=150)
    logging.info("✅ Analysis plots saved to 'double_pendulum_analysis.png'")
    plt.show()

if __name__ == "__main__":
    main()
