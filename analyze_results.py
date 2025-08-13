#!/usr/bin/env python3
"""
Post-Training Analysis and Visualization Script (Corrected)

This script loads the results from a completed two-stage PINN training
and generates all the final analysis, graphs, and age-comparison tests.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import logging
from typing import Dict, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# REQUIRED MODEL & CONFIG DEFINITIONS
# =============================================================================
# We need these definitions to correctly load and use the saved model.

def get_config() -> Dict:
    """
    Re-creates the necessary parts of the training configuration.
    """
    config = {
        'param_bounds': {
            'K': (500.0, 3000.0),
            'B': (20.0, 150.0),
            'tau': (0.05, 0.4)
        }
    }
    return config

class CompactAgeParameterModel(nn.Module):
    """Lightweight age parameter model."""
    def __init__(self, param_bounds: Dict):
        super().__init__()
        self.param_bounds = param_bounds
        self.age_net = nn.Sequential(
            nn.Linear(1, 64), nn.ELU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ELU(), nn.Linear(32, 6)
        )

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
# ANALYSIS AND VISUALIZATION FUNCTION
# =============================================================================

def analyze_and_visualize(age_model: CompactAgeParameterModel, subject_parameters: Dict):
    """
    Creates analysis plots and runs the age comparison test from loaded results.
    """
    logger.info("="*60)
    logger.info("MODEL ANALYSIS & VISUALIZATION")
    logger.info("="*60)

    age_model.eval()
    device = next(age_model.parameters()).device

    # Generate age range predictions for the trend line
    ages_test = torch.linspace(20, 90, 100).unsqueeze(-1).to(device)
    with torch.no_grad():
        pred_means, pred_stds = age_model.predict_parameters(ages_test)
        pred_means = pred_means.cpu().numpy()
        pred_stds = pred_stds.cpu().numpy()

    df = pd.DataFrame(subject_parameters.values())

    try:
        plt.style.use('seaborn-v0_8-darkgrid')
        logger.info("Using plot style: 'seaborn-v0_8-darkgrid'")
    except OSError:
        logger.warning("Could not apply 'seaborn-v0_8-darkgrid', falling back to 'ggplot'.")
        plt.style.use('ggplot')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharex=True)
    param_map = {'K': 'Stiffness (K)', 'B': 'Damping (B)', 'tau': 'Neural Delay (τ)'}
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, 3))

    for i, (p_key, p_name) in enumerate(param_map.items()):
        ax = axes[i]
        ax.scatter(df['age'], df[p_key], alpha=0.7, s=40, color=colors[i], label='Learned Subjects', zorder=3, edgecolors='w', linewidth=0.5)
        ax.plot(ages_test.cpu().numpy(), pred_means[:, i], 'r-', linewidth=3, label='Age Trend', zorder=2)
        ax.fill_between(ages_test.cpu().numpy().flatten(),
                        (pred_means[:, i] - 2 * pred_stds[:, i]),
                        (pred_means[:, i] + 2 * pred_stds[:, i]),
                        alpha=0.2, color='red', label='95% Confidence', zorder=1)
        ax.set_title(f'{p_name} vs. Age', fontsize=14, fontweight='bold')
        ax.set_ylabel(p_name, fontsize=12)
        ax.set_xlabel('Age (years)', fontsize=12)
        ax.legend(fontsize=10)

    plt.suptitle('Learned Biomechanical Parameters vs. Age', fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_filename = 'final_parameter_age_relationships.png'
    plt.savefig(output_filename, dpi=150)
    logger.info(f"✅ Plot saved successfully as '{output_filename}'")
    plt.show()

    # --- Age Comparison Test ---
    logger.info("\n=== AGE COMPARISON TEST ===")
    test_ages = [(30, 60), (40, 70), (60, 80)]

    for age1, age2 in test_ages:
        age1_tensor = torch.tensor([[age1]], dtype=torch.float32, device=device)
        age2_tensor = torch.tensor([[age2]], dtype=torch.float32, device=device)

        with torch.no_grad():
            params1, _ = age_model.predict_parameters(age1_tensor)
            params2, _ = age_model.predict_parameters(age2_tensor)
            params1 = params1.cpu().numpy().squeeze()
            params2 = params2.cpu().numpy().squeeze()
            diff = params2 - params1

        logger.info(f"\nComparing Age {age1} vs {age2}:")
        logger.info(f"  K: {params1[0]:.1f} → {params2[0]:.1f} (Δ={diff[0]:+.1f})")
        logger.info(f"  B: {params1[1]:.1f} → {params2[1]:.1f} (Δ={diff[1]:+.1f})")
        logger.info(f"  τ: {params1[2]:.3f} → {params2[2]:.3f} (Δ={diff[2]:+.3f})")

    logger.info("\n✅ Age comparison functionality working!")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to load results and run analysis."""
    logger.info("🔥 Starting Post-Training Analysis Script 🔥")

    model_path = Path('best_stage2_model.pth')
    params_path = Path('subject_parameters.json')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not model_path.exists() or not params_path.exists():
        logger.error(f"❌ Error: Missing required files!")
        logger.error(f"Make sure '{model_path}' and '{params_path}' are in the current directory.")
        return

    logger.info(f"Loading subject parameters from '{params_path}'...")
    with open(params_path, 'r') as f:
        subject_parameters = json.load(f)
    logger.info(f"Loaded parameters for {len(subject_parameters)} subjects.")

    # === FIX: Re-create config and load the state_dict directly ===
    # The 'best_stage2_model.pth' file only contains the model weights (state_dict),
    # not the config dictionary. We re-create the config and load the weights directly.
    config = get_config()
    logger.info(f"Loading trained age model state_dict from '{model_path}'...")
    age_model_state_dict = torch.load(model_path, map_location=device)

    # Initialize the model architecture and load the trained weights
    age_model = CompactAgeParameterModel(config['param_bounds']).to(device)
    age_model.load_state_dict(age_model_state_dict)
    logger.info("Model weights loaded successfully.")

    analyze_and_visualize(age_model, subject_parameters)

if __name__ == "__main__":
    main()
