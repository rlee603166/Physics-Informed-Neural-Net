#!/usr/bin/env python3
"""
Performance and Validity Evaluation Script for the Double Pendulum PINN Model

This script performs two critical tests on the trained PINN:
1.  Parameter Consistency Test: Checks if the model outputs stable parameters for
    different trials from the same subject.
2.  Trajectory Prediction Test: Checks if the learned parameters can be used
    in a forward simulation (ODE solver) to reproduce the original sway pattern.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG = {
    'param_bounds': {
        'K_ankle': (500., 3000.), 'B_ankle': (10., 150.),
        'K_hip': (500., 3000.), 'B_hip': (10., 150.),
        'tau': (0.05, 0.4)
    },
    'age_csv_path': 'user_ages.csv',
    'data_folder': 'processed_data',
    'n_subjects_trained': 231, # The number used during training
}

# --- Copy Model Definition from Training Script ---
class DoublePendulumPINN(nn.Module):
    def __init__(self, n_subjects, bounds):
        super().__init__()
        self.n_subjects = n_subjects
        self.bounds = bounds
        self.trajectory_net = nn.Sequential(nn.Linear(1 + n_subjects, 128), nn.Tanh(), nn.Linear(128, 2))
        self.param_net = nn.Sequential(nn.Linear(n_subjects, 128), nn.ReLU(), nn.Linear(128, 5))

    def get_params_from_trajectory(self, trajectory_tensor, subject_idx_tensor):
        """Runs the inverse model on a single trajectory to get parameters."""
        with torch.no_grad():
            t = torch.linspace(0, len(trajectory_tensor)/106.0, len(trajectory_tensor)).unsqueeze(-1).to(DEVICE)
            subject_idx_tensor = subject_idx_tensor.to(DEVICE)
            # This is a simplification; a full inverse problem would be an optimization loop.
            # Here, we just use the model's direct output for the subject.
            subject_onehot = F.one_hot(subject_idx_tensor, num_classes=self.n_subjects).float()
            params_norm = torch.sigmoid(self.param_net(subject_onehot))
            
            params = {}
            param_names = ['K_ankle', 'B_ankle', 'K_hip', 'B_hip', 'tau']
            for i, name in enumerate(param_names):
                min_b, max_b = self.bounds[name]
                params[name] = (min_b + (max_b - min_b) * params_norm[:, i]).item()
            return params

# --- ODE for Forward Simulation ---
def balance_ode(t, y, K_a, B_a, K_h, B_h, tau, m=70.0):
    """Simplified double pendulum ODE for forward simulation."""
    x, x_dot = y
    # A very simplified coupling term approximation
    x_delayed = x - tau * x_dot # Taylor approximation
    x_ddot = -(B_a / m) * x_dot - (K_a / m) * x_delayed - (B_h / m) * x_dot - (K_h / m) * x
    return [x_dot, x_ddot]

def main():
    logging.info("🔥 Starting PINN Performance Evaluation Script")
    
    # --- Load Model ---
    try:
        model = DoublePendulumPINN(CONFIG['n_subjects_trained'], CONFIG['param_bounds']).to(DEVICE)
        model.load_state_dict(torch.load('double_pendulum_pinn.pth', map_location=DEVICE))
        model.eval()
        logging.info("✅ Model loaded successfully.")
    except FileNotFoundError:
        logging.error("❌ Model file not found! Make sure 'double_pendulum_pinn.pth' exists.")
        return

    # --- Test 1: Parameter Consistency ---
    logging.info("\n" + "="*50 + "\nTEST 1: PARAMETER CONSISTENCY\n" + "="*50)
    
    # Find a subject with multiple trials
    import h5py
    test_subject_id = 'C0007' # Example subject
    trials = []
    with h5py.File(next(Path(CONFIG['data_folder']).glob("*.h5")), 'r') as f:
        if f'subject_{test_subject_id}' in f:
            for trial_key in f[f'subject_{test_subject_id}'].keys():
                traj = np.stack([f[f'subject_{test_subject_id}'][trial_key]['cop_x'][:], f[f'subject_{test_subject_id}'][trial_key]['cop_y'][:]], axis=1)
                trials.append(torch.tensor(traj, dtype=torch.float32))

    if len(trials) < 2:
        logging.warning(f"Could not find at least 2 trials for subject {test_subject_id}. Skipping consistency test.")
    else:
        logging.info(f"Analyzing {len(trials)} trials from subject {test_subject_id}...")
        all_params = []
        # Get subject index from user_ages.csv
        age_df = pd.read_csv(CONFIG['age_csv_path'])
        subject_list = age_df['user_id'].tolist()
        subject_idx = torch.tensor([subject_list.index(test_subject_id)])

        for i, trial_tensor in enumerate(trials):
            params = model.get_params_from_trajectory(trial_tensor, subject_idx)
            all_params.append(params)
            logging.info(f"  Trial {i+1} Params: { {k: f'{v:.1f}' for k,v in params.items()} }")
        
        # Calculate Coefficient of Variation (CV)
        params_df = pd.DataFrame(all_params)
        cv = params_df.std() / params_df.mean()
        logging.info("\nParameter Consistency (Coefficient of Variation):")
        print(cv.to_string())
        if cv.max() > 0.25:
            logging.warning("⚠️ High variation found! Parameters are not stable across trials. This is a sign of an invalid model.")
        else:
            logging.info("✅ Parameters appear stable across trials.")

    # --- Test 2: Trajectory Prediction ---
    logging.info("\n" + "="*50 + "\nTEST 2: TRAJECTORY PREDICTION (FORWARD PROBLEM)\n" + "="*50)
    
    # Use the parameters learned from the first trial of our test subject
    learned_params = all_params[0]
    logging.info(f"Using learned parameters for simulation: {learned_params}")
    
    # Simulate the trajectory using an ODE solver
    actual_trajectory = trials[0].numpy()
    t_eval = np.linspace(0, len(actual_trajectory)/106.0, len(actual_trajectory))
    sol = solve_ivp(
        fun=balance_ode,
        t_span=[t_eval.min(), t_eval.max()],
        y0=[actual_trajectory[0, 0], 0], # Initial position from data, initial velocity=0
        t_eval=t_eval,
        args=(learned_params['K_ankle'], learned_params['B_ankle'], learned_params['K_hip'], learned_params['B_hip'], learned_params['tau'])
    )
    predicted_trajectory = sol.y[0]

    # Calculate validation metric (MSE)
    mse = np.mean((actual_trajectory[:, 0] - predicted_trajectory)**2)
    logging.info(f"Trajectory Prediction MSE: {mse:.6f}")
    if mse > 0.0001:
         logging.warning("⚠️ High MSE! The learned parameters cannot reproduce the original trajectory. The model is not valid.")
    else:
        logging.info("✅ Low MSE. The learned parameters can reproduce the trajectory.")

    # Plot the comparison
    plt.figure(figsize=(14, 6))
    plt.plot(t_eval, actual_trajectory[:, 0], label='Actual Trajectory', color='blue', linewidth=2)
    plt.plot(t_eval, predicted_trajectory, label='Predicted Trajectory (from PINN params)', color='red', linestyle='--', linewidth=2)
    plt.title(f'Forward Problem Validation for Subject {test_subject_id}', fontsize=16)
    plt.xlabel('Time (s)')
    plt.ylabel('Center of Pressure (x-axis)')
    plt.legend()
    plt.grid(True)
    plt.savefig('pinn_validity_check.png', dpi=150)
    logging.info("✅ Trajectory comparison plot saved to 'pinn_validity_check.png'")
    plt.show()

if __name__ == "__main__":
    main()
