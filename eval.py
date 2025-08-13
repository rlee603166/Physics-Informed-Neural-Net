#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for Balance Models

This script evaluates the performance of two different modeling approaches:
1.  A data-driven Hybrid Embedding Model ('hybrid')
2.  A Physics-Informed Double Pendulum Model ('pinn')

Run from the command line, specifying the model to evaluate:
  - python evaluate_models.py --model hybrid
  - python evaluate_models.py --model pinn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import argparse

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# REQUIRED CLASS DEFINITIONS (for both models)
# ============================================================================

# --- Classes for Hybrid Model ---
class TrajectoryEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, embedding_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, embedding_dim)
    def forward(self, trajectory):
        _, (hidden, _) = self.lstm(trajectory)
        return F.normalize(self.fc(hidden[-1]), p=2, dim=1)

class AgeEncoder(nn.Module):
    def __init__(self, embedding_dim=32):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, embedding_dim))
    def forward(self, age):
        return F.normalize(self.fc((age - 50.0) / 30.0), p=2, dim=1)

class TrajectoryDataset(Dataset):
    def __init__(self, seq_len=512):
        self.data_folder = Path('processed_data')
        self.seq_len = seq_len
        self.age_lookup = pd.read_csv('user_ages.csv', index_col='user_id')['age'].to_dict()
        self.trajectories = self._load_trajectories()
    def _load_trajectories(self):
        data = []
        for h5_file in self.data_folder.glob("*.h5"):
            with h5py.File(h5_file, 'r') as f:
                for sid_full in f.keys():
                    sid = sid_full.replace('subject_', '')
                    if sid in self.age_lookup:
                        age = self.age_lookup[sid]
                        for trial in f[sid_full].keys():
                            traj = np.stack([f[sid_full][trial]['cop_x'][:], f[sid_full][trial]['cop_y'][:]], axis=1)
                            if len(traj) >= self.seq_len: data.append({'trajectory': traj, 'age': age})
        return data
    def __len__(self): return len(self.trajectories)
    def __getitem__(self, idx):
        item = self.trajectories[idx]
        start_idx = np.random.randint(0, len(item['trajectory']) - self.seq_len + 1)
        seq = item['trajectory'][start_idx:start_idx + self.seq_len]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor([item['age']], dtype=torch.float32)


# --- Classes for PINN Model ---
class DoublePendulumPINN(nn.Module):
    def __init__(self, n_subjects, bounds):
        super().__init__()
        self.n_subjects = n_subjects
        self.bounds = bounds
        self.param_net = nn.Sequential(nn.Linear(n_subjects, 128), nn.ReLU(), nn.Linear(128, 5))

    def get_subject_params(self, subject_idx_tensor):
        with torch.no_grad():
            subject_onehot = F.one_hot(subject_idx_tensor, num_classes=self.n_subjects).float()
            params_norm = torch.sigmoid(self.param_net(subject_onehot))
            params = {}
            param_names = ['K_ankle', 'B_ankle', 'K_hip', 'B_hip', 'tau']
            for i, name in enumerate(param_names):
                min_b, max_b = self.bounds[name]
                params[name] = min_b + (max_b - min_b) * params_norm[:, i]
            return params

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_hybrid_model():
    """Loads and evaluates the hybrid data-driven model."""
    logging.info("--- Evaluating Hybrid Data-Driven Model ---")
    
    # --- Load Models ---
    try:
        traj_encoder = TrajectoryEncoder().to(DEVICE)
        traj_encoder.load_state_dict(torch.load('trajectory_encoder.pth', map_location=DEVICE))
        age_encoder = AgeEncoder().to(DEVICE)
        age_encoder.load_state_dict(torch.load('age_encoder.pth', map_location=DEVICE))
        traj_encoder.eval()
        age_encoder.eval()
        logging.info("✅ Models loaded successfully.")
    except FileNotFoundError:
        logging.error("❌ Model files not found! Make sure to run the hybrid training script first.")
        return

    # --- Generate Embeddings ---
    age_range = torch.arange(20, 91, 1).float().view(-1, 1).to(DEVICE)
    with torch.no_grad():
        age_manifold_embeddings = age_encoder(age_range)

    dataset = TrajectoryDataset()
    loader = DataLoader(dataset, batch_size=200, shuffle=True)
    sample_trajectories, sample_ages = next(iter(loader))
    with torch.no_grad():
        sample_traj_embeddings = traj_encoder(sample_trajectories.to(DEVICE))

    # --- t-SNE Visualization ---
    logging.info("Running t-SNE for visualization...")
    all_embeddings = torch.cat([age_manifold_embeddings, sample_traj_embeddings], dim=0).cpu().numpy()
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings)

    age_manifold_2d = embeddings_2d[:len(age_range)]
    sample_traj_2d = embeddings_2d[len(age_range):]

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 10))
    plt.plot(age_manifold_2d[:, 0], age_manifold_2d[:, 1], color='gray', linestyle='--', label='Age Manifold')
    scatter = plt.scatter(sample_traj_2d[:, 0], sample_traj_2d[:, 1], c=sample_ages.numpy().flatten(), cmap='viridis', s=80, alpha=0.8, edgecolors='w')
    plt.title('t-SNE Visualization of Balance & Age Embedding Space', fontsize=18, fontweight='bold')
    cbar = plt.colorbar(scatter); cbar.set_label('Subject Age', fontsize=12)
    plt.legend(fontsize=12)
    plt.savefig('hybrid_model_evaluation.png', dpi=150)
    logging.info("✅ Visualization saved to 'hybrid_model_evaluation.png'")
    plt.show()

    # --- Balance Age Prediction Demo ---
    def predict_balance_age(traj_embedding, age_manifold, age_range_np):
        similarities = F.cosine_similarity(traj_embedding, age_manifold)
        return age_range_np[torch.argmax(similarities)]

    print("\n" + "="*50 + "\nBALANCE AGE PREDICTION DEMO\n" + "="*50)
    for i in range(5):
        true_age = sample_ages[i].item()
        pred_age = predict_balance_age(sample_traj_embeddings[i].unsqueeze(0), age_manifold_embeddings, age_range.cpu().numpy())
        print(f"Sample {i+1}: True Age = {true_age:.0f} -> Predicted Balance Age = {pred_age[0]:.0f}")


def evaluate_pinn_model():
    """Loads and evaluates the Double Pendulum PINN model."""
    logging.info("--- Evaluating Double Pendulum PINN Model ---")
    
    param_bounds = {
        'K_ankle': (500., 3000.), 'B_ankle': (10., 150.),
        'K_hip': (500., 3000.), 'B_hip': (10., 150.),
        'tau': (0.05, 0.4)
    }

    # --- Load Model ---
    try:
        age_df = pd.read_csv('user_ages.csv')
        subject_map = {sid: i for i, sid in enumerate(age_df['user_id'].unique())}
        n_subjects = len(subject_map)
        
        model = DoublePendulumPINN(n_subjects, param_bounds).to(DEVICE)
        model.load_state_dict(torch.load('double_pendulum_pinn.pth', map_location=DEVICE))
        model.eval()
        logging.info("✅ Model loaded successfully.")
    except (FileNotFoundError, RuntimeError) as e:
        logging.error(f"❌ Model file not found or architecture mismatch! Error: {e}")
        logging.error("Make sure to run the PINN training script first and that n_subjects matches.")
        return

    # --- Extract Parameters ---
    logging.info("Extracting parameters for all subjects...")
    subject_indices = torch.arange(n_subjects).to(DEVICE)
    params_dict = model.get_subject_params(subject_indices)
    df = pd.DataFrame({name: tensor.cpu().numpy() for name, tensor in params_dict.items()})
    df['age'] = df.index.map({v: k for k, v in subject_map.items()}).map(pd.read_csv('user_ages.csv').set_index('user_id')['age'])
    
    # --- Boundary Analysis Report ---
    print("\n" + "="*50 + "\nPARAMETER BOUNDARY ANALYSIS\n" + "="*50)
    param_names = list(param_bounds.keys())
    for name in param_names:
        min_b, max_b = param_bounds[name]
        thresh = 0.05 * (max_b - min_b) # 5% margin
        at_min = (df[name] <= min_b + thresh).sum()
        at_max = (df[name] >= max_b - thresh).sum()
        percent = 100 * (at_min + at_max) / len(df)
        status = "✅" if percent < 10 else "⚠️"
        print(f"{status} {name:<10}: {percent:>5.1f}% of subjects at boundaries ({at_min} at min, {at_max} at max)")

    # --- Correlation Matrix ---
    print("\n" + "="*50 + "\nPARAMETER CORRELATION MATRIX\n" + "="*50)
    corr = df[param_names].corr()
    print(corr.to_string(float_format="%.3f"))
    if corr.abs().unstack().sort_values(ascending=False).drop_duplicates()[0] > 0.9:
        print("⚠️ WARNING: High correlation detected between parameters!")

    # --- Plotting ---
    logging.info("Generating analysis plots...")
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle('Double Pendulum PINN: Full Parameter Evaluation', fontsize=22, fontweight='bold')
    axes = axes.flatten()

    for i, name in enumerate(param_names):
        # Distribution Plot
        sns.histplot(df, x=name, ax=axes[i], bins=30, kde=True)
        axes[i].set_title(f'{name} Distribution', fontweight='bold')
        # Age Correlation Plot
        sns.regplot(data=df, x='age', y=name, ax=axes[i+len(param_names)] if i < 3 else axes[5], scatter_kws={'alpha':0.5})
        axes[i+len(param_names) if i < 3 else 5].set_title(f'{name} vs. Age', fontweight='bold')
    
    # Remove the unused subplot if we have 5 params
    if len(param_names) == 5:
        axes[5].set_title('τ vs. Age', fontweight='bold')


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('pinn_model_evaluation.png', dpi=150)
    logging.info("✅ Full evaluation plot saved to 'pinn_model_evaluation.png'")
    plt.show()

# ============================================================================
# SCRIPT ENTRYPOINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained balance models.")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['hybrid', 'pinn'],
        help="The type of model to evaluate: 'hybrid' or 'pinn'."
    )
    args = parser.parse_args()

    if args.model == 'hybrid':
        evaluate_hybrid_model()
    elif args.model == 'pinn':
        evaluate_pinn_model()
