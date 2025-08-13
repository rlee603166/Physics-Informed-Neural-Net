#!/usr/bin/env python3
"""
Analysis Script for the Hybrid Balance Age Model

This script visualizes the learned embedding space from the two-tower model.
It performs the following steps:
1. Loads the trained Trajectory and Age encoders.
2. Generates an "age manifold" by mapping ages 20-90 to the embedding space.
3. Takes a sample of real trajectories and maps them to the same space.
4. Uses t-SNE to visualize the high-dimensional embeddings in 2D.
5. Demonstrates how to calculate a "Balance Age" for a given trajectory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG = {
    'data_folder': 'processed_data',
    'age_csv_path': 'user_ages.csv',
    'embedding_dim': 32,
    'lstm_hidden_dim': 64,
    'sequence_length': 512,
    'batch_size': 64,
}

# --- Copy Model and Dataset Definitions from Training Script ---
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
    def __init__(self, config):
        self.data_folder = Path(config['data_folder'])
        self.seq_len = config['sequence_length']
        self.age_lookup = pd.read_csv(config['age_csv_path'], index_col='user_id')['age'].to_dict()
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

# --- Analysis Functions ---
def predict_balance_age(traj_embedding, age_manifold_embeddings, age_range):
    """Finds the closest age on the manifold to a given trajectory embedding."""
    # Using cosine similarity, since we normalized embeddings
    similarities = F.cosine_similarity(traj_embedding, age_manifold_embeddings)
    best_match_idx = torch.argmax(similarities)
    return age_range[best_match_idx]

def main():
    logging.info("🔥 Starting Hybrid Model Analysis Script")
    
    # --- Load Models ---
    try:
        trajectory_encoder = TrajectoryEncoder(embedding_dim=CONFIG['embedding_dim']).to(DEVICE)
        trajectory_encoder.load_state_dict(torch.load('trajectory_encoder.pth', map_location=DEVICE))
        age_encoder = AgeEncoder(embedding_dim=CONFIG['embedding_dim']).to(DEVICE)
        age_encoder.load_state_dict(torch.load('age_encoder.pth', map_location=DEVICE))
        trajectory_encoder.eval()
        age_encoder.eval()
        logging.info("✅ Models loaded successfully.")
    except FileNotFoundError:
        logging.error("❌ Model files not found! Make sure 'trajectory_encoder.pth' and 'age_encoder.pth' exist.")
        return

    # --- Generate Embeddings for Visualization ---
    # 1. Age Manifold
    age_range = torch.arange(20, 91, 1).float().view(-1, 1).to(DEVICE)
    with torch.no_grad():
        age_manifold_embeddings = age_encoder(age_range)

    # 2. Trajectory Samples
    dataset = TrajectoryDataset(CONFIG)
    loader = DataLoader(dataset, batch_size=200, shuffle=True) # Get 200 random samples
    sample_trajectories, sample_ages = next(iter(loader))
    with torch.no_grad():
        sample_traj_embeddings = trajectory_encoder(sample_trajectories.to(DEVICE))

    # --- Use t-SNE to Visualize in 2D ---
    logging.info("Running t-SNE for visualization... (this may take a moment)")
    all_embeddings = torch.cat([age_manifold_embeddings, sample_traj_embeddings], dim=0).cpu().numpy()
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings)

    age_manifold_2d = embeddings_2d[:len(age_range)]
    sample_traj_2d = embeddings_2d[len(age_range):]

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 10))
    # Plot the smooth age manifold
    plt.plot(age_manifold_2d[:, 0], age_manifold_2d[:, 1], color='gray', linestyle='--', label='Age Manifold')
    # Plot the individual trajectory samples, colored by their true age
    scatter = plt.scatter(sample_traj_2d[:, 0], sample_traj_2d[:, 1], c=sample_ages.numpy().flatten(), cmap='viridis', s=80, alpha=0.8, edgecolors='w')
    plt.title('t-SNE Visualization of Balance & Age Embedding Space', fontsize=18, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Subject Age', fontsize=12)
    plt.legend(fontsize=12)
    plt.savefig('hybrid_model_embedding_space.png', dpi=150)
    logging.info("✅ Visualization saved to 'hybrid_model_embedding_space.png'")
    plt.show()

    # --- Demonstrate Balance Age Prediction ---
    logging.info("\n--- Balance Age Prediction Demo ---")
    for i in range(5):
        true_age = sample_ages[i].item()
        traj_embedding = sample_traj_embeddings[i].unsqueeze(0)
        predicted_age = predict_balance_age(traj_embedding, age_manifold_embeddings, age_range.cpu().numpy())
        logging.info(f"Sample {i+1}: True Age = {true_age:.0f}, Predicted Balance Age = {predicted_age[0]:.0f}")

if __name__ == "__main__":
    main()
