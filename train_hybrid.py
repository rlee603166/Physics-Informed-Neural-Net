#!/usr/bin/env python3
"""
Hybrid Data-Driven Model for Balance Age Comparison

This script trains a two-tower embedding model.
1. An LSTM-based "Trajectory Tower" encodes a sway pattern into a latent vector.
2. An MLP-based "Age Tower" encodes a subject's age into a latent vector.

The model is trained to minimize the distance between these two vectors, learning
a representation of "balance age" without relying on a rigid physics model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
from collections import defaultdict
import h5py

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG = {
    'data_folder': 'processed_data',
    'age_csv_path': 'user_ages.csv',
    'embedding_dim': 32,
    'lstm_hidden_dim': 64,
    'sequence_length': 512, # 512 frames = ~4.8 seconds
    'batch_size': 64,
    'epochs': 50,
    'lr': 5e-4,  # Reduced from 1e-3 to prevent mode collapse
    'weight_decay': 1e-4,  # Increased regularization
    'patience': 15,  # Increased patience for lower learning rate
    'min_delta': 1e-5,  # Smaller threshold for finer improvements
}

# --- Model Architecture ---
class TrajectoryEncoder(nn.Module):
    """Encodes a time-series trajectory into a fixed-size embedding."""
    def __init__(self, input_dim=2, hidden_dim=64, embedding_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, embedding_dim)
        
    def forward(self, trajectory):
        # trajectory shape: (batch, seq_len, input_dim)
        _, (hidden, _) = self.lstm(trajectory)
        # hidden shape: (num_layers, batch, hidden_dim), we take the last layer's hidden state
        features = hidden[-1]
        embedding = self.fc(features)
        # Reduced normalization - don't force unit vectors, allow more diversity
        return embedding / (torch.norm(embedding, dim=1, keepdim=True) + 1e-6)

class AgeEncoder(nn.Module):
    """Encodes an age into the same embedding space as the trajectory."""
    def __init__(self, embedding_dim=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )
        
    def forward(self, age):
        # Normalize age for better network performance
        age_normalized = (age - 50.0) / 30.0
        embedding = self.fc(age_normalized)
        # Reduced normalization - don't force unit vectors, allow more diversity
        return embedding / (torch.norm(embedding, dim=1, keepdim=True) + 1e-6)

# --- Dataset for Sequences ---
class TrajectoryDataset(Dataset):
    """Loads entire trajectories for sequence modeling."""
    def __init__(self, config):
        self.data_folder = Path(config['data_folder'])
        self.seq_len = config['sequence_length']
        self.age_lookup = pd.read_csv(config['age_csv_path'], index_col='user_id')['age'].to_dict()
        
        # Data normalization parameters (from AMD trainer)
        # COP data is in absolute force plate coordinates - need to center around equilibrium
        self.x_mean = 349.0  # Center X position (mm)
        self.x_scale = 10.0  # Scale for ~[-1,1] range (6mm variation / 0.6)
        self.y_mean = 925.0  # Center Y position (mm) 
        self.y_scale = 20.0  # Scale for ~[-1,1] range (37mm variation / 1.8)
        
        self.trajectories = self._load_trajectories()
        logging.info(f"Loaded {len(self.trajectories)} trajectories.")
        logging.info(f"Data normalization: X=({self.x_mean}±{self.x_scale}), Y=({self.y_mean}±{self.y_scale})")

    def _load_trajectories(self):
        data = []
        for h5_file in self.data_folder.glob("*.h5"):
            with h5py.File(h5_file, 'r') as f:
                for subject_id_full in f.keys():
                    subject_id = subject_id_full.replace('subject_', '')
                    if subject_id in self.age_lookup:
                        age = self.age_lookup[subject_id]
                        for trial_key in f[subject_id_full].keys():
                            cop_x = f[subject_id_full][trial_key]['cop_x'][:]
                            cop_y = f[subject_id_full][trial_key]['cop_y'][:]
                            
                            # Normalize COP data to center around equilibrium
                            cop_x_norm = (cop_x - self.x_mean) / self.x_scale
                            cop_y_norm = (cop_y - self.y_mean) / self.y_scale
                            trajectory = np.stack([cop_x_norm, cop_y_norm], axis=1)
                            # Add trajectory if it's long enough
                            if len(trajectory) >= self.seq_len:
                                data.append({'trajectory': trajectory, 'age': age})
        return data

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        item = self.trajectories[idx]
        trajectory = item['trajectory']
        age = item['age']
        
        # Randomly sample a sequence from the full trajectory
        start_idx = np.random.randint(0, len(trajectory) - self.seq_len + 1)
        seq = trajectory[start_idx : start_idx + self.seq_len]
        
        return torch.tensor(seq, dtype=torch.float32), torch.tensor([age], dtype=torch.float32)
    
    def denormalize_positions(self, xy_normalized: torch.Tensor) -> torch.Tensor:
        """Convert normalized positions back to mm coordinates for analysis/visualization."""
        x_mm = xy_normalized[..., 0] * self.x_scale + self.x_mean
        y_mm = xy_normalized[..., 1] * self.y_scale + self.y_mean
        return torch.stack([x_mm, y_mm], dim=-1)

# --- Main Training Logic ---
def main():
    logging.info(f"Starting Hybrid Model training on {DEVICE}")
    
    # Data
    dataset = TrajectoryDataset(CONFIG)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)

    # Models
    trajectory_encoder = TrajectoryEncoder(embedding_dim=CONFIG['embedding_dim']).to(DEVICE)
    age_encoder = AgeEncoder(embedding_dim=CONFIG['embedding_dim']).to(DEVICE)

    # Optimizer and Loss
    params = list(trajectory_encoder.parameters()) + list(age_encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Replace Cosine Embedding Loss with MSE Loss for better diversity
    # MSE loss between trajectory and age embeddings encourages similarity without forcing identical vectors
    loss_fn = nn.MSELoss()

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_trajectory_state = None
    best_age_state = None

    # Training Loop
    for epoch in range(CONFIG['epochs']):
        trajectory_encoder.train()
        age_encoder.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        for trajectory, age in pbar:
            trajectory, age = trajectory.to(DEVICE), age.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Get embeddings from both towers
            traj_embedding = trajectory_encoder(trajectory)
            age_embedding = age_encoder(age)
            
            # MSE loss between embeddings - encourages similarity without forcing identical vectors
            similarity_loss = loss_fn(traj_embedding, age_embedding)
            
            # Add diversity regularization to prevent mode collapse
            # Penalize embeddings that are too similar within the batch
            traj_mean = traj_embedding.mean(dim=0)
            age_mean = age_embedding.mean(dim=0)
            diversity_loss = 0.1 * (torch.norm(traj_embedding - traj_mean, dim=1).mean() + 
                                   torch.norm(age_embedding - age_mean, dim=1).mean())
            
            # Invert diversity loss - we want HIGH diversity, so penalize LOW diversity
            diversity_penalty = 0.01 / (diversity_loss + 1e-6)
            
            loss = similarity_loss + diversity_penalty
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        trajectory_encoder.eval()
        age_encoder.eval()
        val_loss = 0
        with torch.no_grad():
            for trajectory, age in val_loader:
                trajectory, age = trajectory.to(DEVICE), age.to(DEVICE)
                traj_embedding = trajectory_encoder(trajectory)
                age_embedding = age_encoder(age)
                
                similarity_loss = loss_fn(traj_embedding, age_embedding)
                
                # Same diversity calculation for validation
                traj_mean = traj_embedding.mean(dim=0)
                age_mean = age_embedding.mean(dim=0)
                diversity_loss = 0.1 * (torch.norm(traj_embedding - traj_mean, dim=1).mean() + 
                                       torch.norm(age_embedding - age_mean, dim=1).mean())
                diversity_penalty = 0.01 / (diversity_loss + 1e-6)
                
                loss = similarity_loss + diversity_penalty
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        logging.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save models
    torch.save(trajectory_encoder.state_dict(), 'trajectory_encoder.pth')
    torch.save(age_encoder.state_dict(), 'age_encoder.pth')
    logging.info("✅ Models saved successfully!")

if __name__ == "__main__":
    main()
