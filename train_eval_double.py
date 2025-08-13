#!/usr/bin/env python3
"""
Final Hybrid Data-Driven Model for Balance Age Comparison

This script incorporates fixes to prevent model collapse and overfitting:
1. A lower learning rate for more stable training.
2. A 1D CNN-based trajectory encoder, which is often more stable than an LSTM.
3. Strict subject-level data splitting (train/validation/test).
4. Early stopping based on validation loss to prevent overfitting.
5. A final, unbiased performance report on the held-out test set.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG = {
    'data_folder': 'processed_data',
    'age_csv_path': 'user_ages.csv',
    'embedding_dim': 32,
    'sequence_length': 512, # ~4.8 seconds of data
    'batch_size': 64,
    'epochs': 150, # Max epochs; early stopping will likely finish sooner

    # --- FIX: Key changes to prevent model collapse and instability ---
    'lr': 1e-4,            # Lowered learning rate for more stable optimization
    'weight_decay': 1e-4,  # L2 Regularization
    
    # --- Anti-Overfitting ---
    'early_stopping_patience': 15, # Increased patience for slower learning
    'test_split_size': 0.15,
    'validation_split_size': 0.15,
}

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class TrajectoryEncoder(nn.Module):
    """
    Encodes a trajectory using a 1D CNN, which can be more stable than an LSTM.
    It's effective at finding local patterns in time-series data.
    """
    def __init__(self, input_dim=2, embedding_dim=32, dropout=0.4):
        super().__init__()
        self.convnet = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # Global Average Pooling to get a fixed-size output
        )
        self.fc = nn.Sequential(
            nn.Linear(128, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, trajectory):
        # Input shape for Conv1d needs to be (batch, channels, seq_len)
        # Current shape is (batch, seq_len, channels), so we permute
        x = trajectory.permute(0, 2, 1) 
        features = self.convnet(x).squeeze(-1) # Squeeze the last dimension from pooling
        embedding = self.fc(features)
        return F.normalize(embedding, p=2, dim=1)

class AgeEncoder(nn.Module):
    """Encodes an age into the same embedding space as the trajectory."""
    def __init__(self, embedding_dim=32, dropout=0.4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, embedding_dim)
        )
        
    def forward(self, age):
        age_normalized = (age - 50.0) / 30.0
        embedding = self.fc(age_normalized)
        return F.normalize(embedding, p=2, dim=1)

# ============================================================================
# DATASET
# ============================================================================

class TrajectoryDataset(Dataset):
    """Loads entire trajectories and is aware of subject IDs for proper splitting."""
    def __init__(self, config):
        self.data_folder = Path(config['data_folder'])
        self.seq_len = config['sequence_length']
        self.age_lookup = pd.read_csv(config['age_csv_path'], index_col='user_id')['age'].to_dict()
        self.trajectories, self.subject_ids = self._load_trajectories()
        logging.info(f"Loaded {len(self.trajectories)} trajectories from {len(self.subject_ids)} unique subjects.")

    def _load_trajectories(self):
        data, subject_ids = [], set()
        for h5_file in self.data_folder.glob("*.h5"):
            with h5py.File(h5_file, 'r') as f:
                for subject_id_full in f.keys():
                    subject_id = subject_id_full.replace('subject_', '')
                    if subject_id in self.age_lookup:
                        subject_ids.add(subject_id)
                        age = self.age_lookup[subject_id]
                        for trial_key in f[subject_id_full].keys():
                            if 'cop_x' in f[subject_id_full][trial_key]:
                                cop_x = f[subject_id_full][trial_key]['cop_x'][:]
                                cop_y = f[subject_id_full][trial_key]['cop_y'][:]
                                trajectory = np.stack([cop_x, cop_y], axis=1)
                                if len(trajectory) >= self.seq_len:
                                    data.append({'trajectory': trajectory, 'age': age, 'subject_id': subject_id})
        return data, sorted(list(subject_ids))

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        item = self.trajectories[idx]
        trajectory = item['trajectory']
        age = item['age']
        
        # Randomly sample a sequence from the full trajectory (a form of data augmentation)
        start_idx = np.random.randint(0, len(trajectory) - self.seq_len + 1)
        seq = trajectory[start_idx : start_idx + self.seq_len]
        
        return torch.tensor(seq, dtype=torch.float32), torch.tensor([age], dtype=torch.float32)

# ============================================================================
# TRAINER CLASS
# ============================================================================

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.trajectory_encoder = TrajectoryEncoder(embedding_dim=config['embedding_dim']).to(DEVICE)
        self.age_encoder = AgeEncoder(embedding_dim=config['embedding_dim']).to(DEVICE)
        
        params = list(self.trajectory_encoder.parameters()) + list(self.age_encoder.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=config['lr'], weight_decay=config['weight_decay'])
        self.loss_fn = nn.CosineEmbeddingLoss()
        
        self._prepare_data()

    def _prepare_data(self):
        dataset = TrajectoryDataset(self.config)
        
        # Strict Subject-Level Splitting to prevent data leakage
        subject_ids = dataset.subject_ids
        train_val_subjects, test_subjects = train_test_split(subject_ids, test_size=self.config['test_split_size'], random_state=42)
        val_size_adjusted = self.config['validation_split_size'] / (1 - self.config['test_split_size'])
        train_subjects, val_subjects = train_test_split(train_val_subjects, test_size=val_size_adjusted, random_state=42)

        train_indices = [i for i, item in enumerate(dataset.trajectories) if item['subject_id'] in train_subjects]
        val_indices = [i for i, item in enumerate(dataset.trajectories) if item['subject_id'] in val_subjects]
        test_indices = [i for i, item in enumerate(dataset.trajectories) if item['subject_id'] in test_subjects]

        logging.info(f"Data Split: {len(train_indices)} train, {len(val_indices)} validation, {len(test_indices)} test samples.")
        
        self.train_loader = DataLoader(Subset(dataset, train_indices), batch_size=self.config['batch_size'], shuffle=True, num_workers=4)
        self.val_loader = DataLoader(Subset(dataset, val_indices), batch_size=self.config['batch_size'], shuffle=False, num_workers=4)
        self.test_loader = DataLoader(Subset(dataset, test_indices), batch_size=self.config['batch_size'], shuffle=False, num_workers=4)

    def train(self):
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            self.trajectory_encoder.train()
            self.age_encoder.train()
            train_loss = 0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
            for trajectory, age in pbar:
                trajectory, age = trajectory.to(DEVICE), age.to(DEVICE)
                self.optimizer.zero_grad()
                
                traj_embedding = self.trajectory_encoder(trajectory)
                age_embedding = self.age_encoder(age)
                target = torch.ones(trajectory.size(0)).to(DEVICE)
                
                loss = self.loss_fn(traj_embedding, age_embedding, target)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                pbar.set_postfix({'train_loss': loss.item()})
            
            avg_train_loss = train_loss / len(self.train_loader)
            avg_val_loss = self.validate()
            
            logging.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
            # Early Stopping Logic
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.trajectory_encoder.state_dict(), 'best_trajectory_encoder.pth')
                torch.save(self.age_encoder.state_dict(), 'best_age_encoder.pth')
                logging.info(f"  -> New best model saved with validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                logging.info(f"  -> No improvement. Patience: {patience_counter}/{self.config['early_stopping_patience']}")
            
            if patience_counter >= self.config['early_stopping_patience']:
                logging.info("🛑 Early stopping triggered. Training finished.")
                break
        
    def validate(self):
        self.trajectory_encoder.eval()
        self.age_encoder.eval()
        val_loss = 0
        with torch.no_grad():
            for trajectory, age in self.val_loader:
                trajectory, age = trajectory.to(DEVICE), age.to(DEVICE)
                traj_embedding = self.trajectory_encoder(trajectory)
                age_embedding = self.age_encoder(age)
                target = torch.ones(trajectory.size(0)).to(DEVICE)
                loss = self.loss_fn(traj_embedding, age_embedding, target)
                val_loss += loss.item()
        return val_loss / len(self.val_loader)

# ============================================================================
# FINAL PERFORMANCE EVALUATION
# ============================================================================

def evaluate_performance(config, test_loader):
    logging.info("\n" + "="*50 + "\n🔥 FINAL PERFORMANCE EVALUATION ON TEST SET 🔥\n" + "="*50)
    
    try:
        traj_encoder = TrajectoryEncoder(embedding_dim=config['embedding_dim']).to(DEVICE)
        traj_encoder.load_state_dict(torch.load('best_trajectory_encoder.pth', map_location=DEVICE))
        age_encoder = AgeEncoder(embedding_dim=config['embedding_dim']).to(DEVICE)
        age_encoder.load_state_dict(torch.load('best_age_encoder.pth', map_location=DEVICE))
        traj_encoder.eval()
        age_encoder.eval()
    except FileNotFoundError:
        logging.error("❌ Best model files not found. Cannot run final evaluation.")
        return

    age_range = torch.arange(20, 91, 1).float().view(-1, 1).to(DEVICE)
    with torch.no_grad():
        age_manifold_embeddings = age_encoder(age_range)
    
    def predict_balance_age(traj_embedding):
        similarities = F.cosine_similarity(traj_embedding, age_manifold_embeddings)
        return age_range[torch.argmax(similarities)].item()

    true_ages, pred_ages = [], []
    with torch.no_grad():
        for trajectory, age in tqdm(test_loader, desc="Evaluating on Test Set"):
            trajectory = trajectory.to(DEVICE)
            traj_embeddings = traj_encoder(trajectory)
            
            for i in range(len(trajectory)):
                true_ages.append(age[i].item())
                pred_age = predict_balance_age(traj_embeddings[i].unsqueeze(0))
                pred_ages.append(pred_age)

    # Quantitative Metrics
    true_ages, pred_ages = np.array(true_ages), np.array(pred_ages)
    mae = np.mean(np.abs(true_ages - pred_ages))
    correlation = np.corrcoef(true_ages, pred_ages)[0, 1]

    print("\n--- PERFORMANCE REPORT ---")
    print(f"✅ Mean Absolute Error (MAE): {mae:.2f} years")
    print(f"✅ Pearson Correlation (r):   {correlation:.3f}")
    print("--------------------------\n")

    # Example Predictions
    print("--- Example Predictions ---")
    indices = np.random.choice(len(true_ages), 5, replace=False)
    for i in indices:
        print(f"  Sample: True Age = {true_ages[i]:.0f} -> Predicted Balance Age = {pred_ages[i]:.0f}")
    print("---------------------------\n")

    # Visualization
    plt.figure(figsize=(8, 8))
    plt.scatter(true_ages, pred_ages, alpha=0.6, edgecolors='w', c=true_ages, cmap='viridis')
    plt.plot([min(true_ages), max(true_ages)], [min(true_ages), max(true_ages)], 'r--', lw=2, label='Perfect Prediction (y=x)')
    plt.title(f'Predicted Balance Age vs. True Age (MAE = {mae:.2f})', fontsize=16)
    plt.xlabel('True Subject Age', fontsize=12)
    plt.ylabel('Predicted Balance Age', fontsize=12)
    plt.grid(True)
    plt.legend()
    cbar = plt.colorbar()
    cbar.set_label('True Age')
    plt.savefig('final_performance_report.png', dpi=150)
    logging.info("✅ Performance plot saved to 'final_performance_report.png'")
    plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    logging.info(f"🔥 Initializing Hybrid Model Trainer on {DEVICE}")
    trainer = ModelTrainer(CONFIG)
    
    trainer.train()
    
    evaluate_performance(CONFIG, trainer.test_loader)
