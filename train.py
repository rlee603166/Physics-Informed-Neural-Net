#!/usr/bin/env python3
"""
Advanced PINN Training Environment for Balance Assessment
- Loads data from multiple pre-processed HDF5 batch files.
- Implements a memory-efficient "lazy loading" dataset.
- Includes a robust training loop with validation, early stopping, and LR scheduling.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import logging
import random
import gc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 1. DATA LOADING (Updated for multiple HDF5 batch files)
# ==============================================================================
class BalanceAgeDataset(Dataset):
    """
    Memory-efficient PyTorch Dataset for multiple HDF5 batch files.
    It creates a global index of data points and loads them from disk on-the-fly.
    """
    def __init__(self, processed_data_folder: str):
        self.data_folder = Path(processed_data_folder)
        self.batch_files = sorted(list(self.data_folder.glob("batch_*.h5")))
        
        if not self.batch_files:
            raise FileNotFoundError(f"No HDF5 batch files found in '{self.data_folder}'. Please run the processing script first.")
        
        self.sampling_rate = 106.0 # Default, will be updated from the first batch file
        
        # This index will map a global item index to a specific file and location within that file
        self.index_map = []
        self._build_index()

    def _build_index(self):
        """Creates a map of (file_path, subject_key, trial_key, index_in_trial) for every point."""
        logger.info(f"Building dataset index from {len(self.batch_files)} batch files (lazy loading)...")
        
        for file_path in self.batch_files:
            with h5py.File(file_path, 'r') as f:
                # Update sampling rate from the first file that has it
                if 'sampling_rate' in f.attrs and self.sampling_rate == 106.0:
                    self.sampling_rate = f.attrs['sampling_rate']
                
                for subject_key in f.keys():
                    for trial_key in f[subject_key].keys():
                        n_points = f[subject_key][trial_key].attrs['n_points']
                        for i in range(n_points):
                            self.index_map.append((str(file_path), subject_key, trial_key, i))
        
        logger.info(f"Indexing complete. Found {len(self.index_map):,} total data points.")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        # This approach is safe for multiprocessing as each worker opens its own handle.
        file_path, subject_key, trial_key, point_idx = self.index_map[idx]
        
        with h5py.File(file_path, 'r') as f:
            subject_group = f[subject_key]
            trial_group = subject_group[trial_key]
            
            # Load only the required data for this specific point
            age = subject_group.attrs['age']
            
            # Calculate time on-the-fly
            t = (point_idx / self.sampling_rate)
            
            x = trial_group['cop_x'][point_idx]
            y = trial_group['cop_y'][point_idx]
        
        # Reshape for model input
        time_tensor = torch.tensor([t], dtype=torch.float32)
        age_tensor = torch.tensor([age], dtype=torch.float32)
        xy_tensor = torch.tensor([x, y], dtype=torch.float32)

        return time_tensor, age_tensor, xy_tensor


# 2. PINN ARCHITECTURE
# ==============================================================================
class BalancePINN(nn.Module):
    """
    A PINN with two sub-networks:
    1. ParameterNet: Predicts physical parameters (K, B, τ) from age.
    2. SolverNet: Predicts position (x, y) from time and the predicted parameters.
    """
    def __init__(self, hidden_dim=256, num_layers=8):
        super().__init__()
        
        self.parameter_net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 3) # Outputs: K (stiffness), B (damping), τ (delay)
        )
        
        solver_layers = [nn.Linear(1 + 3, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            solver_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        solver_layers.append(nn.Linear(hidden_dim, 2)) # Outputs: x(t), y(t)
        self.solver_net = nn.Sequential(*solver_layers)

    def forward(self, t, age):
        params_raw = self.parameter_net(age)
        K = 2000 * torch.sigmoid(params_raw[:, 0:1]) # Stiffness ~0-2000
        B = 100 * torch.sigmoid(params_raw[:, 1:2])  # Damping ~0-100
        tau = 0.3 * torch.sigmoid(params_raw[:, 2:3]) # Delay ~0-300ms
        
        predicted_params = torch.cat([K, B, tau], dim=1)
        
        solver_input = torch.cat([t, predicted_params], dim=1)
        xy_pred = self.solver_net(solver_input)
        
        return xy_pred, predicted_params


# 3. TRAINING ENVIRONMENT
# ==============================================================================
class Trainer:
    """
    Advanced trainer with early stopping, and LR scheduling.
    """
    def __init__(self, model: BalancePINN, dataset: BalanceAgeDataset, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        
        train_size = int(config['train_split'] * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5)
        self.loss_fn = nn.MSELoss()
        
        self.save_path = Path(config['save_path'])
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

    def _calculate_physics_loss(self, t, age, params):
        t.requires_grad_(True)
        xy_pred_for_grad, _ = self.model(t, age)
        x_pred, y_pred = xy_pred_for_grad[:, 0], xy_pred_for_grad[:, 1]
        
        dx_dt = torch.autograd.grad(x_pred.sum(), t, create_graph=True)[0]
        dy_dt = torch.autograd.grad(y_pred.sum(), t, create_graph=True)[0]
        
        d2x_dt2 = torch.autograd.grad(dx_dt.sum(), t, create_graph=True)[0]
        d2y_dt2 = torch.autograd.grad(dy_dt.sum(), t, create_graph=True)[0]
        
        g, L, m = 9.81, 1.0, 70.0
        K, B, _ = params[:, 0], params[:, 1], params[:, 2]

        residual_x = d2x_dt2 - (g/L)*x_pred + (K/(m*L**2))*x_pred + (B/(m*L**2))*dx_dt
        residual_y = d2y_dt2 - (g/L)*y_pred + (K/(m*L**2))*y_pred + (B/(m*L**2))*dy_dt
        
        loss_physics = self.loss_fn(residual_x, torch.zeros_like(residual_x)) + \
                       self.loss_fn(residual_y, torch.zeros_like(residual_y))
        
        return loss_physics

    def _run_epoch(self, loader, is_train=True):
        self.model.train(is_train)
        total_loss, total_data_loss, total_phys_loss = 0, 0, 0
        
        context = torch.enable_grad() if is_train else torch.no_grad()
        progress_bar = tqdm(loader, desc=f"Epoch {self.current_epoch:03d} ({'Train' if is_train else 'Val'})", leave=False)
        
        for t, age, xy_true in progress_bar:
            t, age, xy_true = t.to(self.device), age.to(self.device), xy_true.to(self.device)
            xy_pred, params = self.model(t, age)
            data_loss = self.loss_fn(xy_pred, xy_true)
            phys_loss = self._calculate_physics_loss(t, age, params)
            loss = data_loss + self.config['physics_weight'] * phys_loss
            
            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            total_loss += loss.item()
            total_data_loss += data_loss.item()
            total_phys_loss += phys_loss.item()
        
        return total_loss / len(loader), total_data_loss / len(loader), total_phys_loss / len(loader)

    def train(self):
        logger.info(f"Starting training on {self.device}...")
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config['epochs']):
            self.current_epoch = epoch + 1
            train_loss, train_data, train_phys = self._run_epoch(self.train_loader, is_train=True)
            val_loss, val_data, val_phys = self._run_epoch(self.val_loader, is_train=False)
            
            logger.info(
                f"Epoch {self.current_epoch:03d} | Train Loss: {train_loss:.4e} | Val Loss: {val_loss:.4e} "
                f"| Data: {val_data:.4e} | Phys: {val_phys:.4e} | LR: {self.optimizer.param_groups[0]['lr']:.2e}"
            )
            
            self.scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.save_path)
                logger.info(f"✅ Validation loss improved. Model saved to {self.save_path}")
            else:
                patience_counter += 1
                if patience_counter >= self.config['patience']:
                    logger.warning(f"Validation loss did not improve for {self.config['patience']} epochs. Stopping early.")
                    break
        
        logger.info("Training complete.")

# 4. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # --- Configuration Dictionary ---
    config = {
        'data_folder': "processed_data",
        'save_path': "trained_models/best_balance_pinn.pth",
        'train_split': 0.8,
        'epochs': 200,
        'batch_size': 8192,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'physics_weight': 0.01,
        'patience': 10
    }

    # --- Setup & Training ---
    try:
        # 1. Initialize the dataset
        dataset = BalanceAgeDataset(processed_data_folder=config['data_folder'])
        
        if len(dataset) == 0:
            logger.error("Dataset is empty. Cannot start training.")
        else:
            # 2. Initialize the model
            model = BalancePINN()
            
            # ✅ Check if a saved model exists to resume training
            model_path = Path(config['save_path'])
            if model_path.exists():
                logger.info(f"Found existing model at '{model_path}'. Loading weights to resume training.")
                model.load_state_dict(torch.load(model_path))
            else:
                logger.info("No existing model found. Starting training from scratch.")

            # 3. Initialize the Trainer and run
            trainer = Trainer(model, dataset, config)
            trainer.train()

    except FileNotFoundError as e:
        logger.error(f"A required file or folder was not found. Please check your paths. Details: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during setup or training: {e}", exc_info=True)
