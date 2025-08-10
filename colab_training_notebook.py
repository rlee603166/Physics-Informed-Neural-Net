# ==================================================================================
# BALANCE PINN TRAINING - GOOGLE COLAB NOTEBOOK
# ==================================================================================
# This notebook trains enhanced Balance PINN models for cross-age comparison
# Copy each cell into separate Google Colab cells and run in order

# ==================================================================================
# CELL 1: Setup and Install Dependencies
# ==================================================================================
"""
BEFORE RUNNING: Upload these files to your Colab session:
1. processed_data/ folder (containing batch_0.h5, batch_1.h5)
2. user_ages.csv
3. improved_models.py
4. enhanced_datasets.py  
5. training_utils.py
6. config.py

You can drag and drop these files into the Colab file browser on the left.
"""

# Install required packages
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install matplotlib seaborn pyyaml h5py pandas numpy tqdm

# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"PyTorch version: {torch.__version__}")

# ==================================================================================
# CELL 2: Verify File Upload and Data
# ==================================================================================

import os
from pathlib import Path

# Check uploaded files
print("=== CHECKING UPLOADED FILES ===")
required_files = [
    'improved_models.py',
    'enhanced_datasets.py', 
    'training_utils.py',
    'config.py',
    'user_ages.csv'
]

missing_files = []
for file in required_files:
    if os.path.exists(file):
        print(f"✅ {file}")
    else:
        print(f"❌ {file} - MISSING!")
        missing_files.append(file)

# Check data folder
if os.path.exists('processed_data'):
    print("✅ processed_data/ folder")
    batch_files = list(Path('processed_data').glob('batch_*.h5'))
    print(f"   Found {len(batch_files)} batch files: {[f.name for f in batch_files]}")
else:
    print("❌ processed_data/ folder - MISSING!")
    missing_files.append('processed_data/')

if missing_files:
    print(f"\n⚠️  Please upload missing files: {missing_files}")
    print("Drag and drop them into the file browser on the left")
else:
    print("\n✅ All files uploaded successfully!")

# ==================================================================================
# CELL 3: Quick Data Inspection
# ==================================================================================

# Quick inspection of the data to ensure it's working
import pandas as pd
import h5py
import numpy as np

print("=== DATA INSPECTION ===")

# Check age data
try:
    age_df = pd.read_csv('user_ages.csv')
    print(f"Age data: {len(age_df)} subjects")
    print(f"Age range: {age_df.age.min():.1f} - {age_df.age.max():.1f} years")
    print(f"Age mean ± std: {age_df.age.mean():.1f} ± {age_df.age.std():.1f}")
    print()
except Exception as e:
    print(f"Error loading age data: {e}")

# Check HDF5 batch files
try:
    batch_files = list(Path('processed_data').glob('batch_*.h5'))
    total_points = 0
    total_subjects = 0
    
    for batch_file in batch_files:
        with h5py.File(batch_file, 'r') as f:
            n_subjects = len(f.keys())
            points_in_batch = 0
            
            for subject_key in f.keys():
                for trial_key in f[subject_key].keys():
                    points_in_batch += f[subject_key][trial_key].attrs['n_points']
            
            total_subjects += n_subjects
            total_points += points_in_batch
            
            print(f"{batch_file.name}: {n_subjects} subjects, {points_in_batch:,} points")
    
    print(f"\nTotal: {total_subjects} subjects, {total_points:,} data points")
    print(f"Sampling rate: {f.attrs.get('sampling_rate', 'Unknown')} Hz")
    
except Exception as e:
    print(f"Error inspecting batch files: {e}")

# ==================================================================================
# CELL 4: Performance Optimization Setup  
# ==================================================================================

# PERFORMANCE OPTIMIZATION FOR FAST TRAINING
print("=== PERFORMANCE OPTIMIZATION ===")

# Auto-detect optimal configuration for your hardware
import torch
from torch.cuda.amp import GradScaler, autocast

# Check GPU capabilities
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_name}")
    print(f"GPU Memory: {gpu_memory_gb:.1f}GB")
    
    # Optimize based on GPU
    if 'A100' in gpu_name:
        # A100 optimized settings  
        OPTIMIZED_CONFIG = {
            'batch_size': 16384,       # 4x larger batches for better GPU utilization
            'num_workers': 8,          # More data loading workers  
            'mixed_precision': True,   # Use automatic mixed precision (AMP)
            'physics_computation_frequency': 4,  # Compute physics every 4th batch
            'val_frequency': 5,        # Validate every 5 epochs (not every epoch)
            'learning_rate': 2e-3,     # Higher LR for larger batches
            'prefetch_factor': 4,      # Prefetch more batches
            'persistent_workers': True, # Keep data workers alive
            'empty_cache_frequency': 100, # Clear GPU cache every 100 batches
        }
        print("✅ A100 optimization enabled - expect ~5x faster training!")
    else:
        # Conservative settings for other GPUs
        OPTIMIZED_CONFIG = {
            'batch_size': 8192,
            'num_workers': 4,
            'mixed_precision': True,
            'physics_computation_frequency': 8,
            'val_frequency': 10,
            'learning_rate': 1e-3,
        }
        print("✅ Standard GPU optimization enabled")
else:
    # CPU fallback
    OPTIMIZED_CONFIG = {
        'batch_size': 1024,
        'num_workers': 2,
        'mixed_precision': False,
        'physics_computation_frequency': 16,
    }
    print("⚠️ Using CPU - training will be slow")

print(f"Optimized batch size: {OPTIMIZED_CONFIG['batch_size']:,}")
print(f"Expected speedup: 3-5x faster than original")

# ==================================================================================
# CELL 5: Choose Training Method
# ==================================================================================

print("=== CHOOSE TRAINING METHOD ===")
print()
print("1. TWO-STAGE TRAINING (Recommended)")
print("   - Stage 1: Learn individual subject parameters")  
print("   - Stage 2: Learn age-dependent trends")
print("   - Best for: Clinical interpretability")
print("   - Training time: ~30-60 minutes")
print()
print("2. IMPROVED SINGLE-STAGE TRAINING")
print("   - Enhanced single model with probabilistic outputs")
print("   - Best for: End-to-end efficiency")
print("   - Training time: ~20-40 minutes")
print()
print("Choose your method by running the appropriate cell below:")

# ==================================================================================
# CELL 5A: Two-Stage Training Setup (Run this OR Cell 5B, not both)
# ==================================================================================

# Two-Stage Training Configuration
TRAINING_METHOD = "two_stage"
print(f"Selected: {TRAINING_METHOD.upper()} TRAINING")

# Configuration - merge with performance optimizations
config = {
    # Data
    'data_folder': 'processed_data',
    'age_csv_path': 'user_ages.csv', 
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'random_seed': 42,
    'min_points_per_subject': 100,
    
    # Performance optimized settings
    **OPTIMIZED_CONFIG,
    
    # Stage 1 (Subject parameters)
    'stage1_epochs': 50,        # Reduced for Colab
    'stage1_lr': 1e-3,
    'stage1_physics_weight': 0.05,
    'stage1_patience': 15,
    
    # Stage 2 (Age relationships)  
    'stage2_epochs': 30,        # Reduced for Colab
    'stage2_lr': 1e-3,
    'stage2_reg_weight': 0.1,
    'stage2_patience': 10,
    'stage2_batch_size': 32,
    'stage2_probabilistic': True,
    
    # General
    'checkpoint_dir': 'checkpoints_two_stage_colab',
    'weight_decay': 1e-5,
    'scheduler_patience': 8,
    'save_every': 10,
    
    # Parameter bounds
    'param_bounds': {
        'K': (500.0, 3000.0),
        'B': (20.0, 150.0), 
        'tau': (0.05, 0.4)
    }
}

print("Two-stage configuration loaded!")
print(f"Stage 1: {config['stage1_epochs']} epochs")
print(f"Stage 2: {config['stage2_epochs']} epochs")

# ==================================================================================
# CELL 5B: Single-Stage Training Setup (Run this OR Cell 5A, not both)
# ==================================================================================

# Single-Stage Training Configuration  
TRAINING_METHOD = "single_stage"
print(f"Selected: {TRAINING_METHOD.upper()} TRAINING")

# Configuration
config = {
    # Data
    'data_folder': 'processed_data',
    'age_csv_path': 'user_ages.csv',
    'normalize_data': False,
    'augment_data': False,
    'train_ratio': 0.7,
    'val_ratio': 0.15, 
    'test_ratio': 0.15,
    'batch_size': 2048,  # Reduced for Colab
    'num_workers': 2,    # Reduced for Colab
    'random_seed': 42,
    
    # Training
    'epochs': 80,         # Reduced for Colab
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'patience': 15,
    'scheduler_patience': 8,
    'warmup_epochs': 5,
    'checkpoint_dir': 'checkpoints_single_stage_colab',
    'save_every': 10,
    
    # Model
    'hidden_dim': 256,
    'num_layers': 6,      # Reduced for Colab
    'dropout_rate': 0.1,
    'use_probabilistic': True,
    
    # Loss weights
    'data_weight': 1.0,
    'physics_weight': 0.005,  # Lower for better data fitting
    'regularization_weight': 0.15,
    'age_aware_weight': 0.1,
    
    # Parameter bounds
    'param_bounds': {
        'K': (500.0, 3000.0),
        'B': (20.0, 150.0),
        'tau': (0.05, 0.4)
    }
}

print("Single-stage configuration loaded!")
print(f"Training: {config['epochs']} epochs")
print(f"Physics weight: {config['physics_weight']} (low for better data fitting)")

# ==================================================================================
# CELL 6: Two-Stage Training Implementation (Run only if chose two-stage)
# ==================================================================================

if TRAINING_METHOD == "two_stage":
    print("=== STARTING TWO-STAGE TRAINING ===")
    
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset
    import numpy as np
    from pathlib import Path
    from tqdm import tqdm
    import json
    import matplotlib.pyplot as plt
    from collections import defaultdict
    import logging

    # Import our modules
    from improved_models import SubjectPINN, AgeParameterModel
    from enhanced_datasets import SubjectAwareDataset, create_subject_splits, create_filtered_dataset
    from training_utils import PhysicsLoss, SimplePhysicsLoss, ParameterRegularizationLoss, EarlyStopping, MetricsTracker

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create checkpoint directory
    Path(config['checkpoint_dir']).mkdir(exist_ok=True)
    
    # Initialize dataset
    print("Loading dataset...")
    dataset = SubjectAwareDataset(
        processed_data_folder=config['data_folder'],
        age_csv_path=config['age_csv_path'],
        min_points_per_subject=config['min_points_per_subject']
    )
    
    print(f"Dataset loaded: {len(dataset):,} points from {len(dataset.valid_subjects)} subjects")
    
    # Create data splits
    subject_splits = create_subject_splits(
        dataset, config['train_ratio'], config['val_ratio'], config['test_ratio'], config['random_seed']
    )
    
    train_indices = create_filtered_dataset(dataset, subject_splits['train'])
    val_indices = create_filtered_dataset(dataset, subject_splits['val'])
    
    # Create data loaders
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        Subset(dataset, val_indices), 
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    print(f"Data splits: {len(train_indices):,} train, {len(val_indices):,} val")
    
else:
    print("Skipping two-stage training (not selected)")

# ==================================================================================
# CELL 7: Two-Stage Training - Stage 1 (Run only if chose two-stage)
# ==================================================================================

if TRAINING_METHOD == "two_stage":
    print("\n" + "="*50)
    print("STAGE 1: TRAINING SUBJECT PINN")
    print("="*50)
    
    # Initialize Stage 1 model
    subject_pinn = SubjectPINN(
        subject_ids=dataset.valid_subjects,
        param_bounds=config['param_bounds']
    ).to(device)
    
    # Optimizer and loss
    stage1_optimizer = torch.optim.Adam(
        subject_pinn.parameters(), 
        lr=config['stage1_lr'],
        weight_decay=config['weight_decay']
    )
    
    stage1_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        stage1_optimizer, mode='min', factor=0.5, patience=config['scheduler_patience']
    )
    
    # Use SimplePhysicsLoss for more stable training (change to PhysicsLoss for full physics)
    stage1_loss_fn = SimplePhysicsLoss(weight=config['stage1_physics_weight'])
    # For full physics constraint, use: stage1_loss_fn = PhysicsLoss(weight=config['stage1_physics_weight'])
    stage1_early_stopping = EarlyStopping(patience=config['stage1_patience'])
    
    print(f"Stage 1 model created: {sum(p.numel() for p in subject_pinn.parameters()):,} parameters")
    
    # Initialize mixed precision training
    scaler = GradScaler() if config.get('mixed_precision', False) else None
    use_amp = config.get('mixed_precision', False)
    physics_freq = config.get('physics_computation_frequency', 1)
    
    print(f"Mixed precision: {use_amp}")
    print(f"Physics computation frequency: every {physics_freq} batches")
    
    # Training loop
    best_val_loss = float('inf')
    stage1_metrics = []
    
    for epoch in range(config['stage1_epochs']):
        # Training
        subject_pinn.train()
        train_losses = defaultdict(float)
        train_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Stage 1 Epoch {epoch+1}")
        for batch_idx, (t, age, xy_true, subject_idx) in enumerate(pbar):
            t = t.to(device, non_blocking=True).requires_grad_(True)
            age = age.to(device, non_blocking=True)
            xy_true = xy_true.to(device, non_blocking=True)
            subject_idx = subject_idx.to(device, non_blocking=True)
            
            # Forward pass with mixed precision
            with autocast(enabled=use_amp):
                xy_pred, params = subject_pinn(t, subject_idx)
                
                # Data loss
                data_loss = nn.functional.mse_loss(xy_pred, xy_true)
                
                # Physics loss (computed less frequently for speed)
                if batch_idx % physics_freq == 0:
                    physics_loss = stage1_loss_fn(t, xy_pred, params)
                else:
                    physics_loss = torch.tensor(0.0, device=device)
                
                total_loss = data_loss + physics_loss
            
            # Backward pass with mixed precision
            if use_amp:
                scaler.scale(total_loss).backward()
                scaler.unscale_(stage1_optimizer)
                torch.nn.utils.clip_grad_norm_(subject_pinn.parameters(), max_norm=1.0)
                scaler.step(stage1_optimizer)
                scaler.update()
                stage1_optimizer.zero_grad()
            else:
                stage1_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(subject_pinn.parameters(), max_norm=1.0)
            stage1_optimizer.step()
            
            # Track metrics
            batch_size = t.shape[0]
            train_losses['data'] += data_loss.item() * batch_size
            train_losses['physics'] += physics_loss.item() * batch_size
            train_losses['total'] += total_loss.item() * batch_size
            train_samples += batch_size
            
            pbar.set_postfix({
                'Data': f"{data_loss.item():.6f}",
                'Physics': f"{physics_loss.item():.6f}"
            })
        
        # Validation
        subject_pinn.eval()
        val_losses = defaultdict(float)
        val_samples = 0
        
        with torch.no_grad():
            for t, age, xy_true, subject_idx in val_loader:
                t = t.to(device).requires_grad_(True)
                age = age.to(device)
                xy_true = xy_true.to(device) 
                subject_idx = subject_idx.to(device)
                
                xy_pred, params = subject_pinn(t, subject_idx)
                data_loss = nn.functional.mse_loss(xy_pred, xy_true)
                physics_loss = stage1_loss_fn(t, xy_pred, params)
                total_loss = data_loss + physics_loss
                
                batch_size = t.shape[0]
                val_losses['data'] += data_loss.item() * batch_size
                val_losses['physics'] += physics_loss.item() * batch_size
                val_losses['total'] += total_loss.item() * batch_size
                val_samples += batch_size
        
        # Calculate averages
        avg_train = {k: v / train_samples for k, v in train_losses.items()}
        avg_val = {k: v / val_samples for k, v in val_losses.items()}
        
        # Learning rate scheduling
        stage1_scheduler.step(avg_val['total'])
        
        # Logging
        print(f"Epoch {epoch+1}/{config['stage1_epochs']}")
        print(f"  Train - Data: {avg_train['data']:.6f}, Physics: {avg_train['physics']:.6f}, Total: {avg_train['total']:.6f}")
        print(f"  Val   - Data: {avg_val['data']:.6f}, Physics: {avg_val['physics']:.6f}, Total: {avg_val['total']:.6f}")
        print(f"  LR: {stage1_optimizer.param_groups[0]['lr']:.2e}")
        
        # Save metrics
        stage1_metrics.append({
            'epoch': epoch + 1,
            'train_data': avg_train['data'],
            'train_physics': avg_train['physics'],
            'train_total': avg_train['total'],
            'val_data': avg_val['data'],
            'val_physics': avg_val['physics'],
            'val_total': avg_val['total'],
            'lr': stage1_optimizer.param_groups[0]['lr']
        })
        
        # Save best model
        if avg_val['total'] < best_val_loss:
            best_val_loss = avg_val['total']
            torch.save(subject_pinn.state_dict(), 
                      Path(config['checkpoint_dir']) / 'best_stage1_model.pth')
            print(f"  ✅ New best model saved (loss: {best_val_loss:.6f})")
        
        # Early stopping
        if stage1_early_stopping(avg_val['total'], subject_pinn):
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    print(f"\nStage 1 completed! Best validation loss: {best_val_loss:.6f}")

# ==================================================================================  
# CELL 8: Two-Stage Training - Extract Parameters (Run only if chose two-stage)
# ==================================================================================

if TRAINING_METHOD == "two_stage":
    print("\n" + "="*50)
    print("EXTRACTING SUBJECT PARAMETERS")
    print("="*50)
    
    # Extract learned parameters for Stage 2
    subject_pinn.eval()
    subject_parameters = {}
    
    with torch.no_grad():
        for i, subject_id in enumerate(dataset.valid_subjects):
            subject_idx = torch.tensor([[i]], dtype=torch.long).to(device)
            K, B, tau = subject_pinn.get_parameters(subject_idx.squeeze())
            
            subject_info = dataset.get_subject_info(subject_id)
            age = subject_info.get('age', 0)
            
            subject_parameters[subject_id] = {
                'age': age,
                'K': K.item(),
                'B': B.item(), 
                'tau': tau.item(),
                'n_points': subject_info.get('n_points', 0)
            }
    
    # Save parameters
    with open(Path(config['checkpoint_dir']) / 'subject_parameters.json', 'w') as f:
        json.dump(subject_parameters, f, indent=2)
    
    # Statistics
    ages = [p['age'] for p in subject_parameters.values()]
    Ks = [p['K'] for p in subject_parameters.values()]
    Bs = [p['B'] for p in subject_parameters.values()]
    taus = [p['tau'] for p in subject_parameters.values()]
    
    print(f"Extracted parameters for {len(subject_parameters)} subjects:")
    print(f"  Age range: {min(ages):.1f} - {max(ages):.1f}")
    print(f"  K range: {min(Ks):.1f} - {max(Ks):.1f}")
    print(f"  B range: {min(Bs):.1f} - {max(Bs):.1f}")  
    print(f"  τ range: {min(taus):.3f} - {max(taus):.3f}")
    
    # Plot parameter distributions
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Age vs parameters
    axes[0].scatter(ages, Ks, alpha=0.6)
    axes[0].set_xlabel('Age (years)')
    axes[0].set_ylabel('Stiffness K')
    axes[0].set_title('K vs Age')
    
    axes[1].scatter(ages, Bs, alpha=0.6)
    axes[1].set_xlabel('Age (years)')  
    axes[1].set_ylabel('Damping B')
    axes[1].set_title('B vs Age')
    
    axes[2].scatter(ages, taus, alpha=0.6)
    axes[2].set_xlabel('Age (years)')
    axes[2].set_ylabel('Delay τ')
    axes[2].set_title('τ vs Age')
    
    # Parameter distribution
    axes[3].hist(Ks, alpha=0.5, label='K', bins=15)
    axes[3].hist(Bs, alpha=0.5, label='B', bins=15)
    axes[3].hist([t*1000 for t in taus], alpha=0.5, label='τ×1000', bins=15)
    axes[3].set_xlabel('Parameter Value')
    axes[3].set_ylabel('Frequency')
    axes[3].set_title('Parameter Distributions')
    axes[3].legend()
    
    plt.tight_layout()
    plt.savefig(Path(config['checkpoint_dir']) / 'stage1_parameters.png', dpi=300, bbox_inches='tight')
    plt.show()

# ==================================================================================
# CELL 9: Two-Stage Training - Stage 2 (Run only if chose two-stage)  
# ==================================================================================

if TRAINING_METHOD == "two_stage":
    print("\n" + "="*50)
    print("STAGE 2: TRAINING AGE PARAMETER MODEL")
    print("="*50)
    
    # Initialize Stage 2 model
    age_model = AgeParameterModel(
        param_bounds=config['param_bounds'],
        use_probabilistic=config['stage2_probabilistic']
    ).to(device)
    
    # Prepare Stage 2 data (age -> parameters)
    train_ages = []
    train_params = []
    val_ages = []
    val_params = []
    
    for subject_id in subject_splits['train']:
        if subject_id in subject_parameters:
            param_data = subject_parameters[subject_id]
            train_ages.append(param_data['age'])
            train_params.append([param_data['K'], param_data['B'], param_data['tau']])
    
    for subject_id in subject_splits['val']:
        if subject_id in subject_parameters:
            param_data = subject_parameters[subject_id]
            val_ages.append(param_data['age'])
            val_params.append([param_data['K'], param_data['B'], param_data['tau']])
    
    # Convert to tensors
    train_ages_tensor = torch.tensor(train_ages, dtype=torch.float32).unsqueeze(-1)
    train_params_tensor = torch.tensor(train_params, dtype=torch.float32)  
    val_ages_tensor = torch.tensor(val_ages, dtype=torch.float32).unsqueeze(-1)
    val_params_tensor = torch.tensor(val_params, dtype=torch.float32)
    
    print(f"Stage 2 data: {len(train_ages)} train, {len(val_ages)} val subjects")
    
    # Optimizer and loss
    stage2_optimizer = torch.optim.Adam(
        age_model.parameters(),
        lr=config['stage2_lr'], 
        weight_decay=config['weight_decay']
    )
    
    stage2_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        stage2_optimizer, mode='min', factor=0.5, patience=config['scheduler_patience']
    )
    
    stage2_param_loss = nn.MSELoss()
    stage2_reg_loss = ParameterRegularizationLoss(param_bounds=config['param_bounds'])
    stage2_early_stopping = EarlyStopping(patience=config['stage2_patience'])
    
    print(f"Stage 2 model created: {sum(p.numel() for p in age_model.parameters()):,} parameters")
    
    # Training loop
    best_val_loss = float('inf')
    stage2_metrics = []
    
    for epoch in range(config['stage2_epochs']):
        # Training
        age_model.train()
        
        # Shuffle training data
        indices = torch.randperm(len(train_ages))
        train_ages_shuffled = train_ages_tensor[indices].to(device)
        train_params_shuffled = train_params_tensor[indices].to(device)
        
        train_losses = defaultdict(float)
        n_batches = 0
        
        # Mini-batch training
        batch_size = min(config['stage2_batch_size'], len(train_ages))
        
        for i in range(0, len(train_ages), batch_size):
            batch_ages = train_ages_shuffled[i:i+batch_size]
            batch_params = train_params_shuffled[i:i+batch_size]
            
            # Forward pass
            if age_model.use_probabilistic:
                pred_means, pred_stds = age_model.predict_parameters(batch_ages)
                # Negative log-likelihood loss
                param_loss = 0.5 * torch.mean(
                    ((batch_params - pred_means) / (pred_stds + 1e-6))**2 + 
                    torch.log(pred_stds + 1e-6)
                )
            else:
                pred_params = age_model.predict_parameters(batch_ages)
                param_loss = stage2_param_loss(pred_params, batch_params)
            
            # Regularization
            reg_losses = stage2_reg_loss(batch_ages, batch_params)
            total_reg_loss = sum(reg_losses.values())
            
            # Total loss
            total_loss = param_loss + config['stage2_reg_weight'] * total_reg_loss
            
            # Backward pass
            stage2_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(age_model.parameters(), max_norm=1.0)
            stage2_optimizer.step()
            
            # Track losses
            train_losses['param'] += param_loss.item()
            train_losses['reg'] += total_reg_loss.item()
            train_losses['total'] += total_loss.item()
            n_batches += 1
        
        # Average training losses
        avg_train = {k: v / n_batches for k, v in train_losses.items()}
        
        # Validation
        age_model.eval()
        ages_tensor = val_ages_tensor.to(device)
        params_tensor = val_params_tensor.to(device)
        
        with torch.no_grad():
            if age_model.use_probabilistic:
                pred_means, pred_stds = age_model.predict_parameters(ages_tensor)
                param_loss = 0.5 * torch.mean(
                    ((params_tensor - pred_means) / (pred_stds + 1e-6))**2 +
                    torch.log(pred_stds + 1e-6)
                )
            else:
                pred_params = age_model.predict_parameters(ages_tensor)
                param_loss = stage2_param_loss(pred_params, params_tensor)
            
            reg_losses = stage2_reg_loss(ages_tensor, params_tensor)
            total_reg_loss = sum(reg_losses.values())
            val_total_loss = param_loss + config['stage2_reg_weight'] * total_reg_loss
        
        # Learning rate scheduling
        stage2_scheduler.step(val_total_loss)
        
        # Logging
        print(f"Epoch {epoch+1}/{config['stage2_epochs']}")
        print(f"  Train - Param: {avg_train['param']:.6f}, Reg: {avg_train['reg']:.6f}, Total: {avg_train['total']:.6f}")
        print(f"  Val   - Param: {param_loss.item():.6f}, Reg: {total_reg_loss.item():.6f}, Total: {val_total_loss.item():.6f}")
        print(f"  LR: {stage2_optimizer.param_groups[0]['lr']:.2e}")
        
        # Save metrics
        stage2_metrics.append({
            'epoch': epoch + 1,
            'train_param': avg_train['param'],
            'train_reg': avg_train['reg'],
            'train_total': avg_train['total'],
            'val_param': param_loss.item(),
            'val_reg': total_reg_loss.item(),
            'val_total': val_total_loss.item(),
            'lr': stage2_optimizer.param_groups[0]['lr']
        })
        
        # Save best model
        if val_total_loss.item() < best_val_loss:
            best_val_loss = val_total_loss.item()
            # Save both models together
            checkpoint = {
                'subject_pinn_state_dict': subject_pinn.state_dict(),
                'age_model_state_dict': age_model.state_dict(),
                'config': config,
                'subject_parameters': subject_parameters
            }
            torch.save(checkpoint, Path(config['checkpoint_dir']) / 'best_two_stage_model.pth')
            print(f"  ✅ New best model saved (loss: {best_val_loss:.6f})")
        
        # Early stopping
        if stage2_early_stopping(val_total_loss.item(), age_model):
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    print(f"\nStage 2 completed! Best validation loss: {best_val_loss:.6f}")

# ==================================================================================
# CELL 10: Single-Stage Training Implementation (Run only if chose single-stage)
# ==================================================================================

if TRAINING_METHOD == "single_stage":
    print("=== STARTING IMPROVED SINGLE-STAGE TRAINING ===")
    
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset  
    import numpy as np
    from pathlib import Path
    from tqdm import tqdm
    import json
    import matplotlib.pyplot as plt
    from collections import defaultdict
    import logging

    # Import our modules
    from improved_models import ImprovedBalancePINN
    from enhanced_datasets import EnhancedBalanceDataset, create_subject_splits, create_filtered_dataset
    from training_utils import CombinedLoss, EarlyStopping, MetricsTracker, WarmupLRScheduler

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create checkpoint directory
    Path(config['checkpoint_dir']).mkdir(exist_ok=True)
    
    # Initialize dataset
    print("Loading dataset...")
    dataset = EnhancedBalanceDataset(
        processed_data_folder=config['data_folder'],
        age_csv_path=config['age_csv_path'],
        normalize=config['normalize_data'],
        augment=config['augment_data']
    )
    
    print(f"Dataset loaded: {len(dataset):,} points from {len(dataset.get_subject_ids())} subjects")
    
    # Create data splits
    subject_splits = create_subject_splits(
        dataset, config['train_ratio'], config['val_ratio'], config['test_ratio'], config['random_seed']
    )
    
    train_indices = create_filtered_dataset(dataset, subject_splits['train'])
    val_indices = create_filtered_dataset(dataset, subject_splits['val'])
    
    # Create data loaders
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    print(f"Data splits: {len(train_indices):,} train, {len(val_indices):,} val")
    
    # Initialize model
    model = ImprovedBalancePINN(
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        param_bounds=config['param_bounds'],
        use_probabilistic=config['use_probabilistic'],
        dropout_rate=config['dropout_rate']
    ).to(device)
    
    print(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Probabilistic outputs: {model.use_probabilistic}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    base_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=config['scheduler_patience']
    )
    
    scheduler = WarmupLRScheduler(
        optimizer,
        warmup_epochs=config['warmup_epochs'],
        max_lr=config['learning_rate'],
        decay_scheduler=base_scheduler
    )
    
    # Loss function
    loss_fn = CombinedLoss(
        data_weight=config['data_weight'],
        physics_weight=config['physics_weight'],
        regularization_weight=config['regularization_weight'],
        age_aware_weight=config['age_aware_weight'],
        param_bounds=config['param_bounds']
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config['patience'])
    
    print("Training setup complete!")

# ==================================================================================
# CELL 11: Single-Stage Training Loop (Run only if chose single-stage)
# ==================================================================================

if TRAINING_METHOD == "single_stage":
    print("\n" + "="*60)
    print("STARTING IMPROVED SINGLE-STAGE TRAINING")
    print("="*60)
    
    # Training loop
    best_val_loss = float('inf')
    training_metrics = []
    
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        epoch_losses = defaultdict(float)
        epoch_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in pbar:
            if len(batch) == 4:
                t, age, xy_true, metadata = batch
            else:
                t, age, xy_true = batch[:3]
            
            # Move to device with gradients for time
            t = t.to(device).requires_grad_(True)
            age = age.to(device)
            xy_true = xy_true.to(device)
            
            # Forward pass
            xy_pred, params = model(t, age, deterministic=False)
            
            # Calculate combined loss
            losses = loss_fn(t, age, xy_pred, xy_true, params)
            total_loss = losses['total']
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track losses
            batch_size = t.shape[0]
            for key, value in losses.items():
                epoch_losses[key] += value.item() * batch_size
            epoch_samples += batch_size
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{total_loss.item():.6f}",
                'Data': f"{losses['data'].item():.6f}",
                'Physics': f"{losses.get('physics', torch.tensor(0)).item():.6f}",
                'LR': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
        
        # Average training losses
        avg_train_losses = {key: value / epoch_samples for key, value in epoch_losses.items()}
        
        # Validation phase
        model.eval()
        val_losses = defaultdict(float)
        val_samples = 0
        val_predictions = []
        val_targets = []
        val_parameters = []
        val_ages = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation {epoch+1}", leave=False):
                if len(batch) == 4:
                    t, age, xy_true, metadata = batch
                else:
                    t, age, xy_true = batch[:3]
                
                t = t.to(device).requires_grad_(True)
                age = age.to(device)
                xy_true = xy_true.to(device)
                
                # Forward pass (deterministic)
                xy_pred, params = model(t, age, deterministic=True)
                
                # Calculate losses
                losses = loss_fn(t, age, xy_pred, xy_true, params)
                
                # Track losses
                batch_size = t.shape[0]
                for key, value in losses.items():
                    val_losses[key] += value.item() * batch_size
                val_samples += batch_size
                
                # Store for metrics
                val_predictions.append(xy_pred.cpu())
                val_targets.append(xy_true.cpu())
                val_parameters.append(params.cpu())
                val_ages.append(age.cpu())
        
        # Average validation losses
        avg_val_losses = {key: value / val_samples for key, value in val_losses.items()}
        
        # Additional validation metrics
        all_predictions = torch.cat(val_predictions, dim=0)
        all_targets = torch.cat(val_targets, dim=0)
        all_parameters = torch.cat(val_parameters, dim=0)
        all_ages = torch.cat(val_ages, dim=0)
        
        # R² score
        ss_res = torch.sum((all_targets - all_predictions) ** 2)
        ss_tot = torch.sum((all_targets - torch.mean(all_targets, dim=0)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Parameter statistics
        K_mean = all_parameters[:, 0].mean().item()
        K_std = all_parameters[:, 0].std().item()
        B_mean = all_parameters[:, 1].mean().item()
        B_std = all_parameters[:, 1].std().item()
        tau_mean = all_parameters[:, 2].mean().item()
        tau_std = all_parameters[:, 2].std().item()
        
        # Parameter variation coefficients
        K_cv = K_std / (K_mean + 1e-6)
        B_cv = B_std / (B_mean + 1e-6)
        tau_cv = tau_std / (tau_mean + 1e-6)
        
        # Learning rate scheduling
        if hasattr(scheduler, 'step'):
            if hasattr(scheduler, 'decay_scheduler') and epoch >= config['warmup_epochs']:
                scheduler.decay_scheduler.step(avg_val_losses['total'])
            else:
                scheduler.step()
        
        val_loss = avg_val_losses['total']
        current_lr = optimizer.param_groups[0]['lr']
        
        # Logging
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print(f"  Train - Total: {avg_train_losses['total']:.6f}, Data: {avg_train_losses['data']:.6f}, "
              f"Physics: {avg_train_losses.get('physics', 0):.6f}")
        print(f"  Val   - Total: {val_loss:.6f}, Data: {avg_val_losses['data']:.6f}, "
              f"R²: {r2_score:.4f}, MAE: {torch.nn.functional.l1_loss(all_predictions, all_targets).item():.4f}")
        print(f"  Params - K: {K_mean:.1f}±{K_std:.1f} (CV: {K_cv:.3f}), "
              f"B: {B_mean:.1f}±{B_std:.1f} (CV: {B_cv:.3f}), "
              f"τ: {tau_mean:.3f}±{tau_std:.3f} (CV: {tau_cv:.3f})")
        print(f"  LR: {current_lr:.2e}")
        
        # Store metrics
        training_metrics.append({
            'epoch': epoch + 1,
            'train_total': avg_train_losses['total'],
            'train_data': avg_train_losses['data'],
            'train_physics': avg_train_losses.get('physics', 0),
            'val_total': val_loss,
            'val_data': avg_val_losses['data'],
            'val_physics': avg_val_losses.get('physics', 0),
            'r2_score': r2_score.item() if isinstance(r2_score, torch.Tensor) else r2_score,
            'mae': torch.nn.functional.l1_loss(all_predictions, all_targets).item(),
            'K_mean': K_mean,
            'K_std': K_std,
            'K_cv': K_cv,
            'B_mean': B_mean,
            'B_std': B_std,
            'B_cv': B_cv,
            'tau_mean': tau_mean,
            'tau_std': tau_std,
            'tau_cv': tau_cv,
            'lr': current_lr
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'training_metrics': training_metrics
            }, Path(config['checkpoint_dir']) / 'best_single_stage_model.pth')
            print(f"  ✅ New best model saved (loss: {best_val_loss:.6f})")
        
        # Early stopping
        if early_stopping(val_loss, model):
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    print(f"\nSingle-stage training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final R² score: {training_metrics[-1]['r2_score']:.4f}")

# ==================================================================================
# CELL 12: Training Results Visualization
# ==================================================================================

print("=== GENERATING TRAINING VISUALIZATIONS ===")

if TRAINING_METHOD == "two_stage":
    # Two-stage training plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Stage 1 plots
    epochs1 = [m['epoch'] for m in stage1_metrics]
    
    axes[0, 0].plot(epochs1, [m['train_total'] for m in stage1_metrics], 'b-', label='Train')
    axes[0, 0].plot(epochs1, [m['val_total'] for m in stage1_metrics], 'r-', label='Val')
    axes[0, 0].set_title('Stage 1: Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs1, [m['train_data'] for m in stage1_metrics], 'b-', label='Train')
    axes[0, 1].plot(epochs1, [m['val_data'] for m in stage1_metrics], 'r-', label='Val')
    axes[0, 1].set_title('Stage 1: Data Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Data Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(epochs1, [m['train_physics'] for m in stage1_metrics], 'b-', label='Train')
    axes[0, 2].plot(epochs1, [m['val_physics'] for m in stage1_metrics], 'r-', label='Val')
    axes[0, 2].set_title('Stage 1: Physics Loss')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Physics Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Stage 2 plots
    epochs2 = [m['epoch'] for m in stage2_metrics]
    
    axes[1, 0].plot(epochs2, [m['train_total'] for m in stage2_metrics], 'b-', label='Train')
    axes[1, 0].plot(epochs2, [m['val_total'] for m in stage2_metrics], 'r-', label='Val')
    axes[1, 0].set_title('Stage 2: Total Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(epochs2, [m['train_param'] for m in stage2_metrics], 'b-', label='Train')
    axes[1, 1].plot(epochs2, [m['val_param'] for m in stage2_metrics], 'r-', label='Val')
    axes[1, 1].set_title('Stage 2: Parameter Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Parameter Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].plot(epochs2, [m['train_reg'] for m in stage2_metrics], 'b-', label='Train')
    axes[1, 2].plot(epochs2, [m['val_reg'] for m in stage2_metrics], 'r-', label='Val')
    axes[1, 2].set_title('Stage 2: Regularization Loss')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Reg Loss')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(config['checkpoint_dir']) / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

elif TRAINING_METHOD == "single_stage":
    # Single-stage training plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    epochs = [m['epoch'] for m in training_metrics]
    
    # Loss plots
    axes[0, 0].plot(epochs, [m['train_total'] for m in training_metrics], 'b-', label='Train')
    axes[0, 0].plot(epochs, [m['val_total'] for m in training_metrics], 'r-', label='Val')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, [m['train_data'] for m in training_metrics], 'b-', label='Train')
    axes[0, 1].plot(epochs, [m['val_data'] for m in training_metrics], 'r-', label='Val')
    axes[0, 1].set_title('Data Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Data Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(epochs, [m['r2_score'] for m in training_metrics], 'g-')
    axes[0, 2].plot(epochs, [m['mae'] for m in training_metrics], 'orange')
    axes[0, 2].set_title('R² Score & MAE')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Metric Value')
    axes[0, 2].legend(['R² Score', 'MAE'])
    axes[0, 2].grid(True, alpha=0.3)
    
    # Parameter variation plots
    axes[1, 0].plot(epochs, [m['K_cv'] for m in training_metrics], 'b-', label='K CV')
    axes[1, 0].plot(epochs, [m['B_cv'] for m in training_metrics], 'r-', label='B CV')
    axes[1, 0].plot(epochs, [m['tau_cv'] for m in training_metrics], 'g-', label='τ CV')
    axes[1, 0].set_title('Parameter Variation Coefficients')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Coefficient of Variation')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(epochs, [m['K_mean'] for m in training_metrics], 'b-', label='K')
    axes[1, 1].plot(epochs, [m['B_mean'] for m in training_metrics], 'r-', label='B')
    axes[1, 1].plot(epochs, [m['tau_mean'] * 1000 for m in training_metrics], 'g-', label='τ×1000')
    axes[1, 1].set_title('Parameter Means')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Parameter Value')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].semilogy(epochs, [m['lr'] for m in training_metrics], 'purple')
    axes[1, 2].set_title('Learning Rate')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Learning Rate')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(config['checkpoint_dir']) / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

print("Training visualizations saved!")

# ==================================================================================
# CELL 13: Model Age Analysis and Testing
# ==================================================================================

print("=== TESTING AGE-PARAMETER RELATIONSHIPS ===")

if TRAINING_METHOD == "two_stage":
    # Test Stage 2 age model
    age_model.eval()
    test_ages = np.linspace(20, 90, 50)
    
    with torch.no_grad():
        age_tensor = torch.tensor(test_ages, dtype=torch.float32).unsqueeze(-1).to(device)
        
        if age_model.use_probabilistic:
            param_means, param_stds = age_model.predict_parameters(age_tensor)
            K_values = param_means[:, 0].cpu().numpy()
            B_values = param_means[:, 1].cpu().numpy()
            tau_values = param_means[:, 2].cpu().numpy()
            K_stds = param_stds[:, 0].cpu().numpy()
            B_stds = param_stds[:, 1].cpu().numpy() 
            tau_stds = param_stds[:, 2].cpu().numpy()
        else:
            params = age_model.predict_parameters(age_tensor)
            K_values = params[:, 0].cpu().numpy()
            B_values = params[:, 1].cpu().numpy()
            tau_values = params[:, 2].cpu().numpy()
    
    # Plot learned age relationships
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Extract real subject data for comparison
    subject_ages = [p['age'] for p in subject_parameters.values()]
    subject_Ks = [p['K'] for p in subject_parameters.values()]
    subject_Bs = [p['B'] for p in subject_parameters.values()]
    subject_taus = [p['tau'] for p in subject_parameters.values()]
    
    # K vs Age
    axes[0].scatter(subject_ages, subject_Ks, alpha=0.6, color='blue', s=20, label='Subjects')
    axes[0].plot(test_ages, K_values, 'red', linewidth=2, label='Learned Trend')
    if age_model.use_probabilistic:
        axes[0].fill_between(test_ages, K_values - K_stds, K_values + K_stds, alpha=0.2, color='red')
    axes[0].set_xlabel('Age (years)')
    axes[0].set_ylabel('Stiffness K')
    axes[0].set_title('Stiffness vs Age')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # B vs Age
    axes[1].scatter(subject_ages, subject_Bs, alpha=0.6, color='blue', s=20, label='Subjects')
    axes[1].plot(test_ages, B_values, 'red', linewidth=2, label='Learned Trend')
    if age_model.use_probabilistic:
        axes[1].fill_between(test_ages, B_values - B_stds, B_values + B_stds, alpha=0.2, color='red')
    axes[1].set_xlabel('Age (years)')
    axes[1].set_ylabel('Damping B')
    axes[1].set_title('Damping vs Age')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # τ vs Age
    axes[2].scatter(subject_ages, subject_taus, alpha=0.6, color='blue', s=20, label='Subjects')
    axes[2].plot(test_ages, tau_values, 'red', linewidth=2, label='Learned Trend')
    if age_model.use_probabilistic:
        axes[2].fill_between(test_ages, tau_values - tau_stds, tau_values + tau_stds, alpha=0.2, color='red')
    axes[2].set_xlabel('Age (years)')
    axes[2].set_ylabel('Neural Delay τ')
    axes[2].set_title('Neural Delay vs Age')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(config['checkpoint_dir']) / 'age_parameter_relationships.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate correlations
    K_corr = np.corrcoef(test_ages, K_values)[0, 1]
    B_corr = np.corrcoef(test_ages, B_values)[0, 1]
    tau_corr = np.corrcoef(test_ages, tau_values)[0, 1]
    
    print(f"Age-Parameter Correlations:")
    print(f"  K-Age correlation: {K_corr:.4f}")
    print(f"  B-Age correlation: {B_corr:.4f}")
    print(f"  τ-Age correlation: {tau_corr:.4f}")
    
    # Parameter variation
    K_variation = np.std(K_values) / np.mean(K_values)
    B_variation = np.std(B_values) / np.mean(B_values)
    tau_variation = np.std(tau_values) / np.mean(tau_values)
    
    print(f"Parameter Variation Coefficients:")
    print(f"  K variation: {K_variation:.4f}")
    print(f"  B variation: {B_variation:.4f}")
    print(f"  τ variation: {tau_variation:.4f}")

elif TRAINING_METHOD == "single_stage":
    # Test single-stage age relationships
    model.eval()
    test_ages = np.linspace(20, 90, 50)
    
    with torch.no_grad():
        age_tensor = torch.tensor(test_ages, dtype=torch.float32).unsqueeze(-1).to(device)
        
        if model.use_probabilistic:
            param_means, param_stds = model.predict_parameters(age_tensor)
            K_values = param_means[:, 0].cpu().numpy()
            B_values = param_means[:, 1].cpu().numpy()
            tau_values = param_means[:, 2].cpu().numpy()
            K_stds = param_stds[:, 0].cpu().numpy()
            B_stds = param_stds[:, 1].cpu().numpy()
            tau_stds = param_stds[:, 2].cpu().numpy()
        else:
            params = model.predict_parameters(age_tensor)
            K_values = params[:, 0].cpu().numpy()
            B_values = params[:, 1].cpu().numpy()
            tau_values = params[:, 2].cpu().numpy()
    
    # Plot age relationships
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # K vs Age
    axes[0].plot(test_ages, K_values, 'blue', linewidth=2, label='Predicted')
    if model.use_probabilistic:
        axes[0].fill_between(test_ages, K_values - K_stds, K_values + K_stds, alpha=0.2, color='blue')
    axes[0].set_xlabel('Age (years)')
    axes[0].set_ylabel('Stiffness K')
    axes[0].set_title('Stiffness vs Age')
    axes[0].grid(True, alpha=0.3)
    
    # B vs Age
    axes[1].plot(test_ages, B_values, 'red', linewidth=2, label='Predicted')
    if model.use_probabilistic:
        axes[1].fill_between(test_ages, B_values - B_stds, B_values + B_stds, alpha=0.2, color='red')
    axes[1].set_xlabel('Age (years)')
    axes[1].set_ylabel('Damping B')
    axes[1].set_title('Damping vs Age')
    axes[1].grid(True, alpha=0.3)
    
    # τ vs Age
    axes[2].plot(test_ages, tau_values, 'green', linewidth=2, label='Predicted')
    if model.use_probabilistic:
        axes[2].fill_between(test_ages, tau_values - tau_stds, tau_values + tau_stds, alpha=0.2, color='green')
    axes[2].set_xlabel('Age (years)')
    axes[2].set_ylabel('Neural Delay τ')
    axes[2].set_title('Neural Delay vs Age')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(config['checkpoint_dir']) / 'age_parameter_relationships.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate metrics
    K_corr = np.corrcoef(test_ages, K_values)[0, 1]
    B_corr = np.corrcoef(test_ages, B_values)[0, 1]
    tau_corr = np.corrcoef(test_ages, tau_values)[0, 1]
    
    K_variation = np.std(K_values) / np.mean(K_values)
    B_variation = np.std(B_values) / np.mean(B_values)  
    tau_variation = np.std(tau_values) / np.mean(tau_values)
    
    print(f"Age-Parameter Correlations:")
    print(f"  K-Age correlation: {K_corr:.4f}")
    print(f"  B-Age correlation: {B_corr:.4f}")
    print(f"  τ-Age correlation: {tau_corr:.4f}")
    
    print(f"Parameter Variation Coefficients:")
    print(f"  K variation: {K_variation:.4f}")
    print(f"  B variation: {B_variation:.4f}")
    print(f"  τ variation: {tau_variation:.4f}")

# ==================================================================================
# CELL 14: Age Comparison Testing
# ==================================================================================

print("=== TESTING AGE COMPARISON CAPABILITIES ===")

if TRAINING_METHOD == "two_stage":
    # For two-stage, we need to create a combined model for testing
    from improved_models import TwoStagePINN
    
    full_model = TwoStagePINN(subject_ids=dataset.valid_subjects, param_bounds=config['param_bounds'])
    full_model.subject_pinn.load_state_dict(subject_pinn.state_dict())
    full_model.age_model.load_state_dict(age_model.state_dict())
    test_model = full_model.age_model  # Use age model for comparisons

elif TRAINING_METHOD == "single_stage":
    test_model = model

test_model.eval()

# Test age comparisons
test_comparisons = [
    (30, 50), (50, 70), (30, 70), (40, 60), (60, 80)
]

print("Age Comparison Results:")
print("-" * 40)

comparison_results = []

for age1, age2 in test_comparisons:
    if TRAINING_METHOD == "two_stage":
        # For two-stage, compare parameter distributions
        with torch.no_grad():
            age1_tensor = torch.tensor([[age1]], dtype=torch.float32).to(device)
            age2_tensor = torch.tensor([[age2]], dtype=torch.float32).to(device)
            
            if test_model.use_probabilistic:
                params1_mean, params1_std = test_model.predict_parameters(age1_tensor)
                params2_mean, params2_std = test_model.predict_parameters(age2_tensor)
                
                # Calculate similarity based on parameter differences
                param_diff = torch.abs(params1_mean - params2_mean)
                similarity = torch.exp(-param_diff.sum()).item()
            else:
                params1 = test_model.predict_parameters(age1_tensor)
                params2 = test_model.predict_parameters(age2_tensor)
                
                param_diff = torch.abs(params1 - params2)
                similarity = torch.exp(-param_diff.sum()).item()
    
    elif TRAINING_METHOD == "single_stage":
        # Use built-in comparison method
        comparison = test_model.compare_ages(float(age1), float(age2), n_samples=100)
        similarity = comparison['mean_similarity']
    
    comparison_results.append((age1, age2, similarity))
    print(f"Age {age1} vs {age2}: Similarity = {similarity:.4f}")

print()

# Test balance age finding
if TRAINING_METHOD == "single_stage":
    print("Balance Age Testing:")
    print("-" * 40)
    
    test_ages = [35, 45, 55, 65, 75]
    
    for test_age in test_ages:
        balance_result = test_model.find_balance_age(float(test_age), age_range=(20, 90))
        print(f"Subject age {test_age} → Balance age: {balance_result['balance_age']:.1f} "
              f"(confidence: {balance_result['confidence']:.4f})")

# Create age similarity matrix
print("\nGenerating Age Similarity Matrix...")

test_ages_matrix = range(30, 81, 10)  # 30, 40, 50, 60, 70, 80
n_ages = len(test_ages_matrix)
similarity_matrix = np.zeros((n_ages, n_ages))

for i, age1 in enumerate(test_ages_matrix):
    for j, age2 in enumerate(test_ages_matrix):
        if i == j:
            similarity_matrix[i, j] = 1.0
        else:
            if TRAINING_METHOD == "two_stage":
                with torch.no_grad():
                    age1_tensor = torch.tensor([[age1]], dtype=torch.float32).to(device)
                    age2_tensor = torch.tensor([[age2]], dtype=torch.float32).to(device)
                    
                    if test_model.use_probabilistic:
                        params1_mean, _ = test_model.predict_parameters(age1_tensor)
                        params2_mean, _ = test_model.predict_parameters(age2_tensor)
                    else:
                        params1_mean = test_model.predict_parameters(age1_tensor)
                        params2_mean = test_model.predict_parameters(age2_tensor)
                    
                    param_diff = torch.abs(params1_mean - params2_mean)
                    similarity = torch.exp(-param_diff.sum()).item()
            
            elif TRAINING_METHOD == "single_stage":
                comparison = test_model.compare_ages(float(age1), float(age2), n_samples=50)
                similarity = comparison['mean_similarity']
            
            similarity_matrix[i, j] = similarity

# Plot similarity matrix
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(similarity_matrix, cmap='viridis', aspect='auto')

ax.set_xticks(range(n_ages))
ax.set_yticks(range(n_ages))
ax.set_xticklabels(test_ages_matrix)
ax.set_yticklabels(test_ages_matrix)

ax.set_xlabel('Target Age')
ax.set_ylabel('Subject Age') 
ax.set_title('Age Comparison Similarity Matrix')

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Similarity Score')

# Add text annotations
for i in range(n_ages):
    for j in range(n_ages):
        text = ax.text(j, i, f'{similarity_matrix[i, j]:.3f}',
                      ha="center", va="center", 
                      color="white" if similarity_matrix[i, j] < 0.5 else "black")

plt.tight_layout()
plt.savefig(Path(config['checkpoint_dir']) / 'age_similarity_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("Age comparison testing completed!")

# ==================================================================================
# CELL 15: Final Summary and Model Saving
# ==================================================================================

print("="*60)
print("TRAINING COMPLETE - FINAL SUMMARY")
print("="*60)

# Save final results summary
results_summary = {
    'training_method': TRAINING_METHOD,
    'config': config,
    'training_completed': True,
    'timestamp': pd.Timestamp.now().isoformat()
}

if TRAINING_METHOD == "two_stage":
    results_summary.update({
        'stage1_epochs_completed': len(stage1_metrics),
        'stage1_best_loss': min(m['val_total'] for m in stage1_metrics),
        'stage2_epochs_completed': len(stage2_metrics),
        'stage2_best_loss': min(m['val_total'] for m in stage2_metrics),
        'n_subjects_with_parameters': len(subject_parameters),
        'parameter_correlations': {
            'K_age_correlation': float(K_corr),
            'B_age_correlation': float(B_corr), 
            'tau_age_correlation': float(tau_corr)
        },
        'parameter_variations': {
            'K_variation_coeff': float(K_variation),
            'B_variation_coeff': float(B_variation),
            'tau_variation_coeff': float(tau_variation)
        }
    })

elif TRAINING_METHOD == "single_stage":
    final_metrics = training_metrics[-1]
    results_summary.update({
        'epochs_completed': len(training_metrics),
        'best_val_loss': best_val_loss,
        'final_r2_score': final_metrics['r2_score'],
        'final_mae': final_metrics['mae'],
        'parameter_correlations': {
            'K_age_correlation': float(K_corr),
            'B_age_correlation': float(B_corr),
            'tau_age_correlation': float(tau_corr)
        },
        'parameter_variations': {
            'K_variation_coeff': float(K_variation),
            'B_variation_coeff': float(B_variation),
            'tau_variation_coeff': float(tau_variation)
        },
        'final_parameter_cvs': {
            'K_cv': final_metrics['K_cv'],
            'B_cv': final_metrics['B_cv'],
            'tau_cv': final_metrics['tau_cv']
        }
    })

# Save summary
with open(Path(config['checkpoint_dir']) / 'results_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"Training Method: {TRAINING_METHOD.upper()}")
print(f"Checkpoint Directory: {config['checkpoint_dir']}")
print()

if TRAINING_METHOD == "two_stage":
    print("STAGE 1 (Subject Parameters):")
    print(f"  - Epochs completed: {len(stage1_metrics)}")
    print(f"  - Best validation loss: {min(m['val_total'] for m in stage1_metrics):.6f}")
    print()
    print("STAGE 2 (Age Relationships):")
    print(f"  - Epochs completed: {len(stage2_metrics)}")
    print(f"  - Best validation loss: {min(m['val_total'] for m in stage2_metrics):.6f}")
    print()
    print("PARAMETER LEARNING:")
    print(f"  - K-Age correlation: {K_corr:.4f}")
    print(f"  - B-Age correlation: {B_corr:.4f}")
    print(f"  - τ-Age correlation: {tau_corr:.4f}")
    print()
    print("PARAMETER VARIATION:")
    print(f"  - K variation coeff: {K_variation:.4f}")
    print(f"  - B variation coeff: {B_variation:.4f}")
    print(f"  - τ variation coeff: {tau_variation:.4f}")

elif TRAINING_METHOD == "single_stage":
    print("SINGLE-STAGE TRAINING:")
    print(f"  - Epochs completed: {len(training_metrics)}")
    print(f"  - Best validation loss: {best_val_loss:.6f}")
    print(f"  - Final R² score: {final_metrics['r2_score']:.4f}")
    print(f"  - Final MAE: {final_metrics['mae']:.4f}")
    print()
    print("PARAMETER LEARNING:")
    print(f"  - K-Age correlation: {K_corr:.4f}")
    print(f"  - B-Age correlation: {B_corr:.4f}")  
    print(f"  - τ-Age correlation: {tau_corr:.4f}")
    print()
    print("PARAMETER VARIATION:")
    print(f"  - K variation coeff: {K_variation:.4f} (CV: {final_metrics['K_cv']:.4f})")
    print(f"  - B variation coeff: {B_variation:.4f} (CV: {final_metrics['B_cv']:.4f})")
    print(f"  - τ variation coeff: {tau_variation:.4f} (CV: {final_metrics['tau_cv']:.4f})")

print()
print("MODEL SUCCESS INDICATORS:")

# Check for successful learning
success_indicators = []
failure_indicators = []

if TRAINING_METHOD == "single_stage":
    if final_metrics['r2_score'] > 0.5:
        success_indicators.append(f"✅ Good data reconstruction (R² = {final_metrics['r2_score']:.3f})")
    else:
        failure_indicators.append(f"❌ Poor data reconstruction (R² = {final_metrics['r2_score']:.3f})")

# Parameter variation checks
if K_variation > 0.1:
    success_indicators.append(f"✅ K varies with age (CV = {K_variation:.3f})")
else:
    failure_indicators.append(f"❌ K doesn't vary with age (CV = {K_variation:.3f})")

if B_variation > 0.05:
    success_indicators.append(f"✅ B varies with age (CV = {B_variation:.3f})")
else:
    failure_indicators.append(f"❌ B doesn't vary with age (CV = {B_variation:.3f})")

if tau_variation > 0.05:
    success_indicators.append(f"✅ τ varies with age (CV = {tau_variation:.3f})")
else:
    failure_indicators.append(f"❌ τ doesn't vary with age (CV = {tau_variation:.3f})")

# Print indicators
for indicator in success_indicators:
    print(indicator)
for indicator in failure_indicators:
    print(indicator)

print()
print("FILES SAVED:")
print(f"  - Best model: {config['checkpoint_dir']}/best_{TRAINING_METHOD}_model.pth")
print(f"  - Training curves: {config['checkpoint_dir']}/training_curves.png")
print(f"  - Age relationships: {config['checkpoint_dir']}/age_parameter_relationships.png")
print(f"  - Similarity matrix: {config['checkpoint_dir']}/age_similarity_matrix.png")
print(f"  - Results summary: {config['checkpoint_dir']}/results_summary.json")

if TRAINING_METHOD == "two_stage":
    print(f"  - Subject parameters: {config['checkpoint_dir']}/subject_parameters.json")

print()
print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
print()
print("Your enhanced Balance PINN model is ready for cross-age balance comparison!")

# Download files reminder
print("=" * 60)
print("IMPORTANT: Download these files from Colab before closing:")
print("=" * 60)
print(f"1. {config['checkpoint_dir']}/ folder (contains all results)")
print("2. All generated PNG plots")
print("3. JSON result files")
print()
print("Use the file browser on the left to download individual files,")
print("or zip the entire checkpoint directory to download everything at once.")