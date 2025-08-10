#!/usr/bin/env python3
"""
Enhanced Dataset Classes for Balance PINN Training

This module provides improved dataset classes that support:
1. Subject-aware data loading for two-stage training
2. Trajectory-based learning vs point-wise learning
3. Memory-efficient data loading with metadata
4. Cross-validation and data splitting utilities
"""

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import h5py
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Iterator
import logging
from collections import defaultdict
import random

logger = logging.getLogger(__name__)

# =============================================================================
# BASE DATASET CLASS
# =============================================================================

class BaseBalanceDataset(Dataset):
    """Base class for balance datasets with common functionality."""
    
    def __init__(self, processed_data_folder: str, age_csv_path: Optional[str] = None):
        self.data_folder = Path(processed_data_folder)
        self.batch_files = sorted(list(self.data_folder.glob("batch_*.h5")))
        
        if not self.batch_files:
            raise FileNotFoundError(f"No HDF5 batch files found in '{self.data_folder}'")
        
        self.sampling_rate = 106.0
        self.age_lookup = {}
        
        # Load age data if provided
        if age_csv_path:
            self._load_age_data(age_csv_path)
        
        # Build dataset index
        self.index_map = []
        self.subject_info = {}
        self._build_index()
    
    def _load_age_data(self, age_csv_path: str):
        """Load age mapping from CSV file."""
        try:
            age_df = pd.read_csv(age_csv_path)
            self.age_lookup = pd.Series(age_df.age.values, index=age_df.user_id).to_dict()
            logger.info(f"Loaded age data for {len(self.age_lookup)} subjects")
        except Exception as e:
            logger.warning(f"Could not load age data: {e}")
            self.age_lookup = {}
    
    def _build_index(self):
        """Build index mapping and collect subject metadata."""
        logger.info(f"Building dataset index from {len(self.batch_files)} batch files...")
        
        subject_trial_counts = defaultdict(int)
        subject_point_counts = defaultdict(int)
        
        for file_path in self.batch_files:
            with h5py.File(file_path, 'r') as f:
                if 'sampling_rate' in f.attrs:
                    self.sampling_rate = f.attrs['sampling_rate']
                
                for subject_key in f.keys():
                    subject_id = subject_key.replace('subject_', '')
                    age = f[subject_key].attrs.get('age', self.age_lookup.get(subject_id, 0))
                    
                    for trial_key in f[subject_key].keys():
                        n_points = f[subject_key][trial_key].attrs['n_points']
                        subject_trial_counts[subject_id] += 1
                        subject_point_counts[subject_id] += n_points
                        
                        for i in range(n_points):
                            self.index_map.append({
                                'file_path': str(file_path),
                                'subject_key': subject_key,
                                'subject_id': subject_id,
                                'trial_key': trial_key,
                                'point_idx': i,
                                'age': age,
                                'total_points': n_points
                            })
        
        # Store subject information
        for subject_id in subject_trial_counts.keys():
            age = self.age_lookup.get(subject_id, 0)
            self.subject_info[subject_id] = {
                'age': age,
                'n_trials': subject_trial_counts[subject_id],
                'n_points': subject_point_counts[subject_id]
            }
        
        logger.info(f"Dataset built: {len(self.index_map):,} points from {len(self.subject_info)} subjects")
    
    def get_subject_ids(self) -> List[str]:
        """Get list of all subject IDs."""
        return list(self.subject_info.keys())
    
    def get_subject_info(self, subject_id: str) -> Dict:
        """Get information for a specific subject."""
        return self.subject_info.get(subject_id, {})
    
    def get_subjects_by_age_range(self, min_age: float, max_age: float) -> List[str]:
        """Get subjects within an age range."""
        subjects = []
        for subject_id, info in self.subject_info.items():
            if min_age <= info.get('age', 0) <= max_age:
                subjects.append(subject_id)
        return subjects
    
    def __len__(self):
        return len(self.index_map)

# =============================================================================
# POINT-WISE DATASET (Enhanced version of original)
# =============================================================================

class EnhancedBalanceDataset(BaseBalanceDataset):
    """
    Enhanced point-wise dataset with subject metadata and improved functionality.
    
    Each item returns a single (time, age, position) tuple with additional metadata.
    """
    
    def __init__(self, processed_data_folder: str, age_csv_path: Optional[str] = None,
                 normalize: bool = False, augment: bool = False):
        super().__init__(processed_data_folder, age_csv_path)
        
        self.normalize = normalize
        self.augment = augment
        
        # Calculate normalization statistics if needed
        if self.normalize:
            self._calculate_stats()
    
    def _calculate_stats(self, sample_size: int = 10000):
        """Calculate dataset statistics for normalization."""
        logger.info("Calculating dataset statistics...")
        
        # Sample random indices
        sample_indices = np.random.choice(len(self.index_map), 
                                        min(sample_size, len(self.index_map)), 
                                        replace=False)
        
        positions = []
        ages = []
        
        for idx in sample_indices:
            item = self.index_map[idx]
            
            with h5py.File(item['file_path'], 'r') as f:
                trial_group = f[item['subject_key']][item['trial_key']]
                x = trial_group['cop_x'][item['point_idx']]
                y = trial_group['cop_y'][item['point_idx']]
                
                positions.append([x, y])
                ages.append(item['age'])
        
        positions = np.array(positions)
        ages = np.array(ages)
        
        self.position_mean = positions.mean(axis=0)
        self.position_std = positions.std(axis=0)
        self.age_mean = ages.mean()
        self.age_std = ages.std()
        
        logger.info(f"Position stats: mean={self.position_mean}, std={self.position_std}")
        logger.info(f"Age stats: mean={self.age_mean:.1f}, std={self.age_std:.1f}")
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Get a single data point with metadata."""
        item = self.index_map[idx]
        
        with h5py.File(item['file_path'], 'r') as f:
            trial_group = f[item['subject_key']][item['trial_key']]
            
            # Load data
            x = trial_group['cop_x'][item['point_idx']]
            y = trial_group['cop_y'][item['point_idx']]
            
            # Calculate time
            t = item['point_idx'] / self.sampling_rate
            age = item['age']
        
        # Apply normalization if enabled
        if self.normalize:
            x = (x - self.position_mean[0]) / self.position_std[0]
            y = (y - self.position_mean[1]) / self.position_std[1] 
            age = (age - self.age_mean) / self.age_std
        
        # Apply augmentation if enabled
        if self.augment and self.training:
            # Add small amount of noise
            noise_scale = 0.01
            x += np.random.normal(0, noise_scale)
            y += np.random.normal(0, noise_scale)
        
        # Convert to tensors
        time_tensor = torch.tensor([t], dtype=torch.float32)
        age_tensor = torch.tensor([age], dtype=torch.float32)
        xy_tensor = torch.tensor([x, y], dtype=torch.float32)
        
        # Metadata
        metadata = {
            'subject_id': item['subject_id'],
            'trial_key': item['trial_key'],
            'point_idx': item['point_idx'],
            'original_age': item['age']
        }
        
        return time_tensor, age_tensor, xy_tensor, metadata

# =============================================================================
# SUBJECT-AWARE DATASET (For two-stage training)
# =============================================================================

class SubjectAwareDataset(BaseBalanceDataset):
    """
    Subject-aware dataset that groups data by subjects.
    
    Returns subject index along with data for two-stage training.
    """
    
    def __init__(self, processed_data_folder: str, age_csv_path: Optional[str] = None,
                 min_points_per_subject: int = 100):
        super().__init__(processed_data_folder, age_csv_path)
        
        self.min_points_per_subject = min_points_per_subject
        
        # Filter subjects with sufficient data
        self._filter_subjects()
        
        # Create subject to index mapping
        self.subject_to_idx = {sid: i for i, sid in enumerate(self.valid_subjects)}
        self.idx_to_subject = {i: sid for sid, i in self.subject_to_idx.items()}
        
        # Rebuild index with only valid subjects
        self._rebuild_index()
        
        logger.info(f"Subject-aware dataset: {len(self.valid_subjects)} subjects, "
                   f"{len(self.index_map):,} points")
    
    def _filter_subjects(self):
        """Filter subjects with minimum data requirements."""
        self.valid_subjects = []
        
        for subject_id, info in self.subject_info.items():
            if info['n_points'] >= self.min_points_per_subject:
                self.valid_subjects.append(subject_id)
        
        logger.info(f"Filtered to {len(self.valid_subjects)} subjects with "
                   f">= {self.min_points_per_subject} points each")
    
    def _rebuild_index(self):
        """Rebuild index with only valid subjects."""
        new_index = []
        
        for item in self.index_map:
            if item['subject_id'] in self.valid_subjects:
                new_index.append(item)
        
        self.index_map = new_index
    
    def get_subject_data(self, subject_id: str) -> List[int]:
        """Get all data point indices for a specific subject."""
        indices = []
        for i, item in enumerate(self.index_map):
            if item['subject_id'] == subject_id:
                indices.append(i)
        return indices
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get data point with subject index."""
        item = self.index_map[idx]
        
        with h5py.File(item['file_path'], 'r') as f:
            trial_group = f[item['subject_key']][item['trial_key']]
            
            x = trial_group['cop_x'][item['point_idx']]
            y = trial_group['cop_y'][item['point_idx']]
            
            t = item['point_idx'] / self.sampling_rate
            age = item['age']
        
        # Get subject index
        subject_idx = self.subject_to_idx[item['subject_id']]
        
        # Convert to tensors
        time_tensor = torch.tensor([t], dtype=torch.float32)
        age_tensor = torch.tensor([age], dtype=torch.float32)
        xy_tensor = torch.tensor([x, y], dtype=torch.float32)
        subject_tensor = torch.tensor([subject_idx], dtype=torch.long)
        
        return time_tensor, age_tensor, xy_tensor, subject_tensor

# =============================================================================
# TRAJECTORY DATASET (For full trajectory learning)
# =============================================================================

class TrajectoryDataset(BaseBalanceDataset):
    """
    Trajectory-based dataset that returns complete time series.
    
    Each item returns a full trajectory (sequence of positions over time)
    instead of individual points.
    """
    
    def __init__(self, processed_data_folder: str, age_csv_path: Optional[str] = None,
                 min_trajectory_length: int = 1000, max_trajectory_length: int = 5000,
                 subsample_rate: int = 1):
        super().__init__(processed_data_folder, age_csv_path)
        
        self.min_trajectory_length = min_trajectory_length
        self.max_trajectory_length = max_trajectory_length
        self.subsample_rate = subsample_rate
        
        # Build trajectory index (one item per trial)
        self._build_trajectory_index()
        
        logger.info(f"Trajectory dataset: {len(self.trajectory_index)} trajectories")
    
    def _build_trajectory_index(self):
        """Build index of complete trajectories."""
        self.trajectory_index = []
        
        # Group by trials
        trials = defaultdict(list)
        for item in self.index_map:
            key = (item['file_path'], item['subject_key'], item['trial_key'])
            trials[key].append(item)
        
        # Filter and store valid trajectories
        for (file_path, subject_key, trial_key), points in trials.items():
            if len(points) >= self.min_trajectory_length:
                # Sort by point index
                points.sort(key=lambda x: x['point_idx'])
                
                # Take first point for metadata
                first_point = points[0]
                
                self.trajectory_index.append({
                    'file_path': file_path,
                    'subject_key': subject_key,
                    'subject_id': first_point['subject_id'],
                    'trial_key': trial_key,
                    'age': first_point['age'],
                    'n_points': len(points)
                })
    
    def __len__(self):
        return len(self.trajectory_index)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a complete trajectory."""
        traj_info = self.trajectory_index[idx]
        
        with h5py.File(traj_info['file_path'], 'r') as f:
            trial_group = f[traj_info['subject_key']][traj_info['trial_key']]
            
            # Load complete trajectory
            cop_x = trial_group['cop_x'][:]
            cop_y = trial_group['cop_y'][:]
        
        # Subsample if needed
        if self.subsample_rate > 1:
            cop_x = cop_x[::self.subsample_rate]
            cop_y = cop_y[::self.subsample_rate]
        
        # Truncate if too long
        if len(cop_x) > self.max_trajectory_length:
            cop_x = cop_x[:self.max_trajectory_length]
            cop_y = cop_y[:self.max_trajectory_length]
        
        # Create time array
        times = np.arange(len(cop_x)) * self.subsample_rate / self.sampling_rate
        
        # Convert to tensors
        times_tensor = torch.tensor(times, dtype=torch.float32).unsqueeze(-1)
        positions_tensor = torch.stack([
            torch.tensor(cop_x, dtype=torch.float32),
            torch.tensor(cop_y, dtype=torch.float32)
        ], dim=-1)
        
        # Age tensor (same for all time points)
        age_tensor = torch.full((len(times), 1), traj_info['age'], dtype=torch.float32)
        
        return times_tensor, age_tensor, positions_tensor

# =============================================================================
# CUSTOM SAMPLERS
# =============================================================================

class SubjectBalancedSampler(Sampler):
    """
    Sampler that ensures balanced sampling across subjects.
    
    Useful for two-stage training to prevent bias toward subjects with more data.
    """
    
    def __init__(self, dataset: SubjectAwareDataset, samples_per_subject: int = 100):
        self.dataset = dataset
        self.samples_per_subject = samples_per_subject
        
        # Group indices by subject
        self.subject_indices = defaultdict(list)
        for i in range(len(dataset)):
            subject_id = dataset.index_map[i]['subject_id']
            self.subject_indices[subject_id].append(i)
    
    def __iter__(self) -> Iterator[int]:
        # Sample from each subject
        all_indices = []
        
        for subject_id, indices in self.subject_indices.items():
            # Sample with replacement if needed
            if len(indices) >= self.samples_per_subject:
                sampled = np.random.choice(indices, self.samples_per_subject, replace=False)
            else:
                sampled = np.random.choice(indices, self.samples_per_subject, replace=True)
            
            all_indices.extend(sampled)
        
        # Shuffle all samples
        np.random.shuffle(all_indices)
        
        return iter(all_indices)
    
    def __len__(self):
        return len(self.subject_indices) * self.samples_per_subject

class AgeStratifiedSampler(Sampler):
    """
    Sampler that ensures balanced sampling across age groups.
    
    Useful for ensuring age representation in training.
    """
    
    def __init__(self, dataset: BaseBalanceDataset, age_bins: List[Tuple[float, float]], 
                 samples_per_bin: int = 100):
        self.dataset = dataset
        self.age_bins = age_bins
        self.samples_per_bin = samples_per_bin
        
        # Group indices by age bins
        self.bin_indices = [[] for _ in age_bins]
        
        for i in range(len(dataset)):
            age = dataset.index_map[i]['age']
            
            for bin_idx, (min_age, max_age) in enumerate(age_bins):
                if min_age <= age <= max_age:
                    self.bin_indices[bin_idx].append(i)
                    break
    
    def __iter__(self) -> Iterator[int]:
        all_indices = []
        
        for bin_indices in self.bin_indices:
            if len(bin_indices) == 0:
                continue
                
            # Sample with replacement if needed
            if len(bin_indices) >= self.samples_per_bin:
                sampled = np.random.choice(bin_indices, self.samples_per_bin, replace=False)
            else:
                sampled = np.random.choice(bin_indices, self.samples_per_bin, replace=True)
            
            all_indices.extend(sampled)
        
        # Shuffle all samples
        np.random.shuffle(all_indices)
        
        return iter(all_indices)
    
    def __len__(self):
        return len([bin_indices for bin_indices in self.bin_indices if len(bin_indices) > 0]) * self.samples_per_bin

# =============================================================================
# DATA SPLITTING UTILITIES
# =============================================================================

def create_subject_splits(dataset: BaseBalanceDataset, 
                         train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15,
                         random_seed: int = 42) -> Dict[str, List[str]]:
    """
    Create train/val/test splits by subjects (not by data points).
    
    This ensures no data leakage between splits.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    subjects = dataset.get_subject_ids()
    random.seed(random_seed)
    random.shuffle(subjects)
    
    n_subjects = len(subjects)
    n_train = int(n_subjects * train_ratio)
    n_val = int(n_subjects * val_ratio)
    
    splits = {
        'train': subjects[:n_train],
        'val': subjects[n_train:n_train + n_val],
        'test': subjects[n_train + n_val:]
    }
    
    logger.info(f"Subject splits: train={len(splits['train'])}, "
               f"val={len(splits['val'])}, test={len(splits['test'])}")
    
    return splits

def create_age_stratified_splits(dataset: BaseBalanceDataset,
                                age_bins: List[Tuple[float, float]],
                                train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15,
                                random_seed: int = 42) -> Dict[str, List[str]]:
    """
    Create train/val/test splits stratified by age groups.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    random.seed(random_seed)
    
    splits = {'train': [], 'val': [], 'test': []}
    
    for min_age, max_age in age_bins:
        subjects_in_bin = dataset.get_subjects_by_age_range(min_age, max_age)
        random.shuffle(subjects_in_bin)
        
        n_subjects = len(subjects_in_bin)
        if n_subjects == 0:
            continue
            
        n_train = int(n_subjects * train_ratio)
        n_val = int(n_subjects * val_ratio)
        
        splits['train'].extend(subjects_in_bin[:n_train])
        splits['val'].extend(subjects_in_bin[n_train:n_train + n_val])
        splits['test'].extend(subjects_in_bin[n_train + n_val:])
    
    logger.info(f"Age-stratified splits: train={len(splits['train'])}, "
               f"val={len(splits['val'])}, test={len(splits['test'])}")
    
    return splits

def create_filtered_dataset(dataset: BaseBalanceDataset, subject_list: List[str]) -> List[int]:
    """Create a filtered dataset with only specified subjects."""
    filtered_indices = []
    
    for i, item in enumerate(dataset.index_map):
        if item['subject_id'] in subject_list:
            filtered_indices.append(i)
    
    return filtered_indices

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_dataset_stats(dataset: BaseBalanceDataset) -> Dict:
    """Get comprehensive dataset statistics."""
    ages = [info['age'] for info in dataset.subject_info.values()]
    n_points_per_subject = [info['n_points'] for info in dataset.subject_info.values()]
    
    stats = {
        'n_subjects': len(dataset.subject_info),
        'n_total_points': len(dataset.index_map),
        'age_stats': {
            'mean': np.mean(ages),
            'std': np.std(ages),
            'min': np.min(ages),
            'max': np.max(ages),
            'median': np.median(ages)
        },
        'points_per_subject_stats': {
            'mean': np.mean(n_points_per_subject),
            'std': np.std(n_points_per_subject),
            'min': np.min(n_points_per_subject),
            'max': np.max(n_points_per_subject),
            'median': np.median(n_points_per_subject)
        },
        'sampling_rate': dataset.sampling_rate
    }
    
    return stats

if __name__ == "__main__":
    # Example usage and testing
    logger.info("Testing enhanced dataset classes...")
    
    # Test enhanced dataset
    try:
        dataset = EnhancedBalanceDataset("processed_data", "user_ages.csv", normalize=True)
        print(f"Enhanced dataset: {len(dataset)} points from {len(dataset.get_subject_ids())} subjects")
        
        # Test data loading
        t, age, xy, metadata = dataset[0]
        print(f"Sample point: t={t.item():.3f}, age={age.item():.1f}, xy=[{xy[0].item():.3f}, {xy[1].item():.3f}]")
        print(f"Metadata: {metadata}")
        
        # Test subject-aware dataset
        subject_dataset = SubjectAwareDataset("processed_data", "user_ages.csv")
        print(f"Subject-aware dataset: {len(subject_dataset)} points from {len(subject_dataset.valid_subjects)} subjects")
        
        # Test data splits
        splits = create_subject_splits(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        print(f"Subject splits: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")
        
        # Test dataset statistics
        stats = get_dataset_stats(dataset)
        print(f"Dataset stats: {stats['n_subjects']} subjects, age range: {stats['age_stats']['min']:.1f}-{stats['age_stats']['max']:.1f}")
        
        logger.info("Dataset tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Dataset test failed: {e}")
        logger.info("This is expected if processed_data folder doesn't exist")