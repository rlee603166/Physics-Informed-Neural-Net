#!/usr/bin/env python3
"""
End-to-End Functional Balance Age Analyzer (Fast, Corrected Version)

This script implements a fast, data-driven pipeline to create a "Functional Balance Age"
based on raw time-series sway data. It bypasses the slow physics-informed models
and instead computes a "Sway Score" directly from the data.

This corrected version includes feature scaling to prevent model collapse and a more
robust functional age lookup.

The pipeline is as follows:
1.  **Create Sway Score**: Calculates the total distance the Center of Pressure (COP)
travels for each subject and converts it to an intuitive 1-10 score.
2.  **Time-Series Feature Engineering**: Calculates advanced features (frequency
power, etc.) from the raw sway data to create a "fingerprint" of each trial.
3.  **Feature Analysis**: Analyzes the correlation of the engineered features with the
Sway Score to ensure they are informative.
4.  **Train a Predictive Model**: Trains a regression model on SCALED features to
predict the Sway Score.
5.  **Generate Functional Age Table**: Uses the trained model to create a lookup
table that maps a person's Sway Score to a "Functional Balance Age."
"""

# --- Core Imports ---
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import logging
import h5py
from tqdm import tqdm
from typing import Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# --- Analysis-specific Imports ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.fft import rfft, rfftfreq

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Configuration ---
PROCESSED_DATA_FOLDER = Path('processed_data')
AGE_CSV_PATH = Path('user_ages.csv')


def create_sway_score(data_folder: Path, age_csv_path: Path) -> Optional[pd.DataFrame]:
    """
    Generates a 'Sway Score' based on the total distance of COP movement.
    """
    logger.info("--- Step 1: Creating Data-Driven 'Sway Score' ---")
    
    try:
        age_df = pd.read_csv(age_csv_path)
        age_df['user_id'] = age_df['user_id'].astype(str)
        age_lookup = {row.user_id: row.age for _, row in age_df.iterrows()}
        logger.info(f"Loaded age data for {len(age_lookup)} subjects.")
    except FileNotFoundError:
        logger.error(f"FATAL: Age file not found at '{age_csv_path}'.")
        return None

    batch_files = sorted(list(data_folder.glob("*.h5")))
    if not batch_files:
        logger.error(f"No HDF5 batch files found in '{data_folder}'.")
        return None

    subject_sway_data = []
    for h5_file in tqdm(batch_files, desc="Calculating Sway Distances"):
        with h5py.File(h5_file, 'r') as f:
            for subject_key in f.keys():
                user_id = subject_key.replace('subject_', '')
                if user_id in age_lookup:
                    total_distances = []
                    subject_group = f[subject_key]
                    for trial_key in subject_group.keys():
                        cop_x = subject_group[trial_key]['cop_x'][:]
                        cop_y = subject_group[trial_key]['cop_y'][:]
                        dist = np.sum(np.sqrt(np.diff(cop_x)**2 + np.diff(cop_y)**2))
                        total_distances.append(dist)
                    
                    if total_distances:
                        avg_sway_dist = np.mean(total_distances)
                        subject_sway_data.append({
                            'subject_id': user_id,
                            'age': age_lookup[user_id],
                            'avg_sway_distance': avg_sway_dist
                        })

    if not subject_sway_data:
        logger.error("Could not process sway data for any subjects.")
        return None

    df = pd.DataFrame(subject_sway_data)
    
    inverse_sway = 1 / (df['avg_sway_distance'].values.reshape(-1, 1) + 1e-6)
    scaler = MinMaxScaler(feature_range=(1, 10))
    df['sway_score'] = scaler.fit_transform(inverse_sway)

    logger.info("Successfully created 'sway_score'.")
    
    plt.figure(figsize=(10, 6))
    sns.regplot(data=df, x='age', y='sway_score', line_kws={"color": "red"})
    plt.title('Data-Driven Sway Score vs. Chronological Age')
    plt.xlabel('Chronological Age (years)')
    plt.ylabel('Functional Sway Score (1-10)')
    plt.grid(True)
    plt.savefig('sway_score_vs_age.png')
    logger.info(f"Saved plot 'sway_score_vs_age.png' to show the relationship.")
    
    return df


def calculate_timeseries_features(ts: np.ndarray, sampling_rate: float = 106.0) -> Dict:
    """
    Calculates frequency-domain features for a single time-series.
    """
    n = len(ts)
    if n < 10:
        return {}
    try:
        yf = rfft(ts)
        xf = rfftfreq(n, 1 / sampling_rate)
        power = np.abs(yf)**2
    except Exception as e:
        logger.warning(f"FFT calculation failed: {e}")
        return {}
    
    low_freq_mask = (xf >= 0.1) & (xf < 0.5)
    mid_freq_mask = (xf >= 0.5) & (xf < 2.0)
    high_freq_mask = (xf >= 2.0) & (xf < 10.0)
    
    total_power = np.sum(power)
    if total_power < 1e-6:
        return {}

    return {
        'power_low_freq': np.sum(power[low_freq_mask]) / total_power,
        'power_mid_freq': np.sum(power[mid_freq_mask]) / total_power,
        'power_high_freq': np.sum(power[high_freq_mask]) / total_power,
        'dominant_freq': xf[np.argmax(power)] if power.any() else 0,
    }


def engineer_timeseries_features(df: pd.DataFrame, data_folder: Path) -> pd.DataFrame:
    """
    Loads raw time-series data and engineers features for each subject.
    """
    logger.info("--- Step 2: Engineering Time-Series Features ---")
    
    batch_files = sorted(list(data_folder.glob("*.h5")))
    if not batch_files:
        logger.error(f"No HDF5 batch files found in '{data_folder}'.")
        return df

    all_features = []
    for subject_id in tqdm(df['subject_id'], desc="Engineering Features"):
        trial_features_list = []
        found_subject = False
        for h5_file in batch_files:
            try:
                with h5py.File(h5_file, 'r') as f:
                    subject_group_key = f"subject_{subject_id}"
                    if subject_group_key in f:
                        found_subject = True
                        subject_group = f[subject_group_key]
                        for trial_key in subject_group.keys():
                            cop_x = subject_group[trial_key]['cop_x'][:]
                            cop_y = subject_group[trial_key]['cop_y'][:]
                            
                            features_x = calculate_timeseries_features(cop_x)
                            features_y = calculate_timeseries_features(cop_y)
                            
                            combined_features = {f"x_{k}": v for k, v in features_x.items()}
                            combined_features.update({f"y_{k}": v for k, v in features_y.items()})
                            if combined_features:
                                trial_features_list.append(combined_features)
                        break
            except Exception as e:
                logger.error(f"Error processing file {h5_file.name} for subject {subject_id}: {e}")
        
        if not found_subject:
            logger.warning(f"Could not find time-series data for subject {subject_id}")
            all_features.append({})
            continue

        if trial_features_list:
            subject_avg_features = pd.DataFrame(trial_features_list).mean().to_dict()
            all_features.append(subject_avg_features)
        else:
            all_features.append({})

    feature_df = pd.DataFrame(all_features, index=df.index)
    
    final_df = df.join(feature_df)
    final_df = final_df.dropna()
    
    logger.info(f"Successfully engineered features for {len(final_df)} subjects.")
    return final_df

def analyze_feature_correlation(df: pd.DataFrame):
    """Analyzes and visualizes the correlation of features with the sway score."""
    logger.info("--- Step 3: Analyzing Feature Correlation ---")
    
    feature_cols = [col for col in df.columns if 'freq' in col or 'power' in col]
    if not feature_cols:
        logger.warning("No feature columns found to analyze for correlation.")
        return

    correlation_matrix = df[feature_cols + ['sway_score']].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='vlag', fmt='.2f', linewidths=.5)
    plt.title('Feature Correlation with Sway Score')
    plt.tight_layout()
    plt.savefig('feature_correlation.png')
    logger.info("Saved feature correlation heatmap to 'feature_correlation.png'")
    
    print("\n" + "-"*25)
    print("Correlation of features with sway_score:")
    print(correlation_matrix['sway_score'].sort_values(ascending=False))
    print("-"*25 + "\n")

class FeatureDataset(Dataset):
    """PyTorch Dataset for the feature data."""
    def __init__(self, features, targets):
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.targets = torch.tensor(targets.values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class ScorePredictor(nn.Module):
    """A simple MLP to predict the sway score from engineered features."""
    def __init__(self, n_features):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layer_stack(x)

def train_score_model(df: pd.DataFrame) -> Optional[ScorePredictor]:
    """Trains the model to predict sway_score from features."""
    logger.info("--- Step 4: Training Model to Predict Sway Score ---")
    
    feature_cols = [col for col in df.columns if col not in ['subject_id', 'age', 'avg_sway_distance', 'sway_score']]
    target_col = 'sway_score'
    
    if not feature_cols:
        logger.error("No feature columns found for training. Stopping.")
        return None

    logger.info(f"Training with {len(feature_cols)} features.")

    X = df[feature_cols]
    y = df[target_col]

    # --- FIX: Scale the features before training ---
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    # ---

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = FeatureDataset(X_train, y_train)
    val_dataset = FeatureDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = ScorePredictor(n_features=len(feature_cols)).to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    num_epochs = 100
    logger.info(f"Starting training for {num_epochs} epochs on {DEVICE}...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for features, targets in train_loader:
            features, targets = features.to(DEVICE), targets.to(DEVICE)
            preds = model(features)
            loss = loss_fn(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(DEVICE), targets.to(DEVICE)
                preds = model(features)
                val_loss += loss_fn(preds, targets).item()
        
        avg_val_loss = val_loss / len(val_loader)

        if (epoch + 1) % 20 == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
    logger.info("Model training complete.")
    return model

def generate_functional_age_table(df: pd.DataFrame, model: ScorePredictor):
    """
    Generates and displays the final Functional Age comparison table.
    """
    logger.info("--- Step 5: Generating Functional Age Comparison ---")

    # Create a lookup table from age to the TRUE sway score
    age_to_score_lookup = df.groupby('age')['sway_score'].mean().reset_index()

    # This function now finds the age in the lookup table whose average score is closest to the predicted score.
    # This is more robust than interpolation.
    def get_functional_age(predicted_score):
        closest_age_index = (age_to_score_lookup['sway_score'] - predicted_score).abs().idxmin()
        return age_to_score_lookup.loc[closest_age_index, 'age']

    logger.info("\n--- Example Patient Analysis ---")
    feature_cols = [col for col in df.columns if 'freq' in col or 'power' in col]
    X = df[feature_cols]

    # Must apply the same scaling as during training
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    model.eval()
    with torch.no_grad():
        features_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
        predicted_scores = model(features_tensor).cpu().numpy().flatten()

    df['predicted_sway_score'] = predicted_scores
    df['functional_age'] = df['predicted_sway_score'].apply(get_functional_age)

    example_df = df.sample(n=min(10, len(df)), random_state=42)

    print("\n{:<12} | {:<12} | {:<18} | {:<20}".format("Subject ID", "Actual Age", "Predicted Score", "Functional Balance Age"))
    print("-" * 70)
    for _, row in example_df.iterrows():
        print("{:<12} | {:<12.0f} | {:<18.2f} | {:<20.0f}".format(
            row['subject_id'],
            row['age'],
            row['predicted_sway_score'],
            row['functional_age']
        ))
    
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data=df, x='age', y='functional_age', hue='sway_score', palette='viridis_r', alpha=0.7, s=80)
    plt.plot([df['age'].min(), df['age'].max()], [df['age'].min(), df['age'].max()], 'r--', linewidth=2, label='Ideal (Func. Age = Chron. Age)')
    plt.title('Functional Balance Age vs. Chronological Age', fontsize=16)
    plt.xlabel('Chronological Age (years)', fontsize=12)
    plt.ylabel('Predicted Functional Balance Age (years)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig('functional_vs_chronological_age.png')
    logger.info("\nSaved final comparison plot to 'functional_vs_chronological_age.png'")

def main():
    """Main execution pipeline for the analyzer."""
    logger.info("--- Starting End-to-End Functional Balance Age Analyzer (Fast Version) ---")
    
    scored_df = create_sway_score(PROCESSED_DATA_FOLDER, AGE_CSV_PATH)
    if scored_df is None:
        return
        
    featured_df = engineer_timeseries_features(scored_df, PROCESSED_DATA_FOLDER)
    if featured_df.empty:
        logger.error("Stopping: No subjects or features available for model training.")
        return

    analyze_feature_correlation(featured_df)

    score_model = train_score_model(featured_df)
    if score_model is None:
        logger.error("Stopping: Model training failed.")
        return

    generate_functional_age_table(featured_df, score_model)

    logger.info("--- End-to-End Analysis Complete ---")

if __name__ == "__main__":
    main()
