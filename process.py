#!/usr/bin/env python3
"""
Refactored Data Processor for Balance PINN Training
- Reads all .mat files for subjects listed in an age CSV.
- Splits the subjects into two batches.
- Saves each batch to a separate HDF5 file (e.g., batch_0.h5, batch_1.h5).
"""

import scipy.io
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import h5py
import re
from tqdm import tqdm
from typing import Dict, List, Optional
import gc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BalanceDataProcessor:
    """Processes all valid .mat files and saves them into two HDF5 batch files."""
    
    def __init__(self, data_folder: str, age_csv_path: str, output_folder: str, num_batches: int = 2, sampling_rate: float = 106.0):
        """
        Initializes the processor.

        Args:
            data_folder (str): Path to folder containing raw .mat files.
            age_csv_path (str): Path to the user_ages.csv file.
            output_folder (str): Path to save the output HDF5 batch files.
            num_batches (int): The number of batches to split the data into.
            sampling_rate (float): The sampling rate of the data in Hz.
        """
        self.data_folder = Path(data_folder)
        self.output_folder = Path(output_folder)
        self.num_batches = num_batches
        self.sampling_rate = sampling_rate

        # Create output directory
        self.output_folder.mkdir(exist_ok=True, parents=True)

        # 1. Load age data and create a lookup dictionary
        try:
            age_df = pd.read_csv(age_csv_path)
            self.age_lookup = pd.Series(age_df.age.values, index=age_df.user_id).to_dict()
            self.valid_user_ids = set(self.age_lookup.keys())
            logger.info(f"Loaded age data for {len(self.valid_user_ids)} subjects from '{age_csv_path}'.")
        except FileNotFoundError:
            logger.error(f"Age file not found at '{age_csv_path}'. Cannot proceed.")
            raise
            
        # 2. Find and filter .mat files based on valid user_ids
        all_mat_files = list(self.data_folder.glob("*.mat"))
        self.mat_files_to_process = self._filter_mat_files(all_mat_files)
        
        if not self.mat_files_to_process:
            raise ValueError(f"No matching .mat files found in '{data_folder}' for the user_ids in '{age_csv_path}'.")

    def _get_userid_from_filename(self, filename: str) -> Optional[str]:
        """Extracts user_id (e.g., 'C0006') from a filename."""
        match = re.search(r'(C\d+)', filename)
        return match.group(1) if match else None

    def _filter_mat_files(self, all_files: List[Path]) -> List[Path]:
        """Filters the list of .mat files to include only those with a valid user_id in the age file."""
        filtered_files = []
        for file_path in all_files:
            user_id = self._get_userid_from_filename(file_path.name)
            if user_id in self.valid_user_ids:
                filtered_files.append(file_path)
        
        logger.info(f"Found {len(all_files)} total .mat files. "
                    f"Found {len(filtered_files)} matching files to process based on age data.")
        return filtered_files

    def _get_trial_timeseries(self, trial_container: np.ndarray) -> Optional[np.ndarray]:
        """Recursively unnests a trial container to extract the 1D time-series data."""
        try:
            payload = trial_container
            while isinstance(payload, np.ndarray) and payload.size == 1:
                payload = payload.item()

            if isinstance(payload, tuple) and len(payload) > 0:
                final_array = payload[0]
            else:
                final_array = payload
            
            if isinstance(final_array, np.ndarray):
                return final_array.flatten()
            return None
        except Exception:
            return None

    def _extract_subject_data(self, mat_file: Path, user_id: str, age: float) -> Optional[Dict]:
        """Extracts all valid trial data from a single .mat file."""
        try:
            data = scipy.io.loadmat(str(mat_file), squeeze_me=False)
            
            if 'TrajX' not in data or 'TrajY' not in data:
                return None
            
            trajx_data = data['TrajX'][0][0] if data['TrajX'].shape == (1, 1) else data['TrajX']
            trajy_data = data['TrajY'][0][0] if data['TrajY'].shape == (1, 1) else data['TrajY']
            
            if not (hasattr(trajx_data, '__len__') and hasattr(trajy_data, '__len__')):
                return None

            valid_trials = []
            n_trials = min(len(trajx_data), len(trajy_data))
            
            for i in range(n_trials):
                cop_x = self._get_trial_timeseries(trajx_data[i])
                cop_y = self._get_trial_timeseries(trajy_data[i])
                
                if cop_x is not None and cop_y is not None and len(cop_x) > 1000:
                    valid_trials.append({
                        'trial_id': i,
                        'cop_x': cop_x.astype(np.float32),
                        'cop_y': cop_y.astype(np.float32),
                        'n_points': len(cop_x)
                    })
            
            if not valid_trials:
                return None

            return {
                'user_id': user_id,
                'age': age,
                'filename': mat_file.name,
                'trials': valid_trials
            }
        except Exception as e:
            logger.error(f"Failed to process {mat_file.name}: {e}")
            return None

    def run(self):
        """Processes all valid files in batches and saves each batch to a separate HDF5 file."""
        file_batches = np.array_split(self.mat_files_to_process, self.num_batches)
        
        total_subjects_processed = 0
        
        for i, batch_files in enumerate(file_batches):
            batch_data = []
            logger.info(f"--- Processing Batch {i+1}/{self.num_batches} ({len(batch_files)} files) ---")
            
            for mat_file in tqdm(batch_files, desc=f"Batch {i+1}"):
                user_id = self._get_userid_from_filename(mat_file.name)
                age = self.age_lookup.get(user_id)
                
                if user_id and age is not None:
                    subject_data = self._extract_subject_data(mat_file, user_id, age)
                    if subject_data:
                        batch_data.append(subject_data)
            
            if batch_data:
                self._save_batch_data(batch_data, i)
                total_subjects_processed += len(batch_data)
            else:
                logger.warning(f"No valid data found in batch {i+1}. No file will be created.")

            # Clean up memory between batches
            del batch_data
            gc.collect()

        logger.info(f"Processing complete. Processed a total of {total_subjects_processed} subjects into {self.num_batches} files.")

    def _save_batch_data(self, batch_data: List[Dict], batch_idx: int):
        """Saves a single batch of processed data into one HDF5 file."""
        output_file = self.output_folder / f"batch_{batch_idx}.h5"
        
        with h5py.File(output_file, 'w') as f:
            logger.info(f"Saving data for {len(batch_data)} subjects to '{output_file}'...")
            
            f.attrs['sampling_rate'] = self.sampling_rate
            
            for subject_data in batch_data:
                subject_group = f.create_group(f"subject_{subject_data['user_id']}")
                subject_group.attrs['user_id'] = subject_data['user_id']
                subject_group.attrs['age'] = subject_data['age']
                subject_group.attrs['filename'] = subject_data['filename']
                
                for trial_data in subject_data['trials']:
                    trial_group = subject_group.create_group(f"trial_{trial_data['trial_id']:02d}")
                    trial_group.attrs['n_points'] = trial_data['n_points']
                    
                    trial_group.create_dataset('cop_x', data=trial_data['cop_x'], compression='gzip')
                    trial_group.create_dataset('cop_y', data=trial_data['cop_y'], compression='gzip')

        logger.info(f"Successfully saved batch {batch_idx} to '{output_file}'.")


def main():
    """Main execution function."""
    # --- Configuration ---
    DATA_FOLDER = "data"
    AGE_CSV = "user_ages.csv"
    OUTPUT_FOLDER = "processed_data"
    NUM_BATCHES = 2

    try:
        processor = BalanceDataProcessor(
            data_folder=DATA_FOLDER,
            age_csv_path=AGE_CSV,
            output_folder=OUTPUT_FOLDER,
            num_batches=NUM_BATCHES
        )
        processor.run()
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Could not run processor. Please check your configuration and file paths. Error: {e}")

if __name__ == "__main__":
    main()

