#!/usr/bin/env python3
"""
MAT File Inspector

This script loads a single .mat file and provides a detailed inspection of its
contents. It is designed to recursively "unwrap" the nested array structures
often found in .mat files loaded via scipy, particularly those that appear
with a shape of (1, 1) but contain the actual data.

The unwrapping logic is based on the data handling in 'process.py'.
"""

import scipy.io
import numpy as np
import sys
from pathlib import Path

def unwrap_item(item: np.ndarray):
    """
    Recursively unwraps a nested numpy array structure to get to the payload.
    
    Args:
        item: The numpy object to unwrap.

    Returns:
        The unwrapped data, which could be a numpy array or another type.
    """
    payload = item
    # Keep unwrapping as long as we have a numpy array of size 1
    while isinstance(payload, np.ndarray) and payload.size == 1:
        try:
            payload = payload.item()
        except ValueError:
            # This can happen if the item is an array of objects
            break

    # After unwrapping, the payload might be a tuple of arrays
    if isinstance(payload, tuple) and len(payload) > 0:
        # Recursively unwrap each item in the tuple
        return [unwrap_item(p) for p in payload]
    
    return payload

def inspect_mat_file(file_path: Path):
    """
    Loads a .mat file and prints its unwrapped structure.

    Args:
        file_path (Path): The path to the .mat file to inspect.
    """
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return

    print(f"--- Inspecting: {file_path.name} ---")
    try:
        # squeeze_me=False is important to preserve the original structure for inspection
        data = scipy.io.loadmat(str(file_path), squeeze_me=False)
        print(f"Successfully loaded. Found {len(data)} top-level keys.")
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    for key, value in data.items():
        # Skip MATLAB's internal header/version keys
        if key.startswith('__'):
            continue
        
        print(f"\n[KEY] '{key}'")
        print(f"  |-- Original Shape: {value.shape}, Original Dtype: {value.dtype}")
        
        unwrapped_value = unwrap_item(value)
        
        if unwrapped_value is value:
            print("  |-- No unwrapping needed.")
        elif isinstance(unwrapped_value, list):
             print(f"  |-- Unwrapped to a list of {len(unwrapped_value)} items:")
             for i, item in enumerate(unwrapped_value):
                 if isinstance(item, np.ndarray):
                     print(f"    |-- Item {i}: shape={item.shape}, dtype={item.dtype}")
                 else:
                     print(f"    |-- Item {i}: type={type(item)}")
        elif isinstance(unwrapped_value, np.ndarray):
            print(f"  |-- Unwrapped Shape: {unwrapped_value.shape}, Unwrapped Dtype: {unwrapped_value.dtype}")
        else:
            print(f"  |-- Unwrapped Type: {type(unwrapped_value)}")


def main():
    """Main execution function."""
    if len(sys.argv) > 1:
        # Use the file provided by the user
        file_path = Path(sys.argv[1])
        inspect_mat_file(file_path)
    else:
        # If no file is provided, find and inspect the first .mat file in the 'data' directory
        print("Usage: python inspect_mat.py <path_to_mat_file>")
        print("No file provided. Attempting to inspect the first .mat file found in 'data/'...")
        
        data_dir = Path("data")
        if not data_dir.exists():
            print("Error: 'data' directory not found.")
            return
            
        first_file = next(data_dir.glob("*.mat"), None)
        
        if first_file:
            inspect_mat_file(first_file)
        else:
            print("No .mat files found in 'data/'.")

if __name__ == "__main__":
    main()
