#!/usr/bin/env python3
"""
Simple test to directly access the MATLAB data like we did before
"""

import scipy.io
import numpy as np
from pathlib import Path

def test_direct_access():
    """Test direct access to match what we saw earlier"""
    
    # Get first .mat file
    data_folder = Path("data")
    mat_files = list(data_folder.glob("*.mat"))
    
    if not mat_files:
        print("No .mat files found!")
        return
        
    mat_file = mat_files[0]
    print(f"Testing: {mat_file.name}")
    
    # Load data
    data = scipy.io.loadmat(str(mat_file))
    
    print(f"\nKeys in file: {list(data.keys())}")
    
    # Check TrajX structure
    if 'TrajX' in data:
        trajx = data['TrajX']
        print(f"\nTrajX shape: {trajx.shape}")
        print(f"TrajX type: {type(trajx)}")
        
        if trajx.shape == (1, 1):
            trajx_nested = trajx[0][0]
            print(f"TrajX[0][0] type: {type(trajx_nested)}")
            print(f"TrajX[0][0] length: {len(trajx_nested)}")
            
            # Try to access first trial like we did before
            print(f"\nTrying to access first trial...")
            try:
                first_trial = trajx_nested[0]
                print(f"First trial type: {type(first_trial)}")
                
                if hasattr(first_trial, '__len__'):
                    print(f"First trial length: {len(first_trial)}")
                    
                    # Try to get first array in trial
                    if len(first_trial) > 0:
                        first_array = first_trial[0]
                        print(f"First array type: {type(first_array)}")
                        if hasattr(first_array, 'shape'):
                            print(f"First array shape: {first_array.shape}")
                            print(f"First array size: {first_array.size}")
                            if first_array.size > 0:
                                print(f"First few values: {first_array.flatten()[:5]}")
                        
                        # Also show what's in sample_values from inspector output
                        print(f"\nFull first trial (limited):")
                        if len(first_trial) <= 5:
                            for i, arr in enumerate(first_trial):
                                if hasattr(arr, 'shape'):
                                    print(f"  Array {i}: shape={arr.shape}, first_val={arr.flatten()[0] if arr.size > 0 else 'empty'}")
                        else:
                            print(f"  Trial contains {len(first_trial)} arrays")
                            
            except Exception as e:
                print(f"Error accessing first trial: {e}")
                
                # Try alternative access pattern
                print("\nTrying alternative access...")
                try:
                    # Maybe the structure is different
                    print(f"TrajX[0][0] first element: {trajx_nested[0]}")
                    if hasattr(trajx_nested[0], 'shape'):
                        print(f"Direct shape: {trajx_nested[0].shape}")
                except Exception as e2:
                    print(f"Alternative access failed: {e2}")

if __name__ == "__main__":
    test_direct_access()
