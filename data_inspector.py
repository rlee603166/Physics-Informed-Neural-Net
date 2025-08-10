#!/usr/bin/env python3
"""
Balance Data Inspector - Debug what's actually in the .mat files
"""

import scipy.io
import numpy as np
from pathlib import Path
import pandas as pd
from typing import Dict, Any

def inspect_mat_file(mat_file: Path, verbose: bool = True) -> Dict[str, Any]:
    """
    Deep inspection of a .mat file structure
    """
    try:
        data = scipy.io.loadmat(str(mat_file), squeeze_me=False)
        
        results = {
            'filename': mat_file.name,
            'keys': list(data.keys()),
            'trajx_info': {},
            'trajy_info': {},
            'issues': []
        }
        
        if verbose:
            print(f"\n=== Inspecting {mat_file.name} ===")
        
        # Check TrajX structure
        if 'TrajX' in data:
            trajx = data['TrajX']
            results['trajx_info']['shape'] = trajx.shape
            
            if verbose:
                print(f"TrajX shape: {trajx.shape}")
            
            # Navigate nested structure
            if trajx.shape == (1, 1):
                trajx_nested = trajx[0][0]
                results['trajx_info']['nested_type'] = type(trajx_nested).__name__
                results['trajx_info']['nested_length'] = len(trajx_nested) if hasattr(trajx_nested, '__len__') else 'N/A'
                
                if verbose:
                    print(f"TrajX nested type: {type(trajx_nested)}")
                    print(f"TrajX nested length: {len(trajx_nested) if hasattr(trajx_nested, '__len__') else 'N/A'}")
                
                # Inspect each trial/array
                if hasattr(trajx_nested, '__len__'):
                    trial_info = []
                    for i, item in enumerate(trajx_nested):
                        if i >= 10:  # Limit to first 10 for inspection
                            break
                        
                        try:
                            # This loop recursively unnests the data. MATLAB cell arrays often
                            # load as object arrays of shape (1,) or (1,1) containing the
                            # actual data. We use .item() to safely extract the single element.
                            data_payload = item
                            while isinstance(data_payload, np.ndarray) and data_payload.size == 1:
                                data_payload = data_payload.item()

                            # After un-nesting, the payload is often a tuple of arrays.
                            # We'll use the first array in the tuple for length/shape info.
                            if isinstance(data_payload, tuple) and len(data_payload) > 0:
                                first_array = data_payload[0]
                                n_arrays_in_trial = len(data_payload)
                                data_type = "time_series_from_tuple"
                            else:
                                # Fallback if it's not a tuple (e.g., just a single array)
                                first_array = data_payload
                                n_arrays_in_trial = 1
                                data_type = "single_array"

                            # Now, extract info from the actual data array
                            if isinstance(first_array, np.ndarray):
                                shape = first_array.shape
                                length = first_array.size # .size is the total number of elements
                                sample_vals = first_array.flatten()[:5] if first_array.size > 0 else []
                                
                                trial_info.append({
                                    'index': i,
                                    'shape': shape,
                                    'length': length,
                                    'sample_values': np.round(sample_vals, 2).tolist(), # Round for cleaner printing
                                    'type': data_type,
                                    'n_arrays_in_trial': n_arrays_in_trial,
                                    'original_type': type(item).__name__
                                })

                                if verbose and i < 5:
                                    print(f"  Trial {i}: Found tuple with {n_arrays_in_trial} arrays")
                                    print(f"    First array shape: {shape}, length: {length}")
                                    print(f"    Sample values: {np.round(sample_vals, 2).tolist()}")

                            else:
                                raise TypeError(f"Expected a numpy array after un-nesting, but got {type(first_array).__name__}")

                        except Exception as e:
                            trial_info.append({
                                'index': i,
                                'error': str(e),
                                'original_type': type(item).__name__ if hasattr(item, '__class__') else 'unknown'
                            })
                            if verbose:
                                print(f"  Trial {i}: ERROR - {e}")
                    
                    results['trajx_info']['trials'] = trial_info
        
        # Check TrajY structure (You would apply the same un-nesting logic here if needed)
        if 'TrajY' in data:
            trajy = data['TrajY']
            results['trajy_info']['shape'] = trajy.shape
            
            if trajy.shape == (1, 1) and hasattr(trajy[0][0], '__len__'):
                trajy_nested = trajy[0][0]
                results['trajy_info']['nested_length'] = len(trajy_nested)
                
                if verbose:
                    print(f"\nTrajY nested length: {len(trajy_nested)}")
        
        # Check for data quality issues
        if 'trajx_info' in results and 'trials' in results['trajx_info']:
            for trial in results['trajx_info']['trials']:
                if 'length' in trial:
                    if trial['length'] < 100:
                        results['issues'].append(f"Trial {trial['index']} very short: {trial['length']} points")
                    elif trial['length'] < 1000:
                        results['issues'].append(f"Trial {trial['index']} short: {trial['length']} points")
        
        if verbose and results['issues']:
            print(f"\nIssues found:")
            for issue in results['issues']:
                print(f"  - {issue}")
        
        return results
        
    except Exception as e:
        return {
            'filename': mat_file.name,
            'error': str(e),
            'issues': [f"Failed to load file: {e}"]
        }

def batch_inspect_files(data_folder: str, n_files: int = 10) -> pd.DataFrame:
    """
    Inspect multiple files to understand data patterns
    """
    data_path = Path(data_folder)
    mat_files = list(data_path.glob("*.mat"))[:n_files]
    
    all_results = []
    
    for mat_file in mat_files:
        print(f"\n{'='*60}")
        result = inspect_mat_file(mat_file, verbose=True)
        
        # Flatten for DataFrame
        flat_result = {
            'filename': result['filename'],
            'has_trajx': 'TrajX' in result.get('keys', []),
            'has_trajy': 'TrajY' in result.get('keys', []),
            'trajx_nested_length': result.get('trajx_info', {}).get('nested_length', 0),
            'trajy_nested_length': result.get('trajy_info', {}).get('nested_length', 0),
            'n_issues': len(result.get('issues', [])),
            'issues': '; '.join(result.get('issues', [])),
        }
        
        # Add trial length statistics
        if 'trials' in result.get('trajx_info', {}):
            trial_lengths = [t.get('length', 0) for t in result['trajx_info']['trials'] if 'length' in t]
            if trial_lengths:
                flat_result.update({
                    'min_trial_length': min(trial_lengths),
                    'max_trial_length': max(trial_lengths),
                    'avg_trial_length': np.mean(trial_lengths),
                    'n_trials': len(trial_lengths)
                })
        
        all_results.append(flat_result)
    
    return pd.DataFrame(all_results)

def suggest_processing_parameters(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Suggest optimal processing parameters based on inspection
    """
    # Analyze trial lengths
    all_lengths = []
    for _, row in df.iterrows():
        if pd.notna(row.get('min_trial_length')):
            all_lengths.extend([row['min_trial_length'], row['max_trial_length']])
    
    suggestions = {}
    
    if all_lengths:
        all_lengths = [l for l in all_lengths if l > 0]  # Remove zeros
        suggestions['trial_lengths'] = {
            'min': min(all_lengths),
            'max': max(all_lengths), 
            'median': np.median(all_lengths),
            'mean': np.mean(all_lengths)
        }
        
        # Suggest minimum length threshold
        p10 = np.percentile(all_lengths, 10)
        suggestions['recommended_min_length'] = int(p10)
        suggestions['rationale'] = f"10th percentile is {p10:.0f} points. Using this as minimum keeps 90% of trials."
    
    # Check sampling rate estimates
    if 'avg_trial_length' in df.columns:
        avg_lengths = df['avg_trial_length'].dropna()
        if not avg_lengths.empty:
            # Assume 60-second trials
            estimated_rates = avg_lengths / 60
            suggestions['estimated_sampling_rates'] = {
                'min': estimated_rates.min(),
                'max': estimated_rates.max(),
                'median': estimated_rates.median()
            }
    
    return suggestions

def main():
    """Main inspection function"""
    DATA_FOLDER = "data"  # Adjust this path to where your .mat files are
    
    print("🔍 Inspecting Balance Data Files...")
    print("This will help us understand why trials are being rejected as 'too short'")
    
    # Inspect first 10 files in detail
    df = batch_inspect_files(DATA_FOLDER, n_files=10)
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 SUMMARY OF INSPECTION")
    print(f"{'='*60}")
    print(f"Files inspected: {len(df)}")
    print(f"Files with TrajX: {df['has_trajx'].sum()}")
    print(f"Files with TrajY: {df['has_trajy'].sum()}")
    print(f"Files with issues: {df['n_issues'].sum()}")
    
    if 'n_trials' in df.columns:
        print(f"Average trials per file: {df['n_trials'].mean():.1f}")
    
    if 'min_trial_length' in df.columns:
        lengths = df[['min_trial_length', 'max_trial_length', 'avg_trial_length']].describe()
        print(f"\nTrial Length Statistics:")
        print(lengths)
    
    # Get processing suggestions
    suggestions = suggest_processing_parameters(df)
    
    print(f"\n{'='*60}")
    print("💡 PROCESSING RECOMMENDATIONS")
    print(f"{'='*60}")
    
    if 'recommended_min_length' in suggestions:
        print(f"Recommended minimum trial length: {suggestions['recommended_min_length']} points")
        print(f"Rationale: {suggestions['rationale']}")
    
    if 'estimated_sampling_rates' in suggestions:
        rates = suggestions['estimated_sampling_rates']
        print(f"Estimated sampling rates: {rates['min']:.1f} - {rates['max']:.1f} Hz (median: {rates['median']:.1f} Hz)")
    
    # Save detailed results
    df.to_csv('data_inspection_results.csv', index=False)
    print(f"\n💾 Detailed results saved to 'data_inspection_results.csv'")
    
    # Show files with issues
    problem_files = df[df['n_issues'] > 0]
    if not problem_files.empty:
        print(f"\n⚠️  FILES WITH ISSUES ({len(problem_files)}):")
        for _, row in problem_files.iterrows():
            print(f"  {row['filename']}: {row['issues']}")

if __name__ == "__main__":
    main()
