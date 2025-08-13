#!/usr/bin/env python3
"""
Subject Parameter Boundary Analysis Script

Analyzes the distribution of learned parameters to identify how many subjects
are stuck at or near the parameter boundaries.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from scipy import stats

def analyze_boundary_clustering(params_file='subject_parameters.json'):
    """
    Analyzes how many subjects have parameters at or near the boundaries.
    """
    # Load the subject parameters
    if not Path(params_file).exists():
        print(f"❌ Error: '{params_file}' not found!")
        return
    
    with open(params_file, 'r') as f:
        subject_params = json.load(f)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(subject_params.values())
    n_subjects = len(df)
    
    print("="*70)
    print("SUBJECT PARAMETER BOUNDARY ANALYSIS")
    print("="*70)
    print(f"\nTotal subjects analyzed: {n_subjects}")
    
    # Define parameter bounds and thresholds for "near boundary"
    param_bounds = {
        'K': {'min': 500.0, 'max': 3000.0, 'threshold': 100},  # Within 100 of bounds
        'B': {'min': 20.0, 'max': 150.0, 'threshold': 10},    # Within 10 of bounds  
        'tau': {'min': 0.05, 'max': 0.4, 'threshold': 0.02}   # Within 0.02 of bounds
    }
    
    # Analyze each parameter
    boundary_stats = {}
    
    for param_name, bounds in param_bounds.items():
        print(f"\n{'='*50}")
        print(f"Parameter: {param_name}")
        print(f"Bounds: [{bounds['min']}, {bounds['max']}]")
        print(f"Boundary threshold: ±{bounds['threshold']}")
        
        values = df[param_name].values
        
        # Count subjects at or near boundaries
        at_min = np.sum(np.abs(values - bounds['min']) < bounds['threshold'])
        at_max = np.sum(np.abs(values - bounds['max']) < bounds['threshold'])
        at_boundary = at_min + at_max
        
        # Calculate statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        median_val = np.median(values)
        cv = std_val / mean_val if mean_val != 0 else 0
        
        # Store stats
        boundary_stats[param_name] = {
            'at_min': at_min,
            'at_max': at_max,
            'at_boundary': at_boundary,
            'pct_at_boundary': (at_boundary / n_subjects) * 100,
            'mean': mean_val,
            'std': std_val,
            'median': median_val,
            'cv': cv,
            'min_val': np.min(values),
            'max_val': np.max(values),
            'range': np.max(values) - np.min(values)
        }
        
        print(f"\nDistribution Statistics:")
        print(f"  Mean: {mean_val:.2f}")
        print(f"  Std: {std_val:.2f}")
        print(f"  Median: {median_val:.2f}")
        print(f"  CV: {cv:.3f}")
        print(f"  Actual range: [{np.min(values):.3f}, {np.max(values):.3f}]")
        
        print(f"\nBoundary Analysis:")
        print(f"  At/near MIN ({bounds['min']}): {at_min} subjects ({at_min/n_subjects*100:.1f}%)")
        print(f"  At/near MAX ({bounds['max']}): {at_max} subjects ({at_max/n_subjects*100:.1f}%)")
        print(f"  Total at boundaries: {at_boundary} subjects ({at_boundary/n_subjects*100:.1f}%)")
        
        # Warning if too many at boundaries
        if at_boundary / n_subjects > 0.3:
            print(f"  ⚠️ WARNING: {at_boundary/n_subjects*100:.1f}% of subjects at boundaries!")
        elif at_boundary / n_subjects > 0.2:
            print(f"  ⚠️ CAUTION: {at_boundary/n_subjects*100:.1f}% of subjects near boundaries")
        else:
            print(f"  ✅ Good: Only {at_boundary/n_subjects*100:.1f}% at boundaries")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot histograms for each parameter
    for i, (param_name, bounds) in enumerate(param_bounds.items()):
        ax = axes[0, i]
        values = df[param_name].values
        
        # Create histogram
        n_bins = 30
        counts, bins, patches = ax.hist(values, bins=n_bins, edgecolor='black', alpha=0.7)
        
        # Color bars near boundaries in red
        for j, patch in enumerate(patches):
            bin_center = (bins[j] + bins[j+1]) / 2
            if (abs(bin_center - bounds['min']) < bounds['threshold'] or 
                abs(bin_center - bounds['max']) < bounds['threshold']):
                patch.set_facecolor('red')
                patch.set_alpha(0.8)
            else:
                patch.set_facecolor('blue')
                patch.set_alpha(0.6)
        
        # Add boundary lines
        ax.axvline(bounds['min'], color='red', linestyle='--', linewidth=2, label='Min bound')
        ax.axvline(bounds['max'], color='red', linestyle='--', linewidth=2, label='Max bound')
        ax.axvline(bounds['min'] + bounds['threshold'], color='orange', linestyle=':', linewidth=1)
        ax.axvline(bounds['max'] - bounds['threshold'], color='orange', linestyle=':', linewidth=1)
        
        ax.set_title(f'{param_name} Distribution\n({boundary_stats[param_name]["pct_at_boundary"]:.1f}% at boundaries)', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel(f'{param_name} value')
        ax.set_ylabel('Count')
        ax.legend()
        
        # Add text with stats
        stats_text = f"μ={boundary_stats[param_name]['mean']:.1f}\nσ={boundary_stats[param_name]['std']:.1f}"
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot parameter vs age scatter plots
    for i, param_name in enumerate(['K', 'B', 'tau']):
        ax = axes[1, i]
        values = df[param_name].values
        ages = df['age'].values
        
        # Color points based on proximity to boundaries
        colors = []
        for val in values:
            if abs(val - param_bounds[param_name]['min']) < param_bounds[param_name]['threshold']:
                colors.append('red')
            elif abs(val - param_bounds[param_name]['max']) < param_bounds[param_name]['threshold']:
                colors.append('darkred')
            else:
                colors.append('blue')
        
        ax.scatter(ages, values, c=colors, alpha=0.6, s=20)
        
        # Add boundary lines
        ax.axhline(param_bounds[param_name]['min'], color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(param_bounds[param_name]['max'], color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        ax.set_title(f'{param_name} vs Age (colored by boundary proximity)')
        ax.set_xlabel('Age (years)')
        ax.set_ylabel(f'{param_name} value')
        
        # Add correlation
        corr = np.corrcoef(ages, values)[0, 1]
        ax.text(0.02, 0.98, f'r={corr:.3f}', transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Subject Parameter Boundary Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('parameter_boundary_analysis.png', dpi=150)
    plt.show()
    
    # Summary report
    print("\n" + "="*70)
    print("SUMMARY REPORT")
    print("="*70)
    
    total_boundary_subjects = 0
    for param_name, stats in boundary_stats.items():
        if stats['pct_at_boundary'] > 20:
            print(f"⚠️ {param_name}: {stats['pct_at_boundary']:.1f}% subjects at boundaries - NEEDS ATTENTION")
        else:
            print(f"✅ {param_name}: {stats['pct_at_boundary']:.1f}% subjects at boundaries - Acceptable")
    
    # Check for parameter correlations
    print("\n" + "="*70)
    print("PARAMETER CORRELATIONS")
    print("="*70)
    
    correlation_matrix = df[['K', 'B', 'tau']].corr()
    print("\nCorrelation Matrix:")
    print(correlation_matrix.round(3))
    
    # Check if parameters are correlated (might indicate coupling issues)
    for i, param1 in enumerate(['K', 'B', 'tau']):
        for j, param2 in enumerate(['K', 'B', 'tau']):
            if i < j:
                corr = correlation_matrix.loc[param1, param2]
                if abs(corr) > 0.5:
                    print(f"⚠️ High correlation between {param1} and {param2}: {corr:.3f}")
    
    # Identify specific problem subjects
    print("\n" + "="*70)
    print("SUBJECTS WITH MULTIPLE PARAMETERS AT BOUNDARIES")
    print("="*70)
    
    problem_subjects = []
    for subject_id, params in subject_params.items():
        boundary_count = 0
        for param_name, bounds in param_bounds.items():
            if abs(params[param_name] - bounds['min']) < bounds['threshold']:
                boundary_count += 1
            elif abs(params[param_name] - bounds['max']) < bounds['threshold']:
                boundary_count += 1
        
        if boundary_count >= 2:
            problem_subjects.append((subject_id, boundary_count, params))
    
    if problem_subjects:
        print(f"\nFound {len(problem_subjects)} subjects with 2+ parameters at boundaries:")
        for subject_id, count, params in problem_subjects[:10]:  # Show first 10
            print(f"  Subject {subject_id}: {count} params at boundaries")
            print(f"    K={params['K']:.1f}, B={params['B']:.1f}, τ={params['tau']:.3f}")
    else:
        print("✅ No subjects have multiple parameters at boundaries")
    
    return boundary_stats, df

if __name__ == "__main__":
    boundary_stats, df = analyze_boundary_clustering()
    print("\n✅ Analysis complete! Check 'parameter_boundary_analysis.png' for visualizations.")
