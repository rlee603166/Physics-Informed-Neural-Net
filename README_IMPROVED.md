# Enhanced Balance PINN Training System

This directory contains a complete, enhanced training system for Physics-Informed Neural Networks (PINNs) designed for balance assessment and cross-age comparison.

## 🚀 Quick Start

### Two-Stage Training (Recommended)
```bash
# Activate virtual environment
source .venv/bin/activate

# Install additional dependencies
pip install matplotlib seaborn pyyaml

# Run two-stage training
python train_two_stage.py --data-folder processed_data --age-csv-path user_ages.csv --experiment-name two_stage_run1
```

### Improved Single-Stage Training
```bash
python train_improved_single.py --data-folder processed_data --age-csv-path user_ages.csv --experiment-name single_stage_run1 --use-probabilistic
```

## 📁 File Structure

### Core Files
- **`improved_models.py`** - Enhanced PINN architectures (two-stage & improved single-stage)
- **`enhanced_datasets.py`** - Advanced dataset classes with subject awareness
- **`training_utils.py`** - Comprehensive loss functions and training utilities
- **`train_two_stage.py`** - Two-stage training pipeline
- **`train_improved_single.py`** - Improved single-stage training pipeline
- **`config.py`** - Configuration management system
- **`model_tester.py`** - Comprehensive model evaluation framework

### Legacy Files (Original Implementation)
- `train.py` - Original PINN implementation
- `process.py` - Data processing utilities
- `download.py` - Data download utilities

## 🏗️ Architecture Overview

### Two-Stage Architecture (Recommended)
1. **Stage 1: SubjectPINN** - Learn individual parameters for each subject
2. **Stage 2: AgeParameterModel** - Learn population-level age trends

**Benefits:**
- Decoupled learning for better convergence
- Individual subject parameter interpretability
- Clinical relevance for age comparisons

### Improved Single-Stage Architecture
- Enhanced parameter network with probabilistic outputs
- Better loss balancing and regularization
- Age-aware training strategies

## 🔧 Configuration System

### Using Presets
```bash
# Quick test (reduced parameters)
python train_improved_single.py --preset quick_test

# High quality training
python train_improved_single.py --preset high_quality

# Physics-focused training
python train_improved_single.py --preset physics_focused
```

### Creating Custom Configurations
```python
from config import ConfigManager

manager = ConfigManager()
config = manager.create_experiment_config(
    base_preset="default",
    overrides={
        "training": {"epochs": 300},
        "loss": {"physics_weight": 0.005}
    },
    experiment_name="my_experiment"
)
```

### Environment Variables
```bash
export PINN_DATA_FOLDER="processed_data"
export PINN_EPOCHS=200
export PINN_PHYSICS_WEIGHT=0.01
```

## 📊 Key Features

### Enhanced Loss Functions
- **PhysicsLoss**: Inverted pendulum dynamics enforcement
- **ParameterRegularizationLoss**: Age-aware parameter smoothness
- **AgeAwareLoss**: Balanced learning across age groups
- **CombinedLoss**: Integrated multi-component loss

### Advanced Training Utilities
- **EarlyStopping**: Prevent overfitting with patience
- **WarmupLRScheduler**: Gradual learning rate increase
- **ModelCheckpointer**: Automatic model saving
- **MetricsTracker**: Comprehensive training monitoring

### Comprehensive Evaluation
- Physics compliance testing
- Age-parameter relationship analysis
- Cross-age comparison capabilities
- Parameter distribution analysis

## 🎯 Model Capabilities

### Age Comparison Functions
```python
# Compare balance at two different ages
comparison = model.compare_ages(subject_age=65, target_age=75)
print(f"Similarity: {comparison['mean_similarity']:.3f}")

# Find "balance age" for a subject
balance_result = model.find_balance_age(subject_age=70)
print(f"Balance age: {balance_result['balance_age']:.1f}")
```

### Parameter Analysis
```python
# Get age-dependent parameters
age_tensor = torch.tensor([[65.0]])
if model.use_probabilistic:
    param_means, param_stds = model.predict_parameters(age_tensor)
else:
    params = model.predict_parameters(age_tensor)
```

## 📈 Training Monitoring

### Automatic Visualization
- Training/validation curves
- Parameter vs age relationships
- Physics residual distributions
- Age comparison matrices

### Comprehensive Logging
- Loss decomposition tracking
- Parameter variation monitoring
- Physics compliance metrics
- Age correlation analysis

## 🔍 Model Testing

### Performance Evaluation
```bash
python model_tester.py  # Tests existing trained models
```

The tester evaluates:
- Data reconstruction performance (R², MAE, RMSE)
- Physics constraint compliance
- Parameter learning effectiveness
- Age-dependent parameter variation

### Previous Results (Original Model)
Based on your existing model test:
- ❌ Poor R² score (-0.002) indicates overfitting to physics
- ❌ No age-dependent parameter learning
- ⚠️ Physics residual (0.01) acceptable but could improve

## 🎛️ Training Recommendations

### For Research/Development
```bash
python train_two_stage.py --preset high_quality --experiment-name research_run
```

### For Quick Testing
```bash
python train_improved_single.py --preset quick_test --experiment-name test_run
```

### For Production Models
```bash
python train_two_stage.py \
    --preset high_quality \
    --stage1-epochs 200 \
    --stage2-epochs 100 \
    --physics-weight 0.005 \
    --experiment-name production_model
```

## 📋 Training Checklist

### Before Training
- [ ] Ensure `processed_data/` contains batch files
- [ ] Verify `user_ages.csv` exists and has correct format
- [ ] Activate virtual environment with required packages
- [ ] Choose appropriate configuration preset

### During Training
- [ ] Monitor loss convergence (total, data, physics)
- [ ] Check parameter variation coefficients > 0.1
- [ ] Verify age correlation coefficients are meaningful
- [ ] Watch for early stopping triggers

### After Training
- [ ] Run comprehensive evaluation with `model_tester.py`
- [ ] Check R² > 0.8 for good data fitting
- [ ] Verify physics residuals < 1e-3
- [ ] Test age comparison functionality
- [ ] Generate analysis plots and reports

## 🚨 Troubleshooting

### Common Issues

**Low R² Score**
- Reduce physics weight (try 0.001-0.01)
- Increase data weight
- Use more training epochs

**No Parameter Variation**
- Increase regularization weight
- Add age-aware loss component
- Check parameter bounds are appropriate

**Poor Physics Compliance**
- Increase physics weight gradually
- Check derivative calculations
- Verify parameter ranges make physical sense

**Training Instability**
- Use gradient clipping (max_norm=1.0)
- Reduce learning rate
- Enable warmup scheduler

### Performance Tips
- Use GPU for faster training
- Increase batch size for stable gradients
- Use multiple workers for data loading
- Monitor GPU memory usage

## 📚 References

### Model Architecture
- Physics-Informed Neural Networks (Raissi et al., 2019)
- Inverted pendulum balance model
- Age-dependent physiological parameter modeling

### Implementation Details
- PyTorch deep learning framework
- HDF5 for efficient data storage
- Matplotlib/Seaborn for visualization
- YAML/JSON for configuration management

---

## 🎉 Ready to Train!

Your enhanced Balance PINN training system is ready to produce reliable models for cross-age balance comparison. The two-stage approach is recommended for best results, but both architectures address the issues found in your original model.

Run with your data and compare the results!