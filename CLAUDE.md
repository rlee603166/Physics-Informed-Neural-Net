# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Physics-Informed Neural Network (PINN) implementation for balance assessment analysis. The project processes MATLAB (.mat) files containing balance data, trains neural networks to predict physical parameters from age, and models balance dynamics using physics constraints.

## Architecture

The codebase follows a data pipeline architecture:

1. **Data Download** (`download.py`): Downloads balance data files from Box API using async HTTP
2. **Data Processing** (`process.py`): Converts MATLAB files to HDF5 batches for efficient training
3. **Neural Network Training** (`train.py`): Implements dual-network PINN architecture
4. **Testing & Validation** (`test.py`, `data_inspector.py`): Debug and validation utilities

### Core Components

**BalancePINN Architecture** (`train.py:92-126`):
- ParameterNet: Predicts physical parameters (K=stiffness, B=damping, τ=delay) from age
- SolverNet: Predicts position (x,y) from time and predicted parameters
- Physics loss enforces balance dynamics equations

**Data Pipeline** (`process.py:24-194`):
- Lazy-loading HDF5 dataset for memory efficiency
- Batch processing of MATLAB files with subject age mapping
- Handles nested MATLAB cell array structures

## Common Commands

### Training
```bash
python train.py
```

### Data Processing
```bash
python process.py
```

### Data Inspection/Debug
```bash
python data_inspector.py
python test.py
```

### Download Data
```bash
python download.py
```

## Key Data Structures

- **MATLAB Files**: Contain `TrajX`/`TrajY` nested cell arrays with balance trajectory data
- **HDF5 Batches**: Processed data split into `batch_*.h5` files in `processed_data/`
- **Subject Ages**: CSV mapping user IDs to ages for parameter prediction
- **Model Checkpoints**: Saved in `trained_models/` directory

## Configuration

Training parameters are configured in `train.py:234-245`:
- Batch size: 8192 (high for GPU efficiency)
- Physics weight: 0.01 (balance between data and physics loss)
- Early stopping patience: 10 epochs

Data processing uses 2 batches by default to manage memory usage with large datasets.

## Dependencies

Uses PyTorch, scipy, h5py, pandas, numpy, and aiohttp for async downloads. No requirements.txt present - dependencies must be installed manually.