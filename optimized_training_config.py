#!/usr/bin/env python3
"""
Optimized training configuration for A100 GPU performance.
This provides settings tuned for maximum A100 utilization.
"""

def get_a100_optimized_config():
    """
    Get configuration optimized for A100 GPU training.
    
    Key optimizations:
    - Larger batch sizes for better GPU utilization
    - More efficient data loading
    - Reduced gradient computation overhead
    - Memory-optimized settings
    """
    return {
        #  loading optimizations
        'batch_size': 16384,  # Increased from 4096 for better GPU utilization
        'num_workers': 8,     # More workers for data loading
        'pin_memory': True,
        'prefetch_factor': 4,  # Prefetch batches
        'persistent_workers': True,  # Keep workers alive between epochs
        
        # Training optimizations  
        'gradient_accumulation_steps': 1,  # Can increase if memory allows
        'mixed_precision': True,  # Use automatic mixed precision (AMP)
        'compile_model': True,    # Use torch.compile for faster execution
        
        # Memory optimizations
        'max_memory_fraction': 0.9,  # Use 90% of GPU memory
        'empty_cache_frequency': 100,  # Clear cache every N batches
        
        # Physics loss optimizations
        'use_simplified_physics': True,  # Start with SimplePhysicsLoss
        'physics_computation_frequency': 4,  # Compute physics loss every N batches
        'gradient_checkpointing': False,  # Disable if not memory constrained
        
        # Learning optimizations
        'learning_rate': 2e-3,  # Higher LR for larger batches
        'warmup_steps': 1000,   # Warmup for stability
        'scheduler': 'cosine_annealing',  # Better than plateau for large scale
        
        # Validation optimizations
        'val_frequency': 5,  # Validate every N epochs (not every epoch)
        'val_batch_size': 32768,  # Larger batch for validation
        
        # Checkpointing optimizations
        'save_frequency': 10,  # Save less frequently
        'keep_best_only': True,  # Don't save all checkpoints
        
        # Stage-specific optimizations
        'stage1': {
            'epochs': 50,  # Reduced epochs with better optimization
            'lr': 2e-3,
            'physics_weight': 0.01,  # Lower physics weight initially
            'early_stopping_patience': 15,
        },
        'stage2': {
            'epochs': 25,  # Reduced epochs
            'lr': 1e-3,
            'batch_size': 128,  # Smaller batch for stage 2
            'early_stopping_patience': 10,
        }
    }

def get_memory_optimized_config():
    """
    Configuration for systems with limited GPU memory.
    """
    return {
        'batch_size': 8192,   # Smaller batch size
        'gradient_accumulation_steps': 2,  # Simulate larger batches
        'mixed_precision': True,
        'gradient_checkpointing': True,  # Trade compute for memory
        'num_workers': 4,
        'prefetch_factor': 2,
        
        # More frequent cache clearing
        'empty_cache_frequency': 50,
        'max_memory_fraction': 0.8,
        
        # Simplified physics loss
        'use_simplified_physics': True,
        'physics_computation_frequency': 8,
        
        # Conservative learning
        'learning_rate': 1e-3,
        'warmup_steps': 500,
    }

def get_speed_optimized_config():
    """
    Configuration prioritizing training speed over memory efficiency.
    """
    return {
        # Maximum batch sizes
        'batch_size': 32768,   # Very large batches
        'val_batch_size': 65536,
        
        # Aggressive data loading
        'num_workers': 12,
        'prefetch_factor': 6,
        'persistent_workers': True,
        'pin_memory': True,
        
        # Speed optimizations
        'mixed_precision': True,
        'compile_model': True,
        'gradient_accumulation_steps': 1,
        
        # Reduced precision where possible
        'use_simplified_physics': True,
        'physics_computation_frequency': 1,  # Every batch
        
        # Fast scheduling
        'learning_rate': 3e-3,  # Higher for large batches
        'scheduler': 'cosine_annealing',
        'warmup_steps': 2000,
        
        # Less frequent validation
        'val_frequency': 10,
        'save_frequency': 20,
    }

# Benchmark configurations for different hardware
HARDWARE_CONFIGS = {
    'a100_80gb': get_a100_optimized_config(),
    'a100_40gb': get_a100_optimized_config(),
    'v100_32gb': get_memory_optimized_config(),
    'rtx_4090': get_memory_optimized_config(),
    'colab_free': {
        'batch_size': 2048,
        'num_workers': 2,
        'mixed_precision': True,
        'use_simplified_physics': True,
        'physics_computation_frequency': 16,
        'val_frequency': 20,
    }
}

def detect_optimal_config():
    """
    Automatically detect optimal configuration based on available hardware.
    """
    import torch
    
    if not torch.cuda.is_available():
        return HARDWARE_CONFIGS['colab_free']
    
    # Get GPU memory
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    gpu_name = torch.cuda.get_device_name(0).lower()
    
    if 'a100' in gpu_name:
        if gpu_memory_gb > 70:
            return HARDWARE_CONFIGS['a100_80gb']
        else:
            return HARDWARE_CONFIGS['a100_40gb']
    elif 'v100' in gpu_name:
        return HARDWARE_CONFIGS['v100_32gb']
    elif '4090' in gpu_name or 'rtx' in gpu_name:
        return HARDWARE_CONFIGS['rtx_4090']
    else:
        # Default to memory optimized for unknown GPUs
        return get_memory_optimized_config()

if __name__ == "__main__":
    # Test configuration detection
    import torch
    config = detect_optimal_config()
    print(f"Detected configuration for {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}:")
    for key, value in config.items():
        print(f"  {key}: {value}")