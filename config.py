#!/usr/bin/env python3
"""
Configuration Management for Balance PINN Training

This module provides centralized configuration management with:
- Default configurations for different training modes
- Configuration validation and type checking
- Environment-specific overrides
- Hyperparameter optimization support
- Configuration versioning and reproducibility
"""

import json
import yaml
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
import os
import argparse

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION DATA CLASSES
# =============================================================================

@dataclass
class DataConfig:
    """Data-related configuration."""
    data_folder: str = "processed_data"
    age_csv_path: str = "user_ages.csv"
    normalize_data: bool = False
    augment_data: bool = False
    min_points_per_subject: int = 100
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    num_workers: int = 4
    batch_size: int = 4096

@dataclass  
class ModelConfig:
    """Model architecture configuration."""
    hidden_dim: int = 256
    num_layers: int = 8
    dropout_rate: float = 0.1
    use_probabilistic: bool = True
    param_bounds: Dict[str, tuple] = field(default_factory=lambda: {
        'K': (500.0, 3000.0),
        'B': (20.0, 150.0),
        'tau': (0.05, 0.4)
    })

@dataclass
class TrainingConfig:
    """Training-related configuration."""
    epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 20
    scheduler_patience: int = 10
    warmup_epochs: int = 5
    save_every: int = 10
    checkpoint_dir: str = "checkpoints"

@dataclass
class LossConfig:
    """Loss function configuration."""
    data_weight: float = 1.0
    physics_weight: float = 0.01
    regularization_weight: float = 0.1
    age_aware_weight: float = 0.1

@dataclass
class Stage1Config:
    """Stage 1 specific configuration (for two-stage training)."""
    epochs: int = 100
    learning_rate: float = 1e-3
    physics_weight: float = 0.1
    patience: int = 20
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 256, 256])

@dataclass
class Stage2Config:
    """Stage 2 specific configuration (for two-stage training)."""
    epochs: int = 50
    learning_rate: float = 1e-3
    reg_weight: float = 0.1
    patience: int = 15
    batch_size: int = 32
    probabilistic: bool = True

@dataclass
class Config:
    """Main configuration class."""
    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    
    # Meta information
    experiment_name: str = "balance_pinn_experiment"
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create from dictionary."""
        # Handle nested configurations
        config = cls()
        
        for key, value in config_dict.items():
            if hasattr(config, key):
                if key in ['data', 'model', 'training', 'loss', 'stage1', 'stage2']:
                    # Handle nested dataclass
                    nested_class = getattr(config, key).__class__
                    setattr(config, key, nested_class(**value))
                else:
                    setattr(config, key, value)
        
        return config
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Data validation
        if not (0 < self.data.train_ratio < 1):
            issues.append("train_ratio must be between 0 and 1")
        if not (0 < self.data.val_ratio < 1):
            issues.append("val_ratio must be between 0 and 1")
        if not (0 < self.data.test_ratio < 1):
            issues.append("test_ratio must be between 0 and 1")
        
        total_ratio = self.data.train_ratio + self.data.val_ratio + self.data.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            issues.append(f"train/val/test ratios must sum to 1.0, got {total_ratio}")
        
        if self.data.batch_size <= 0:
            issues.append("batch_size must be positive")
        
        # Model validation
        if self.model.hidden_dim <= 0:
            issues.append("hidden_dim must be positive")
        if self.model.num_layers <= 0:
            issues.append("num_layers must be positive")
        if not (0 <= self.model.dropout_rate < 1):
            issues.append("dropout_rate must be between 0 and 1")
        
        # Parameter bounds validation
        for param, (min_val, max_val) in self.model.param_bounds.items():
            if min_val >= max_val:
                issues.append(f"param_bounds[{param}]: min_val must be < max_val")
        
        # Training validation
        if self.training.epochs <= 0:
            issues.append("epochs must be positive")
        if self.training.learning_rate <= 0:
            issues.append("learning_rate must be positive")
        if self.training.weight_decay < 0:
            issues.append("weight_decay must be non-negative")
        
        # Loss validation
        if any(w < 0 for w in [self.loss.data_weight, self.loss.physics_weight, 
                               self.loss.regularization_weight, self.loss.age_aware_weight]):
            issues.append("All loss weights must be non-negative")
        
        return issues

# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

def get_default_config() -> Config:
    """Get default configuration."""
    return Config(
        experiment_name="balance_pinn_default",
        description="Default Balance PINN configuration"
    )

def get_quick_test_config() -> Config:
    """Get configuration for quick testing."""
    config = Config(
        experiment_name="balance_pinn_quick_test",
        description="Quick test configuration with reduced parameters"
    )
    
    # Reduce training time for testing
    config.training.epochs = 10
    config.stage1.epochs = 5
    config.stage2.epochs = 5
    config.data.batch_size = 512
    config.training.patience = 5
    
    return config

def get_high_quality_config() -> Config:
    """Get configuration for high-quality training."""
    config = Config(
        experiment_name="balance_pinn_high_quality",
        description="High-quality training configuration"
    )
    
    # Extended training
    config.training.epochs = 500
    config.stage1.epochs = 200
    config.stage2.epochs = 100
    
    # Better regularization
    config.loss.regularization_weight = 0.2
    config.loss.age_aware_weight = 0.2
    
    # Lower learning rates for stability
    config.training.learning_rate = 5e-4
    config.stage1.learning_rate = 5e-4
    config.stage2.learning_rate = 5e-4
    
    return config

def get_two_stage_focused_config() -> Config:
    """Get configuration optimized for two-stage training."""
    config = Config(
        experiment_name="balance_pinn_two_stage",
        description="Two-stage training focused configuration"
    )
    
    # Emphasize Stage 1
    config.stage1.epochs = 150
    config.stage1.physics_weight = 0.05  # Reduced physics weight
    
    # Stage 2 with more regularization
    config.stage2.epochs = 75
    config.stage2.reg_weight = 0.2
    config.stage2.probabilistic = True
    
    return config

def get_single_stage_focused_config() -> Config:
    """Get configuration optimized for single-stage training."""
    config = Config(
        experiment_name="balance_pinn_single_stage",
        description="Single-stage training focused configuration"  
    )
    
    # Longer training with careful loss balancing
    config.training.epochs = 300
    config.loss.physics_weight = 0.005  # Very low physics weight
    config.loss.regularization_weight = 0.15
    config.loss.age_aware_weight = 0.15
    
    # Probabilistic outputs
    config.model.use_probabilistic = True
    config.model.dropout_rate = 0.15
    
    return config

def get_physics_focused_config() -> Config:
    """Get configuration that emphasizes physics compliance."""
    config = Config(
        experiment_name="balance_pinn_physics_focused",
        description="Physics-focused training configuration"
    )
    
    # Higher physics weights
    config.loss.physics_weight = 0.1
    config.stage1.physics_weight = 0.2
    
    # More regularization for parameter smoothness
    config.loss.regularization_weight = 0.3
    
    return config

# =============================================================================
# CONFIGURATION MANAGEMENT
# =============================================================================

class ConfigManager:
    """Configuration manager with loading, saving, and validation."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
    
    def save_config(self, config: Config, filename: Optional[str] = None) -> str:
        """Save configuration to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{config.experiment_name}_{timestamp}.yaml"
        
        filepath = self.config_dir / filename
        
        with open(filepath, 'w') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to: {filepath}")
        return str(filepath)
    
    def load_config(self, filepath: Union[str, Path]) -> Config:
        """Load configuration from file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            if filepath.suffix.lower() == '.json':
                config_dict = json.load(f)
            elif filepath.suffix.lower() in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {filepath.suffix}")
        
        config = Config.from_dict(config_dict)
        logger.info(f"Configuration loaded from: {filepath}")
        
        return config
    
    def validate_config(self, config: Config, strict: bool = True) -> bool:
        """Validate configuration."""
        issues = config.validate()
        
        if issues:
            logger.error("Configuration validation failed:")
            for issue in issues:
                logger.error(f"  - {issue}")
            
            if strict:
                raise ValueError("Configuration validation failed")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    def merge_configs(self, base_config: Config, override_config: Dict[str, Any]) -> Config:
        """Merge base configuration with overrides."""
        base_dict = base_config.to_dict()
        
        # Deep merge dictionaries
        def deep_merge(base: Dict, override: Dict) -> Dict:
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        merged_dict = deep_merge(base_dict, override_config)
        return Config.from_dict(merged_dict)
    
    def create_experiment_config(self, base_preset: str = "default", 
                                overrides: Optional[Dict[str, Any]] = None,
                                experiment_name: Optional[str] = None) -> Config:
        """Create experiment configuration with presets and overrides."""
        # Get base configuration
        preset_configs = {
            "default": get_default_config,
            "quick_test": get_quick_test_config,
            "high_quality": get_high_quality_config,
            "two_stage": get_two_stage_focused_config,
            "single_stage": get_single_stage_focused_config,
            "physics_focused": get_physics_focused_config
        }
        
        if base_preset not in preset_configs:
            raise ValueError(f"Unknown preset: {base_preset}. Available: {list(preset_configs.keys())}")
        
        config = preset_configs[base_preset]()
        
        # Apply overrides
        if overrides:
            config = self.merge_configs(config, overrides)
        
        # Set experiment name
        if experiment_name:
            config.experiment_name = experiment_name
            config.training.checkpoint_dir = f"checkpoints_{experiment_name}"
        
        # Update timestamp
        config.created_at = datetime.now().isoformat()
        
        return config

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def create_config_from_args(args: argparse.Namespace) -> Config:
    """Create configuration from command line arguments."""
    # Start with base preset
    config_manager = ConfigManager()
    config = config_manager.create_experiment_config(
        base_preset=getattr(args, 'preset', 'default'),
        experiment_name=getattr(args, 'experiment_name', None)
    )
    
    # Override with command line arguments
    overrides = {}
    
    # Data overrides
    if hasattr(args, 'data_folder'):
        overrides.setdefault('data', {})['data_folder'] = args.data_folder
    if hasattr(args, 'age_csv_path'):
        overrides.setdefault('data', {})['age_csv_path'] = args.age_csv_path
    if hasattr(args, 'batch_size'):
        overrides.setdefault('data', {})['batch_size'] = args.batch_size
    if hasattr(args, 'normalize_data'):
        overrides.setdefault('data', {})['normalize_data'] = args.normalize_data
    if hasattr(args, 'augment_data'):
        overrides.setdefault('data', {})['augment_data'] = args.augment_data
    
    # Training overrides
    if hasattr(args, 'epochs'):
        overrides.setdefault('training', {})['epochs'] = args.epochs
    if hasattr(args, 'learning_rate'):
        overrides.setdefault('training', {})['learning_rate'] = args.learning_rate
    if hasattr(args, 'checkpoint_dir'):
        overrides.setdefault('training', {})['checkpoint_dir'] = args.checkpoint_dir
    
    # Model overrides
    if hasattr(args, 'hidden_dim'):
        overrides.setdefault('model', {})['hidden_dim'] = args.hidden_dim
    if hasattr(args, 'num_layers'):
        overrides.setdefault('model', {})['num_layers'] = args.num_layers
    if hasattr(args, 'use_probabilistic'):
        overrides.setdefault('model', {})['use_probabilistic'] = args.use_probabilistic
    
    # Loss overrides
    if hasattr(args, 'physics_weight'):
        overrides.setdefault('loss', {})['physics_weight'] = args.physics_weight
    if hasattr(args, 'data_weight'):
        overrides.setdefault('loss', {})['data_weight'] = args.data_weight
    
    # Apply overrides
    if overrides:
        config = config_manager.merge_configs(config, overrides)
    
    return config

def add_config_args(parser: argparse.ArgumentParser):
    """Add configuration arguments to argument parser."""
    
    # Configuration management
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--preset', type=str, default='default',
                       choices=['default', 'quick_test', 'high_quality', 'two_stage', 'single_stage', 'physics_focused'],
                       help='Configuration preset to use')
    parser.add_argument('--experiment-name', type=str, help='Experiment name')
    parser.add_argument('--save-config', type=str, help='Save configuration to file')
    
    # Data arguments
    parser.add_argument('--data-folder', type=str, help='Path to processed data folder')
    parser.add_argument('--age-csv-path', type=str, help='Path to age CSV file')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--normalize-data', action='store_true', help='Normalize input data')
    parser.add_argument('--augment-data', action='store_true', help='Apply data augmentation')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--checkpoint-dir', type=str, help='Checkpoint directory')
    
    # Model arguments
    parser.add_argument('--hidden-dim', type=int, help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, help='Number of layers')
    parser.add_argument('--use-probabilistic', action='store_true', help='Use probabilistic outputs')
    
    # Loss arguments
    parser.add_argument('--physics-weight', type=float, help='Physics loss weight')
    parser.add_argument('--data-weight', type=float, help='Data loss weight')

# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

def load_config_from_env() -> Dict[str, Any]:
    """Load configuration overrides from environment variables."""
    env_overrides = {}
    
    # Data configuration from environment
    if 'PINN_DATA_FOLDER' in os.environ:
        env_overrides.setdefault('data', {})['data_folder'] = os.environ['PINN_DATA_FOLDER']
    if 'PINN_BATCH_SIZE' in os.environ:
        env_overrides.setdefault('data', {})['batch_size'] = int(os.environ['PINN_BATCH_SIZE'])
    
    # Training configuration from environment
    if 'PINN_EPOCHS' in os.environ:
        env_overrides.setdefault('training', {})['epochs'] = int(os.environ['PINN_EPOCHS'])
    if 'PINN_LEARNING_RATE' in os.environ:
        env_overrides.setdefault('training', {})['learning_rate'] = float(os.environ['PINN_LEARNING_RATE'])
    if 'PINN_CHECKPOINT_DIR' in os.environ:
        env_overrides.setdefault('training', {})['checkpoint_dir'] = os.environ['PINN_CHECKPOINT_DIR']
    
    # Loss configuration from environment
    if 'PINN_PHYSICS_WEIGHT' in os.environ:
        env_overrides.setdefault('loss', {})['physics_weight'] = float(os.environ['PINN_PHYSICS_WEIGHT'])
    
    return env_overrides

# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def main():
    """Main function for config management CLI."""
    parser = argparse.ArgumentParser(description='Balance PINN Configuration Management')
    
    parser.add_argument('command', choices=['create', 'validate', 'show'],
                       help='Command to run')
    parser.add_argument('--preset', type=str, default='default',
                       choices=['default', 'quick_test', 'high_quality', 'two_stage', 'single_stage', 'physics_focused'],
                       help='Configuration preset')
    parser.add_argument('--output', type=str, help='Output file for created config')
    parser.add_argument('--config', type=str, help='Configuration file to validate/show')
    parser.add_argument('--experiment-name', type=str, help='Experiment name')
    
    args = parser.parse_args()
    
    config_manager = ConfigManager()
    
    if args.command == 'create':
        # Create configuration
        config = config_manager.create_experiment_config(
            base_preset=args.preset,
            experiment_name=args.experiment_name
        )
        
        if args.output:
            config_manager.save_config(config, args.output)
        else:
            print(yaml.dump(config.to_dict(), default_flow_style=False, indent=2))
    
    elif args.command == 'validate':
        # Validate configuration
        if not args.config:
            print("Error: --config required for validate command")
            return
        
        config = config_manager.load_config(args.config)
        is_valid = config_manager.validate_config(config, strict=False)
        
        if is_valid:
            print("✅ Configuration is valid")
        else:
            print("❌ Configuration has issues")
    
    elif args.command == 'show':
        # Show configuration
        if args.config:
            config = config_manager.load_config(args.config)
        else:
            config = config_manager.create_experiment_config(base_preset=args.preset)
        
        print(yaml.dump(config.to_dict(), default_flow_style=False, indent=2))

if __name__ == "__main__":
    main()