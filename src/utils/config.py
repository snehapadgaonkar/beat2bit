"""
Configuration management utilities for Beat2Bit experiments.
Handles loading, validation, and management of experiment configurations.
"""

import yaml
import json
import os
from typing import Dict, Any, Optional, Union, Tuple, List
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Data class for experiment configuration."""
    # Experiment metadata
    experiment_name: str
    description: str
    version: str = "1.0.0"

    # Data configuration
    dataset_name: str = "MIT-BIH"
    dataset_version: str = "1.0.0"
    data_split_strategy: str = "patient_aware_aami"
    test_size: float = 0.2
    validation_size: float = 0.1
    random_seed: int = 42

    # Preprocessing configuration
    window_size: int = 180
    sampling_rate: int = 360
    normalization_method: str = "zscore"
    remove_baseline_wander: bool = True

    # Model configuration
    model_architecture: str = "baseline_1d_cnn"
    input_shape: tuple = (180, 1)
    num_classes: int = 1

    # Training configuration
    epochs: int = 10
    batch_size: int = 128
    optimizer: str = "adam"
    learning_rate: float = 0.001
    validation_split: float = 0.1
    class_weight_method: str = "balanced"

    # Optimization configuration
    apply_pruning: bool = False
    pruning_sparsity: float = 0.5
    pruning_schedule: str = "polynomial_decay"
    apply_quantization: bool = False
    quantization_type: str = "int8"

    # Evaluation configuration
    metrics_to_report: list = None
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000

    # Output configuration
    save_model: bool = True
    save_predictions: bool = False
    generate_plots: bool = True

    def __post_init__(self):
        """Set default values for mutable fields."""
        if self.metrics_to_report is None:
            self.metrics_to_report = [
                'accuracy', 'precision', 'recall', 'f1_score',
                'ami_sensitivity', 'ami_positive_predictivity', 'ami_effectiveness'
            ]


class ConfigManager:
    """Manages experiment configurations."""

    def __init__(self, config_dir: str = "configs"):
        """
        Initialize ConfigManager.

        Args:
            config_dir: Directory to store/load configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        logger.info(f"ConfigManager initialized with directory: {self.config_dir}")

    def create_config(self,
                     experiment_name: str,
                     description: str,
                     **kwargs) -> ExperimentConfig:
        """
        Create a new experiment configuration.

        Args:
            experiment_name: Name of the experiment
            description: Description of the experiment
            **kwargs: Additional configuration parameters

        Returns:
            ExperimentConfig object
        """
        config = ExperimentConfig(
            experiment_name=experiment_name,
            description=description,
            **kwargs
        )

        logger.info(f"Created configuration for experiment: {experiment_name}")
        return config

    def save_config(self, config: ExperimentConfig,
                   filename: Optional[str] = None) -> str:
        """
        Save configuration to YAML file.

        Args:
            config: ExperimentConfig to save
            filename: Optional filename (without extension)

        Returns:
            Path to saved configuration file
        """
        if filename is None:
            filename = f"{config.experiment_name}_{config.version}"

        # Convert config to dict and handle tuple conversion
        config_dict = asdict(config)
        # Convert tuple to list for YAML serialization
        if isinstance(config_dict['input_shape'], tuple):
            config_dict['input_shape'] = list(config_dict['input_shape'])

        # Save as YAML
        yaml_path = self.config_dir / f"{filename}.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        # Also save as JSON for easy parsing
        json_path = self.config_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"Configuration saved to {yaml_path} and {json_path}")
        return str(yaml_path)

    def load_config(self, filepath: Union[str, Path]) -> ExperimentConfig:
        """
        Load configuration from YAML or JSON file.

        Args:
            filepath: Path to configuration file

        Returns:
            ExperimentConfig object
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        # Determine file type and load accordingly
        if filepath.suffix in ['.yaml', '.yml']:
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

        # Create ExperimentConfig from dictionary
        config = ExperimentConfig(**config_dict)
        logger.info(f"Configuration loaded from {filepath}")
        return config

    def list_configs(self) -> list:
        """
        List all available configuration files.

        Returns:
            List of configuration file paths
        """
        yaml_files = list(self.config_dir.glob("*.yaml"))
        json_files = list(self.config_dir.glob("*.json"))
        return sorted(list(set([f.stem for f in yaml_files + json_files])))

    def validate_config(self, config: ExperimentConfig) -> Tuple[bool, list]:
        """
        Validate experiment configuration.

        Args:
            config: ExperimentConfig to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Validate required fields
        if not config.experiment_name:
            errors.append("Experiment name is required")

        if not config.description:
            errors.append("Experiment description is required")

        # Validate numeric ranges
        if not 0 < config.test_size <= 1:
            errors.append("Test size must be between 0 and 1")

        if not 0 < config.validation_size <= 1:
            errors.append("Validation size must be between 0 and 1")

        if config.test_size + config.validation_size >= 1:
            errors.append("Test size + validation size must be less than 1")

        if config.window_size <= 0:
            errors.append("Window size must be positive")

        if config.sampling_rate <= 0:
            errors.append("Sampling rate must be positive")

        if config.epochs <= 0:
            errors.append("Number of epochs must be positive")

        if config.batch_size <= 0:
            errors.append("Batch size must be positive")

        if not 0 < config.learning_rate <= 1:
            errors.append("Learning rate must be between 0 and 1")

        if config.apply_pruning:
            if not 0 <= config.pruning_sparsity <= 1:
                errors.append("Pruning sparsity must be between 0 and 1")

        if config.random_seed < 0:
            errors.append("Random seed must be non-negative")

        is_valid = len(errors) == 0
        if not is_valid:
            logger.warning(f"Configuration validation failed: {errors}")

        return is_valid, errors

    def merge_configs(self, base_config: ExperimentConfig,
                     override_dict: Dict[str, Any]) -> ExperimentConfig:
        """
        Merge base configuration with override values.

        Args:
            base_config: Base ExperimentConfig
            override_dict: Dictionary of values to override

        Returns:
            New ExperimentConfig with merged values
        """
        # Convert base config to dict
        config_dict = asdict(base_config)

        # Update with override values
        config_dict.update(override_dict)

        # Create new config
        merged_config = ExperimentConfig(**config_dict)
        logger.info("Configurations merged successfully")
        return merged_config

    def create_experiment_directory(self, config: ExperimentConfig) -> Path:
        """
        Create directory structure for an experiment.

        Args:
            config: ExperimentConfig for the experiment

        Returns:
            Path to the experiment directory
        """
        # Create experiment directory name
        dir_name = f"{config.experiment_name}_v{config.version}"
        experiment_dir = self.config_dir.parent / "experiments" / dir_name

        # Create subdirectories
        experiment_dir.mkdir(parents=True, exist_ok=True)
        (experiment_dir / "data").mkdir(exist_ok=True)
        (experiment_dir / "models").mkdir(exist_ok=True)
        (experiment_dir / "results").mkdir(exist_ok=True)
        (experiment_dir / "logs").mkdir(exist_ok=True)
        (experiment_dir / "plots").mkdir(exist_ok=True)

        # Save config in experiment directory
        self.save_config(config, experiment_dir / "config")

        logger.info(f"Experiment directory created: {experiment_dir}")
        return experiment_dir


def load_config_from_file(filepath: str) -> ExperimentConfig:
    """
    Convenience function to load configuration from file.

    Args:
        filepath: Path to configuration file

    Returns:
        ExperimentConfig object
    """
    manager = ConfigManager()
    return manager.load_config(filepath)


def save_config_to_file(config: ExperimentConfig,
                       config_dir: str = "configs",
                       filename: Optional[str] = None) -> str:
    """
    Convenience function to save configuration to file.

    Args:
        config: ExperimentConfig to save
        config_dir: Directory to save configuration
        filename: Optional filename

    Returns:
        Path to saved configuration file
    """
    manager = ConfigManager(config_dir)
    return manager.save_config(config, filename)


if __name__ == "__main__":
    # Example usage
    config_manager = ConfigManager()

    # Create a sample configuration
    config = config_manager.create_config(
        experiment_name="baseline_ecg_detection",
        description="Baseline 1D CNN for ECG arrhythmia detection on MIT-BIH dataset",
        epochs=15,
        batch_size=64,
        apply_pruning=True,
        pruning_sparsity=0.7,
        apply_quantization=True,
        quantization_type="int8"
    )

    # Validate configuration
    is_valid, errors = config_manager.validate_config(config)
    print(f"Configuration valid: {is_valid}")
    if not is_valid:
        print(f"Errors: {errors}")

    # Save configuration
    config_path = config_manager.save_config(config)
    print(f"Configuration saved to: {config_path}")

    # Load configuration back
    loaded_config = config_manager.load_config(config_path)
    print(f"Loaded experiment name: {loaded_config.experiment_name}")
    print(f"Loaded epochs: {loaded_config.epochs}")

    # Create experiment directory
    exp_dir = config_manager.create_experiment_directory(config)
    print(f"Experiment directory created: {exp_dir}")