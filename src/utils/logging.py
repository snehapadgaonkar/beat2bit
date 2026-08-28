"""
Logging utilities for Beat2Bit experiments.
Provides structured logging for experiment tracking and reproducibility.
"""

import logging
import os
import sys
from typing import Optional, Dict, Any
from datetime import datetime
import json


class ExperimentLogger:
    """Structured logger for experiment tracking."""

    def __init__(self,
                 experiment_name: str,
                 log_dir: str = "logs",
                 log_level: int = logging.INFO,
                 console_output: bool = True,
                 file_output: bool = True):
        """
        Initialize experiment logger.

        Args:
            experiment_name: Name of the experiment
            log_dir: Directory to store log files
            log_level: Logging level
            console_output: Whether to output to console
            file_output: Whether to output to file
        """
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.console_output = console_output
        self.file_output = file_output

        # Create logger
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(log_level)
        self.logger.handlers.clear()  # Clear any existing handlers

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(level)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Add console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # Add file handler
        if file_output:
            os.makedirs(log_dir, exist_ok=True)
            log_filename = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            log_path = os.path.join(log_dir, log_filename)
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # Prevent propagation to root logger
        self.logger.propagate = False

        self.logger.info(f"Experiment logger initialized for: {experiment_name}")

    def info(self, message: str, **kwargs):
        """Log info message with optional structured data."""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.info(message)

    def warning(self, message: str, **kwargs):
        """Log warning message with optional structured data."""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.warning(message)

    def error(self, message: str, **kwargs):
        """Log error message with optional structured data."""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.error(message)

    def debug(self, message: str, **kwargs):
        """Log debug message with optional structured data."""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.debug(message)

    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters in a structured format."""
        self.logger.info("HYPERPARAMETERS | " + json.dumps(params, indent=2))

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics in a structured format."""
        log_data = {"metrics": metrics}
        if step is not None:
            log_data["step"] = step
        self.logger.info("METRICS | " + json.dumps(log_data, indent=2))

    def log_artifact(self, artifact_name: str, artifact_path: str):
        """Log artifact information."""
        self.logger.info(f"ARTIFACT | {artifact_name}: {artifact_path}")

    def log_git_info(self, repo_path: str = "."):
        """Log git repository information."""
        try:
            import subprocess
            # Get current commit
            commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                cwd=repo_path,
                stderr=subprocess.DEVNULL
            ).decode().strip()

            # Get current branch
            branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=repo_path,
                stderr=subprocess.DEVNULL
            ).decode().strip()

            # Get commit message
            message = subprocess.check_output(
                ['git', 'log', '-1', '--pretty=format:%s'],
                cwd=repo_path,
                stderr=subprocess.DEVNULL
            ).decode().strip()

            self.logger.info(f"GIT_INFO | commit: {commit}, branch: {branch}, message: '{message}'")
        except Exception as e:
            self.logger.warning(f"Could not retrieve git info: {e}")

    def log_environment_info(self):
        """Log environment information."""
        try:
            import platform
            import sys

            env_info = {
                "platform": platform.platform(),
                "python_version": sys.version,
                "executable": sys.executable
            }

            # Try to get package versions
            try:
                import tensorflow as tf
                env_info["tensorflow_version"] = tf.__version__
            except ImportError:
                env_info["tensorflow_version"] = "not installed"

            try:
                import numpy as np
                env_info["numpy_version"] = np.__version__
            except ImportError:
                env_info["numpy_version"] = "not installed"

            self.logger.info("ENVIRONMENT_INFO | " + json.dumps(env_info, indent=2))
        except Exception as e:
            self.logger.warning(f"Could not retrieve environment info: {e}")


def setup_experiment_logging(experiment_name: str,
                           log_dir: str = "logs",
                           console_output: bool = True,
                           file_output: bool = True) -> ExperimentLogger:
    """
    Set up experiment logging.

    Args:
        experiment_name: Name of the experiment
        log_dir: Directory to store log files
        console_output: Whether to output to console
        file_output: Whether to output to file

    Returns:
        Configured ExperimentLogger instance
    """
    return ExperimentLogger(
        experiment_name=experiment_name,
        log_dir=log_dir,
        console_output=console_output,
        file_output=file_output
    )


class ExperimentTracker:
    """Tracks experiment runs and maintains history."""

    def __init__(self, tracking_dir: str = "experiments"):
        """
        Initialize experiment tracker.

        Args:
            tracking_dir: Directory to store experiment tracking data
        """
        self.tracking_dir = tracking_dir
        os.makedirs(tracking_dir, exist_ok=True)
        self.history_file = os.path.join(tracking_dir, "experiment_history.json")
        self._load_history()

    def _load_history(self):
        """Load experiment history from file."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    self.history = json.load(f)
            except Exception:
                self.history = []
        else:
            self.history = []

    def _save_history(self):
        """Save experiment history to file."""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def log_experiment_start(self,
                           experiment_name: str,
                           config: Dict[str, Any],
                           git_commit: Optional[str] = None) -> str:
        """
        Log the start of an experiment.

        Args:
            experiment_name: Name of the experiment
            config: Experiment configuration
            git_commit: Git commit hash (optional)

        Returns:
            Experiment ID
        """
        experiment_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        experiment_record = {
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "config": config,
            "git_commit": git_commit,
            "status": "running",
            "metrics": {},
            "artifacts": []
        }

        self.history.append(experiment_record)
        self._save_history()

        return experiment_id

    def log_experiment_end(self,
                         experiment_id: str,
                         status: str = "completed",
                         metrics: Optional[Dict[str, Any]] = None,
                         artifacts: Optional[list] = None):
        """
        Log the end of an experiment.

        Args:
            experiment_id: Experiment ID
            status: Experiment status (completed, failed, interrupted)
            metrics: Final metrics (optional)
            artifacts: List of artifacts (optional)
        """
        for exp in self.history:
            if exp["experiment_id"] == experiment_id:
                exp["end_time"] = datetime.now().isoformat()
                exp["status"] = status
                if metrics:
                    exp["metrics"] = metrics
                if artifacts:
                    exp["artifacts"] = artifacts
                break

        self._save_history()

    def get_experiment_history(self) -> list:
        """Get experiment history."""
        return self.history.copy()

    def get_experiment_by_id(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment record by ID."""
        for exp in self.history:
            if exp["experiment_id"] == experiment_id:
                return exp.copy()
        return None

    def get_experiments_by_name(self, experiment_name: str) -> list:
        """Get all experiments with given name."""
        return [exp.copy() for exp in self.history
                if exp["experiment_name"] == experiment_name]


if __name__ == "__main__":
    # Example usage
    logger = setup_experiment_logging("test_experiment")
    logger.info("This is an info message")
    logger.warning("This is a warning message", component="data_loader")
    logger.error("This is an error message", error_code=500)

    # Log hyperparameters
    params = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "optimizer": "Adam"
    }
    logger.log_hyperparameters(params)

    # Log metrics
    metrics = {
        "accuracy": 0.95,
        "loss": 0.05,
        "f1_score": 0.93
    }
    logger.log_metrics(metrics, step=100)

    # Test experiment tracker
    tracker = ExperimentTracker()
    exp_id = tracker.log_experiment_start(
        "test_experiment",
        {"learning_rate": 0.001, "batch_size": 32},
        git_commit="abc123"
    )
    print(f"Started experiment: {exp_id}")

    tracker.log_experiment_end(
        exp_id,
        status="completed",
        metrics={"accuracy": 0.95},
        artifacts=["model.h5", "results.json"]
    )

    history = tracker.get_experiment_history()
    print(f"Tracking {len(history)} experiments")