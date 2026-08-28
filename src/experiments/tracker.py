"""
Experiment tracking utilities for Beat2Bit project.
Manages experiment logging, configuration, and results tracking.
"""

import os
import json
import yaml
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Tracks experiments and maintains reproducible records."""

    def __init__(self, base_dir: str = "experiments"):
        """
        Initialize experiment tracker.

        Args:
            base_dir: Base directory for experiment tracking
        """
        self.base_dir = base_dir
        self.experiments_dir = os.path.join(base_dir, "experiments")
        self.registry_file = os.path.join(base_dir, "registry.json")

        # Create directories
        os.makedirs(self.experiments_dir, exist_ok=True)

        # Load or create registry
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        """Load experiment registry from file."""
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load registry: {e}")
                return {"experiments": {}, "last_updated": None}
        else:
            return {"experiments": {}, "last_updated": None}

    def _save_registry(self):
        """Save experiment registry to file."""
        self.registry["last_updated"] = datetime.now().isoformat()
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def create_experiment(self,
                         name: str,
                         config: Dict[str, Any],
                         description: str = "",
                         tags: Optional[List[str]] = None) -> str:
        """
        Create a new experiment record.

        Args:
            name: Experiment name
            config: Experiment configuration
            description: Experiment description
            tags: Optional tags for categorization

        Returns:
            Experiment ID
        """
        # Generate unique experiment ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_hash = hashlib.md5(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()[:8]
        exp_id = f"{name}_{timestamp}_{config_hash}"

        # Create experiment directory
        exp_dir = os.path.join(self.experiments_dir, exp_id)
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "data"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "results"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)

        # Create experiment record
        experiment_record = {
            "experiment_id": exp_id,
            "name": name,
            "description": description,
            "config": config,
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "status": "created",
            "directory": exp_dir,
            "artifacts": {},
            "metrics": {},
            "git_info": self._get_git_info()
        }

        # Save experiment record
        self._save_experiment_record(experiment_record)

        # Add to registry
        self.registry["experiments"][exp_id] = {
            "name": name,
            "created_at": experiment_record["created_at"],
            "status": "created",
            "directory": exp_dir
        }
        self._save_registry()

        logger.info(f"Created experiment: {exp_id}")
        return exp_id

    def _save_experiment_record(self, record: Dict[str, Any]):
        """Save experiment record to its directory."""
        record_path = os.path.join(record["directory"], "experiment.json")
        with open(record_path, 'w') as f:
            json.dump(record, f, indent=2)

    def _load_experiment_record(self, exp_id: str) -> Optional[Dict[str, Any]]:
        """Load experiment record from its directory."""
        if exp_id not in self.registry["experiments"]:
            return None

        exp_dir = self.registry["experiments"][exp_id]["directory"]
        record_path = os.path.join(exp_dir, "experiment.json")

        if os.path.exists(record_path):
            try:
                with open(record_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load experiment record {exp_id}: {e}")
                return None
        return None

    def update_experiment_status(self, exp_id: str, status: str):
        """
        Update experiment status.

        Args:
            exp_id: Experiment ID
            status: New status (created, running, completed, failed, interrupted)
        """
        record = self._load_experiment_record(exp_id)
        if record is None:
            logger.warning(f"Experiment {exp_id} not found")
            return False

        record["status"] = status
        record["updated_at"] = datetime.now().isoformat()
        self._save_experiment_record(record)

        # Update registry
        if exp_id in self.registry["experiments"]:
            self.registry["experiments"][exp_id]["status"] = status
            self._save_registry()

        logger.info(f"Updated experiment {exp_id} status to: {status}")
        return True

    def log_metrics(self, exp_id: str, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics for an experiment.

        Args:
            exp_id: Experiment ID
            metrics: Dictionary of metric names and values
            step: Optional step/epoch number
        """
        record = self._load_experiment_record(exp_id)
        if record is None:
            logger.warning(f"Experiment {exp_id} not found")
            return False

        # Initialize metrics storage if needed
        if "metrics_history" not in record:
            record["metrics_history"] = []

        # Add timestamp and step to metrics
        metrics_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "metrics": metrics
        }
        record["metrics_history"].append(metrics_entry)

        # Also update latest metrics
        record["metrics"].update(metrics)
        record["updated_at"] = datetime.now().isoformat()

        self._save_experiment_record(record)
        logger.info(f"Logged metrics for experiment {exp_id}")
        return True

    def log_artifact(self, exp_id: str, name: str, file_path: str,
                    artifact_type: str = "file", description: str = ""):
        """
        Log an artifact for an experiment.

        Args:
            exp_id: Experiment ID
            name: Artifact name
            file_path: Path to the artifact (relative to experiment directory)
            artifact_type: Type of artifact (file, model, plot, data, etc.)
            description: Description of the artifact
        """
        record = self._load_experiment_record(exp_id)
        if record is None:
            logger.warning(f"Experiment {exp_id} not found")
            return False

        # Ensure artifact directory exists
        artifact_dir = os.path.join(record["directory"], "artifacts")
        os.makedirs(artifact_dir, exist_ok=True)

        # Copy or reference the artifact
        if os.path.exists(file_path):
            # For simplicity, we'll just store the reference
            # In a more advanced system, we might copy the file
            artifact_record = {
                "name": name,
                "path": file_path,
                "type": artifact_type,
                "description": description,
                "logged_at": datetime.now().isoformat(),
                "size": os.path.getsize(file_path) if os.path.isfile(file_path) else None
            }

            # Initialize artifacts storage if needed
            if "artifacts" not in record:
                record["artifacts"] = {}

            record["artifacts"][name] = artifact_record
            record["updated_at"] = datetime.now().isoformat()

            self._save_experiment_record(record)
            logger.info(f"Logged artifact '{name}' for experiment {exp_id}")
            return True
        else:
            logger.warning(f"Artifact file not found: {file_path}")
            return False

    def get_experiment(self, exp_id: str) -> Optional[Dict[str, Any]]:
        """
        Get experiment record by ID.

        Args:
            exp_id: Experiment ID

        Returns:
            Experiment record or None if not found
        """
        return self._load_experiment_record(exp_id)

    def list_experiments(self, status: Optional[str] = None,
                        tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        List experiments with optional filtering.

        Args:
            status: Filter by status (optional)
            tags: Filter by tags (optional)

        Returns:
            List of experiment records
        """
        experiments = []

        for exp_id, exp_info in self.registry["experiments"].items():
            # Apply filters
            if status and exp_info["status"] != status:
                continue
            if tags:
                exp_record = self._load_experiment_record(exp_id)
                if exp_record is None:
                    continue
                exp_tags = set(exp_record.get("tags", []))
                if not set(tags).issubset(exp_tags):
                    continue

            # Load full record
            exp_record = self._load_experiment_record(exp_id)
            if exp_record:
                experiments.append(exp_record)

        # Sort by creation time (newest first)
        experiments.sort(
            key=lambda x: x.get("created_at", ""),
            reverse=True
        )
        return experiments

    def get_completed_experiments(self) -> List[Dict[str, Any]]:
        """Get all completed experiments."""
        return self.list_experiments(status="completed")

    def delete_experiment(self, exp_id: str) -> bool:
        """
        Delete an experiment and its data.

        Args:
            exp_id: Experiment ID

        Returns:
            True if deleted successfully
        """
        record = self._load_experiment_record(exp_id)
        if record is None:
            logger.warning(f"Experiment {exp_id} not found")
            return False

        # Remove experiment directory
        exp_dir = record["directory"]
        if os.path.exists(exp_dir):
            import shutil
            shutil.rmtree(exp_dir)
            logger.info(f"Removed experiment directory: {exp_dir}")

        # Remove from registry
        if exp_id in self.registry["experiments"]:
            del self.registry["experiments"][exp_id]
            self._save_registry()

        logger.info(f"Deleted experiment: {exp_id}")
        return True

    def export_experiment(self, exp_id: str, export_path: str) -> bool:
        """
        Export experiment to a tarball or zip file.

        Args:
            exp_id: Experiment ID
            export_path: Path for export file

        Returns:
            True if exported successfully
        """
        record = self._load_experiment_record(exp_id)
        if record is None:
            logger.warning(f"Experiment {exp_id} not found")
            return False

        exp_dir = record["directory"]

        try:
            import shutil
            # Create tar.gz archive
            archive_path = shutil.make_archive(
                export_path.replace('.tar.gz', '').replace('.tgz', ''),
                'gztar',
                exp_dir
            )
            logger.info(f"Exported experiment {exp_id} to {archive_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export experiment {exp_id}: {e}")
            return False

    def _get_git_info(self) -> Dict[str, Any]:
        """Get git repository information."""
        git_info = {
            "commit": None,
            "branch": None,
            "remote_url": None,
            "dirty": False
        }

        try:
            import subprocess
            # Get current commit
            git_info["commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()

            # Get current branch
            git_info["branch"] = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()

            # Get remote URL
            try:
                git_info["remote_url"] = subprocess.check_output(
                    ["git", "config", "--get", "remote.origin.url"],
                    stderr=subprocess.DEVNULL
                ).decode().strip()
            except subprocess.CalledProcessError:
                pass  # No remote configured

            # Check if working directory is dirty
            try:
                subprocess.check_output(
                    ["git", "diff-index", "--quiet", "HEAD", "--"],
                    stderr=subprocess.DEVNULL
                )
                # If we get here, no differences
            except subprocess.CalledProcessError:
                git_info["dirty"] = True

        except Exception as e:
            logger.warning(f"Could not get git info: {e}")

        return git_info


def create_experiment_from_config(config_path: str,
                                name: Optional[str] = None,
                                description: Optional[str] = None) -> str:
    """
    Create experiment from configuration file.

    Args:
        config_path: Path to configuration file (YAML or JSON)
        name: Optional experiment name (overrides config)
        description: Optional experiment description (overrides config)

    Returns:
        Experiment ID
    """
    # Load configuration
    if config_path.endswith(('.yaml', '.yml')):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError("Configuration file must be YAML or JSON")

    # Override name and description if provided
    if name is None:
        name = config.get('experiment_name', 'unnamed_experiment')
    if description is None:
        description = config.get('description', '')

    # Remove experiment-specific fields from config if they exist
    config_for_experiment = config.copy()
    config_for_experiment.pop('experiment_name', None)
    config_for_experiment.pop('description', None)

    # Create experiment
    tracker = ExperimentTracker()
    return tracker.create_experiment(
        name=name,
        config=config_for_experiment,
        description=description
    )


if __name__ == "__main__":
    # Example usage
    tracker = ExperimentTracker()

    # Create an experiment
    config = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100,
        "model_architecture": "baseline_1d_cnn",
        "dataset": "MIT-BIH",
        "apply_pruning": True,
        "pruning_sparsity": 0.7,
        "apply_quantization": True,
        "quantization_type": "int8"
    }

    exp_id = tracker.create_experiment(
        name="ecg_arrhythmia_detection",
        config=config,
        description="Baseline model with pruning and quantization for ECG arrhythmia detection",
        tags=["baseline", "pruning", "quantization"]
    )

    print(f"Created experiment: {exp_id}")

    # Update status
    tracker.update_experiment_status(exp_id, "running")

    # Log some metrics
    tracker.log_metrics(exp_id, {
        "accuracy": 0.87,
        "loss": 0.34,
        "val_accuracy": 0.85,
        "val_loss": 0.38
    }, step=10)

    tracker.log_metrics(exp_id, {
        "accuracy": 0.92,
        "loss": 0.18,
        "val_accuracy": 0.90,
        "val_loss": 0.22
    }, step=20)

    # Log an artifact (example)
    # tracker.log_artifact(exp_id, "model_weights", "./models/baseline.h5", "model", "Trained model weights")

    # Complete experiment
    tracker.update_experiment_status(exp_id, "completed")

    # List experiments
    experiments = tracker.list_experiments()
    print(f"\nTotal experiments: {len(experiments)}")
    for exp in experiments[:3]:  # Show first 3
        print(f"- {exp['experiment_id']}: {exp['name']} ({exp['status']})")