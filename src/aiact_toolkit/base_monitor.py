"""
Base Monitor Module

Provides the abstract base class for framework-specific monitors (PyTorch, TensorFlow).
Centralizes common functionality to reduce code duplication.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datetime import datetime
from pathlib import Path
from .metadata_storage import MetadataStorage


class BaseFrameworkMonitor(ABC):
    """
    Abstract base class for ML framework monitors.

    Provides common functionality for metadata capture including:
    - Lifecycle management (start/stop)
    - Training configuration logging
    - Training metrics logging
    - Metadata persistence
    """

    framework_name: str = "Unknown"

    def __init__(self, system_name: str, output_dir: str = "."):
        """
        Initialize framework monitor.

        Args:
            system_name: Name of the AI system being monitored
            output_dir: Directory for output files
        """
        self.system_name = system_name
        self.output_dir = Path(output_dir)
        self.storage = MetadataStorage(system_name)
        self.is_active = False

    def start(self):
        """Start monitoring framework operations."""
        self.is_active = True
        self.storage.metadata["framework"] = self.framework_name
        self.storage.metadata["framework_components"] = []

    def stop(self):
        """Stop monitoring framework operations."""
        self.is_active = False

    def log_training_config(self, **kwargs):
        """
        Log training configuration parameters.

        Args:
            **kwargs: Training configuration parameters (epochs, batch_size, etc.)
        """
        if not self.is_active:
            return

        config_info = {
            "timestamp": datetime.now().isoformat(),
            "component_type": "training_config",
            "parameters": kwargs
        }
        self.storage.metadata["framework_components"].append(config_info)

    def log_training_metrics(self, epoch: int, metrics: Dict[str, float]):
        """
        Log training metrics for an epoch.

        Args:
            epoch: Epoch number
            metrics: Dictionary of metric names to values
        """
        if not self.is_active:
            return

        if "training_history" not in self.storage.metadata:
            self.storage.metadata["training_history"] = []

        self.storage.metadata["training_history"].append({
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "metrics": metrics
        })

    def get_metadata(self) -> Dict[str, Any]:
        """Get all collected metadata."""
        return self.storage.get_all_metadata()

    def save_to_file(self, filename: str):
        """
        Save metadata to JSON file.

        Args:
            filename: Output filename
        """
        output_path = self.output_dir / filename
        self.storage.save_to_file(str(output_path))

    @abstractmethod
    def register_model(self, model: Any, name: Optional[str] = None,
                      description: Optional[str] = None):
        """Register a model and extract its metadata."""
        pass

    @abstractmethod
    def register_optimizer(self, optimizer: Any, name: Optional[str] = None):
        """Register optimizer configuration."""
        pass

    @abstractmethod
    def register_loss_function(self, loss_fn: Any, name: Optional[str] = None):
        """Register loss function."""
        pass

    @abstractmethod
    def register_dataset(self, dataset: Any, name: str, split: str = "train",
                        description: Optional[str] = None):
        """Register dataset information."""
        pass
