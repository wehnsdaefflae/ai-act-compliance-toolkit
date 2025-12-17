"""
PyTorch Monitor

This module provides monitoring capabilities for PyTorch models and training operations
to support EU AI Act compliance documentation.

It captures:
- Model architecture and parameters
- Training configurations (optimizer, loss function, learning rate)
- Dataset information
- Hardware utilization (GPU/CPU)
- Hyperparameters
"""

from typing import Any, Dict, Optional, List
from datetime import datetime
import json
from pathlib import Path
from .metadata_storage import MetadataStorage


class PyTorchMonitor:
    """
    Monitor for PyTorch models that captures compliance-relevant metadata.

    Usage:
        monitor = PyTorchMonitor(system_name="my_pytorch_model")
        monitor.start()

        # Register your model
        monitor.register_model(model, name="ResNet50")

        # Register training components
        monitor.register_optimizer(optimizer)
        monitor.register_dataset(train_dataset, name="training_data", split="train")

        # Capture training configuration
        monitor.log_training_config(epochs=100, batch_size=32, learning_rate=0.001)

        # Save metadata
        monitor.save_to_file("pytorch_metadata.json")
    """

    def __init__(self, system_name: str, output_dir: str = "."):
        """
        Initialize PyTorch monitor.

        Args:
            system_name: Name of the AI system being monitored
            output_dir: Directory for output files
        """
        self.system_name = system_name
        self.output_dir = Path(output_dir)
        self.storage = MetadataStorage(system_name)
        self.is_active = False

    def start(self):
        """Start monitoring PyTorch operations."""
        self.is_active = True
        self.storage.metadata["framework"] = "PyTorch"
        self.storage.metadata["framework_components"] = []

    def stop(self):
        """Stop monitoring PyTorch operations."""
        self.is_active = False

    def register_model(self, model: Any, name: Optional[str] = None,
                      description: Optional[str] = None):
        """
        Register a PyTorch model and extract its metadata.

        Args:
            model: PyTorch model (torch.nn.Module)
            name: Optional model name
            description: Optional model description
        """
        if not self.is_active:
            return

        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("PyTorch is required but not installed. Install with: pip install torch")

        if not isinstance(model, nn.Module):
            raise ValueError("Model must be a torch.nn.Module instance")

        # Extract model information
        model_info = {
            "timestamp": datetime.now().isoformat(),
            "model_name": name or model.__class__.__name__,
            "model_type": "pytorch_model",
            "provider": "PyTorch",
            "framework_component": "Model",
            "description": description or f"PyTorch {model.__class__.__name__}",
        }

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        model_info["parameters"] = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
            "architecture": str(model.__class__.__name__),
        }

        # Extract layer information
        layers = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Only leaf modules
                layers.append({
                    "name": name,
                    "type": module.__class__.__name__
                })

        model_info["architecture_details"] = {
            "layers": layers[:20],  # Limit to first 20 layers for readability
            "total_layers": len(layers)
        }

        # Check if model is on GPU
        try:
            device = next(model.parameters()).device
            model_info["parameters"]["device"] = str(device)
        except StopIteration:
            model_info["parameters"]["device"] = "unknown"

        self.storage.add_model(model_info)

    def register_optimizer(self, optimizer: Any, name: Optional[str] = None):
        """
        Register optimizer configuration.

        Args:
            optimizer: PyTorch optimizer instance
            name: Optional optimizer name
        """
        if not self.is_active:
            return

        try:
            import torch.optim as optim
        except ImportError:
            raise ImportError("PyTorch is required but not installed. Install with: pip install torch")

        optimizer_info = {
            "timestamp": datetime.now().isoformat(),
            "component_type": "optimizer",
            "optimizer_name": name or optimizer.__class__.__name__,
            "parameters": {}
        }

        # Extract optimizer hyperparameters
        if hasattr(optimizer, 'defaults'):
            optimizer_info["parameters"] = optimizer.defaults.copy()

        # Get current learning rates from param groups
        if hasattr(optimizer, 'param_groups'):
            learning_rates = [group['lr'] for group in optimizer.param_groups]
            optimizer_info["parameters"]["learning_rates"] = learning_rates

        self.storage.metadata["framework_components"].append(optimizer_info)

    def register_loss_function(self, loss_fn: Any, name: Optional[str] = None):
        """
        Register loss function.

        Args:
            loss_fn: PyTorch loss function
            name: Optional loss function name
        """
        if not self.is_active:
            return

        loss_info = {
            "timestamp": datetime.now().isoformat(),
            "component_type": "loss_function",
            "loss_name": name or (loss_fn.__class__.__name__ if hasattr(loss_fn, '__class__') else str(loss_fn)),
        }

        self.storage.metadata["framework_components"].append(loss_info)

    def register_dataset(self, dataset: Any, name: str, split: str = "train",
                        description: Optional[str] = None):
        """
        Register dataset information.

        Args:
            dataset: PyTorch dataset instance
            name: Dataset name
            split: Dataset split (train/val/test)
            description: Optional dataset description
        """
        if not self.is_active:
            return

        dataset_info = {
            "timestamp": datetime.now().isoformat(),
            "source_name": name,
            "source_type": "pytorch_dataset",
            "data_type": dataset.__class__.__name__,
            "split": split,
            "description": description or f"PyTorch {dataset.__class__.__name__}",
        }

        # Try to get dataset size
        try:
            dataset_info["size"] = len(dataset)
        except (TypeError, AttributeError):
            dataset_info["size"] = "unknown"

        self.storage.add_data_source(dataset_info)

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

    def register_data_loader(self, data_loader: Any, name: str):
        """
        Register DataLoader configuration.

        Args:
            data_loader: PyTorch DataLoader instance
            name: DataLoader name
        """
        if not self.is_active:
            return

        loader_info = {
            "timestamp": datetime.now().isoformat(),
            "component_type": "data_loader",
            "name": name,
            "parameters": {
                "batch_size": getattr(data_loader, 'batch_size', None),
                "num_workers": getattr(data_loader, 'num_workers', None),
                "shuffle": getattr(data_loader, 'shuffle', None),
                "drop_last": getattr(data_loader, 'drop_last', None),
            }
        }

        # Remove None values
        loader_info["parameters"] = {k: v for k, v in loader_info["parameters"].items()
                                     if v is not None}

        self.storage.metadata["framework_components"].append(loader_info)

    def get_metadata(self) -> Dict[str, Any]:
        """Get all collected metadata."""
        return self.storage.get_metadata()

    def save_to_file(self, filename: str):
        """
        Save metadata to JSON file.

        Args:
            filename: Output filename
        """
        output_path = self.output_dir / filename
        self.storage.save_to_file(str(output_path))

    def load_from_file(self, filename: str):
        """
        Load metadata from JSON file.

        Args:
            filename: Input filename
        """
        input_path = self.output_dir / filename
        self.storage.load_from_file(str(input_path))
