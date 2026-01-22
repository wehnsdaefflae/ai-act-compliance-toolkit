"""
PyTorch Monitor

Monitoring capabilities for PyTorch models and training operations
to support EU AI Act compliance documentation.

Captures: model architecture, training configurations, dataset info, hyperparameters.
"""

from typing import Any, Dict, Optional
from datetime import datetime
from .base_monitor import BaseFrameworkMonitor


class PyTorchMonitor(BaseFrameworkMonitor):
    """
    Monitor for PyTorch models that captures compliance-relevant metadata.

    Usage:
        monitor = PyTorchMonitor(system_name="my_pytorch_model")
        monitor.start()
        monitor.register_model(model, name="ResNet50")
        monitor.register_optimizer(optimizer)
        monitor.register_dataset(train_dataset, name="training_data", split="train")
        monitor.log_training_config(epochs=100, batch_size=32)
        monitor.save_to_file("pytorch_metadata.json")
    """

    framework_name = "PyTorch"

    def register_model(self, model: Any, name: Optional[str] = None,
                      description: Optional[str] = None):
        """Register a PyTorch model and extract its metadata."""
        if not self.is_active:
            return

        try:
            import torch.nn as nn
        except ImportError:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        if not isinstance(model, nn.Module):
            raise ValueError("Model must be a torch.nn.Module instance")

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

        # Extract layer information (limit for readability)
        layers = [
            {"name": name, "type": module.__class__.__name__}
            for name, module in model.named_modules()
            if len(list(module.children())) == 0
        ]
        model_info["architecture_details"] = {
            "layers": layers[:20],
            "total_layers": len(layers)
        }

        # Check device
        try:
            device = next(model.parameters()).device
            model_info["parameters"]["device"] = str(device)
        except StopIteration:
            model_info["parameters"]["device"] = "unknown"

        self.storage.add_model(model_info)

    def register_optimizer(self, optimizer: Any, name: Optional[str] = None):
        """Register optimizer configuration."""
        if not self.is_active:
            return

        optimizer_info = {
            "timestamp": datetime.now().isoformat(),
            "component_type": "optimizer",
            "optimizer_name": name or optimizer.__class__.__name__,
            "parameters": {}
        }

        if hasattr(optimizer, 'defaults'):
            optimizer_info["parameters"] = optimizer.defaults.copy()

        if hasattr(optimizer, 'param_groups'):
            optimizer_info["parameters"]["learning_rates"] = [
                group['lr'] for group in optimizer.param_groups
            ]

        self.storage.metadata["framework_components"].append(optimizer_info)

    def register_loss_function(self, loss_fn: Any, name: Optional[str] = None):
        """Register loss function."""
        if not self.is_active:
            return

        loss_info = {
            "timestamp": datetime.now().isoformat(),
            "component_type": "loss_function",
            "loss_name": name or getattr(loss_fn, '__class__', type(loss_fn)).__name__,
        }
        self.storage.metadata["framework_components"].append(loss_info)

    def register_dataset(self, dataset: Any, name: str, split: str = "train",
                        description: Optional[str] = None):
        """Register dataset information."""
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

        try:
            dataset_info["size"] = len(dataset)
        except (TypeError, AttributeError):
            dataset_info["size"] = "unknown"

        self.storage.add_data_source(dataset_info)

    def register_data_loader(self, data_loader: Any, name: str):
        """Register DataLoader configuration."""
        if not self.is_active:
            return

        loader_info = {
            "timestamp": datetime.now().isoformat(),
            "component_type": "data_loader",
            "name": name,
            "parameters": {
                k: getattr(data_loader, k, None)
                for k in ['batch_size', 'num_workers', 'shuffle', 'drop_last']
            }
        }
        loader_info["parameters"] = {k: v for k, v in loader_info["parameters"].items() if v is not None}
        self.storage.metadata["framework_components"].append(loader_info)
