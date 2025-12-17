"""
TensorFlow Monitor

This module provides monitoring capabilities for TensorFlow/Keras models and training operations
to support EU AI Act compliance documentation.

It captures:
- Model architecture and parameters
- Training configurations (optimizer, loss function, learning rate)
- Dataset information
- Hardware utilization (GPU/CPU/TPU)
- Hyperparameters
- Callbacks and training history
"""

from typing import Any, Dict, Optional, List
from datetime import datetime
import json
from pathlib import Path
from .metadata_storage import MetadataStorage


class TensorFlowMonitor:
    """
    Monitor for TensorFlow/Keras models that captures compliance-relevant metadata.

    Usage:
        monitor = TensorFlowMonitor(system_name="my_tf_model")
        monitor.start()

        # Register your model
        monitor.register_model(model, name="MobileNetV2")

        # Register training components
        monitor.register_optimizer(model.optimizer)
        monitor.register_dataset(train_dataset, name="training_data", split="train")

        # Log training configuration
        monitor.log_training_config(epochs=50, batch_size=32)

        # Can also use as a Keras callback
        callback = monitor.create_keras_callback()
        model.fit(train_data, callbacks=[callback])

        # Save metadata
        monitor.save_to_file("tensorflow_metadata.json")
    """

    def __init__(self, system_name: str, output_dir: str = "."):
        """
        Initialize TensorFlow monitor.

        Args:
            system_name: Name of the AI system being monitored
            output_dir: Directory for output files
        """
        self.system_name = system_name
        self.output_dir = Path(output_dir)
        self.storage = MetadataStorage(system_name)
        self.is_active = False

    def start(self):
        """Start monitoring TensorFlow operations."""
        self.is_active = True
        self.storage.metadata["framework"] = "TensorFlow"
        self.storage.metadata["framework_components"] = []

    def stop(self):
        """Stop monitoring TensorFlow operations."""
        self.is_active = False

    def register_model(self, model: Any, name: Optional[str] = None,
                      description: Optional[str] = None):
        """
        Register a TensorFlow/Keras model and extract its metadata.

        Args:
            model: TensorFlow/Keras model
            name: Optional model name
            description: Optional model description
        """
        if not self.is_active:
            return

        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow is required but not installed. Install with: pip install tensorflow")

        if not isinstance(model, tf.keras.Model):
            raise ValueError("Model must be a tf.keras.Model instance")

        # Extract model information
        model_info = {
            "timestamp": datetime.now().isoformat(),
            "model_name": name or (model.name if hasattr(model, 'name') else "unnamed_model"),
            "model_type": "tensorflow_model",
            "provider": "TensorFlow",
            "framework_component": "Model",
            "description": description or f"TensorFlow/Keras model",
        }

        # Count parameters
        try:
            total_params = model.count_params()
            trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
            non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])

            model_info["parameters"] = {
                "total_parameters": int(total_params),
                "trainable_parameters": int(trainable_params),
                "non_trainable_parameters": int(non_trainable_params),
            }
        except Exception:
            model_info["parameters"] = {"total_parameters": "unknown"}

        # Extract layer information
        layers = []
        for layer in model.layers:
            layer_config = {
                "name": layer.name,
                "type": layer.__class__.__name__,
            }

            # Add layer-specific configuration
            if hasattr(layer, 'units'):
                layer_config["units"] = layer.units
            if hasattr(layer, 'activation') and layer.activation:
                layer_config["activation"] = layer.activation.__name__ if hasattr(layer.activation, '__name__') else str(layer.activation)
            if hasattr(layer, 'kernel_size'):
                layer_config["kernel_size"] = layer.kernel_size
            if hasattr(layer, 'filters'):
                layer_config["filters"] = layer.filters

            layers.append(layer_config)

        model_info["architecture_details"] = {
            "layers": layers[:30],  # Limit for readability
            "total_layers": len(layers),
            "input_shape": str(model.input_shape) if hasattr(model, 'input_shape') else "unknown",
            "output_shape": str(model.output_shape) if hasattr(model, 'output_shape') else "unknown",
        }

        # Get TensorFlow version
        model_info["tensorflow_version"] = tf.__version__

        # Check device placement
        try:
            gpus = tf.config.list_physical_devices('GPU')
            model_info["parameters"]["available_gpus"] = len(gpus)
            model_info["parameters"]["gpu_names"] = [gpu.name for gpu in gpus]
        except Exception:
            model_info["parameters"]["available_gpus"] = 0

        self.storage.add_model(model_info)

    def register_optimizer(self, optimizer: Any, name: Optional[str] = None):
        """
        Register optimizer configuration.

        Args:
            optimizer: TensorFlow/Keras optimizer instance
            name: Optional optimizer name
        """
        if not self.is_active:
            return

        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow is required but not installed. Install with: pip install tensorflow")

        optimizer_info = {
            "timestamp": datetime.now().isoformat(),
            "component_type": "optimizer",
            "optimizer_name": name or optimizer.__class__.__name__,
            "parameters": {}
        }

        # Extract optimizer configuration
        if hasattr(optimizer, 'get_config'):
            try:
                config = optimizer.get_config()
                # Convert numpy types to native Python types
                optimizer_info["parameters"] = {
                    k: float(v) if hasattr(v, 'item') else v
                    for k, v in config.items()
                    if k not in ['name']  # Exclude redundant fields
                }
            except Exception:
                pass

        # Extract learning rate
        if hasattr(optimizer, 'learning_rate'):
            try:
                lr = optimizer.learning_rate
                if hasattr(lr, 'numpy'):
                    optimizer_info["parameters"]["learning_rate"] = float(lr.numpy())
                else:
                    optimizer_info["parameters"]["learning_rate"] = float(lr)
            except Exception:
                pass

        self.storage.metadata["framework_components"].append(optimizer_info)

    def register_loss_function(self, loss: Any, name: Optional[str] = None):
        """
        Register loss function.

        Args:
            loss: TensorFlow/Keras loss function or string
            name: Optional loss function name
        """
        if not self.is_active:
            return

        loss_name = name
        if not loss_name:
            if isinstance(loss, str):
                loss_name = loss
            elif hasattr(loss, '__name__'):
                loss_name = loss.__name__
            elif hasattr(loss, '__class__'):
                loss_name = loss.__class__.__name__
            else:
                loss_name = str(loss)

        loss_info = {
            "timestamp": datetime.now().isoformat(),
            "component_type": "loss_function",
            "loss_name": loss_name,
        }

        self.storage.metadata["framework_components"].append(loss_info)

    def register_metrics(self, metrics: List[Any]):
        """
        Register evaluation metrics.

        Args:
            metrics: List of metric functions or strings
        """
        if not self.is_active:
            return

        metric_names = []
        for metric in metrics:
            if isinstance(metric, str):
                metric_names.append(metric)
            elif hasattr(metric, '__name__'):
                metric_names.append(metric.__name__)
            elif hasattr(metric, '__class__'):
                metric_names.append(metric.__class__.__name__)
            else:
                metric_names.append(str(metric))

        metrics_info = {
            "timestamp": datetime.now().isoformat(),
            "component_type": "metrics",
            "metric_names": metric_names,
        }

        self.storage.metadata["framework_components"].append(metrics_info)

    def register_dataset(self, dataset: Any, name: str, split: str = "train",
                        description: Optional[str] = None):
        """
        Register dataset information.

        Args:
            dataset: TensorFlow dataset instance (tf.data.Dataset)
            name: Dataset name
            split: Dataset split (train/val/test)
            description: Optional dataset description
        """
        if not self.is_active:
            return

        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow is required but not installed. Install with: pip install tensorflow")

        dataset_info = {
            "timestamp": datetime.now().isoformat(),
            "source_name": name,
            "source_type": "tensorflow_dataset",
            "data_type": type(dataset).__name__,
            "split": split,
            "description": description or f"TensorFlow dataset",
        }

        # Try to get dataset information
        if isinstance(dataset, tf.data.Dataset):
            try:
                # Get element spec
                element_spec = dataset.element_spec
                if isinstance(element_spec, tuple):
                    dataset_info["input_shape"] = str(element_spec[0].shape) if hasattr(element_spec[0], 'shape') else "unknown"
                    dataset_info["output_shape"] = str(element_spec[1].shape) if hasattr(element_spec[1], 'shape') else "unknown"
                else:
                    dataset_info["element_shape"] = str(element_spec.shape) if hasattr(element_spec, 'shape') else "unknown"
            except Exception:
                pass

            # Try to get cardinality
            try:
                cardinality = dataset.cardinality().numpy()
                if cardinality >= 0:
                    dataset_info["size"] = int(cardinality)
            except Exception:
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

    def log_training_metrics(self, epoch: int, logs: Dict[str, float]):
        """
        Log training metrics for an epoch.

        Args:
            epoch: Epoch number
            logs: Dictionary of metric names to values
        """
        if not self.is_active:
            return

        if "training_history" not in self.storage.metadata:
            self.storage.metadata["training_history"] = []

        self.storage.metadata["training_history"].append({
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "metrics": logs
        })

    def create_keras_callback(self):
        """
        Create a Keras callback for automatic monitoring during training.

        Returns:
            MonitorCallback: Keras callback instance
        """
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow is required but not installed. Install with: pip install tensorflow")

        monitor = self

        class MonitorCallback(tf.keras.callbacks.Callback):
            """Keras callback for automatic monitoring."""

            def on_train_begin(self, logs=None):
                """Called at the beginning of training."""
                if monitor.is_active:
                    # Register model if not already registered
                    if not monitor.storage.metadata.get("models"):
                        monitor.register_model(self.model)

            def on_epoch_end(self, epoch, logs=None):
                """Called at the end of each epoch."""
                if monitor.is_active and logs:
                    monitor.log_training_metrics(epoch, logs)

        return MonitorCallback()

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
