"""
TensorFlow Monitor

Monitoring capabilities for TensorFlow/Keras models and training operations
to support EU AI Act compliance documentation.

Captures: model architecture, training configurations, dataset info, callbacks.
"""

from typing import Any, Dict, Optional, List
from datetime import datetime
from .base_monitor import BaseFrameworkMonitor


class TensorFlowMonitor(BaseFrameworkMonitor):
    """
    Monitor for TensorFlow/Keras models that captures compliance-relevant metadata.

    Usage:
        monitor = TensorFlowMonitor(system_name="my_tf_model")
        monitor.start()
        monitor.register_model(model, name="MobileNetV2")
        monitor.register_optimizer(model.optimizer)
        monitor.log_training_config(epochs=50, batch_size=32)
        callback = monitor.create_keras_callback()
        model.fit(train_data, callbacks=[callback])
        monitor.save_to_file("tensorflow_metadata.json")
    """

    framework_name = "TensorFlow"

    def register_model(self, model: Any, name: Optional[str] = None,
                      description: Optional[str] = None):
        """Register a TensorFlow/Keras model and extract its metadata."""
        if not self.is_active:
            return

        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow is required. Install with: pip install tensorflow")

        if not isinstance(model, tf.keras.Model):
            raise ValueError("Model must be a tf.keras.Model instance")

        model_info = {
            "timestamp": datetime.now().isoformat(),
            "model_name": name or getattr(model, 'name', "unnamed_model"),
            "model_type": "tensorflow_model",
            "provider": "TensorFlow",
            "framework_component": "Model",
            "description": description or "TensorFlow/Keras model",
            "tensorflow_version": tf.__version__,
        }

        # Count parameters
        try:
            total_params = model.count_params()
            trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
            non_trainable = sum(tf.size(w).numpy() for w in model.non_trainable_weights)
            model_info["parameters"] = {
                "total_parameters": int(total_params),
                "trainable_parameters": int(trainable),
                "non_trainable_parameters": int(non_trainable),
            }
        except Exception:
            model_info["parameters"] = {"total_parameters": "unknown"}

        # Extract layer information
        layers = []
        for layer in model.layers:
            layer_config = {"name": layer.name, "type": layer.__class__.__name__}
            for attr in ['units', 'kernel_size', 'filters']:
                if hasattr(layer, attr):
                    layer_config[attr] = getattr(layer, attr)
            if hasattr(layer, 'activation') and layer.activation:
                layer_config["activation"] = getattr(layer.activation, '__name__', str(layer.activation))
            layers.append(layer_config)

        model_info["architecture_details"] = {
            "layers": layers[:30],
            "total_layers": len(layers),
            "input_shape": str(getattr(model, 'input_shape', "unknown")),
            "output_shape": str(getattr(model, 'output_shape', "unknown")),
        }

        # GPU info
        try:
            gpus = tf.config.list_physical_devices('GPU')
            model_info["parameters"]["available_gpus"] = len(gpus)
            model_info["parameters"]["gpu_names"] = [gpu.name for gpu in gpus]
        except Exception:
            model_info["parameters"]["available_gpus"] = 0

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

        if hasattr(optimizer, 'get_config'):
            try:
                config = optimizer.get_config()
                optimizer_info["parameters"] = {
                    k: float(v) if hasattr(v, 'item') else v
                    for k, v in config.items() if k != 'name'
                }
            except Exception:
                pass

        if hasattr(optimizer, 'learning_rate'):
            try:
                lr = optimizer.learning_rate
                optimizer_info["parameters"]["learning_rate"] = float(lr.numpy() if hasattr(lr, 'numpy') else lr)
            except Exception:
                pass

        self.storage.metadata["framework_components"].append(optimizer_info)

    def register_loss_function(self, loss: Any, name: Optional[str] = None):
        """Register loss function."""
        if not self.is_active:
            return

        if not name:
            if isinstance(loss, str):
                name = loss
            else:
                name = getattr(loss, '__name__', None) or getattr(loss, '__class__', type(loss)).__name__

        loss_info = {
            "timestamp": datetime.now().isoformat(),
            "component_type": "loss_function",
            "loss_name": name,
        }
        self.storage.metadata["framework_components"].append(loss_info)

    def register_metrics(self, metrics: List[Any]):
        """Register evaluation metrics."""
        if not self.is_active:
            return

        metric_names = []
        for metric in metrics:
            if isinstance(metric, str):
                metric_names.append(metric)
            else:
                metric_names.append(getattr(metric, '__name__', None) or metric.__class__.__name__)

        metrics_info = {
            "timestamp": datetime.now().isoformat(),
            "component_type": "metrics",
            "metric_names": metric_names,
        }
        self.storage.metadata["framework_components"].append(metrics_info)

    def register_dataset(self, dataset: Any, name: str, split: str = "train",
                        description: Optional[str] = None):
        """Register dataset information."""
        if not self.is_active:
            return

        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow is required. Install with: pip install tensorflow")

        dataset_info = {
            "timestamp": datetime.now().isoformat(),
            "source_name": name,
            "source_type": "tensorflow_dataset",
            "data_type": type(dataset).__name__,
            "split": split,
            "description": description or "TensorFlow dataset",
        }

        if isinstance(dataset, tf.data.Dataset):
            try:
                element_spec = dataset.element_spec
                if isinstance(element_spec, tuple):
                    dataset_info["input_shape"] = str(getattr(element_spec[0], 'shape', "unknown"))
                    dataset_info["output_shape"] = str(getattr(element_spec[1], 'shape', "unknown"))
                else:
                    dataset_info["element_shape"] = str(getattr(element_spec, 'shape', "unknown"))
            except Exception:
                pass

            try:
                cardinality = dataset.cardinality().numpy()
                dataset_info["size"] = int(cardinality) if cardinality >= 0 else "unknown"
            except Exception:
                dataset_info["size"] = "unknown"

        self.storage.add_data_source(dataset_info)

    def create_keras_callback(self):
        """Create a Keras callback for automatic monitoring during training."""
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow is required. Install with: pip install tensorflow")

        monitor = self

        class MonitorCallback(tf.keras.callbacks.Callback):
            """Keras callback for automatic monitoring."""

            def on_train_begin(self, logs=None):
                if monitor.is_active and not monitor.storage.metadata.get("models"):
                    monitor.register_model(self.model)

            def on_epoch_end(self, epoch, logs=None):
                if monitor.is_active and logs:
                    monitor.log_training_metrics(epoch, logs)

        return MonitorCallback()
