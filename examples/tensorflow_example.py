"""
TensorFlow/Keras Integration Example

This example demonstrates how to use the AI Act Compliance Toolkit
with TensorFlow/Keras models to automatically capture compliance metadata.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aiact_toolkit import TensorFlowMonitor


def create_sample_model():
    """Create a sample TensorFlow/Keras model for demonstration."""
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        print("TensorFlow is not installed. Install with: pip install tensorflow")
        return None

    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(784,), name='dense_1'),
        keras.layers.Dropout(0.2, name='dropout_1'),
        keras.layers.Dense(64, activation='relu', name='dense_2'),
        keras.layers.Dropout(0.2, name='dropout_2'),
        keras.layers.Dense(10, activation='softmax', name='output')
    ], name='mnist_classifier')

    return model


def create_sample_dataset():
    """Create a sample TensorFlow dataset for demonstration."""
    try:
        import tensorflow as tf
        import numpy as np
    except ImportError:
        return None

    # Create dummy data
    X = np.random.randn(1000, 784).astype(np.float32)
    y = np.random.randint(0, 10, (1000,)).astype(np.int32)

    # Create tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.batch(32)

    return dataset


def main():
    """Main example demonstrating TensorFlow monitoring."""

    print("=" * 70)
    print("TensorFlow/Keras AI Act Compliance Toolkit - Example")
    print("=" * 70)
    print()

    # Initialize monitor
    print("1. Initializing TensorFlow monitor...")
    monitor = TensorFlowMonitor(system_name="mnist_keras_classifier")
    monitor.start()
    print("   ✓ Monitor started\n")

    # Create and register model
    print("2. Creating and registering Keras model...")
    model = create_sample_model()
    if model is None:
        print("   ✗ Could not create model (TensorFlow not installed)")
        return

    monitor.register_model(
        model,
        name="MNISTClassifier",
        description="Keras Sequential model for MNIST digit classification"
    )
    print("   ✓ Model registered\n")

    # Compile model and register components
    print("3. Compiling model and registering optimizer...")
    try:
        import tensorflow as tf
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        monitor.register_optimizer(optimizer, name="Adam")
        monitor.register_loss_function(loss, name="SparseCategoricalCrossentropy")
        monitor.register_metrics(metrics)
        print("   ✓ Optimizer, loss, and metrics registered\n")
    except ImportError:
        print("   ✗ Could not compile model\n")

    # Register dataset
    print("4. Registering training dataset...")
    dataset = create_sample_dataset()
    if dataset is not None:
        monitor.register_dataset(
            dataset,
            name="MNIST_training_data",
            split="train",
            description="MNIST handwritten digits training dataset (tf.data format)"
        )
        print("   ✓ Dataset registered\n")
    else:
        print("   ✗ Could not create dataset\n")

    # Log training configuration
    print("5. Logging training configuration...")
    monitor.log_training_config(
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        early_stopping=True,
        reduce_lr_on_plateau=True
    )
    print("   ✓ Training config logged\n")

    # Create Keras callback for automatic monitoring
    print("6. Creating Keras callback for automatic monitoring...")
    callback = monitor.create_keras_callback()
    print("   ✓ Callback created\n")

    # Simulate training with callback
    print("7. Simulating training with automatic metric logging...")
    print("   (In real usage, you would call: model.fit(train_data, callbacks=[callback]))")

    # Manually log some simulated metrics
    for epoch in range(1, 4):  # Just first 3 epochs for demo
        logs = {
            "loss": 2.3 - (epoch * 0.3),
            "accuracy": 0.1 + (epoch * 0.15),
            "val_loss": 2.4 - (epoch * 0.25),
            "val_accuracy": 0.09 + (epoch * 0.14)
        }
        monitor.log_training_metrics(epoch, logs)
        print(f"   ✓ Epoch {epoch} metrics logged")
    print()

    # Get collected metadata
    print("8. Retrieving collected metadata...")
    metadata = monitor.get_metadata()
    print(f"   ✓ Collected {len(metadata.get('models', []))} model(s)")
    print(f"   ✓ Collected {len(metadata.get('data_sources', []))} data source(s)")
    print(f"   ✓ Collected {len(metadata.get('framework_components', []))} framework component(s)")
    print()

    # Save to file
    output_file = "tensorflow_metadata.json"
    print(f"9. Saving metadata to {output_file}...")
    monitor.save_to_file(output_file)
    print(f"   ✓ Metadata saved\n")

    print("=" * 70)
    print("Example completed successfully!")
    print("=" * 70)
    print()
    print(f"Next steps:")
    print(f"1. Review the generated metadata: cat {output_file}")
    print(f"2. Use the metadata to generate compliance documents")
    print(f"3. Integrate with your actual TensorFlow/Keras training pipeline")
    print(f"4. Use the Keras callback in model.fit() for automatic tracking")
    print()


if __name__ == "__main__":
    main()
