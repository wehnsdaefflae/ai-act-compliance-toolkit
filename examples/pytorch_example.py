"""
PyTorch Integration Example

This example demonstrates how to use the AI Act Compliance Toolkit
with PyTorch models to automatically capture compliance metadata.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aiact_toolkit import PyTorchMonitor


def create_sample_model():
    """Create a sample PyTorch model for demonstration."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("PyTorch is not installed. Install with: pip install torch")
        return None

    class SimpleClassifier(nn.Module):
        """Simple neural network for classification."""

        def __init__(self, input_size=784, hidden_size=128, num_classes=10):
            super(SimpleClassifier, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
            self.fc2 = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    return SimpleClassifier()


def create_sample_dataset():
    """Create a sample PyTorch dataset for demonstration."""
    try:
        import torch
        from torch.utils.data import TensorDataset
    except ImportError:
        return None

    # Create dummy data
    X = torch.randn(1000, 784)  # 1000 samples, 784 features
    y = torch.randint(0, 10, (1000,))  # 1000 labels (10 classes)

    return TensorDataset(X, y)


def main():
    """Main example demonstrating PyTorch monitoring."""

    print("=" * 70)
    print("PyTorch AI Act Compliance Toolkit - Example")
    print("=" * 70)
    print()

    # Initialize monitor
    print("1. Initializing PyTorch monitor...")
    monitor = PyTorchMonitor(system_name="mnist_classifier")
    monitor.start()
    print("   ✓ Monitor started\n")

    # Create and register model
    print("2. Creating and registering PyTorch model...")
    model = create_sample_model()
    if model is None:
        print("   ✗ Could not create model (PyTorch not installed)")
        return

    monitor.register_model(
        model,
        name="SimpleClassifier",
        description="Simple neural network for MNIST digit classification"
    )
    print("   ✓ Model registered\n")

    # Register optimizer
    print("3. Registering optimizer...")
    try:
        import torch.optim as optim
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        monitor.register_optimizer(optimizer, name="Adam")
        print("   ✓ Optimizer registered\n")
    except ImportError:
        print("   ✗ Could not register optimizer\n")

    # Register loss function
    print("4. Registering loss function...")
    try:
        import torch.nn as nn
        criterion = nn.CrossEntropyLoss()
        monitor.register_loss_function(criterion, name="CrossEntropyLoss")
        print("   ✓ Loss function registered\n")
    except ImportError:
        print("   ✗ Could not register loss function\n")

    # Register dataset
    print("5. Registering training dataset...")
    dataset = create_sample_dataset()
    if dataset is not None:
        monitor.register_dataset(
            dataset,
            name="MNIST_training_data",
            split="train",
            description="MNIST handwritten digits training dataset"
        )
        print("   ✓ Dataset registered\n")
    else:
        print("   ✗ Could not create dataset\n")

    # Register DataLoader
    print("6. Registering DataLoader...")
    try:
        from torch.utils.data import DataLoader
        if dataset is not None:
            train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
            monitor.register_data_loader(train_loader, name="train_loader")
            print("   ✓ DataLoader registered\n")
    except ImportError:
        print("   ✗ Could not register DataLoader\n")

    # Log training configuration
    print("7. Logging training configuration...")
    monitor.log_training_config(
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        early_stopping=True,
        validation_split=0.2
    )
    print("   ✓ Training config logged\n")

    # Simulate training metrics
    print("8. Logging training metrics (simulated)...")
    for epoch in range(1, 4):  # Just first 3 epochs for demo
        metrics = {
            "loss": 2.3 - (epoch * 0.3),
            "accuracy": 0.1 + (epoch * 0.15),
            "val_loss": 2.4 - (epoch * 0.25),
            "val_accuracy": 0.09 + (epoch * 0.14)
        }
        monitor.log_training_metrics(epoch, metrics)
        print(f"   ✓ Epoch {epoch} metrics logged")
    print()

    # Get collected metadata
    print("9. Retrieving collected metadata...")
    metadata = monitor.get_metadata()
    print(f"   ✓ Collected {len(metadata.get('models', []))} model(s)")
    print(f"   ✓ Collected {len(metadata.get('data_sources', []))} data source(s)")
    print(f"   ✓ Collected {len(metadata.get('framework_components', []))} framework component(s)")
    print()

    # Save to file
    output_file = "pytorch_metadata.json"
    print(f"10. Saving metadata to {output_file}...")
    monitor.save_to_file(output_file)
    print(f"    ✓ Metadata saved\n")

    print("=" * 70)
    print("Example completed successfully!")
    print("=" * 70)
    print()
    print(f"Next steps:")
    print(f"1. Review the generated metadata: cat {output_file}")
    print(f"2. Use the metadata to generate compliance documents")
    print(f"3. Integrate with your actual PyTorch training pipeline")
    print()


if __name__ == "__main__":
    main()
