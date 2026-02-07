"""
Unit Tests for PyTorch Monitor

Tests the PyTorch monitoring capabilities of the AI Act Compliance Toolkit.
"""

import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aiact_toolkit import PyTorchMonitor

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import pytest


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestPyTorchMonitor:
    """Test suite for PyTorch monitor functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.monitor = PyTorchMonitor(system_name="test_pytorch")
        self.monitor.start()
        self.model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def teardown_method(self):
        """Clean up."""
        self.monitor.stop()

    def test_initialization(self):
        """Test monitor initializes correctly."""
        assert self.monitor.system_name == "test_pytorch"
        assert self.monitor.is_active
        assert self.monitor.framework_name == "PyTorch"

    def test_start_sets_framework(self):
        """Test that start() sets framework metadata."""
        assert self.monitor.storage.metadata["framework"] == "PyTorch"
        assert isinstance(self.monitor.storage.metadata["framework_components"], list)

    def test_register_model(self):
        """Test registering a PyTorch model captures architecture info."""
        self.monitor.register_model(self.model, name="TestNet")

        metadata = self.monitor.get_metadata()
        assert len(metadata["models"]) == 1

        model = metadata["models"][0]
        assert model["model_name"] == "TestNet"
        assert model["provider"] == "PyTorch"
        assert model["parameters"]["total_parameters"] > 0
        assert model["parameters"]["trainable_parameters"] > 0
        assert "architecture_details" in model
        assert model["architecture_details"]["total_layers"] > 0

    def test_register_model_inactive(self):
        """Test that register_model does nothing when monitor is inactive."""
        self.monitor.stop()
        self.monitor.register_model(self.model, name="Ignored")

        metadata = self.monitor.get_metadata()
        assert len(metadata["models"]) == 0

    def test_register_model_invalid(self):
        """Test that register_model raises for non-Module objects."""
        with pytest.raises(ValueError, match="torch.nn.Module"):
            self.monitor.register_model("not_a_model")

    def test_register_optimizer(self):
        """Test registering an optimizer captures hyperparameters."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.monitor.register_optimizer(optimizer, name="Adam")

        components = self.monitor.storage.metadata["framework_components"]
        opt_components = [c for c in components if c.get("component_type") == "optimizer"]
        assert len(opt_components) == 1
        assert opt_components[0]["optimizer_name"] == "Adam"
        assert "learning_rate" in str(opt_components[0]["parameters"]) or "lr" in str(opt_components[0]["parameters"])

    def test_register_loss_function(self):
        """Test registering a loss function."""
        loss_fn = nn.CrossEntropyLoss()
        self.monitor.register_loss_function(loss_fn)

        components = self.monitor.storage.metadata["framework_components"]
        loss_components = [c for c in components if c.get("component_type") == "loss_function"]
        assert len(loss_components) == 1
        assert loss_components[0]["loss_name"] == "CrossEntropyLoss"

    def test_register_dataset(self):
        """Test registering a dataset."""
        dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 10), torch.randint(0, 2, (100,))
        )
        self.monitor.register_dataset(dataset, name="train_data", split="train")

        metadata = self.monitor.get_metadata()
        assert len(metadata["data_sources"]) == 1
        assert metadata["data_sources"][0]["source_name"] == "train_data"
        assert metadata["data_sources"][0]["size"] == 100

    def test_register_data_loader(self):
        """Test registering a DataLoader."""
        dataset = torch.utils.data.TensorDataset(torch.randn(50, 10))
        loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
        self.monitor.register_data_loader(loader, name="train_loader")

        components = self.monitor.storage.metadata["framework_components"]
        loader_components = [c for c in components if c.get("component_type") == "data_loader"]
        assert len(loader_components) == 1
        assert loader_components[0]["parameters"]["batch_size"] == 16

    def test_log_training_config(self):
        """Test logging training configuration."""
        self.monitor.log_training_config(epochs=10, batch_size=32, learning_rate=0.001)

        components = self.monitor.storage.metadata["framework_components"]
        config_components = [c for c in components if c.get("component_type") == "training_config"]
        assert len(config_components) == 1
        assert config_components[0]["parameters"]["epochs"] == 10

    def test_log_training_metrics(self):
        """Test logging training metrics per epoch."""
        self.monitor.log_training_metrics(0, {"loss": 1.5, "accuracy": 0.6})
        self.monitor.log_training_metrics(1, {"loss": 0.8, "accuracy": 0.85})

        history = self.monitor.storage.metadata["training_history"]
        assert len(history) == 2
        assert history[0]["epoch"] == 0
        assert history[1]["metrics"]["accuracy"] == 0.85

    def test_save_and_load(self):
        """Test saving metadata to file."""
        self.monitor.register_model(self.model, name="SaveTest")

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_file = f.name

        try:
            self.monitor.save_to_file(temp_file)
            with open(temp_file) as f:
                data = json.load(f)
            assert data["system_name"] == "test_pytorch"
            assert len(data["models"]) == 1
            assert data["framework"] == "PyTorch"
        finally:
            os.unlink(temp_file)

    def test_full_workflow(self):
        """Test a complete monitoring workflow."""
        self.monitor.register_model(self.model, name="FullWorkflow")
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.monitor.register_optimizer(optimizer)
        self.monitor.register_loss_function(nn.MSELoss())

        dataset = torch.utils.data.TensorDataset(
            torch.randn(200, 10), torch.randn(200, 2)
        )
        self.monitor.register_dataset(dataset, name="synthetic_data")

        self.monitor.log_training_config(epochs=5, batch_size=32)
        for epoch in range(3):
            self.monitor.log_training_metrics(epoch, {"loss": 1.0 / (epoch + 1)})

        metadata = self.monitor.get_metadata()
        assert len(metadata["models"]) == 1
        assert len(metadata["data_sources"]) == 1
        assert metadata["framework"] == "PyTorch"
        assert len(metadata["training_history"]) == 3


class TestPyTorchMonitorWithoutTorch:
    """Tests that work without PyTorch installed."""

    def test_framework_name(self):
        """Test that framework name is set correctly."""
        assert PyTorchMonitor.framework_name == "PyTorch"

    def test_inactive_by_default(self):
        """Test monitor is inactive before start."""
        monitor = PyTorchMonitor(system_name="inactive_test")
        assert not monitor.is_active
