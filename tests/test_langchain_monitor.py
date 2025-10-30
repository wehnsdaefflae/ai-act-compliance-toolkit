"""
Unit Tests for LangChain Monitor

Tests the core functionality of the AI Act Compliance Toolkit's
LangChain monitoring capabilities.
"""

import unittest
import sys
import os
import json
import tempfile

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aiact_toolkit import LangChainMonitor, MetadataStorage


class TestMetadataStorage(unittest.TestCase):
    """Test the MetadataStorage class."""

    def setUp(self):
        """Set up test fixtures."""
        self.storage = MetadataStorage(system_name="test_system")

    def test_initialization(self):
        """Test storage initialization."""
        self.assertEqual(self.storage.system_name, "test_system")
        self.assertEqual(len(self.storage.models), 0)
        self.assertEqual(len(self.storage.components), 0)
        self.assertEqual(len(self.storage.data_sources), 0)

    def test_add_model(self):
        """Test adding model metadata."""
        model_info = {
            "model_name": "gpt-4",
            "provider": "OpenAI",
            "parameters": {"temperature": 0.7}
        }
        self.storage.add_model(model_info)
        self.assertEqual(len(self.storage.models), 1)
        self.assertEqual(self.storage.models[0]["model_name"], "gpt-4")

    def test_add_model_deduplication(self):
        """Test that duplicate models are not added."""
        model_info = {
            "model_name": "gpt-4",
            "provider": "OpenAI",
            "parameters": {"temperature": 0.7}
        }
        self.storage.add_model(model_info)
        self.storage.add_model(model_info)
        self.assertEqual(len(self.storage.models), 1)

    def test_add_data_source(self):
        """Test adding data source metadata."""
        data_source = {
            "data_source": "./data/test.txt",
            "data_type": "text",
            "loader_type": "TextLoader"
        }
        self.storage.add_data_source(data_source)
        self.assertEqual(len(self.storage.data_sources), 1)

    def test_add_data_source_deduplication(self):
        """Test that duplicate data sources are not added."""
        data_source = {
            "data_source": "./data/test.txt",
            "data_type": "text",
            "loader_type": "TextLoader"
        }
        self.storage.add_data_source(data_source)
        self.storage.add_data_source(data_source)
        self.assertEqual(len(self.storage.data_sources), 1)

    def test_get_all_metadata(self):
        """Test retrieving all metadata."""
        model_info = {
            "model_name": "gpt-4",
            "provider": "OpenAI",
            "parameters": {"temperature": 0.7}
        }
        self.storage.add_model(model_info)

        metadata = self.storage.get_all_metadata()
        self.assertEqual(metadata["system_name"], "test_system")
        self.assertEqual(len(metadata["models"]), 1)
        self.assertEqual(metadata["summary"]["total_models"], 1)

    def test_save_and_load_file(self):
        """Test saving and loading metadata from file."""
        model_info = {
            "model_name": "gpt-4",
            "provider": "OpenAI",
            "parameters": {"temperature": 0.7}
        }
        self.storage.add_model(model_info)

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name

        try:
            self.storage.save_to_file(temp_file)

            # Load into new storage
            new_storage = MetadataStorage()
            new_storage.load_from_file(temp_file)

            self.assertEqual(new_storage.system_name, "test_system")
            self.assertEqual(len(new_storage.models), 1)
            self.assertEqual(new_storage.models[0]["model_name"], "gpt-4")
        finally:
            os.unlink(temp_file)


class TestLangChainMonitor(unittest.TestCase):
    """Test the LangChainMonitor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.monitor = LangChainMonitor(system_name="test_monitor")

    def tearDown(self):
        """Clean up after tests."""
        if self.monitor._is_started:
            self.monitor.stop()

    def test_initialization(self):
        """Test monitor initialization."""
        self.assertEqual(self.monitor.system_name, "test_monitor")
        self.assertFalse(self.monitor._is_started)

    def test_start_stop(self):
        """Test starting and stopping the monitor."""
        self.monitor.start()
        self.assertTrue(self.monitor._is_started)

        self.monitor.stop()
        self.assertFalse(self.monitor._is_started)

    def test_context_manager(self):
        """Test using monitor as context manager."""
        with LangChainMonitor(system_name="context_test") as monitor:
            self.assertTrue(monitor._is_started)
        # Should be stopped after exiting context
        self.assertFalse(monitor._is_started)

    def test_get_metadata(self):
        """Test retrieving metadata."""
        self.monitor.start()
        metadata = self.monitor.get_metadata()

        self.assertIn("system_name", metadata)
        self.assertIn("models", metadata)
        self.assertIn("components", metadata)
        self.assertIn("data_sources", metadata)
        self.assertEqual(metadata["system_name"], "test_monitor")

    def test_save_to_file(self):
        """Test saving metadata to file."""
        self.monitor.start()

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_file = f.name

        try:
            self.monitor.save_to_file(temp_file)
            self.assertTrue(os.path.exists(temp_file))

            # Verify file contents
            with open(temp_file, 'r') as f:
                data = json.load(f)
                self.assertEqual(data["system_name"], "test_monitor")
        finally:
            os.unlink(temp_file)


class TestIntegration(unittest.TestCase):
    """Integration tests with LangChain components."""

    def test_capture_chatopenai(self):
        """Test capturing ChatOpenAI initialization."""
        try:
            from langchain_openai import ChatOpenAI

            monitor = LangChainMonitor(system_name="integration_test")
            monitor.start()

            # Create ChatOpenAI instance
            llm = ChatOpenAI(
                model_name="gpt-4",
                temperature=0.7,
                max_tokens=500
            )

            # Check metadata
            metadata = monitor.get_metadata()
            monitor.stop()

            self.assertGreater(len(metadata["models"]), 0)
            model = metadata["models"][0]
            self.assertEqual(model["model_name"], "gpt-4")
            self.assertIn("temperature", model["parameters"])
            self.assertEqual(model["parameters"]["temperature"], 0.7)

        except ImportError:
            self.skipTest("langchain-openai not installed")

    def test_capture_document_loader(self):
        """Test capturing document loader."""
        try:
            from langchain_community.document_loaders import TextLoader

            monitor = LangChainMonitor(system_name="loader_test")
            monitor.start()

            # Create temp file
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                temp_file = f.name
                f.write("Test content")

            try:
                # Create loader
                loader = TextLoader(temp_file)

                # Check metadata
                metadata = monitor.get_metadata()
                monitor.stop()

                self.assertGreater(len(metadata["data_sources"]), 0)
                source = metadata["data_sources"][0]
                self.assertEqual(source["data_source"], temp_file)
                self.assertEqual(source["data_type"], "text")
                self.assertEqual(source["loader_type"], "TextLoader")

            finally:
                os.unlink(temp_file)

        except ImportError:
            self.skipTest("langchain-community not installed")


def run_tests():
    """Run all tests."""
    print("=" * 70)
    print("Running AI Act Compliance Toolkit Tests")
    print("=" * 70)
    print()

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMetadataStorage))
    suite.addTests(loader.loadTestsFromTestCase(TestLangChainMonitor))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print()
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print()

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
