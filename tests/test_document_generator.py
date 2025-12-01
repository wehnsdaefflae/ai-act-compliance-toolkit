"""
Unit tests for Document Generator
"""

import sys
import os
import json
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aiact_toolkit.document_generator import DocumentGenerator


class TestDocumentGenerator:
    """Test suite for DocumentGenerator class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create temp directory for test outputs
        self.temp_dir = tempfile.mkdtemp()

        # Sample metadata for testing
        self.sample_metadata = {
            "system_name": "test_system",
            "created_at": "2024-01-01T00:00:00",
            "timestamp": "2024-01-01T00:00:00",
            "models": [
                {
                    "model_name": "gpt-4",
                    "provider": "OpenAI",
                    "parameters": {
                        "temperature": 0.7,
                        "max_tokens": 500
                    }
                }
            ],
            "data_sources": [
                {
                    "data_source": "./data/test.txt",
                    "loader_type": "TextLoader"
                }
            ],
            "components": [],
            "summary": {
                "total_models": 1,
                "total_components": 0,
                "total_data_sources": 1
            }
        }

    def test_init_with_default_templates(self):
        """Test initialization with default templates directory."""
        generator = DocumentGenerator()
        assert generator.templates_dir.exists()
        assert generator.templates_dir.name == "templates"

    def test_init_with_custom_templates(self):
        """Test initialization with custom templates directory."""
        # Create temp templates dir
        temp_templates = Path(self.temp_dir) / "templates"
        temp_templates.mkdir()

        generator = DocumentGenerator(templates_dir=str(temp_templates))
        assert generator.templates_dir == temp_templates

    def test_init_with_nonexistent_dir_raises_error(self):
        """Test that initialization fails with non-existent directory."""
        try:
            DocumentGenerator(templates_dir="/nonexistent/path")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "not found" in str(e)

    def test_list_templates(self):
        """Test listing available templates."""
        generator = DocumentGenerator()
        templates = generator.list_templates()

        assert isinstance(templates, list)
        assert len(templates) > 0
        assert all(t.endswith('.jinja2') for t in templates)

    def test_load_metadata(self):
        """Test loading metadata from JSON file."""
        # Create temp metadata file
        metadata_file = Path(self.temp_dir) / "test_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.sample_metadata, f)

        generator = DocumentGenerator()
        loaded = generator.load_metadata(str(metadata_file))

        assert loaded["system_name"] == "test_system"
        assert len(loaded["models"]) == 1

    def test_load_metadata_file_not_found(self):
        """Test loading non-existent metadata file."""
        generator = DocumentGenerator()

        try:
            generator.load_metadata("/nonexistent/metadata.json")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass

    def test_validate_metadata_valid(self):
        """Test validation of valid metadata."""
        generator = DocumentGenerator()
        validation = generator.validate_metadata(self.sample_metadata)

        assert validation["valid"] is True
        assert len(validation["missing_fields"]) == 0

    def test_validate_metadata_missing_fields(self):
        """Test validation with missing required fields."""
        incomplete_metadata = {"system_name": "test"}

        generator = DocumentGenerator()
        validation = generator.validate_metadata(incomplete_metadata)

        assert validation["valid"] is False
        assert "models" in validation["missing_fields"]
        assert "data_sources" in validation["missing_fields"]

    def test_validate_metadata_empty_models(self):
        """Test validation with empty models list."""
        metadata = {
            "system_name": "test",
            "models": [],
            "data_sources": []
        }

        generator = DocumentGenerator()
        validation = generator.validate_metadata(metadata)

        assert validation["valid"] is True
        assert len(validation["warnings"]) > 0
        assert any("No models captured" in w for w in validation["warnings"])

    def test_generate_document(self):
        """Test generating document from template."""
        generator = DocumentGenerator()
        templates = generator.list_templates()

        if not templates:
            print("Skipping: No templates available")
            return

        # Generate document without saving
        template_name = templates[0]
        document = generator.generate_document(
            template_name=template_name,
            metadata=self.sample_metadata
        )

        assert isinstance(document, str)
        assert len(document) > 0

    def test_generate_document_with_output(self):
        """Test generating document and saving to file."""
        generator = DocumentGenerator()
        templates = generator.list_templates()

        if not templates:
            print("Skipping: No templates available")
            return

        output_file = Path(self.temp_dir) / "test_output.md"
        template_name = templates[0]

        document = generator.generate_document(
            template_name=template_name,
            metadata=self.sample_metadata,
            output_path=str(output_file)
        )

        assert output_file.exists()
        assert output_file.read_text() == document

    def test_generate_all_documents(self):
        """Test generating all documents."""
        generator = DocumentGenerator()
        output_dir = Path(self.temp_dir) / "all_docs"

        generated = generator.generate_all_documents(
            metadata=self.sample_metadata,
            output_dir=str(output_dir)
        )

        assert len(generated) > 0
        assert output_dir.exists()

        for filepath in generated:
            assert Path(filepath).exists()


def run_tests():
    """Run all tests."""
    import pytest
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()
