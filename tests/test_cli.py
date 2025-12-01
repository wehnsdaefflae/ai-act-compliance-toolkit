"""
Unit tests for CLI module
"""

import sys
import os
import json
import tempfile
from pathlib import Path
from io import StringIO

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aiact_toolkit.cli import create_parser, cmd_validate, cmd_list_templates


class TestCLI:
    """Test suite for CLI functionality."""

    def setup_method(self):
        """Set up test fixtures."""
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

        # Create test metadata file
        self.metadata_file = Path(self.temp_dir) / "test_metadata.json"
        with open(self.metadata_file, 'w') as f:
            json.dump(self.sample_metadata, f)

    def test_create_parser(self):
        """Test parser creation."""
        parser = create_parser()
        assert parser is not None
        assert parser.prog == "aiact-toolkit"

    def test_parser_generate_command(self):
        """Test parsing generate command."""
        parser = create_parser()
        args = parser.parse_args([
            "generate",
            "metadata.json",
            "-t", "template.jinja2",
            "-o", "output.md"
        ])

        assert args.command == "generate"
        assert args.metadata == "metadata.json"
        assert args.template == "template.jinja2"
        assert args.output == "output.md"

    def test_parser_list_templates_command(self):
        """Test parsing list-templates command."""
        parser = create_parser()
        args = parser.parse_args(["list-templates"])

        assert args.command == "list-templates"

    def test_parser_validate_command(self):
        """Test parsing validate command."""
        parser = create_parser()
        args = parser.parse_args([
            "validate",
            "metadata.json",
            "-v"
        ])

        assert args.command == "validate"
        assert args.metadata == "metadata.json"
        assert args.verbose is True

    def test_cmd_list_templates(self):
        """Test list-templates command execution."""
        # Create mock args
        class Args:
            templates_dir = None

        # Capture output
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            exit_code = cmd_list_templates(Args())
            output = sys.stdout.getvalue()

            assert exit_code == 0
            assert "Available templates" in output

        finally:
            sys.stdout = old_stdout

    def test_cmd_validate_valid_metadata(self):
        """Test validate command with valid metadata."""
        class Args:
            metadata = str(self.metadata_file)
            verbose = False

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            exit_code = cmd_validate(Args())
            output = sys.stdout.getvalue()

            assert exit_code == 0
            assert "validation passed" in output.lower()

        finally:
            sys.stdout = old_stdout

    def test_cmd_validate_invalid_file(self):
        """Test validate command with non-existent file."""
        class Args:
            metadata = "/nonexistent/metadata.json"
            verbose = False

        old_stderr = sys.stderr
        sys.stderr = StringIO()

        try:
            exit_code = cmd_validate(Args())
            assert exit_code == 1

        finally:
            sys.stderr = old_stderr

    def test_cmd_validate_verbose(self):
        """Test validate command with verbose output."""
        class Args:
            metadata = str(self.metadata_file)
            verbose = True

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            exit_code = cmd_validate(Args())
            output = sys.stdout.getvalue()

            assert exit_code == 0
            assert "Metadata summary" in output

        finally:
            sys.stdout = old_stdout


def run_tests():
    """Run all tests."""
    import pytest
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()
