#!/bin/bash

# CLI Usage Example for AI Act Compliance Toolkit
# This script demonstrates how to use the aiact-toolkit CLI

echo "=========================================="
echo "AI Act Compliance Toolkit - CLI Examples"
echo "=========================================="
echo ""

# First, ensure the package is installed
echo "Step 1: Install the toolkit (if not already installed)"
echo "  pip install -e ."
echo ""

# List available templates
echo "Step 2: List available templates"
echo "  Command: aiact-toolkit list-templates"
echo ""
python -m aiact_toolkit.cli list-templates
echo ""

# Validate existing metadata
echo "Step 3: Validate metadata completeness"
echo "  Command: aiact-toolkit validate examples/generated_outputs/example_metadata.json"
echo ""
python -m aiact_toolkit.cli validate examples/generated_outputs/example_metadata.json -v
echo ""

# Generate all documents
echo "Step 4: Generate all compliance documents"
echo "  Command: aiact-toolkit generate examples/generated_outputs/example_metadata.json -o compliance_output/"
echo ""
python -m aiact_toolkit.cli generate examples/generated_outputs/example_metadata.json -o compliance_output/
echo ""

# Generate specific document
echo "Step 5: Generate specific document (GDPR DPIA)"
echo "  Command: aiact-toolkit generate examples/generated_outputs/example_metadata.json -t dsgvo_dsfa.md.jinja2 -o dpia_output.md"
echo ""
python -m aiact_toolkit.cli generate examples/generated_outputs/example_metadata.json -t dsgvo_dsfa.md.jinja2 -o dpia_output.md
echo ""

echo "=========================================="
echo "CLI examples complete!"
echo ""
echo "Generated files:"
echo "  - compliance_output/ (all documents)"
echo "  - dpia_output.md (GDPR DPIA only)"
echo "=========================================="
