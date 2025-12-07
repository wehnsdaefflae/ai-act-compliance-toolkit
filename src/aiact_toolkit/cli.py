"""
CLI Tool for AI Act Compliance Toolkit

Command-line interface for generating compliance documents from captured metadata.
"""

import sys
import json
from pathlib import Path
from typing import Optional
import argparse

from .document_generator import DocumentGenerator
from .metadata_storage import MetadataStorage
from .risk_assessment import AIActRiskAssessor


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        prog="aiact-toolkit",
        description="AI Act Compliance Toolkit - Generate compliance documents from LangChain metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all documents from metadata
  aiact-toolkit generate metadata.json -o output/

  # Generate specific document
  aiact-toolkit generate metadata.json -t dsgvo_dsfa.md.jinja2 -o dpia.md

  # List available templates
  aiact-toolkit list-templates

  # Validate metadata completeness
  aiact-toolkit validate metadata.json

  # Assess risk level for EU AI Act compliance
  aiact-toolkit assess-risk metadata.json --use-case "Medical diagnosis chatbot"
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate command
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate compliance documents from metadata"
    )
    generate_parser.add_argument(
        "metadata",
        help="Path to metadata JSON file"
    )
    generate_parser.add_argument(
        "-t", "--template",
        help="Specific template to use (default: generate all)"
    )
    generate_parser.add_argument(
        "-o", "--output",
        help="Output file or directory path"
    )
    generate_parser.add_argument(
        "--templates-dir",
        help="Custom templates directory"
    )

    # List templates command
    list_parser = subparsers.add_parser(
        "list-templates",
        help="List all available templates"
    )
    list_parser.add_argument(
        "--templates-dir",
        help="Custom templates directory"
    )

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate metadata completeness"
    )
    validate_parser.add_argument(
        "metadata",
        help="Path to metadata JSON file"
    )
    validate_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed validation results"
    )

    # Assess risk command
    assess_parser = subparsers.add_parser(
        "assess-risk",
        help="Assess EU AI Act risk level for the system"
    )
    assess_parser.add_argument(
        "metadata",
        help="Path to metadata JSON file"
    )
    assess_parser.add_argument(
        "-u", "--use-case",
        help="Description of system use case (helps determine risk level)"
    )
    assess_parser.add_argument(
        "-d", "--domain",
        help="Application domain (e.g., healthcare, education, employment)"
    )
    assess_parser.add_argument(
        "-o", "--output",
        help="Save risk assessment report to file"
    )
    assess_parser.add_argument(
        "--save-to-metadata",
        action="store_true",
        help="Save risk assessment results back to metadata file"
    )

    return parser


def cmd_generate(args) -> int:
    """Handle generate command."""
    try:
        # Initialize generator
        generator = DocumentGenerator(templates_dir=args.templates_dir)

        # Load metadata
        print(f"Loading metadata from: {args.metadata}")
        metadata = generator.load_metadata(args.metadata)
        print(f"✓ Loaded metadata for system: {metadata.get('system_name', 'unknown')}")

        # Validate metadata
        validation = generator.validate_metadata(metadata)
        if not validation["valid"]:
            print("\n⚠ Warning: Metadata validation failed")
            print("Missing fields:", ", ".join(validation["missing_fields"]))
            print("Continue anyway? [y/N]: ", end="")
            if input().lower() != 'y':
                return 1

        if validation["warnings"]:
            print("\n⚠ Metadata warnings:")
            for warning in validation["warnings"]:
                print(f"  - {warning}")

        # Generate documents
        if args.template:
            # Single template
            output_path = args.output or args.template.replace('.jinja2', '')
            print(f"\nGenerating document from template: {args.template}")
            generator.generate_document(
                template_name=args.template,
                metadata=metadata,
                output_path=output_path
            )
            print(f"✓ Generated: {output_path}")
        else:
            # All templates
            output_dir = args.output or "compliance_docs"
            print(f"\nGenerating all documents to: {output_dir}")
            generated = generator.generate_all_documents(metadata, output_dir)
            print(f"✓ Generated {len(generated)} document(s):")
            for filepath in generated:
                print(f"  - {filepath}")

        print("\n✓ Document generation complete!")
        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_list_templates(args) -> int:
    """Handle list-templates command."""
    try:
        generator = DocumentGenerator(templates_dir=args.templates_dir)
        templates = generator.list_templates()

        print(f"Available templates in {generator.templates_dir}:")
        print()
        for template in templates:
            # Derive document type from filename
            doc_type = template.replace('.md.jinja2', '').replace('_', ' ').title()
            print(f"  - {template}")
            print(f"    → {doc_type}")

        print(f"\nTotal: {len(templates)} template(s)")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_validate(args) -> int:
    """Handle validate command."""
    try:
        generator = DocumentGenerator()
        metadata = generator.load_metadata(args.metadata)

        print(f"Validating metadata from: {args.metadata}")
        print(f"System: {metadata.get('system_name', 'unknown')}")
        print()

        validation = generator.validate_metadata(metadata)

        # Print validation status
        if validation["valid"]:
            print("✓ Metadata validation passed")
        else:
            print("✗ Metadata validation failed")

        # Print missing fields
        if validation["missing_fields"]:
            print("\nMissing required fields:")
            for field in validation["missing_fields"]:
                print(f"  ✗ {field}")

        # Print warnings
        if validation["warnings"]:
            print("\n⚠ Warnings:")
            for warning in validation["warnings"]:
                print(f"  - {warning}")

        # Print recommendations
        if validation["recommendations"]:
            print("\n💡 Recommendations:")
            for rec in validation["recommendations"]:
                print(f"  - {rec}")

        # Verbose output
        if args.verbose:
            print("\nMetadata summary:")
            if "models" in metadata:
                print(f"  Models: {len(metadata['models'])}")
            if "data_sources" in metadata:
                print(f"  Data sources: {len(metadata['data_sources'])}")
            if "components" in metadata:
                print(f"  Components: {len(metadata['components'])}")

        return 0 if validation["valid"] else 1

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_assess_risk(args) -> int:
    """Handle assess-risk command."""
    try:
        # Load metadata
        generator = DocumentGenerator()
        metadata = generator.load_metadata(args.metadata)

        print(f"Assessing risk for system: {metadata.get('system_name', 'unknown')}")
        if args.use_case:
            print(f"Use case: {args.use_case}")
        if args.domain:
            print(f"Domain: {args.domain}")
        print()

        # Perform risk assessment
        assessor = AIActRiskAssessor()
        assessment = assessor.assess_risk(
            metadata=metadata,
            use_case=args.use_case,
            application_domain=args.domain
        )

        # Display results
        risk_level = assessment["risk_level"].upper()
        confidence = assessment["confidence"] * 100

        # Color-coded output based on risk level
        risk_symbols = {
            "unacceptable": "⛔",
            "high": "⚠️",
            "limited": "ℹ️",
            "minimal": "✓",
            "unknown": "❓"
        }

        symbol = risk_symbols.get(assessment["risk_level"], "?")
        print(f"{symbol} Risk Level: {risk_level}")
        print(f"Confidence: {confidence}%")
        print()

        print("Risk Factors:")
        for factor in assessment["risk_factors"]:
            print(f"  • {factor}")
        print()

        print("Compliance Requirements:")
        for i, req in enumerate(assessment["compliance_requirements"], 1):
            print(f"  {i}. {req}")
        print()

        print("Recommendations:")
        for i, rec in enumerate(assessment["recommendations"], 1):
            print(f"  {i}. {rec}")
        print()

        # Save to metadata if requested
        if args.save_to_metadata:
            storage = MetadataStorage()
            storage.load_from_file(args.metadata)
            storage.set_risk_assessment(assessment)
            storage.save_to_file(args.metadata)
            print(f"✓ Risk assessment saved to metadata file: {args.metadata}")
            print()

        # Generate report if output specified
        if args.output:
            report_data = assessor.generate_risk_report(metadata, assessment)
            generator.generate_document(
                template_name="risk_assessment_report.md.jinja2",
                metadata=report_data,
                output_path=args.output
            )
            print(f"✓ Risk assessment report generated: {args.output}")

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Route to command handlers
    if args.command == "generate":
        return cmd_generate(args)
    elif args.command == "list-templates":
        return cmd_list_templates(args)
    elif args.command == "validate":
        return cmd_validate(args)
    elif args.command == "assess-risk":
        return cmd_assess_risk(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
