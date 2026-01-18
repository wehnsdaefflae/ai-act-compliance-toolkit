#!/usr/bin/env python3
"""
Showcase Script - AI Act Compliance Toolkit Demo

This script demonstrates the core concept: automatic compliance documentation
from captured AI/ML metadata. No API keys required - uses pre-captured data.

Run this to see what the toolkit produces for EU AI Act compliance.
"""

import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aiact_toolkit import (
    MetadataStorage,
    AIActRiskAssessor,
    ConformityAssessor,
    DocumentGenerator
)


def main():
    print("=" * 70)
    print(" AI ACT COMPLIANCE TOOLKIT - PROOF OF CONCEPT DEMONSTRATION")
    print("=" * 70)
    print()
    print("This toolkit implements 'Compliance-as-Code' for EU AI Act.")
    print("It captures metadata from AI/ML frameworks and generates compliance docs.")
    print()

    # Load pre-captured metadata from a real LangChain application
    metadata_file = os.path.join(
        os.path.dirname(__file__),
        "generated_outputs",
        "example_metadata.json"
    )

    if not os.path.exists(metadata_file):
        print(f"Error: Example metadata not found at {metadata_file}")
        print("Run 'python examples/llama2_medical_chatbot_integration.py' first.")
        return 1

    print(f"Loading captured metadata from: {os.path.basename(metadata_file)}")
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    print(f"System: {metadata.get('system_name', 'Unknown')}")
    print()

    # Show what was captured
    print("-" * 70)
    print(" CAPTURED METADATA SUMMARY")
    print("-" * 70)
    models = metadata.get('models', [])
    print(f"  Models captured: {len(models)}")
    for m in models[:3]:
        print(f"    - {m.get('model_name', 'unknown')} ({m.get('provider', 'unknown')})")

    components = metadata.get('components', [])
    print(f"  Components: {len(components)}")

    data_sources = metadata.get('data_sources', [])
    print(f"  Data sources: {len(data_sources)}")
    print()

    # Risk Assessment
    print("-" * 70)
    print(" AUTOMATED RISK ASSESSMENT (EU AI Act)")
    print("-" * 70)
    assessor = AIActRiskAssessor()
    risk = assessor.assess_risk(
        metadata,
        use_case="Medical chatbot for patient inquiries",
        application_domain="healthcare"
    )

    symbols = {"unacceptable": "PROHIBITED", "high": "HIGH", "limited": "LIMITED", "minimal": "MINIMAL"}
    print(f"  Risk Level: {symbols.get(risk['risk_level'], risk['risk_level']).upper()}")
    print(f"  Confidence: {risk['confidence'] * 100:.0f}%")
    print()
    print("  Risk Factors:")
    for factor in risk['risk_factors'][:3]:
        print(f"    - {factor}")
    print()

    # Conformity Assessment
    print("-" * 70)
    print(" CONFORMITY ASSESSMENT PREVIEW")
    print("-" * 70)
    conformity = ConformityAssessor()
    result = conformity.assess_compliance(metadata)

    print(f"  Requirements Checked: {result.requirements_checked}")
    print(f"  Passed: {result.requirements_passed}")
    print(f"  Partial: {result.requirements_partial}")
    print(f"  Failed: {result.requirements_failed}")
    print(f"  Overall Status: {result.overall_status.value.upper()}")
    print()

    # Available outputs
    print("-" * 70)
    print(" COMPLIANCE DOCUMENTS THE TOOLKIT CAN GENERATE")
    print("-" * 70)
    documents = [
        ("Model Card", "Article 13 transparency - model documentation"),
        ("Technical Documentation", "Article 11 - high-risk system documentation"),
        ("Conformity Report", "Articles 43-46 - compliance assessment"),
        ("DSGVO/GDPR DPIA", "Data Protection Impact Assessment"),
        ("Bias/Fairness Report", "Article 10.2f - bias detection analysis"),
        ("Audit Trail Report", "Article 12 - automatic logging"),
    ]

    for name, description in documents:
        print(f"  * {name}")
        print(f"      {description}")

    print()
    print("-" * 70)
    print(" NEXT STEPS")
    print("-" * 70)
    print("  Generate documents using the CLI:")
    print()
    print("    # Quick status overview")
    print("    aiact-toolkit status examples/generated_outputs/example_metadata.json")
    print()
    print("    # Generate model card")
    print("    aiact-toolkit generate-model-card examples/generated_outputs/example_metadata.json")
    print()
    print("    # Full conformity assessment")
    print("    aiact-toolkit conformity-assessment examples/generated_outputs/example_metadata.json")
    print()
    print("=" * 70)
    print(" Proof-of-concept demonstrating Compliance-as-Code for EU AI Act")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
