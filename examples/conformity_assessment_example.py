"""
Example: EU AI Act Conformity Assessment

Demonstrates automated conformity assessment for EU AI Act compliance.
Uses pre-captured metadata to show the complete workflow:
1. Load metadata from a monitored AI system
2. Perform risk assessment
3. Conduct conformity assessment
4. Generate compliance reports
"""

import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aiact_toolkit.risk_assessment import AIActRiskAssessor
from aiact_toolkit.conformity_assessment import ConformityAssessor, generate_conformity_report


def main():
    # Load pre-captured metadata
    metadata_file = os.path.join(
        os.path.dirname(__file__),
        "generated_outputs",
        "example_metadata.json"
    )

    if not os.path.exists(metadata_file):
        print(f"Error: Example metadata not found at {metadata_file}")
        print("Run 'python examples/llama2_medical_chatbot_integration.py' first.")
        return 1

    print("Loading captured metadata...")
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    print(f"System: {metadata.get('system_name', 'Unknown')}")
    print()

    # Perform risk assessment
    print("=" * 70)
    print("STEP 1: RISK ASSESSMENT")
    print("=" * 70)
    print()

    risk_assessor = AIActRiskAssessor()
    risk_assessment = risk_assessor.assess_risk(
        metadata=metadata,
        use_case="Medical advisory chatbot providing general health advice",
        application_domain="healthcare"
    )

    print(f"Risk Level: {risk_assessment['risk_level'].upper()}")
    print(f"Confidence: {risk_assessment['confidence'] * 100:.0f}%")
    print("\nRisk Factors:")
    for factor in risk_assessment['risk_factors']:
        print(f"  - {factor}")

    # Add risk assessment to metadata for conformity assessment
    metadata['risk_assessment'] = risk_assessment

    # Perform conformity assessment
    print("\n" + "=" * 70)
    print("STEP 2: CONFORMITY ASSESSMENT")
    print("=" * 70)
    print()

    conformity_assessor = ConformityAssessor()
    conformity_result = conformity_assessor.assess_compliance(metadata)

    # Print summary report
    summary = generate_conformity_report(conformity_result)
    print(summary)

    # Show detailed findings for critical gaps
    if conformity_result.critical_gaps:
        print("\n" + "=" * 70)
        print("CRITICAL GAPS")
        print("=" * 70)
        print()
        for i, gap in enumerate(conformity_result.critical_gaps[:5], 1):
            print(f"{i}. {gap}")

    # Show recommendations
    if conformity_result.recommendations:
        print("\n" + "=" * 70)
        print("RECOMMENDATIONS")
        print("=" * 70)
        print()
        for i, rec in enumerate(conformity_result.recommendations[:5], 1):
            print(f"{i}. {rec}")

    # Summary
    compliance_rate = (
        conformity_result.requirements_passed / conformity_result.requirements_checked * 100
        if conformity_result.requirements_checked > 0 else 0
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"Overall Status: {conformity_result.overall_status.value.upper()}")
    print(f"Compliance Rate: {compliance_rate:.0f}%")
    print(f"Requirements: {conformity_result.requirements_passed}/{conformity_result.requirements_checked} passed")
    print()
    print("Generate detailed reports using the CLI:")
    print(f"  aiact-toolkit conformity-assessment {metadata_file} -o conformity_report.md")
    print(f"  aiact-toolkit generate-technical-doc {metadata_file} -o technical_doc.md")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
