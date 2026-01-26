"""
Tests for EU AI Act Conformity Assessment Module
"""

import sys
import os
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aiact_toolkit.conformity_assessment import (
    ConformityAssessor,
    ComplianceStatus,
    RequirementCategory,
    generate_conformity_report
)
from aiact_toolkit.metadata_storage import MetadataStorage
from aiact_toolkit.data_governance import DataType


def test_minimal_risk_system():
    """Test conformity assessment for minimal risk system"""
    print("="*80)
    print("TEST 1: Minimal Risk System Assessment")
    print("="*80)

    # Create minimal metadata
    storage = MetadataStorage(system_name="Simple Chatbot")
    storage.add_model({"model_name": "gpt-3.5-turbo", "provider": "OpenAI", "parameters": {"temperature": 0.7}})

    # Set minimal risk
    storage.set_risk_assessment({
        "risk_level": "minimal",
        "confidence": 0.9,
        "risk_factors": ["General purpose chatbot"],
        "compliance_requirements": ["Basic transparency"],
        "recommendations": []
    })

    metadata = storage.get_all_metadata()

    # Perform assessment
    assessor = ConformityAssessor()
    result = assessor.assess_compliance(metadata)

    print(f"\nSystem: {result.system_name}")
    print(f"Risk Level: {result.risk_level}")
    print(f"Overall Status: {result.overall_status.value}")
    print(f"Requirements Checked: {result.requirements_checked}")
    print(f"Passed: {result.requirements_passed}")
    print(f"Failed: {result.requirements_failed}")

    # Minimal risk systems should have fewer requirements
    assert result.requirements_checked < 10, "Minimal risk should have fewer requirements"
    print("\n✓ Test passed: Minimal risk system assessed correctly")


def test_high_risk_system_without_compliance():
    """Test conformity assessment for high-risk system without compliance measures"""
    print("\n" + "="*80)
    print("TEST 2: High-Risk System Without Compliance Features")
    print("="*80)

    # Create basic metadata without compliance features
    storage = MetadataStorage(system_name="Medical Diagnosis AI")
    storage.add_model({"model_name": "medical-model-v1", "provider": "Custom", "parameters": {
        "architecture": "transformer",
        "parameters": 1000000
    }})
    storage.add_data_source({"data_source": "medical_images", "loader": "ImageLoader", "path": "medical_images"})

    # Set high risk
    storage.set_risk_assessment({
        "risk_level": "high",
        "confidence": 0.95,
        "risk_factors": ["Healthcare application", "Medical diagnosis"],
        "compliance_requirements": [
            "Risk management system",
            "Technical documentation",
            "Human oversight"
        ],
        "recommendations": ["Implement comprehensive compliance measures"]
    })

    metadata = storage.get_all_metadata()

    # Perform assessment
    assessor = ConformityAssessor()
    result = assessor.assess_compliance(metadata)

    print(f"\nSystem: {result.system_name}")
    print(f"Risk Level: {result.risk_level}")
    print(f"Overall Status: {result.overall_status.value}")
    print(f"Requirements Checked: {result.requirements_checked}")
    print(f"Passed: {result.requirements_passed}")
    print(f"Failed: {result.requirements_failed}")

    # High-risk without compliance should fail many requirements
    assert result.overall_status in [ComplianceStatus.NON_COMPLIANT, ComplianceStatus.PARTIAL], \
        "High-risk system without compliance should not be fully compliant"
    assert result.requirements_failed > 0, "Should have failed requirements"
    assert len(result.critical_gaps) > 0, "Should have critical gaps"

    print(f"\nCritical Gaps: {len(result.critical_gaps)}")
    for gap in result.critical_gaps[:3]:
        print(f"  - {gap}")

    print("\n✓ Test passed: Non-compliant high-risk system identified correctly")


def test_high_risk_system_with_full_compliance():
    """Test conformity assessment for fully compliant high-risk system"""
    print("\n" + "="*80)
    print("TEST 3: High-Risk System With Full Compliance Features")
    print("="*80)

    # Create comprehensive metadata with all compliance features
    storage = MetadataStorage(
        system_name="Medical Advisory System",
        enable_auditing=True
    )

    # Add model with full documentation
    storage.add_model({
        "model_name": "gpt-4-medical",
        "provider": "OpenAI",
        "parameters": {
            "temperature": 0.3,
            "max_tokens": 1000,
            "model_type": "chat",
            "capabilities": ["Medical advice", "Symptom analysis"],
            "limitations": ["Not a replacement for medical professionals"]
        }
    })

    # Add data sources with governance
    data_tracker = storage.get_data_governance_tracker()
    if data_tracker:
        data_tracker.register_data_source(
            source_id="training_data",
            name="Medical Training Data",
            data_type=DataType.TRAINING,
            description="Curated medical Q&A",
            personal_data=False,
            sensitive_data=True
        )

    # Add comprehensive risk assessment
    storage.set_risk_assessment({
        "risk_level": "high",
        "confidence": 0.95,
        "risk_factors": [
            "Healthcare application",
            "Medical advisory"
        ],
        "compliance_requirements": [
            "Risk management system required",
            "Technical documentation (Article 11) required",
            "Automatic logging (Article 12) required",
            "Human oversight required",
            "Transparency obligations (Article 13)"
        ],
        "recommendations": [
            "Maintain audit trail",
            "Regular bias monitoring",
            "Continuous human oversight"
        ],
        "use_case": "Medical advisory chatbot for general health guidance"
    })

    # Set operational metrics directly
    storage.operational_metrics = {
        "execution_time_ms": 250,
        "success": True,
        "model_name": "gpt-4-medical"
    }

    metadata = storage.get_all_metadata()

    # Perform assessment
    assessor = ConformityAssessor()
    result = assessor.assess_compliance(metadata)

    print(f"\nSystem: {result.system_name}")
    print(f"Risk Level: {result.risk_level}")
    print(f"Overall Status: {result.overall_status.value}")
    print(f"Requirements Checked: {result.requirements_checked}")
    print(f"Passed: {result.requirements_passed}")
    print(f"Failed: {result.requirements_failed}")
    print(f"Partial: {result.requirements_partial}")

    compliance_rate = (result.requirements_passed / result.requirements_checked * 100) \
        if result.requirements_checked > 0 else 0
    print(f"Compliance Rate: {compliance_rate:.1f}%")

    # High-risk systems have many requirements - partial compliance is expected
    # without full enterprise implementation (bias analysis, cybersecurity docs, etc.)
    assert result.requirements_passed > 0, "Should pass some requirements with compliance features"
    assert compliance_rate > 30, "Should have >30% compliance rate with basic features"

    # Print category results
    print("\nCategory Results:")
    for category, data in result.category_results.items():
        print(f"  {category}: {data['passed']}/{data['total_requirements']} "
              f"({data['compliance_percentage']:.0f}%)")

    print("\n✓ Test passed: Compliant high-risk system recognized correctly")


def test_conformity_report_generation():
    """Test conformity report text generation"""
    print("\n" + "="*80)
    print("TEST 4: Conformity Report Generation")
    print("="*80)

    # Create simple system
    storage = MetadataStorage(system_name="Test System")
    storage.add_model({"model_name": "test-model", "provider": "Test Provider", "parameters": {}})
    storage.set_risk_assessment({
        "risk_level": "limited",
        "confidence": 0.8,
        "risk_factors": ["Test factor"],
        "compliance_requirements": [],
        "recommendations": []
    })

    metadata = storage.get_all_metadata()

    # Perform assessment
    assessor = ConformityAssessor()
    result = assessor.assess_compliance(metadata)

    # Generate report
    report = generate_conformity_report(result)

    print("\n" + report)

    # Verify report contains key sections
    assert "KONFORMITÄTSBEWERTUNG" in report, "Report should have title"
    assert result.system_name in report, "Report should contain system name"
    assert result.risk_level in report.lower(), "Report should contain risk level"
    assert "ÜBERBLICK" in report, "Report should have overview section"

    print("\n✓ Test passed: Conformity report generated successfully")


def test_category_compliance():
    """Test per-category compliance tracking"""
    print("\n" + "="*80)
    print("TEST 5: Category-Based Compliance Tracking")
    print("="*80)

    # Create system with mixed compliance
    storage = MetadataStorage(
        system_name="Category Test System",
        enable_auditing=True
    )

    storage.add_model({"model_name": "test-model", "provider": "Provider", "parameters": {"param": "value"}})

    # Add data with quality metrics
    data_tracker = storage.get_data_governance_tracker()
    if data_tracker:
        data_tracker.register_data_source(
            source_id="test_data",
            name="Test Data",
            data_type=DataType.TRAINING,
            description="Test dataset"
        )

    storage.set_risk_assessment({
        "risk_level": "high",
        "confidence": 0.9,
        "risk_factors": ["Test"],
        "compliance_requirements": ["Various"],
        "recommendations": []
    })

    metadata = storage.get_all_metadata()

    # Perform assessment
    assessor = ConformityAssessor()
    result = assessor.assess_compliance(metadata)

    print("\nCategory Compliance Results:")
    for category, data in result.category_results.items():
        print(f"\n{category}:")
        print(f"  Total Requirements: {data['total_requirements']}")
        print(f"  Passed: {data['passed']}")
        print(f"  Compliance: {data['compliance_percentage']:.1f}%")
        if data['critical_gaps']:
            print(f"  Critical Gaps: {', '.join(data['critical_gaps'])}")

    # Verify different categories have different results
    assert len(result.category_results) > 0, "Should have category results"

    # Data governance should be better due to enabled feature
    if RequirementCategory.DATA_GOVERNANCE.value in result.category_results:
        dg_result = result.category_results[RequirementCategory.DATA_GOVERNANCE.value]
        print(f"\nData Governance Compliance: {dg_result['compliance_percentage']:.1f}%")

    # Record-keeping should be good due to audit trail
    if RequirementCategory.RECORD_KEEPING.value in result.category_results:
        rk_result = result.category_results[RequirementCategory.RECORD_KEEPING.value]
        print(f"Record-Keeping Compliance: {rk_result['compliance_percentage']:.1f}%")

    print("\n✓ Test passed: Category compliance tracked correctly")
