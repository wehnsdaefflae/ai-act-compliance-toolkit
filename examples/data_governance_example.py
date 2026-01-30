"""
Data Governance Tracking Example

Demonstrates data lineage and quality tracking for EU AI Act Article 10.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aiact_toolkit import (
    DataGovernanceTracker,
    DataType,
    DataQualityStatus,
    TransformationType,
    MetadataStorage
)


def main():
    """Demonstrate data governance tracking."""

    print("=" * 70)
    print("EU AI Act Article 10 - Data Governance Example")
    print("=" * 70)
    print()

    # Initialize tracker
    print("1. Initializing Data Governance Tracker...")
    tracker = DataGovernanceTracker(system_name="medical_diagnosis_ai")
    print(f"   System: {tracker.system_name}")
    print()

    # Register raw data source
    print("2. Registering Data Sources...")

    raw_records = tracker.register_data_source(
        source_id="raw_medical_records",
        name="Raw Medical Records",
        description="Electronic health records from partner hospitals",
        data_type=DataType.TRAINING,
        location="/data/raw/medical_records.csv",
        size_records=50000,
        personal_data=True,
        sensitive_data=True
    )
    raw_records.set_quality_status(DataQualityStatus.GOOD)
    print(f"   - {raw_records.name}: {raw_records.size_records} records")

    # Register cleaned data
    cleaned_records = tracker.register_data_source(
        source_id="cleaned_medical_records",
        name="Cleaned Medical Records",
        description="Medical records after data cleaning",
        data_type=DataType.TRAINING,
        location="/data/processed/cleaned_records.csv",
        size_records=48500,
        personal_data=True,
        sensitive_data=True
    )
    cleaned_records.set_quality_status(DataQualityStatus.GOOD)
    print(f"   - {cleaned_records.name}: {cleaned_records.size_records} records")

    # Register anonymized data
    anonymized_records = tracker.register_data_source(
        source_id="anonymized_medical_records",
        name="Anonymized Medical Records",
        description="De-identified records for model training",
        data_type=DataType.TRAINING,
        location="/data/processed/anonymized_records.csv",
        size_records=48500,
        personal_data=False,  # No longer contains PII
        sensitive_data=True
    )
    anonymized_records.set_quality_status(DataQualityStatus.GOOD)
    print(f"   - {anonymized_records.name}: {anonymized_records.size_records} records")
    print()

    # Record transformations
    print("3. Recording Data Transformations...")

    cleaning = tracker.register_transformation(
        transformation_id="transform_001_cleaning",
        transformation_type=TransformationType.CLEANING,
        description="Remove incomplete records and validate data types",
        input_source_id="raw_medical_records",
        output_source_id="cleaned_medical_records"
    )
    print(f"   - {cleaning.transformation_type.value}: {cleaning.description}")

    anonymization = tracker.register_transformation(
        transformation_id="transform_002_anonymization",
        transformation_type=TransformationType.ANONYMIZATION,
        description="Remove PII and apply k-anonymity",
        input_source_id="cleaned_medical_records",
        output_source_id="anonymized_medical_records"
    )
    print(f"   - {anonymization.transformation_type.value}: {anonymization.description}")
    print()

    # Display summaries
    print("=" * 70)
    print("DATA GOVERNANCE SUMMARY")
    print("=" * 70)
    print()

    quality_summary = tracker.get_data_quality_summary()
    print("Data Quality:")
    print(f"  Total Sources: {quality_summary['total_sources']}")
    print(f"  Assessed: {quality_summary['sources_with_quality_metrics']}")
    print()

    privacy_summary = tracker.get_privacy_summary()
    print("Privacy:")
    print(f"  Sources with Personal Data: {privacy_summary['personal_data_sources']}")
    print(f"  Sources with Sensitive Data: {privacy_summary['sensitive_data_sources']}")
    print()

    # Generate Article 10 report
    print("4. Generating Article 10 Report...")
    report = tracker.generate_article10_report()
    print(f"   Data Sources: {report['data_sources']['total']}")
    print(f"   Transformations: {report['transformations']['total']}")
    print(f"   Compliance Checks: {report['compliance_checks']['passed']}/{report['compliance_checks']['total']}")
    print()

    # Integrate with metadata storage
    print("5. Integration with MetadataStorage...")
    storage = MetadataStorage(system_name="medical_diagnosis_ai")
    # The data governance tracker is automatically initialized
    metadata = storage.get_all_metadata()
    print(f"   System: {metadata['system_name']}")
    print(f"   Data Governance: {'included' if 'data_governance' in metadata else 'not included'}")
    print()

    print("=" * 70)
    print("Example Complete")
    print("=" * 70)
    print()
    print("Key Points:")
    print("  1. Track data sources with origin, quality, and privacy info")
    print("  2. Record transformations to establish data lineage")
    print("  3. Generate Article 10 compliance reports")
    print("  4. Integrate with MetadataStorage for complete compliance metadata")


if __name__ == "__main__":
    main()
