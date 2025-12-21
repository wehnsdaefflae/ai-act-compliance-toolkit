"""
Data Governance and Provenance Tracking Example

This example demonstrates how to use the Data Governance module to track
data lineage, quality, and compliance for EU AI Act Article 10 requirements.

This is particularly important for high-risk AI systems that need to document:
- Training data sources and provenance
- Data quality assessment
- Data transformations and lineage
- Privacy and licensing compliance
"""

from aiact_toolkit import (
    DataGovernanceTracker,
    DataType,
    DataQualityStatus,
    TransformationType,
    MetadataStorage
)

def main():
    """Demonstrate data governance tracking capabilities."""

    print("=" * 80)
    print("EU AI Act Article 10 - Data Governance Example")
    print("=" * 80)
    print()

    # Initialize data governance tracker
    print("1. Initializing Data Governance Tracker...")
    tracker = DataGovernanceTracker(system_name="medical_diagnosis_ai")
    print(f"   ✓ Initialized tracker for system: {tracker.system_name}")
    print()

    # Register raw data sources
    print("2. Registering Raw Data Sources...")

    # Register raw medical records
    raw_records = tracker.register_data_source(
        source_id="raw_medical_records_2024",
        name="Raw Medical Records 2024",
        description="Electronic health records from partner hospitals",
        data_type=DataType.TRAINING,
        location="/data/raw/medical_records_2024.csv",
        format="CSV",
        size_records=50000,
        size_bytes=524288000,  # ~500MB
        source_origin="Partner Hospital Network",
        license="Restricted - Medical Data Use Agreement",
        copyright_info="© 2024 Partner Hospital Network",
        personal_data=True,  # Contains patient data
        sensitive_data=True  # Medical data is sensitive
    )

    # Add quality metrics
    raw_records.add_quality_metric(
        "completeness",
        0.95,
        "95% of required fields are populated"
    )
    raw_records.add_quality_metric(
        "accuracy_verified",
        0.88,
        "88% of records verified by medical professionals"
    )
    raw_records.set_quality_status(DataQualityStatus.GOOD)

    print(f"   ✓ Registered: {raw_records.name}")
    print(f"     - {raw_records.size_records} records")
    print(f"     - Quality: {raw_records.quality_status.value}")
    print(f"     - Contains personal data: {raw_records.personal_data}")
    print()

    # Register reference medical guidelines
    guidelines = tracker.register_data_source(
        source_id="medical_guidelines_2024",
        name="WHO Medical Guidelines 2024",
        description="World Health Organization diagnostic guidelines",
        data_type=DataType.REFERENCE,
        location="/data/reference/who_guidelines.json",
        format="JSON",
        size_records=1500,
        source_origin="World Health Organization",
        license="CC BY-NC-SA 3.0 IGO",
        copyright_info="© 2024 WHO",
        personal_data=False,
        sensitive_data=False
    )
    guidelines.set_quality_status(DataQualityStatus.EXCELLENT)

    print(f"   ✓ Registered: {guidelines.name}")
    print(f"     - License: {guidelines.license}")
    print()

    # Step 1: Data Cleaning Transformation
    print("3. Recording Data Transformations - Step 1: Cleaning...")

    cleaned_records = tracker.register_data_source(
        source_id="cleaned_medical_records",
        name="Cleaned Medical Records",
        description="Medical records after data cleaning and validation",
        data_type=DataType.TRAINING,
        location="/data/processed/cleaned_records.csv",
        format="CSV",
        size_records=48500,
        personal_data=True,
        sensitive_data=True
    )
    cleaned_records.set_quality_status(DataQualityStatus.GOOD)

    cleaning_transform = tracker.register_transformation(
        transformation_id="transform_001_cleaning",
        transformation_type=TransformationType.CLEANING,
        description="Remove incomplete records and validate data types",
        input_source_ids=["raw_medical_records_2024"],
        output_source_id="cleaned_medical_records",
        parameters={
            "min_completeness": 0.9,
            "validation_rules": "medical_data_schema_v2"
        },
        performed_by="data_engineer",
        tool_used="pandas + custom validators"
    )

    print(f"   ✓ Transformation: {cleaning_transform.description}")
    print(f"     - Reduced from {raw_records.size_records} to {cleaned_records.size_records} records")
    print()

    # Step 2: Anonymization
    print("4. Recording Data Transformations - Step 2: Anonymization...")

    anonymized_records = tracker.register_data_source(
        source_id="anonymized_medical_records",
        name="Anonymized Medical Records",
        description="De-identified medical records for model training",
        data_type=DataType.TRAINING,
        location="/data/processed/anonymized_records.csv",
        format="CSV",
        size_records=48500,
        license="Internal Use - Anonymized Medical Data",
        personal_data=False,  # No longer contains personal data after anonymization
        sensitive_data=True   # Still medically sensitive
    )
    anonymized_records.add_quality_metric(
        "k_anonymity",
        5,
        "K-anonymity level of 5 achieved"
    )
    anonymized_records.set_quality_status(DataQualityStatus.EXCELLENT)

    anonymization_transform = tracker.register_transformation(
        transformation_id="transform_002_anonymization",
        transformation_type=TransformationType.ANONYMIZATION,
        description="Remove PII and apply k-anonymity techniques",
        input_source_ids=["cleaned_medical_records"],
        output_source_id="anonymized_medical_records",
        parameters={
            "k_value": 5,
            "quasi_identifiers": ["age", "gender", "zip_code"],
            "suppression_threshold": 0.05
        },
        performed_by="privacy_engineer",
        tool_used="ARX Data Anonymization Tool"
    )

    print(f"   ✓ Transformation: {anonymization_transform.description}")
    print(f"     - Personal data removed: {cleaned_records.personal_data} → {anonymized_records.personal_data}")
    print()

    # Step 3: Feature Extraction
    print("5. Recording Data Transformations - Step 3: Feature Engineering...")

    feature_dataset = tracker.register_data_source(
        source_id="training_features_final",
        name="Training Feature Dataset",
        description="Engineered features for model training",
        data_type=DataType.TRAINING,
        location="/data/final/training_features.parquet",
        format="Parquet",
        size_records=48500,
        personal_data=False,
        sensitive_data=True
    )
    feature_dataset.set_quality_status(DataQualityStatus.EXCELLENT)

    feature_transform = tracker.register_transformation(
        transformation_id="transform_003_features",
        transformation_type=TransformationType.FEATURE_EXTRACTION,
        description="Extract diagnostic features from medical records",
        input_source_ids=["anonymized_medical_records", "medical_guidelines_2024"],
        output_source_id="training_features_final",
        parameters={
            "feature_count": 127,
            "normalization": "z-score",
            "guideline_alignment": True
        },
        performed_by="ml_engineer",
        tool_used="scikit-learn + custom feature extractors"
    )

    print(f"   ✓ Transformation: {feature_transform.description}")
    print(f"     - Feature count: {feature_transform.parameters['feature_count']}")
    print()

    # Add governance policies
    print("6. Adding Data Governance Policies...")

    tracker.add_governance_policy(
        "data_retention",
        {
            "retention_period": "7 years (regulatory requirement)",
            "deletion_procedure": "Secure wipe after retention period",
            "backup_policy": "Encrypted backups, same retention period"
        }
    )

    tracker.add_governance_policy(
        "privacy_protection",
        {
            "anonymization_required": True,
            "k_anonymity_minimum": 5,
            "gdpr_compliance": "Full GDPR Article 25 compliance",
            "access_control": "Role-based, minimum 2FA"
        }
    )

    tracker.add_governance_policy(
        "quality_standards",
        {
            "minimum_completeness": 0.90,
            "validation_required": True,
            "quality_review_frequency": "Quarterly"
        }
    )

    print(f"   ✓ Added {len(tracker.governance_policies)} governance policies")
    print()

    # Run compliance checks
    print("7. Running Compliance Checks...")

    tracker.run_compliance_check(
        check_name="License Documentation Complete",
        check_type="legal",
        passed=True,
        details={"sources_checked": 4, "all_documented": True}
    )

    tracker.run_compliance_check(
        check_name="Personal Data Anonymization",
        check_type="privacy",
        passed=True,
        details={"anonymization_verified": True, "k_anonymity": 5}
    )

    tracker.run_compliance_check(
        check_name="Data Quality Standards Met",
        check_type="data_quality",
        passed=True,
        details={"min_quality": "good", "all_sources_assessed": True}
    )

    tracker.run_compliance_check(
        check_name="Data Lineage Documented",
        check_type="traceability",
        passed=True,
        details={"transformations_tracked": 3, "lineage_complete": True}
    )

    print(f"   ✓ Completed {len(tracker.compliance_checks)} compliance checks")
    print()

    # Display summaries
    print("=" * 80)
    print("DATA GOVERNANCE SUMMARY")
    print("=" * 80)
    print()

    # Data quality summary
    quality_summary = tracker.get_data_quality_summary()
    print("Data Quality:")
    print(f"  Total Sources: {quality_summary['total_sources']}")
    print(f"  Sources with Quality Metrics: {quality_summary['sources_with_quality_metrics']}")
    print("  Quality Distribution:")
    for status, count in quality_summary['quality_distribution'].items():
        if count > 0:
            print(f"    {status}: {count}")
    print()

    # Privacy summary
    privacy_summary = tracker.get_privacy_summary()
    print("Privacy & Compliance:")
    print(f"  Sources with Personal Data: {privacy_summary['personal_data_sources']}")
    print(f"  Sources with Sensitive Data: {privacy_summary['sensitive_data_sources']}")
    print(f"  Sources with License: {privacy_summary['sources_with_license']}")
    print(f"  Sources with Copyright: {privacy_summary['sources_with_copyright']}")
    print()

    # Lineage example
    print("Data Lineage Example (training_features_final):")
    lineage = tracker.get_lineage_report("training_features_final")
    print(f"  Lineage Depth: {lineage['lineage_depth']}")
    print(f"  Total Transformations: {lineage['total_transformations']}")
    print("  Transformation Chain:")
    for trans in lineage['transformations']:
        print(f"    → {trans['transformation_type']}: {trans['description']}")
    print()

    # Generate Article 10 compliance report
    print("8. Generating EU AI Act Article 10 Compliance Report...")
    article10_report = tracker.generate_article10_report()

    print(f"   ✓ Report generated")
    print(f"     - Data Sources: {article10_report['data_sources']['total']}")
    print(f"     - Transformations: {article10_report['transformations']['total']}")
    print(f"     - Compliance Checks Passed: {article10_report['compliance_checks']['passed']}/{article10_report['compliance_checks']['total']}")
    print()

    # Save data governance information
    print("9. Saving Data Governance Information...")
    tracker.save_to_file("examples/generated_outputs/data_governance.json")
    print("   ✓ Saved to: examples/generated_outputs/data_governance.json")
    print()

    # Integrate with metadata storage
    print("10. Integrating with Metadata Storage...")
    storage = MetadataStorage(
        system_name="medical_diagnosis_ai",
        enable_data_governance=True
    )
    storage.set_data_governance_tracker(tracker)
    storage.save_to_file("examples/generated_outputs/metadata_with_governance.json")
    print("   ✓ Saved integrated metadata to: examples/generated_outputs/metadata_with_governance.json")
    print()

    # Generate documentation using template
    print("11. Generating Article 10 Compliance Document...")
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template('article10_data_governance.md.jinja2')
    document = template.render(**article10_report)

    with open("examples/generated_outputs/article10_compliance.md", "w", encoding="utf-8") as f:
        f.write(document)

    print("   ✓ Generated: examples/generated_outputs/article10_compliance.md")
    print()

    print("=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)
    print()
    print("Generated Files:")
    print("  - examples/generated_outputs/data_governance.json")
    print("  - examples/generated_outputs/metadata_with_governance.json")
    print("  - examples/generated_outputs/article10_compliance.md")
    print()
    print("Next Steps:")
    print("  1. Review the generated Article 10 compliance document")
    print("  2. Complete [To be specified] sections with actual information")
    print("  3. Use CLI commands to query data lineage:")
    print("     aiact-toolkit data-lineage metadata_with_governance.json --source-id training_features_final")
    print("  4. Generate data quality reports:")
    print("     aiact-toolkit data-quality metadata_with_governance.json --detailed")
    print()


if __name__ == "__main__":
    main()
