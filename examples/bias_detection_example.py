"""
Example: Bias Detection and Fairness Analysis

This example demonstrates how to use the BiasDetector to analyze datasets and
model predictions for potential bias, in compliance with EU AI Act Article 10
(Data Governance) and Article 15 (Accuracy and Robustness) requirements.
"""

import sys
from pathlib import Path

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aiact_toolkit import (
    BiasDetector,
    BiasReportGenerator,
    MetadataStorage,
    DocumentGenerator
)


def example_dataset_bias_analysis():
    """
    Example 1: Analyzing a dataset for representation bias

    This is critical for EU AI Act compliance - Article 10(2)(f) requires
    "appropriate measures to detect, prevent and mitigate possible biases"
    """
    print("="*80)
    print("Example 1: Dataset Representation Bias Analysis")
    print("="*80)

    # Simulate a hiring dataset with potential gender and age bias
    dataset_name = "hiring_candidates"
    data = {
        'gender': ['male', 'male', 'male', 'male', 'male', 'male', 'male',
                   'female', 'female', 'female'],  # 70% male, 30% female
        'age_group': ['25-35', '25-35', '25-35', '25-35', '36-50', '36-50',
                      '36-50', '51+', '51+', '25-35'],  # Uneven age distribution
        'hired': [1, 1, 0, 1, 1, 0, 1, 0, 0, 1]  # Binary outcome
    }

    # Create bias detector
    detector = BiasDetector()

    # Analyze the dataset
    result = detector.analyze_dataset(
        dataset_name=dataset_name,
        data=data,
        protected_attributes=['gender', 'age_group'],
        target_column='hired',
        metadata={'source': 'recruitment_system', 'year': 2024}
    )

    # Print results
    print(f"\nDataset: {result.dataset_name}")
    print(f"Overall Fairness Score: {result.overall_fairness_score:.1%}")
    print(f"Risk Level: {result.risk_level.upper()}")
    print(f"\nDetected Metrics:")

    for metric in result.metrics:
        status = "✓ PASS" if metric.passed else "✗ FAIL"
        print(f"\n  {status} - {metric.metric_name}")
        print(f"    Protected Attribute: {metric.protected_attribute}")
        print(f"    Value: {metric.value:.4f} (threshold: {metric.threshold:.4f})")
        print(f"    Groups: {', '.join(str(g) for g in metric.groups_analyzed)}")

        if metric.details and not metric.passed:
            if 'proportions' in metric.details:
                print(f"    Distribution:")
                for group, prop in metric.details['proportions'].items():
                    print(f"      {group}: {prop:.1%}")

    print(f"\nRecommendations:")
    for i, rec in enumerate(result.recommendations, 1):
        print(f"  {i}. {rec}")

    return result


def example_model_predictions_bias_analysis():
    """
    Example 2: Analyzing model predictions for fairness

    Analyzes whether a model's predictions exhibit bias across protected groups
    """
    print("\n" + "="*80)
    print("Example 2: Model Predictions Fairness Analysis")
    print("="*80)

    # Simulate model predictions on a loan approval dataset
    # Ground truth: 1 = should be approved, 0 = should be rejected
    ground_truth = [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]

    # Model predictions (showing potential bias)
    predictions = [1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]

    # Protected attributes
    protected_attrs = {
        'ethnicity': ['group_A', 'group_A', 'group_A', 'group_A', 'group_A',
                     'group_B', 'group_B', 'group_B', 'group_B', 'group_B',
                     'group_A', 'group_A', 'group_B', 'group_B', 'group_A',
                     'group_B', 'group_A', 'group_B', 'group_A', 'group_B'],
        'gender': ['male', 'male', 'female', 'male', 'female',
                  'male', 'female', 'male', 'female', 'male',
                  'female', 'male', 'female', 'male', 'female',
                  'male', 'female', 'male', 'female', 'male']
    }

    # Create detector and analyze
    detector = BiasDetector()
    result = detector.analyze_model_predictions(
        model_name="loan_approval_model",
        predictions=predictions,
        ground_truth=ground_truth,
        protected_attributes_data=protected_attrs,
        metadata={'model_type': 'random_forest', 'accuracy': 0.75}
    )

    # Print results
    print(f"\nModel: {result.dataset_name}")
    print(f"Overall Fairness Score: {result.overall_fairness_score:.1%}")
    print(f"Risk Level: {result.risk_level.upper()}")

    # Group metrics by type
    metric_types = {}
    for metric in result.metrics:
        if metric.metric_type not in metric_types:
            metric_types[metric.metric_type] = []
        metric_types[metric.metric_type].append(metric)

    print(f"\nFairness Metrics by Type:")
    for metric_type, metrics in metric_types.items():
        print(f"\n  {metric_type.upper()}:")
        for metric in metrics:
            status = "✓" if metric.passed else "✗"
            print(f"    {status} {metric.protected_attribute}: {metric.value:.4f} (threshold: {metric.threshold})")

            # Show group-level details
            if metric.details:
                if 'tpr_by_group' in metric.details:
                    print(f"       True Positive Rates:")
                    for group, rate in metric.details['tpr_by_group'].items():
                        print(f"         {group}: {rate:.1%}")
                elif 'positive_rates' in metric.details:
                    print(f"       Positive Prediction Rates:")
                    for group, rate in metric.details['positive_rates'].items():
                        print(f"         {group}: {rate:.1%}")
                elif 'ppv_by_group' in metric.details:
                    print(f"       Precision by Group:")
                    for group, ppv in metric.details['ppv_by_group'].items():
                        print(f"         {group}: {ppv:.1%}")

    print(f"\nRecommendations:")
    for i, rec in enumerate(result.recommendations, 1):
        print(f"  {i}. {rec}")

    return result


def example_integration_with_metadata_storage():
    """
    Example 3: Integrating bias analysis with metadata storage

    Shows how to store bias analysis results alongside other compliance metadata
    """
    print("\n" + "="*80)
    print("Example 3: Integration with Metadata Storage")
    print("="*80)

    # Create metadata storage for a medical diagnosis system
    storage = MetadataStorage(
        system_name="medical_diagnosis_assistant",
        enable_auditing=True,
        enable_versioning=True
    )

    # Add some model info
    storage.add_model({
        "model_name": "diagnosis_classifier",
        "provider": "pytorch",
        "parameters": {"architecture": "ResNet50", "input_size": 224}
    })

    # Simulate dataset analysis
    detector = BiasDetector()

    # Analyze training data
    training_data = {
        'age_group': ['18-30'] * 100 + ['31-50'] * 150 + ['51-70'] * 80 + ['71+'] * 20,
        'gender': ['male'] * 180 + ['female'] * 170,
        'ethnicity': ['group_A'] * 200 + ['group_B'] * 100 + ['group_C'] * 50,
        'diagnosis_positive': [1] * 175 + [0] * 175
    }

    dataset_result = detector.analyze_dataset(
        dataset_name="medical_training_data",
        data=training_data,
        protected_attributes=['age_group', 'gender', 'ethnicity'],
        target_column='diagnosis_positive'
    )

    # Add bias analysis to storage
    storage.add_bias_analysis(dataset_result.to_dict())

    print(f"\nBias analysis added to metadata storage")
    print(f"System: {storage.system_name}")
    print(f"Bias analyses recorded: {len(storage.bias_analyses)}")

    # Save to file
    output_file = "medical_diagnosis_metadata.json"
    storage.save_to_file(output_file)
    print(f"\n✓ Metadata saved to: {output_file}")

    # Generate bias report
    print("\nGenerating bias and fairness report...")
    generator = DocumentGenerator()
    metadata = storage.get_all_metadata()

    generator.generate_document(
        template_name="bias_fairness_report.md.jinja2",
        metadata=metadata,
        output_path="medical_diagnosis_bias_report.md"
    )
    print("✓ Bias report generated: medical_diagnosis_bias_report.md")

    # Show summary from metadata
    if 'bias_summary' in metadata:
        summary = metadata['bias_summary']
        print(f"\nBias Summary:")
        print(f"  Total Analyses: {summary['total_analyses']}")
        print(f"  Overall Risk: {summary['overall_risk_level'].upper()}")
        print(f"  Avg Fairness Score: {summary['average_fairness_score']:.1%}")

    return storage


def example_comprehensive_bias_report():
    """
    Example 4: Generate comprehensive bias report with multiple analyses
    """
    print("\n" + "="*80)
    print("Example 4: Comprehensive Bias Report Generation")
    print("="*80)

    detector = BiasDetector()
    analyses = []

    # Analyze multiple datasets/models
    datasets_configs = [
        {
            'name': 'recruitment_dataset',
            'data': {
                'gender': ['male'] * 60 + ['female'] * 40,
                'age': ['young'] * 55 + ['senior'] * 45,
                'outcome': [1] * 52 + [0] * 48
            },
            'protected': ['gender', 'age'],
            'target': 'outcome'
        },
        {
            'name': 'promotion_dataset',
            'data': {
                'gender': ['male'] * 70 + ['female'] * 30,
                'department': ['tech'] * 60 + ['non_tech'] * 40,
                'promoted': [1] * 45 + [0] * 55
            },
            'protected': ['gender', 'department'],
            'target': 'promoted'
        }
    ]

    for config in datasets_configs:
        result = detector.analyze_dataset(
            dataset_name=config['name'],
            data=config['data'],
            protected_attributes=config['protected'],
            target_column=config['target']
        )
        analyses.append(result)
        print(f"\n✓ Analyzed: {config['name']}")
        print(f"  Risk Level: {result.risk_level.upper()}")
        print(f"  Fairness Score: {result.overall_fairness_score:.1%}")

    # Generate comprehensive report
    report = BiasReportGenerator.generate_report(analyses)

    print(f"\n{'='*60}")
    print("Comprehensive Bias Report Summary")
    print(f"{'='*60}")
    print(f"Total Analyses: {report['summary']['total_analyses']}")
    print(f"Overall Risk: {report['summary']['overall_risk_level'].upper()}")
    print(f"Average Fairness: {report['summary']['average_fairness_score']:.1%}")

    print(f"\nCompliance Notes:")
    for note in report['compliance_notes']:
        print(f"  • {note}")

    print(f"\nAggregated Recommendations ({len(report['aggregated_recommendations'])} total):")
    for i, rec in enumerate(report['aggregated_recommendations'][:5], 1):  # Show first 5
        print(f"  {i}. {rec}")

    return report


if __name__ == "__main__":
    print("\n" + "="*80)
    print("AI Act Compliance Toolkit - Bias Detection Examples")
    print("EU AI Act Article 10 (Data Governance) & Article 15 (Robustness)")
    print("="*80 + "\n")

    # Run all examples
    try:
        # Example 1: Dataset bias analysis
        dataset_result = example_dataset_bias_analysis()

        # Example 2: Model predictions bias analysis
        model_result = example_model_predictions_bias_analysis()

        # Example 3: Integration with metadata storage
        storage = example_integration_with_metadata_storage()

        # Example 4: Comprehensive report
        comprehensive_report = example_comprehensive_bias_report()

        print("\n" + "="*80)
        print("All examples completed successfully!")
        print("="*80)
        print("\nKey Takeaways:")
        print("  1. Use BiasDetector to analyze datasets for representation bias")
        print("  2. Analyze model predictions across protected groups for fairness")
        print("  3. Integrate bias analysis with MetadataStorage for compliance tracking")
        print("  4. Generate comprehensive bias reports for EU AI Act documentation")
        print("\nGenerated Files:")
        print("  • medical_diagnosis_metadata.json - Complete system metadata")
        print("  • medical_diagnosis_bias_report.md - Bias analysis report")
        print("\nNext Steps:")
        print("  • Review generated bias reports")
        print("  • Implement recommended bias mitigation strategies")
        print("  • Continuously monitor for bias during production")
        print("  • Document all fairness measures for regulatory compliance")

    except Exception as e:
        print(f"\nError running examples: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
