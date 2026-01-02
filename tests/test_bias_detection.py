"""
Tests for Bias Detection Module

Tests the BiasDetector, BiasMetric, BiasAnalysisResult, and BiasReportGenerator classes
"""

import sys
from pathlib import Path
import json

# Add source directory to path
src_path = Path(__file__).parent.parent / "src" / "aiact_toolkit"
sys.path.insert(0, str(src_path.parent))

# Import directly from modules without triggering __init__.py
import importlib.util

# Load bias_detection module
spec = importlib.util.spec_from_file_location("bias_detection", src_path / "bias_detection.py")
bias_detection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bias_detection)

BiasDetector = bias_detection.BiasDetector
BiasMetric = bias_detection.BiasMetric
BiasAnalysisResult = bias_detection.BiasAnalysisResult
BiasReportGenerator = bias_detection.BiasReportGenerator

# Load metadata_storage module
spec = importlib.util.spec_from_file_location("metadata_storage", src_path / "metadata_storage.py")
metadata_storage = importlib.util.module_from_spec(spec)

# Need to load dependencies first
spec_audit = importlib.util.spec_from_file_location("audit_trail", src_path / "audit_trail.py")
audit_trail = importlib.util.module_from_spec(spec_audit)
sys.modules['audit_trail'] = audit_trail
spec_audit.loader.exec_module(audit_trail)

spec_vc = importlib.util.spec_from_file_location("version_control", src_path / "version_control.py")
version_control = importlib.util.module_from_spec(spec_vc)
sys.modules['version_control'] = version_control
spec_vc.loader.exec_module(version_control)

spec_dg = importlib.util.spec_from_file_location("data_governance", src_path / "data_governance.py")
data_governance = importlib.util.module_from_spec(spec_dg)
sys.modules['data_governance'] = data_governance
spec_dg.loader.exec_module(data_governance)

sys.modules['metadata_storage'] = metadata_storage
spec.loader.exec_module(metadata_storage)

MetadataStorage = metadata_storage.MetadataStorage


def test_bias_detector_initialization():
    """Test BiasDetector initialization with default and custom thresholds"""
    print("Testing BiasDetector initialization...")

    # Test with default thresholds
    detector = BiasDetector()
    assert detector.thresholds['demographic_parity'] == 0.1
    assert detector.thresholds['equal_opportunity'] == 0.1

    # Test with custom thresholds
    custom_thresholds = {
        'demographic_parity': 0.05,
        'equal_opportunity': 0.15
    }
    detector_custom = BiasDetector(thresholds=custom_thresholds)
    assert detector_custom.thresholds['demographic_parity'] == 0.05
    assert detector_custom.thresholds['equal_opportunity'] == 0.15

    print("✓ BiasDetector initialization tests passed")


def test_dataset_representation_bias():
    """Test detection of representation bias in datasets"""
    print("\nTesting dataset representation bias detection...")

    detector = BiasDetector()

    # Create dataset with imbalanced gender representation (70% male, 30% female)
    data = {
        'gender': ['male'] * 7 + ['female'] * 3,
        'outcome': [1, 1, 0, 1, 0, 1, 0, 0, 1, 0]
    }

    result = detector.analyze_dataset(
        dataset_name="test_dataset",
        data=data,
        protected_attributes=['gender'],
        target_column='outcome'
    )

    # Verify result structure
    assert result.dataset_name == "test_dataset"
    assert result.analysis_type == "dataset"
    assert isinstance(result.metrics, list)
    assert len(result.metrics) > 0
    assert result.overall_fairness_score >= 0 and result.overall_fairness_score <= 1
    assert result.risk_level in ['low', 'medium', 'high', 'critical', 'unknown']

    # Check that representation metric was calculated
    representation_metrics = [m for m in result.metrics if m.metric_type == 'representation']
    assert len(representation_metrics) > 0

    print(f"  Dataset: {result.dataset_name}")
    print(f"  Fairness Score: {result.overall_fairness_score:.2f}")
    print(f"  Risk Level: {result.risk_level}")
    print(f"  Metrics Calculated: {len(result.metrics)}")
    print("✓ Dataset representation bias tests passed")


def test_dataset_outcome_bias():
    """Test detection of outcome bias in datasets"""
    print("\nTesting dataset outcome bias detection...")

    detector = BiasDetector()

    # Create dataset with outcome disparity
    # Males: 80% positive outcomes, Females: 40% positive outcomes
    data = {
        'gender': ['male'] * 5 + ['female'] * 5,
        'outcome': [1, 1, 1, 1, 0] + [1, 1, 0, 0, 0]  # 4/5 vs 2/5
    }

    result = detector.analyze_dataset(
        dataset_name="outcome_bias_test",
        data=data,
        protected_attributes=['gender'],
        target_column='outcome'
    )

    # Check that outcome metric was calculated
    outcome_metrics = [m for m in result.metrics if m.metric_type == 'outcome_distribution']
    assert len(outcome_metrics) > 0

    # The bias should be detected (difference of 0.4 > threshold of 0.1)
    outcome_metric = outcome_metrics[0]
    assert not outcome_metric.passed  # Should fail the fairness check

    print(f"  Outcome Bias Value: {outcome_metric.value:.4f}")
    print(f"  Threshold: {outcome_metric.threshold:.4f}")
    print(f"  Passed: {outcome_metric.passed}")
    print("✓ Dataset outcome bias tests passed")


def test_model_demographic_parity():
    """Test demographic parity metric on model predictions"""
    print("\nTesting model demographic parity...")

    detector = BiasDetector()

    # Create predictions with demographic parity violation
    # Group A: 80% positive predictions, Group B: 40% positive predictions
    predictions = [1, 1, 1, 1, 0] + [1, 1, 0, 0, 0]
    ground_truth = [1, 1, 0, 1, 0] + [1, 0, 0, 1, 0]
    protected_attrs = {
        'group': ['A', 'A', 'A', 'A', 'A'] + ['B', 'B', 'B', 'B', 'B']
    }

    result = detector.analyze_model_predictions(
        model_name="demographic_parity_test",
        predictions=predictions,
        ground_truth=ground_truth,
        protected_attributes_data=protected_attrs
    )

    # Check demographic parity metric
    dp_metrics = [m for m in result.metrics if m.metric_type == 'demographic_parity']
    assert len(dp_metrics) > 0

    dp_metric = dp_metrics[0]
    print(f"  Demographic Parity Difference: {dp_metric.value:.4f}")
    print(f"  Groups Analyzed: {dp_metric.groups_analyzed}")
    print(f"  Passed: {dp_metric.passed}")

    # Should detect violation (0.4 difference > 0.1 threshold)
    assert not dp_metric.passed

    print("✓ Model demographic parity tests passed")


def test_model_equal_opportunity():
    """Test equal opportunity metric on model predictions"""
    print("\nTesting model equal opportunity...")

    detector = BiasDetector()

    # Create predictions with equal opportunity violation
    # Both groups have same positive prediction rate, but different TPR
    predictions = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
    ground_truth = [1, 1, 1, 0, 0, 1, 1, 1, 0, 0]
    # Group A: TP=2 out of 4 positives (TPR=0.5)
    # Group B: TP=1 out of 4 positives (TPR=0.25)
    protected_attrs = {
        'group': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B']
    }

    result = detector.analyze_model_predictions(
        model_name="equal_opportunity_test",
        predictions=predictions,
        ground_truth=ground_truth,
        protected_attributes_data=protected_attrs
    )

    # Check equal opportunity metric
    eo_metrics = [m for m in result.metrics if m.metric_type == 'equal_opportunity']
    assert len(eo_metrics) > 0

    eo_metric = eo_metrics[0]
    print(f"  Equal Opportunity Difference: {eo_metric.value:.4f}")
    print(f"  Details: {eo_metric.details}")
    print(f"  Passed: {eo_metric.passed}")

    print("✓ Model equal opportunity tests passed")


def test_model_predictive_parity():
    """Test predictive parity metric on model predictions"""
    print("\nTesting model predictive parity...")

    detector = BiasDetector()

    # Create predictions with predictive parity violation
    predictions = [1, 1, 1, 1, 0, 0, 1, 1, 1, 0]
    ground_truth = [1, 1, 0, 0, 0, 0, 1, 0, 0, 0]
    # Group A: PPV = 2/4 = 0.5
    # Group B: PPV = 1/3 ≈ 0.33
    protected_attrs = {
        'group': ['A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
    }

    result = detector.analyze_model_predictions(
        model_name="predictive_parity_test",
        predictions=predictions,
        ground_truth=ground_truth,
        protected_attributes_data=protected_attrs
    )

    # Check predictive parity metric
    pp_metrics = [m for m in result.metrics if m.metric_type == 'predictive_parity']
    assert len(pp_metrics) > 0

    pp_metric = pp_metrics[0]
    print(f"  Predictive Parity Difference: {pp_metric.value:.4f}")
    print(f"  Details: {pp_metric.details}")
    print(f"  Passed: {pp_metric.passed}")

    print("✓ Model predictive parity tests passed")


def test_fair_model():
    """Test that a fair model passes all metrics"""
    print("\nTesting fair model...")

    detector = BiasDetector()

    # Create perfectly fair predictions
    predictions = [1, 1, 0, 0, 1, 1, 0, 0]
    ground_truth = [1, 1, 0, 0, 1, 1, 0, 0]
    protected_attrs = {
        'group': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
    }

    result = detector.analyze_model_predictions(
        model_name="fair_model_test",
        predictions=predictions,
        ground_truth=ground_truth,
        protected_attributes_data=protected_attrs
    )

    # All metrics should pass
    all_passed = all(m.passed for m in result.metrics)
    print(f"  Total Metrics: {len(result.metrics)}")
    print(f"  All Passed: {all_passed}")
    print(f"  Fairness Score: {result.overall_fairness_score:.2f}")
    print(f"  Risk Level: {result.risk_level}")

    assert result.overall_fairness_score == 1.0
    assert result.risk_level == 'low'

    print("✓ Fair model tests passed")


def test_bias_report_generator():
    """Test BiasReportGenerator with multiple analyses"""
    print("\nTesting BiasReportGenerator...")

    detector = BiasDetector()

    # Create multiple analyses
    analyses = []

    # Analysis 1: High bias
    data1 = {
        'gender': ['male'] * 8 + ['female'] * 2,
        'outcome': [1] * 5 + [0] * 5
    }
    result1 = detector.analyze_dataset(
        dataset_name="high_bias_dataset",
        data=data1,
        protected_attributes=['gender'],
        target_column='outcome'
    )
    analyses.append(result1)

    # Analysis 2: Low bias
    data2 = {
        'gender': ['male'] * 5 + ['female'] * 5,
        'outcome': [1, 1, 0, 0, 1] + [1, 1, 0, 0, 1]
    }
    result2 = detector.analyze_dataset(
        dataset_name="low_bias_dataset",
        data=data2,
        protected_attributes=['gender'],
        target_column='outcome'
    )
    analyses.append(result2)

    # Generate comprehensive report
    report = BiasReportGenerator.generate_report(analyses)

    # Verify report structure
    assert 'summary' in report
    assert 'analyses' in report
    assert 'aggregated_recommendations' in report
    assert 'compliance_notes' in report

    summary = report['summary']
    assert summary['total_analyses'] == 2
    assert 'overall_risk_level' in summary
    assert 'average_fairness_score' in summary

    print(f"  Total Analyses: {summary['total_analyses']}")
    print(f"  Overall Risk: {summary['overall_risk_level']}")
    print(f"  Avg Fairness: {summary['average_fairness_score']:.2f}")
    print(f"  Recommendations: {len(report['aggregated_recommendations'])}")
    print(f"  Compliance Notes: {len(report['compliance_notes'])}")

    print("✓ BiasReportGenerator tests passed")


def test_metadata_storage_integration():
    """Test integration with MetadataStorage"""
    print("\nTesting MetadataStorage integration...")

    # Create storage
    storage = MetadataStorage(
        system_name="bias_test_system",
        enable_auditing=True,
        enable_versioning=True
    )

    # Create and add bias analysis
    detector = BiasDetector()
    data = {
        'age': ['young'] * 5 + ['old'] * 5,
        'outcome': [1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    }

    result = detector.analyze_dataset(
        dataset_name="integration_test",
        data=data,
        protected_attributes=['age'],
        target_column='outcome'
    )

    # Add to storage
    storage.add_bias_analysis(result.to_dict())

    # Verify storage
    assert len(storage.bias_analyses) == 1
    assert storage.bias_analyses[0]['dataset_name'] == "integration_test"

    # Get metadata
    metadata = storage.get_all_metadata()
    assert 'bias_analyses' in metadata
    assert 'bias_summary' in metadata

    bias_summary = metadata['bias_summary']
    assert bias_summary['total_analyses'] == 1
    print(f"  Bias analyses stored: {len(storage.bias_analyses)}")
    print(f"  Bias summary generated: {bias_summary}")

    # Verify audit trail
    if storage.audit_trail:
        events = storage.audit_trail.get_all_events()
        bias_events = [e for e in events if 'bias' in e.event_type.lower()]
        assert len(bias_events) > 0
        print(f"  Audit events recorded: {len(bias_events)}")

    print("✓ MetadataStorage integration tests passed")


def test_bias_metric_serialization():
    """Test BiasMetric and BiasAnalysisResult serialization"""
    print("\nTesting serialization...")

    # Create a BiasMetric
    metric = BiasMetric(
        metric_name="test_metric",
        metric_type="demographic_parity",
        protected_attribute="gender",
        value=0.15,
        threshold=0.1,
        passed=False,
        groups_analyzed=['male', 'female'],
        timestamp="2024-01-01T00:00:00",
        details={'test': 'data'}
    )

    # Serialize to dict
    metric_dict = metric.to_dict()
    assert metric_dict['metric_name'] == "test_metric"
    assert metric_dict['value'] == 0.15
    assert metric_dict['passed'] is False

    # Create a BiasAnalysisResult
    result = BiasAnalysisResult(
        analysis_id="test_123",
        dataset_name="test_dataset",
        analysis_type="dataset",
        timestamp="2024-01-01T00:00:00",
        metrics=[metric],
        overall_fairness_score=0.5,
        risk_level="medium",
        recommendations=["Test recommendation"]
    )

    # Serialize to dict
    result_dict = result.to_dict()
    assert result_dict['analysis_id'] == "test_123"
    assert result_dict['overall_fairness_score'] == 0.5
    assert len(result_dict['metrics']) == 1

    # Verify it can be JSON serialized
    json_str = json.dumps(result_dict)
    assert json_str is not None

    print("  ✓ BiasMetric serialization")
    print("  ✓ BiasAnalysisResult serialization")
    print("  ✓ JSON serialization")
    print("✓ Serialization tests passed")


def test_empty_data_handling():
    """Test handling of empty or invalid data"""
    print("\nTesting empty data handling...")

    detector = BiasDetector()

    # Test with empty data
    result = detector.analyze_dataset(
        dataset_name="empty_test",
        data={},
        protected_attributes=['gender']
    )

    assert result.risk_level == 'unknown'
    assert result.overall_fairness_score == 0.0
    assert len(result.metrics) == 0
    print("  ✓ Empty data handled correctly")

    # Test with mismatched lengths
    result2 = detector.analyze_model_predictions(
        model_name="mismatch_test",
        predictions=[1, 0, 1],
        ground_truth=[1, 0],  # Different length
        protected_attributes_data={'group': ['A', 'B', 'C']}
    )

    assert result2.risk_level == 'unknown'
    print("  ✓ Mismatched lengths handled correctly")

    print("✓ Empty data handling tests passed")


def run_all_tests():
    """Run all bias detection tests"""
    print("="*80)
    print("Running Bias Detection Tests")
    print("="*80)

    tests = [
        test_bias_detector_initialization,
        test_dataset_representation_bias,
        test_dataset_outcome_bias,
        test_model_demographic_parity,
        test_model_equal_opportunity,
        test_model_predictive_parity,
        test_fair_model,
        test_bias_report_generator,
        test_metadata_storage_integration,
        test_bias_metric_serialization,
        test_empty_data_handling
    ]

    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*80)
    print(f"Test Results: {len(tests) - failed}/{len(tests)} passed")
    if failed == 0:
        print("✓ All tests passed!")
    else:
        print(f"✗ {failed} test(s) failed")
    print("="*80)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
