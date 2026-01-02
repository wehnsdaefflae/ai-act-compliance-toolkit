"""
Bias and Fairness Detection Module

Implements bias detection and fairness metrics for EU AI Act compliance,
particularly Article 10 (Data Governance) and Article 15 (Accuracy and Robustness).

Supports multiple fairness metrics and bias detection strategies across different
protected attributes (gender, age, ethnicity, etc.).
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json


@dataclass
class BiasMetric:
    """Represents a single bias metric measurement"""
    metric_name: str
    metric_type: str  # 'demographic_parity', 'equal_opportunity', 'disparate_impact', etc.
    protected_attribute: str  # e.g., 'gender', 'age', 'ethnicity'
    value: float
    threshold: float
    passed: bool
    groups_analyzed: List[str]
    timestamp: str
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BiasAnalysisResult:
    """Complete bias analysis result for a dataset or model"""
    analysis_id: str
    dataset_name: str
    analysis_type: str  # 'dataset', 'model_predictions', 'combined'
    timestamp: str
    metrics: List[BiasMetric]
    overall_fairness_score: float  # 0-1, higher is better
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    recommendations: List[str]
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['metrics'] = [m.to_dict() if isinstance(m, BiasMetric) else m for m in self.metrics]
        return result


class BiasDetector:
    """
    Bias and fairness detector for AI systems.

    Implements various fairness metrics aligned with EU AI Act requirements:
    - Demographic Parity (Statistical Parity)
    - Equal Opportunity (True Positive Rate Equality)
    - Equalized Odds
    - Disparate Impact
    - Predictive Parity
    """

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize bias detector with configurable thresholds.

        Args:
            thresholds: Dictionary mapping metric names to threshold values
        """
        self.thresholds = thresholds or {
            'demographic_parity': 0.1,  # Max difference of 10%
            'equal_opportunity': 0.1,
            'disparate_impact': 0.8,  # Minimum ratio of 0.8 (80% rule)
            'predictive_parity': 0.1,
        }

    def analyze_dataset(
        self,
        dataset_name: str,
        data: Dict[str, List[Any]],
        protected_attributes: List[str],
        target_column: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BiasAnalysisResult:
        """
        Analyze a dataset for bias in representation and distribution.

        Args:
            dataset_name: Name of the dataset
            data: Dictionary mapping column names to lists of values
            protected_attributes: List of protected attribute column names
            target_column: Target variable column name (optional)
            metadata: Additional metadata about the analysis

        Returns:
            BiasAnalysisResult with detected biases
        """
        metrics = []
        timestamp = datetime.utcnow().isoformat()

        # Check data availability
        if not data:
            return BiasAnalysisResult(
                analysis_id=f"bias_analysis_{timestamp}",
                dataset_name=dataset_name,
                analysis_type='dataset',
                timestamp=timestamp,
                metrics=[],
                overall_fairness_score=0.0,
                risk_level='unknown',
                recommendations=['No data available for bias analysis'],
                metadata=metadata
            )

        # Analyze representation for each protected attribute
        for attr in protected_attributes:
            if attr not in data:
                continue

            values = data[attr]
            if not values:
                continue

            # Calculate representation bias
            representation_metric = self._calculate_representation_bias(
                attr, values, timestamp
            )
            metrics.append(representation_metric)

            # If target column exists, analyze outcome bias
            if target_column and target_column in data:
                outcome_metric = self._calculate_outcome_bias(
                    attr, data[attr], data[target_column], timestamp
                )
                if outcome_metric:
                    metrics.append(outcome_metric)

        # Calculate overall fairness score and risk level
        overall_score, risk_level = self._calculate_overall_fairness(metrics)

        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, risk_level)

        return BiasAnalysisResult(
            analysis_id=f"bias_analysis_{timestamp.replace(':', '').replace('.', '')}",
            dataset_name=dataset_name,
            analysis_type='dataset',
            timestamp=timestamp,
            metrics=metrics,
            overall_fairness_score=overall_score,
            risk_level=risk_level,
            recommendations=recommendations,
            metadata=metadata
        )

    def analyze_model_predictions(
        self,
        model_name: str,
        predictions: List[Any],
        ground_truth: List[Any],
        protected_attributes_data: Dict[str, List[Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> BiasAnalysisResult:
        """
        Analyze model predictions for bias across protected groups.

        Args:
            model_name: Name of the model
            predictions: List of model predictions
            ground_truth: List of actual labels
            protected_attributes_data: Dict mapping protected attributes to their values
            metadata: Additional metadata

        Returns:
            BiasAnalysisResult with detected biases in predictions
        """
        metrics = []
        timestamp = datetime.utcnow().isoformat()

        # Validate inputs
        if not predictions or not ground_truth:
            return BiasAnalysisResult(
                analysis_id=f"bias_analysis_{timestamp}",
                dataset_name=model_name,
                analysis_type='model_predictions',
                timestamp=timestamp,
                metrics=[],
                overall_fairness_score=0.0,
                risk_level='unknown',
                recommendations=['Insufficient prediction data for bias analysis'],
                metadata=metadata
            )

        if len(predictions) != len(ground_truth):
            return BiasAnalysisResult(
                analysis_id=f"bias_analysis_{timestamp}",
                dataset_name=model_name,
                analysis_type='model_predictions',
                timestamp=timestamp,
                metrics=[],
                overall_fairness_score=0.0,
                risk_level='unknown',
                recommendations=['Prediction and ground truth length mismatch'],
                metadata=metadata
            )

        # Analyze fairness for each protected attribute
        for attr_name, attr_values in protected_attributes_data.items():
            if len(attr_values) != len(predictions):
                continue

            # Equal Opportunity (TPR equality)
            eq_opp_metric = self._calculate_equal_opportunity(
                attr_name, attr_values, predictions, ground_truth, timestamp
            )
            if eq_opp_metric:
                metrics.append(eq_opp_metric)

            # Demographic Parity
            dem_parity_metric = self._calculate_demographic_parity(
                attr_name, attr_values, predictions, timestamp
            )
            if dem_parity_metric:
                metrics.append(dem_parity_metric)

            # Predictive Parity
            pred_parity_metric = self._calculate_predictive_parity(
                attr_name, attr_values, predictions, ground_truth, timestamp
            )
            if pred_parity_metric:
                metrics.append(pred_parity_metric)

        # Calculate overall fairness
        overall_score, risk_level = self._calculate_overall_fairness(metrics)
        recommendations = self._generate_recommendations(metrics, risk_level)

        return BiasAnalysisResult(
            analysis_id=f"bias_analysis_{timestamp.replace(':', '').replace('.', '')}",
            dataset_name=model_name,
            analysis_type='model_predictions',
            timestamp=timestamp,
            metrics=metrics,
            overall_fairness_score=overall_score,
            risk_level=risk_level,
            recommendations=recommendations,
            metadata=metadata
        )

    def _calculate_representation_bias(
        self,
        attr_name: str,
        values: List[Any],
        timestamp: str
    ) -> BiasMetric:
        """Calculate representation bias (class imbalance) for a protected attribute"""
        # Count occurrences
        from collections import Counter
        counts = Counter(values)
        total = len(values)

        if total == 0:
            return BiasMetric(
                metric_name='representation_balance',
                metric_type='representation',
                protected_attribute=attr_name,
                value=0.0,
                threshold=0.2,
                passed=False,
                groups_analyzed=[],
                timestamp=timestamp
            )

        # Calculate proportions
        proportions = {k: v/total for k, v in counts.items()}
        groups = list(counts.keys())

        # Calculate max deviation from uniform distribution
        expected_proportion = 1.0 / len(groups) if groups else 0
        max_deviation = max(abs(p - expected_proportion) for p in proportions.values()) if proportions else 0

        # Threshold: max 20% deviation from uniform
        threshold = 0.2
        passed = max_deviation <= threshold

        return BiasMetric(
            metric_name='representation_balance',
            metric_type='representation',
            protected_attribute=attr_name,
            value=max_deviation,
            threshold=threshold,
            passed=passed,
            groups_analyzed=groups,
            timestamp=timestamp,
            details={
                'proportions': {str(k): v for k, v in proportions.items()},
                'counts': {str(k): v for k, v in counts.items()},
                'total_samples': total
            }
        )

    def _calculate_outcome_bias(
        self,
        attr_name: str,
        attr_values: List[Any],
        target_values: List[Any],
        timestamp: str
    ) -> Optional[BiasMetric]:
        """Calculate bias in outcome distribution across groups"""
        if len(attr_values) != len(target_values):
            return None

        # Group by protected attribute
        from collections import defaultdict
        groups = defaultdict(list)
        for attr_val, target_val in zip(attr_values, target_values):
            groups[attr_val].append(target_val)

        if len(groups) < 2:
            return None

        # Calculate positive outcome rate for each group
        positive_rates = {}
        for group, outcomes in groups.items():
            # Assume positive outcome is 1 or True
            positive_count = sum(1 for o in outcomes if o in [1, True, '1', 'true', 'True'])
            positive_rates[group] = positive_count / len(outcomes) if outcomes else 0

        # Calculate max difference in positive rates
        rates = list(positive_rates.values())
        max_diff = max(rates) - min(rates) if rates else 0

        threshold = self.thresholds.get('demographic_parity', 0.1)
        passed = max_diff <= threshold

        return BiasMetric(
            metric_name='outcome_balance',
            metric_type='outcome_distribution',
            protected_attribute=attr_name,
            value=max_diff,
            threshold=threshold,
            passed=passed,
            groups_analyzed=list(groups.keys()),
            timestamp=timestamp,
            details={
                'positive_rates': {str(k): v for k, v in positive_rates.items()}
            }
        )

    def _calculate_equal_opportunity(
        self,
        attr_name: str,
        attr_values: List[Any],
        predictions: List[Any],
        ground_truth: List[Any],
        timestamp: str
    ) -> Optional[BiasMetric]:
        """Calculate Equal Opportunity (TPR equality across groups)"""
        from collections import defaultdict

        # Group by protected attribute, only for positive ground truth
        groups_tpr = defaultdict(lambda: {'tp': 0, 'fn': 0})

        for attr_val, pred, truth in zip(attr_values, predictions, ground_truth):
            if truth in [1, True, '1', 'true', 'True']:  # Positive cases only
                if pred in [1, True, '1', 'true', 'True']:
                    groups_tpr[attr_val]['tp'] += 1
                else:
                    groups_tpr[attr_val]['fn'] += 1

        if len(groups_tpr) < 2:
            return None

        # Calculate TPR for each group
        tpr_by_group = {}
        for group, counts in groups_tpr.items():
            total_pos = counts['tp'] + counts['fn']
            tpr_by_group[group] = counts['tp'] / total_pos if total_pos > 0 else 0

        # Calculate max difference
        tprs = list(tpr_by_group.values())
        max_diff = max(tprs) - min(tprs) if tprs else 0

        threshold = self.thresholds.get('equal_opportunity', 0.1)
        passed = max_diff <= threshold

        return BiasMetric(
            metric_name='equal_opportunity',
            metric_type='equal_opportunity',
            protected_attribute=attr_name,
            value=max_diff,
            threshold=threshold,
            passed=passed,
            groups_analyzed=list(groups_tpr.keys()),
            timestamp=timestamp,
            details={
                'tpr_by_group': {str(k): v for k, v in tpr_by_group.items()}
            }
        )

    def _calculate_demographic_parity(
        self,
        attr_name: str,
        attr_values: List[Any],
        predictions: List[Any],
        timestamp: str
    ) -> Optional[BiasMetric]:
        """Calculate Demographic Parity (equal positive prediction rates)"""
        from collections import defaultdict

        groups = defaultdict(lambda: {'total': 0, 'positive': 0})

        for attr_val, pred in zip(attr_values, predictions):
            groups[attr_val]['total'] += 1
            if pred in [1, True, '1', 'true', 'True']:
                groups[attr_val]['positive'] += 1

        if len(groups) < 2:
            return None

        # Calculate positive rate for each group
        pos_rates = {}
        for group, counts in groups.items():
            pos_rates[group] = counts['positive'] / counts['total'] if counts['total'] > 0 else 0

        rates = list(pos_rates.values())
        max_diff = max(rates) - min(rates) if rates else 0

        threshold = self.thresholds.get('demographic_parity', 0.1)
        passed = max_diff <= threshold

        return BiasMetric(
            metric_name='demographic_parity',
            metric_type='demographic_parity',
            protected_attribute=attr_name,
            value=max_diff,
            threshold=threshold,
            passed=passed,
            groups_analyzed=list(groups.keys()),
            timestamp=timestamp,
            details={
                'positive_rates': {str(k): v for k, v in pos_rates.items()}
            }
        )

    def _calculate_predictive_parity(
        self,
        attr_name: str,
        attr_values: List[Any],
        predictions: List[Any],
        ground_truth: List[Any],
        timestamp: str
    ) -> Optional[BiasMetric]:
        """Calculate Predictive Parity (PPV equality across groups)"""
        from collections import defaultdict

        groups_ppv = defaultdict(lambda: {'tp': 0, 'fp': 0})

        for attr_val, pred, truth in zip(attr_values, predictions, ground_truth):
            if pred in [1, True, '1', 'true', 'True']:  # Positive predictions only
                if truth in [1, True, '1', 'true', 'True']:
                    groups_ppv[attr_val]['tp'] += 1
                else:
                    groups_ppv[attr_val]['fp'] += 1

        if len(groups_ppv) < 2:
            return None

        # Calculate PPV (precision) for each group
        ppv_by_group = {}
        for group, counts in groups_ppv.items():
            total_pos_pred = counts['tp'] + counts['fp']
            ppv_by_group[group] = counts['tp'] / total_pos_pred if total_pos_pred > 0 else 0

        ppvs = list(ppv_by_group.values())
        max_diff = max(ppvs) - min(ppvs) if ppvs else 0

        threshold = self.thresholds.get('predictive_parity', 0.1)
        passed = max_diff <= threshold

        return BiasMetric(
            metric_name='predictive_parity',
            metric_type='predictive_parity',
            protected_attribute=attr_name,
            value=max_diff,
            threshold=threshold,
            passed=passed,
            groups_analyzed=list(groups_ppv.keys()),
            timestamp=timestamp,
            details={
                'ppv_by_group': {str(k): v for k, v in ppv_by_group.items()}
            }
        )

    def _calculate_overall_fairness(
        self,
        metrics: List[BiasMetric]
    ) -> Tuple[float, str]:
        """
        Calculate overall fairness score and risk level.

        Returns:
            Tuple of (fairness_score, risk_level)
            - fairness_score: 0-1, higher is better
            - risk_level: 'low', 'medium', 'high', 'critical'
        """
        if not metrics:
            return 0.0, 'unknown'

        # Calculate percentage of passed metrics
        passed_count = sum(1 for m in metrics if m.passed)
        pass_rate = passed_count / len(metrics)

        # Fairness score is the pass rate
        fairness_score = pass_rate

        # Determine risk level based on pass rate
        if pass_rate >= 0.9:
            risk_level = 'low'
        elif pass_rate >= 0.7:
            risk_level = 'medium'
        elif pass_rate >= 0.5:
            risk_level = 'high'
        else:
            risk_level = 'critical'

        return fairness_score, risk_level

    def _generate_recommendations(
        self,
        metrics: List[BiasMetric],
        risk_level: str
    ) -> List[str]:
        """Generate recommendations based on detected biases"""
        recommendations = []

        # Group metrics by type
        failed_metrics = [m for m in metrics if not m.passed]

        if not failed_metrics:
            recommendations.append("System shows good fairness across all analyzed metrics")
            return recommendations

        # Analyze failed metrics
        representation_issues = [m for m in failed_metrics if m.metric_type == 'representation']
        outcome_issues = [m for m in failed_metrics if m.metric_type == 'outcome_distribution']
        demographic_parity_issues = [m for m in failed_metrics if m.metric_type == 'demographic_parity']
        equal_opp_issues = [m for m in failed_metrics if m.metric_type == 'equal_opportunity']
        pred_parity_issues = [m for m in failed_metrics if m.metric_type == 'predictive_parity']

        if representation_issues:
            attrs = [m.protected_attribute for m in representation_issues]
            recommendations.append(
                f"Class imbalance detected for: {', '.join(attrs)}. "
                f"Consider data augmentation, resampling, or collecting more diverse data."
            )

        if outcome_issues:
            attrs = [m.protected_attribute for m in outcome_issues]
            recommendations.append(
                f"Outcome disparity detected for: {', '.join(attrs)}. "
                f"Review data collection process for potential bias sources."
            )

        if demographic_parity_issues:
            attrs = [m.protected_attribute for m in demographic_parity_issues]
            recommendations.append(
                f"Demographic parity violation for: {', '.join(attrs)}. "
                f"Model predictions show different positive rates across groups. "
                f"Consider fairness constraints during training or post-processing."
            )

        if equal_opp_issues:
            attrs = [m.protected_attribute for m in equal_opp_issues]
            recommendations.append(
                f"Equal opportunity violation for: {', '.join(attrs)}. "
                f"Model has different true positive rates across groups. "
                f"Consider equalizing opportunity through threshold adjustment or retraining."
            )

        if pred_parity_issues:
            attrs = [m.protected_attribute for m in pred_parity_issues]
            recommendations.append(
                f"Predictive parity violation for: {', '.join(attrs)}. "
                f"Model predictions have different precision across groups. "
                f"Review feature engineering and model calibration."
            )

        # Add risk-level specific recommendations
        if risk_level in ['high', 'critical']:
            recommendations.append(
                "CRITICAL: Significant fairness issues detected. "
                "System may not comply with EU AI Act Article 10 requirements. "
                "Recommend thorough bias mitigation before deployment."
            )
            recommendations.append(
                "Consider consulting with bias/fairness experts and affected communities."
            )

        return recommendations


class BiasReportGenerator:
    """Generate bias and fairness reports for compliance documentation"""

    @staticmethod
    def generate_report(analysis_results: List[BiasAnalysisResult]) -> Dict[str, Any]:
        """
        Generate a comprehensive bias report from multiple analyses.

        Args:
            analysis_results: List of BiasAnalysisResult objects

        Returns:
            Dictionary containing the complete bias report
        """
        if not analysis_results:
            return {
                'summary': 'No bias analyses available',
                'analyses': [],
                'overall_risk': 'unknown'
            }

        # Determine overall risk level
        risk_levels = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3, 'unknown': -1}
        max_risk_level = max(
            risk_levels.get(r.risk_level, -1) for r in analysis_results
        )
        overall_risk = next(
            (k for k, v in risk_levels.items() if v == max_risk_level),
            'unknown'
        )

        # Aggregate all recommendations
        all_recommendations = []
        for result in analysis_results:
            all_recommendations.extend(result.recommendations)

        # Remove duplicates while preserving order
        unique_recommendations = list(dict.fromkeys(all_recommendations))

        # Calculate aggregate fairness score
        avg_fairness = sum(r.overall_fairness_score for r in analysis_results) / len(analysis_results)

        return {
            'summary': {
                'total_analyses': len(analysis_results),
                'overall_risk_level': overall_risk,
                'average_fairness_score': round(avg_fairness, 3),
                'timestamp': datetime.utcnow().isoformat()
            },
            'analyses': [r.to_dict() for r in analysis_results],
            'aggregated_recommendations': unique_recommendations,
            'compliance_notes': BiasReportGenerator._generate_compliance_notes(overall_risk)
        }

    @staticmethod
    def _generate_compliance_notes(risk_level: str) -> List[str]:
        """Generate EU AI Act compliance notes based on risk level"""
        notes = [
            "EU AI Act Article 10 requires appropriate data governance and management practices",
            "Article 15 requires accuracy, robustness and cybersecurity appropriate to the risk level"
        ]

        if risk_level in ['high', 'critical']:
            notes.extend([
                "HIGH-RISK SYSTEM: Enhanced bias monitoring and mitigation required",
                "Article 10(3): Training, validation and testing data shall be subject to data governance practices",
                "Article 10(2)(f): Appropriate measures to detect, prevent and mitigate possible biases",
                "Consider implementing continuous bias monitoring in production"
            ])
        elif risk_level == 'medium':
            notes.append("Regular bias monitoring recommended to ensure continued compliance")

        return notes
