"""Tests for the Operational Metrics module."""

import pytest
from aiact_toolkit.operational_metrics import OperationalMetricsTracker, MetricsAnalyzer


class TestOperationalMetricsTracker:
    """Test OperationalMetricsTracker class."""

    def setup_method(self):
        self.tracker = OperationalMetricsTracker()

    def test_empty_summary(self):
        summary = self.tracker.get_summary_statistics()
        assert summary["total_operations"] == 0
        assert summary["total_errors"] == 0
        assert summary["error_rate"] == 0.0

    def test_record_successful_operation(self):
        self.tracker.record_operation(
            operation_type="llm_call",
            model_name="gpt-4",
            provider="OpenAI",
            execution_time_ms=150.0,
            success=True,
        )
        assert len(self.tracker.operations) == 1
        assert len(self.tracker.errors) == 0

    def test_record_failed_operation(self):
        self.tracker.record_operation(
            operation_type="llm_call",
            model_name="gpt-4",
            provider="OpenAI",
            execution_time_ms=500.0,
            success=False,
            error_message="Rate limit exceeded",
        )
        assert len(self.tracker.operations) == 1
        assert len(self.tracker.errors) == 1
        assert self.tracker.errors[0]["error_message"] == "Rate limit exceeded"

    def test_record_with_token_usage(self):
        self.tracker.record_operation(
            operation_type="llm_call",
            model_name="gpt-4",
            provider="OpenAI",
            execution_time_ms=200.0,
            token_usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        )
        assert self.tracker.operations[0]["token_usage"]["total_tokens"] == 150

    def test_summary_statistics(self):
        for i in range(8):
            self.tracker.record_operation("llm_call", "gpt-4", "OpenAI", 100.0 + i * 10)
        self.tracker.record_operation("llm_call", "gpt-4", "OpenAI", 500.0, success=False, error_message="err")
        self.tracker.record_operation("llm_call", "gpt-4", "OpenAI", 200.0, success=False, error_message="err2")

        summary = self.tracker.get_summary_statistics()
        assert summary["total_operations"] == 10
        assert summary["successful_operations"] == 8
        assert summary["total_errors"] == 2
        assert summary["error_rate"] == 0.2
        assert summary["average_execution_time_ms"] > 0
        assert summary["median_execution_time_ms"] > 0

    def test_operations_by_model(self):
        self.tracker.record_operation("llm_call", "gpt-4", "OpenAI", 100.0)
        self.tracker.record_operation("llm_call", "gpt-4", "OpenAI", 120.0)
        self.tracker.record_operation("llm_call", "claude-3", "Anthropic", 150.0)

        summary = self.tracker.get_summary_statistics()
        assert summary["operations_by_model"]["gpt-4"] == 2
        assert summary["operations_by_model"]["claude-3"] == 1

    def test_token_usage_aggregation(self):
        self.tracker.record_operation(
            "llm_call", "gpt-4", "OpenAI", 100.0,
            token_usage={"total_tokens": 100}
        )
        self.tracker.record_operation(
            "llm_call", "gpt-4", "OpenAI", 120.0,
            token_usage={"total_tokens": 200}
        )

        summary = self.tracker.get_summary_statistics()
        assert summary["total_tokens_used"] == 300

    def test_to_dict(self):
        self.tracker.record_operation("llm_call", "gpt-4", "OpenAI", 100.0)
        d = self.tracker.to_dict()
        assert "summary" in d
        assert "operations" in d
        assert "errors" in d


class TestMetricsAnalyzer:
    """Test MetricsAnalyzer class."""

    def test_healthy_metrics(self):
        metrics = {
            "error_rate": 0.01,
            "average_execution_time_ms": 200,
        }
        result = MetricsAnalyzer.analyze_from_dict(metrics)
        assert result["health_status"] == "healthy"
        assert len(result["issues_detected"]) == 0

    def test_high_error_rate_detected(self):
        metrics = {"error_rate": 0.10}
        result = MetricsAnalyzer.analyze_from_dict(metrics)
        assert result["health_status"] == "needs_attention"
        assert any("error rate" in issue.lower() for issue in result["issues_detected"])

    def test_slow_response_detected(self):
        metrics = {"average_execution_time_ms": 10000}
        result = MetricsAnalyzer.analyze_from_dict(metrics)
        assert any("slow" in issue.lower() for issue in result["issues_detected"])

    def test_nested_summary_format(self):
        metrics = {
            "summary": {
                "error_rate": 0.10,
                "average_execution_time_ms": 200,
            }
        }
        result = MetricsAnalyzer.analyze_from_dict(metrics)
        assert any("error rate" in issue.lower() for issue in result["issues_detected"])
