"""
Operational Metrics Module

Basic operational metrics tracking for AI systems including execution times
and error rates to support transparency requirements.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict
import statistics


class OperationalMetricsTracker:
    """Tracks operational metrics for AI system operations."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.operations: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []
        self.start_time = datetime.now()

    def record_operation(
        self,
        operation_type: str,
        model_name: str,
        provider: str,
        execution_time_ms: float,
        token_usage: Optional[Dict[str, int]] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record an operation with its metrics."""
        operation = {
            "timestamp": datetime.now().isoformat(),
            "operation_type": operation_type,
            "model_name": model_name,
            "provider": provider,
            "execution_time_ms": execution_time_ms,
            "success": success
        }

        if token_usage:
            operation["token_usage"] = token_usage

        if error_message:
            operation["error_message"] = error_message
            self.errors.append({
                "timestamp": operation["timestamp"],
                "operation_type": operation_type,
                "model_name": model_name,
                "error_message": error_message
            })

        if metadata:
            operation["metadata"] = metadata

        self.operations.append(operation)

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics of all operations."""
        if not self.operations:
            return {
                "total_operations": 0,
                "total_errors": 0,
                "error_rate": 0.0,
                "uptime_hours": 0.0
            }

        successful_ops = [op for op in self.operations if op.get("success", True)]
        execution_times = [op["execution_time_ms"] for op in self.operations if "execution_time_ms" in op]

        # Token usage statistics
        total_tokens = 0
        for op in self.operations:
            if "token_usage" in op:
                total_tokens += op["token_usage"].get("total_tokens", 0)

        # Operations by model
        ops_by_model = defaultdict(int)
        for op in self.operations:
            ops_by_model[op["model_name"]] += 1

        uptime_hours = (datetime.now() - self.start_time).total_seconds() / 3600

        return {
            "total_operations": len(self.operations),
            "successful_operations": len(successful_ops),
            "total_errors": len(self.errors),
            "error_rate": len(self.errors) / len(self.operations) if self.operations else 0.0,
            "uptime_hours": round(uptime_hours, 2),
            "total_tokens_used": total_tokens,
            "average_execution_time_ms": round(statistics.mean(execution_times), 2) if execution_times else 0,
            "median_execution_time_ms": round(statistics.median(execution_times), 2) if execution_times else 0,
            "operations_by_model": dict(ops_by_model),
            "start_time": self.start_time.isoformat(),
            "last_operation": self.operations[-1]["timestamp"] if self.operations else None
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "summary": self.get_summary_statistics(),
            "operations": self.operations,
            "errors": self.errors
        }


class MetricsAnalyzer:
    """Analyzes operational metrics for compliance reporting."""

    @staticmethod
    def analyze_from_dict(metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze metrics from a dictionary (used by CLI)."""
        issues = []
        summary = metrics.get("summary", metrics)

        error_rate = summary.get("error_rate", 0)
        if error_rate > 0.05:
            issues.append(f"High error rate: {error_rate:.2%}")

        avg_time = summary.get("average_execution_time_ms", 0)
        if avg_time > 5000:
            issues.append(f"Slow average response time: {avg_time:.0f}ms")

        return {
            "issues_detected": issues,
            "health_status": "healthy" if not issues else "needs_attention",
            "analyzed_at": datetime.now().isoformat()
        }
