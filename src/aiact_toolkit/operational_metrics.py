"""
Operational Metrics Module

Tracks and analyzes operational metrics for AI systems including token usage,
costs, execution times, and error rates. Supports transparency and accountability
requirements under EU AI Act.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict
import statistics


class OperationalMetricsTracker:
    """
    Tracks operational metrics for AI system operations.

    Captures:
    - LLM invocations and token usage
    - Execution times and performance metrics
    - Error rates and failure patterns
    - Cost estimations
    - API call patterns
    """

    # Cost estimates per 1K tokens (approximate, as of 2024)
    COST_ESTIMATES = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
        "llama-2": {"input": 0.0, "output": 0.0},  # Open source
        "mistral": {"input": 0.0, "output": 0.0},  # Open source
    }

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
        """
        Record an operation with its metrics.

        Args:
            operation_type: Type of operation (llm_call, chain_execution, etc.)
            model_name: Name of the model used
            provider: Provider of the model
            execution_time_ms: Execution time in milliseconds
            token_usage: Token usage statistics (input_tokens, output_tokens, total_tokens)
            success: Whether the operation succeeded
            error_message: Error message if operation failed
            metadata: Additional metadata about the operation
        """
        operation = {
            "timestamp": datetime.now().isoformat(),
            "operation_type": operation_type,
            "model_name": model_name,
            "provider": provider,
            "execution_time_ms": execution_time_ms,
            "success": success,
        }

        if token_usage:
            operation["token_usage"] = token_usage
            # Estimate cost
            cost = self._estimate_cost(model_name, token_usage)
            if cost > 0:
                operation["estimated_cost_usd"] = cost

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

    def _estimate_cost(self, model_name: str, token_usage: Dict[str, int]) -> float:
        """
        Estimate cost based on token usage and model pricing.

        Args:
            model_name: Name of the model
            token_usage: Token usage statistics

        Returns:
            Estimated cost in USD
        """
        model_name_lower = model_name.lower()

        # Find matching cost estimate
        cost_config = None
        for model_key, costs in self.COST_ESTIMATES.items():
            if model_key in model_name_lower:
                cost_config = costs
                break

        if not cost_config:
            return 0.0

        input_tokens = token_usage.get("input_tokens", 0) or token_usage.get("prompt_tokens", 0)
        output_tokens = token_usage.get("output_tokens", 0) or token_usage.get("completion_tokens", 0)

        input_cost = (input_tokens / 1000) * cost_config["input"]
        output_cost = (output_tokens / 1000) * cost_config["output"]

        return round(input_cost + output_cost, 6)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of operational metrics.

        Returns:
            Dictionary with aggregated metrics and statistics
        """
        if not self.operations:
            return {
                "total_operations": 0,
                "message": "No operations recorded"
            }

        total_operations = len(self.operations)
        successful_operations = sum(1 for op in self.operations if op["success"])
        failed_operations = total_operations - successful_operations

        # Calculate execution time statistics
        execution_times = [op["execution_time_ms"] for op in self.operations]
        avg_execution_time = statistics.mean(execution_times) if execution_times else 0
        min_execution_time = min(execution_times) if execution_times else 0
        max_execution_time = max(execution_times) if execution_times else 0

        # Calculate token usage statistics
        total_input_tokens = 0
        total_output_tokens = 0
        total_tokens = 0

        for op in self.operations:
            if "token_usage" in op:
                usage = op["token_usage"]
                total_input_tokens += usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
                total_output_tokens += usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
                total_tokens += usage.get("total_tokens", 0) or (total_input_tokens + total_output_tokens)

        # Calculate total estimated cost
        total_cost = sum(
            op.get("estimated_cost_usd", 0) for op in self.operations
        )

        # Group operations by model
        operations_by_model = defaultdict(int)
        cost_by_model = defaultdict(float)
        tokens_by_model = defaultdict(int)

        for op in self.operations:
            model = op["model_name"]
            operations_by_model[model] += 1
            cost_by_model[model] += op.get("estimated_cost_usd", 0)
            if "token_usage" in op:
                tokens_by_model[model] += op["token_usage"].get("total_tokens", 0)

        # Group operations by type
        operations_by_type = defaultdict(int)
        for op in self.operations:
            operations_by_type[op["operation_type"]] += 1

        # Calculate error rate
        error_rate = (failed_operations / total_operations * 100) if total_operations > 0 else 0

        # Calculate uptime
        uptime_seconds = (datetime.now() - self.start_time).total_seconds()

        return {
            "tracking_period": {
                "start": self.start_time.isoformat(),
                "end": datetime.now().isoformat(),
                "duration_seconds": round(uptime_seconds, 2)
            },
            "operations": {
                "total": total_operations,
                "successful": successful_operations,
                "failed": failed_operations,
                "error_rate_percent": round(error_rate, 2)
            },
            "performance": {
                "avg_execution_time_ms": round(avg_execution_time, 2),
                "min_execution_time_ms": round(min_execution_time, 2),
                "max_execution_time_ms": round(max_execution_time, 2)
            },
            "token_usage": {
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_tokens
            },
            "costs": {
                "total_estimated_usd": round(total_cost, 6),
                "by_model": {model: round(cost, 6) for model, cost in cost_by_model.items()}
            },
            "operations_by_model": dict(operations_by_model),
            "operations_by_type": dict(operations_by_type),
            "tokens_by_model": dict(tokens_by_model),
            "errors": self.errors
        }

    def get_all_operations(self) -> List[Dict[str, Any]]:
        """Get all recorded operations."""
        return self.operations

    def get_errors(self) -> List[Dict[str, Any]]:
        """Get all recorded errors."""
        return self.errors

    def clear(self):
        """Clear all recorded metrics."""
        self.operations.clear()
        self.errors.clear()
        self.start_time = datetime.now()

    def export_metrics(self) -> Dict[str, Any]:
        """
        Export metrics in a format suitable for storage and analysis.

        Returns:
            Dictionary with all metrics and operations
        """
        return {
            "summary": self.get_summary(),
            "operations": self.operations,
            "errors": self.errors
        }


class MetricsAnalyzer:
    """
    Analyzes operational metrics to identify patterns, anomalies, and optimization opportunities.
    """

    @staticmethod
    def analyze_cost_trends(operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze cost trends and identify high-cost operations.

        Args:
            operations: List of operation records

        Returns:
            Cost analysis report
        """
        if not operations:
            return {"message": "No operations to analyze"}

        ops_with_cost = [op for op in operations if "estimated_cost_usd" in op]

        if not ops_with_cost:
            return {"message": "No cost data available"}

        total_cost = sum(op["estimated_cost_usd"] for op in ops_with_cost)
        avg_cost = statistics.mean([op["estimated_cost_usd"] for op in ops_with_cost])

        # Find most expensive operations
        sorted_ops = sorted(ops_with_cost, key=lambda x: x["estimated_cost_usd"], reverse=True)
        top_expensive = sorted_ops[:5]

        return {
            "total_cost_usd": round(total_cost, 6),
            "average_cost_per_operation_usd": round(avg_cost, 6),
            "most_expensive_operations": [
                {
                    "model": op["model_name"],
                    "cost_usd": op["estimated_cost_usd"],
                    "timestamp": op["timestamp"]
                }
                for op in top_expensive
            ]
        }

    @staticmethod
    def analyze_performance(operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze performance metrics and identify bottlenecks.

        Args:
            operations: List of operation records

        Returns:
            Performance analysis report
        """
        if not operations:
            return {"message": "No operations to analyze"}

        execution_times = [op["execution_time_ms"] for op in operations]

        # Calculate percentiles
        sorted_times = sorted(execution_times)
        p50 = sorted_times[len(sorted_times) // 2] if sorted_times else 0
        p95_idx = int(len(sorted_times) * 0.95)
        p95 = sorted_times[p95_idx] if sorted_times and p95_idx < len(sorted_times) else 0
        p99_idx = int(len(sorted_times) * 0.99)
        p99 = sorted_times[p99_idx] if sorted_times and p99_idx < len(sorted_times) else 0

        # Find slowest operations
        sorted_ops = sorted(operations, key=lambda x: x["execution_time_ms"], reverse=True)
        slowest_ops = sorted_ops[:5]

        return {
            "percentiles": {
                "p50_ms": round(p50, 2),
                "p95_ms": round(p95, 2),
                "p99_ms": round(p99, 2)
            },
            "slowest_operations": [
                {
                    "model": op["model_name"],
                    "execution_time_ms": op["execution_time_ms"],
                    "timestamp": op["timestamp"]
                }
                for op in slowest_ops
            ]
        }

    @staticmethod
    def identify_issues(summary: Dict[str, Any]) -> List[str]:
        """
        Identify potential issues from metrics summary.

        Args:
            summary: Metrics summary from get_summary()

        Returns:
            List of identified issues and recommendations
        """
        issues = []

        # Check error rate
        error_rate = summary.get("operations", {}).get("error_rate_percent", 0)
        if error_rate > 5:
            issues.append(f"High error rate detected: {error_rate}% - investigate failure patterns")
        elif error_rate > 1:
            issues.append(f"Elevated error rate: {error_rate}% - monitor for trends")

        # Check performance
        avg_time = summary.get("performance", {}).get("avg_execution_time_ms", 0)
        if avg_time > 10000:  # 10 seconds
            issues.append(f"High average execution time: {avg_time}ms - consider optimization")

        # Check costs
        total_cost = summary.get("costs", {}).get("total_estimated_usd", 0)
        if total_cost > 10:
            issues.append(f"High operational costs detected: ${total_cost} - review token usage and model selection")

        # Check token usage patterns
        token_usage = summary.get("token_usage", {})
        total_tokens = token_usage.get("total_tokens", 0)
        if total_tokens > 1000000:  # 1M tokens
            issues.append(f"High token consumption: {total_tokens} tokens - optimize prompts and context length")

        if not issues:
            issues.append("No significant issues detected - system operating normally")

        return issues
