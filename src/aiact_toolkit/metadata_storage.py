"""
Metadata Storage

Simple storage mechanism for captured compliance metadata.
Stores data in-memory and provides methods to serialize to JSON.
"""

from typing import Any, Dict, List
from datetime import datetime
import json


class MetadataStorage:
    """
    Storage class for compliance metadata captured from LangChain operations.
    """

    def __init__(self, system_name: str = "unnamed_system"):
        """Initialize metadata storage."""
        self.system_name = system_name
        self.models: List[Dict[str, Any]] = []
        self.components: List[Dict[str, Any]] = []
        self.data_sources: List[Dict[str, Any]] = []
        self.risk_assessment: Dict[str, Any] = {}
        self.operational_metrics: Dict[str, Any] = {}
        self.created_at = datetime.now().isoformat()

    def add_model(self, model_info: Dict[str, Any]):
        """Add model metadata."""
        # Avoid duplicates by checking if similar model already exists
        for existing in self.models:
            if (existing.get("model_name") == model_info.get("model_name") and
                existing.get("provider") == model_info.get("provider") and
                existing.get("parameters") == model_info.get("parameters")):
                return  # Already captured
        self.models.append(model_info)

    def add_component(self, component_info: Dict[str, Any]):
        """Add framework component metadata."""
        self.components.append(component_info)

    def add_data_source(self, data_source_info: Dict[str, Any]):
        """Add data source metadata."""
        # Avoid duplicates
        for existing in self.data_sources:
            if existing.get("data_source") == data_source_info.get("data_source"):
                return
        self.data_sources.append(data_source_info)

    def set_risk_assessment(self, risk_assessment: Dict[str, Any]):
        """
        Store risk assessment results.

        Args:
            risk_assessment: Risk assessment data from AIActRiskAssessor
        """
        self.risk_assessment = risk_assessment

    def set_operational_metrics(self, metrics: Dict[str, Any]):
        """
        Store operational metrics.

        Args:
            metrics: Operational metrics from OperationalMetricsTracker
        """
        self.operational_metrics = metrics

    def get_all_metadata(self) -> Dict[str, Any]:
        """
        Get all captured metadata in a structured format.

        Returns:
            Dictionary with all metadata suitable for template rendering
        """
        metadata = {
            "system_name": self.system_name,
            "created_at": self.created_at,
            "timestamp": datetime.now().isoformat(),
            "models": self._deduplicate_models(),
            "components": self._deduplicate_components(),
            "data_sources": self.data_sources,
            "summary": {
                "total_models": len(self._deduplicate_models()),
                "total_components": len(self._deduplicate_components()),
                "total_data_sources": len(self.data_sources),
            }
        }

        # Include risk assessment if available
        if self.risk_assessment:
            metadata["risk_assessment"] = self.risk_assessment

        # Include operational metrics if available
        if self.operational_metrics:
            metadata["operational_metrics"] = self.operational_metrics

        return metadata

    def _deduplicate_models(self) -> List[Dict[str, Any]]:
        """Remove duplicate model entries."""
        seen = set()
        unique_models = []
        for model in self.models:
            key = (model.get("model_name"), model.get("provider"),
                   json.dumps(model.get("parameters", {}), sort_keys=True))
            if key not in seen:
                seen.add(key)
                unique_models.append(model)
        return unique_models

    def _deduplicate_components(self) -> List[Dict[str, Any]]:
        """Remove duplicate component entries."""
        seen = set()
        unique_components = []
        for component in self.components:
            key = component.get("chain_type") or component.get("tool_name")
            if key and key not in seen:
                seen.add(key)
                unique_components.append(component)
        return unique_components

    def save_to_file(self, filepath: str):
        """
        Save metadata to a JSON file.

        Args:
            filepath: Path to save the JSON file
        """
        metadata = self.get_all_metadata()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def load_from_file(self, filepath: str):
        """
        Load metadata from a JSON file.

        Args:
            filepath: Path to the JSON file
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.system_name = data.get("system_name", "unnamed_system")
            self.models = data.get("models", [])
            self.components = data.get("components", [])
            self.data_sources = data.get("data_sources", [])
            self.risk_assessment = data.get("risk_assessment", {})
            self.operational_metrics = data.get("operational_metrics", {})
            self.created_at = data.get("created_at", datetime.now().isoformat())

    def clear(self):
        """Clear all stored metadata."""
        self.models.clear()
        self.components.clear()
        self.data_sources.clear()
        self.risk_assessment = {}
        self.operational_metrics = {}
