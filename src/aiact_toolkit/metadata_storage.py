"""
Metadata Storage

Simple storage mechanism for captured compliance metadata.
Stores data in-memory and provides methods to serialize to JSON.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import json
from .audit_trail import AuditTrail, AuditEventType
from .data_governance import DataGovernanceTracker


class MetadataStorage:
    """Storage class for compliance metadata."""

    def __init__(self, system_name: str = "unnamed_system", enable_auditing: bool = True):
        """Initialize metadata storage."""
        self.system_name = system_name
        self.models: List[Dict[str, Any]] = []
        self.components: List[Dict[str, Any]] = []
        self.data_sources: List[Dict[str, Any]] = []
        self.risk_assessment: Dict[str, Any] = {}
        self.operational_metrics: Dict[str, Any] = {}
        self.bias_analyses: List[Dict[str, Any]] = []
        self.created_at = datetime.now().isoformat()

        # Optional audit trail and data governance
        self.audit_trail: Optional[AuditTrail] = AuditTrail(system_name) if enable_auditing else None
        self.data_governance_tracker: Optional[DataGovernanceTracker] = DataGovernanceTracker(system_name)

        if self.audit_trail:
            self.audit_trail.record_event(
                event_type=AuditEventType.SYSTEM_CREATED,
                description=f"AI system '{system_name}' metadata storage initialized",
                actor="system"
            )

    def add_model(self, model_info: Dict[str, Any]):
        """Add model metadata."""
        # Avoid duplicates
        is_duplicate = any(
            existing.get("model_name") == model_info.get("model_name") and
            existing.get("provider") == model_info.get("provider") and
            existing.get("parameters") == model_info.get("parameters")
            for existing in self.models
        )

        if is_duplicate:
            return

        self.models.append(model_info)

        if self.audit_trail:
            self.audit_trail.record_event(
                event_type=AuditEventType.MODEL_ADDED,
                description=f"Added model: {model_info.get('model_name', 'unknown')}",
                actor="system",
                metadata={"model_info": model_info}
            )

    def add_component(self, component_info: Dict[str, Any]):
        """Add framework component metadata."""
        self.components.append(component_info)

    def add_data_source(self, data_source_info: Dict[str, Any]):
        """Add data source metadata."""
        # Avoid duplicates
        is_duplicate = any(
            existing.get("data_source") == data_source_info.get("data_source")
            for existing in self.data_sources
        )

        if is_duplicate:
            return

        self.data_sources.append(data_source_info)

        if self.audit_trail:
            self.audit_trail.record_event(
                event_type=AuditEventType.DATA_SOURCE_ADDED,
                description=f"Added data source: {data_source_info.get('data_source', 'unknown')}",
                actor="system",
                metadata={"data_source_info": data_source_info}
            )

    def set_risk_assessment(self, risk_assessment: Dict[str, Any]):
        """Store risk assessment results."""
        self.risk_assessment = risk_assessment

        if self.audit_trail:
            self.audit_trail.record_event(
                event_type=AuditEventType.RISK_ASSESSMENT_PERFORMED,
                description=f"Risk assessment performed: {risk_assessment.get('risk_level')}",
                actor="system",
                metadata=risk_assessment
            )

    def set_operational_metrics(self, metrics: Dict[str, Any]):
        """Store operational metrics."""
        self.operational_metrics = metrics

        if self.audit_trail:
            self.audit_trail.record_event(
                event_type=AuditEventType.METADATA_UPDATED,
                description="Operational metrics updated",
                actor="system"
            )

    def add_bias_analysis(self, bias_analysis: Dict[str, Any]):
        """Add bias analysis results."""
        self.bias_analyses.append(bias_analysis)

        if self.audit_trail:
            self.audit_trail.record_event(
                event_type=AuditEventType.BIAS_ANALYSIS_PERFORMED,
                description=f"Bias analysis performed: {bias_analysis.get('risk_level', 'unknown')} risk",
                actor="system",
                metadata={"risk_level": bias_analysis.get("risk_level")}
            )

    def get_all_metadata(self) -> Dict[str, Any]:
        """Get all captured metadata in a structured format."""
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

        if self.risk_assessment:
            metadata["risk_assessment"] = self.risk_assessment

        if self.operational_metrics:
            metadata["operational_metrics"] = self.operational_metrics

        if self.audit_trail:
            metadata["audit_summary"] = self.audit_trail.generate_summary()

        if self.data_governance_tracker:
            metadata["data_governance"] = self.data_governance_tracker.to_dict()
            metadata["data_quality_summary"] = self.data_governance_tracker.get_data_quality_summary()
            metadata["privacy_summary"] = self.data_governance_tracker.get_privacy_summary()

        if self.bias_analyses:
            metadata["bias_analyses"] = self.bias_analyses

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
        """Save metadata to a JSON file."""
        metadata = self.get_all_metadata()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def clear(self):
        """Clear all stored metadata."""
        self.models.clear()
        self.components.clear()
        self.data_sources.clear()
        self.risk_assessment = {}
        self.operational_metrics = {}
        self.bias_analyses.clear()
