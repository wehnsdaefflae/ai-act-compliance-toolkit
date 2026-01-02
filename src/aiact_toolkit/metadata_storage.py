"""
Metadata Storage

Simple storage mechanism for captured compliance metadata.
Stores data in-memory and provides methods to serialize to JSON.
Integrates with audit trail and version control for compliance tracking.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import json
from .audit_trail import AuditTrail, AuditEventType
from .version_control import VersionControl
from .data_governance import DataGovernanceTracker


class MetadataStorage:
    """
    Storage class for compliance metadata captured from LangChain operations.
    Includes integrated audit trail and version control for EU AI Act compliance.
    """

    def __init__(self, system_name: str = "unnamed_system", enable_auditing: bool = True, enable_versioning: bool = True, enable_data_governance: bool = True):
        """
        Initialize metadata storage.

        Args:
            system_name: Name of the AI system
            enable_auditing: Enable audit trail tracking
            enable_versioning: Enable version control
            enable_data_governance: Enable data governance and lineage tracking
        """
        self.system_name = system_name
        self.models: List[Dict[str, Any]] = []
        self.components: List[Dict[str, Any]] = []
        self.data_sources: List[Dict[str, Any]] = []
        self.risk_assessment: Dict[str, Any] = {}
        self.operational_metrics: Dict[str, Any] = {}
        self.bias_analyses: List[Dict[str, Any]] = []
        self.created_at = datetime.now().isoformat()

        # Audit trail and version control
        self.enable_auditing = enable_auditing
        self.enable_versioning = enable_versioning
        self.enable_data_governance = enable_data_governance
        self.audit_trail: Optional[AuditTrail] = AuditTrail(system_name) if enable_auditing else None
        self.version_control: Optional[VersionControl] = VersionControl(system_name) if enable_versioning else None
        self.data_governance_tracker: Optional[DataGovernanceTracker] = DataGovernanceTracker(system_name) if enable_data_governance else None

        # Record initial creation event
        if self.audit_trail:
            self.audit_trail.record_event(
                event_type=AuditEventType.SYSTEM_CREATED,
                description=f"AI system '{system_name}' metadata storage initialized",
                actor="system"
            )

        # Create initial version
        if self.version_control:
            self.version_control.commit(
                metadata=self.get_all_metadata(),
                description="Initial system creation",
                author="system"
            )

    def add_model(self, model_info: Dict[str, Any]):
        """Add model metadata."""
        # Avoid duplicates by checking if similar model already exists
        is_duplicate = False
        for existing in self.models:
            if (existing.get("model_name") == model_info.get("model_name") and
                existing.get("provider") == model_info.get("provider") and
                existing.get("parameters") == model_info.get("parameters")):
                is_duplicate = True
                break

        if is_duplicate:
            return  # Already captured

        self.models.append(model_info)

        # Record audit event
        if self.audit_trail:
            self.audit_trail.record_event(
                event_type=AuditEventType.MODEL_ADDED,
                description=f"Added model: {model_info.get('model_name', 'unknown')}",
                actor="system",
                metadata={"model_info": model_info}
            )

        # Create new version
        if self.version_control:
            self.version_control.commit(
                metadata=self.get_all_metadata(),
                description=f"Added model: {model_info.get('model_name', 'unknown')}",
                author="system"
            )

    def add_component(self, component_info: Dict[str, Any]):
        """Add framework component metadata."""
        self.components.append(component_info)

    def add_data_source(self, data_source_info: Dict[str, Any]):
        """Add data source metadata."""
        # Avoid duplicates
        is_duplicate = False
        for existing in self.data_sources:
            if existing.get("data_source") == data_source_info.get("data_source"):
                is_duplicate = True
                break

        if is_duplicate:
            return

        self.data_sources.append(data_source_info)

        # Record audit event
        if self.audit_trail:
            self.audit_trail.record_event(
                event_type=AuditEventType.DATA_SOURCE_ADDED,
                description=f"Added data source: {data_source_info.get('data_source', 'unknown')}",
                actor="system",
                metadata={"data_source_info": data_source_info}
            )

        # Create new version
        if self.version_control:
            self.version_control.commit(
                metadata=self.get_all_metadata(),
                description=f"Added data source: {data_source_info.get('data_source', 'unknown')}",
                author="system"
            )

    def set_risk_assessment(self, risk_assessment: Dict[str, Any]):
        """
        Store risk assessment results.

        Args:
            risk_assessment: Risk assessment data from AIActRiskAssessor
        """
        old_risk_level = self.risk_assessment.get("risk_level")
        new_risk_level = risk_assessment.get("risk_level")

        self.risk_assessment = risk_assessment

        # Record audit event
        if self.audit_trail:
            if old_risk_level and old_risk_level != new_risk_level:
                self.audit_trail.record_event(
                    event_type=AuditEventType.RISK_LEVEL_CHANGED,
                    description=f"Risk level changed from {old_risk_level} to {new_risk_level}",
                    actor="system",
                    metadata={"old_level": old_risk_level, "new_level": new_risk_level}
                )
            else:
                self.audit_trail.record_event(
                    event_type=AuditEventType.RISK_ASSESSMENT_PERFORMED,
                    description=f"Risk assessment performed: {new_risk_level}",
                    actor="system",
                    metadata=risk_assessment
                )

        # Create new version
        if self.version_control:
            description = f"Risk assessment: {new_risk_level}"
            if old_risk_level and old_risk_level != new_risk_level:
                description = f"Risk level changed: {old_risk_level} → {new_risk_level}"
            self.version_control.commit(
                metadata=self.get_all_metadata(),
                description=description,
                author="system"
            )

    def set_operational_metrics(self, metrics: Dict[str, Any]):
        """
        Store operational metrics.

        Args:
            metrics: Operational metrics from OperationalMetricsTracker
        """
        self.operational_metrics = metrics

        # Record audit event
        if self.audit_trail:
            self.audit_trail.record_event(
                event_type=AuditEventType.METRICS_RECORDED,
                description="Operational metrics updated",
                actor="system",
                metadata={"total_operations": metrics.get("operations", {}).get("total", 0)}
            )

    def add_bias_analysis(self, bias_analysis: Dict[str, Any]):
        """
        Add bias analysis results to metadata storage.

        Args:
            bias_analysis: Bias analysis results from BiasDetector
        """
        self.bias_analyses.append(bias_analysis)

        # Record audit event
        if self.audit_trail:
            risk_level = bias_analysis.get("risk_level", "unknown")
            dataset_name = bias_analysis.get("dataset_name", "unknown")
            self.audit_trail.record_event(
                event_type=AuditEventType.BIAS_ANALYSIS_PERFORMED,
                description=f"Bias analysis performed on {dataset_name}: {risk_level} risk",
                actor="system",
                metadata={
                    "analysis_id": bias_analysis.get("analysis_id"),
                    "risk_level": risk_level,
                    "fairness_score": bias_analysis.get("overall_fairness_score")
                }
            )

        # Create new version
        if self.version_control:
            self.version_control.commit(
                metadata=self.get_all_metadata(),
                description=f"Bias analysis: {dataset_name} - {risk_level} risk",
                author="system"
            )

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

        # Include audit trail summary if available
        if self.audit_trail:
            metadata["audit_summary"] = self.audit_trail.generate_summary()

        # Include version control summary if available
        if self.version_control:
            metadata["version_info"] = self.version_control.to_dict()

        # Include data governance information if available
        if self.data_governance_tracker:
            metadata["data_governance"] = self.data_governance_tracker.to_dict()
            metadata["data_quality_summary"] = self.data_governance_tracker.get_data_quality_summary()
            metadata["privacy_summary"] = self.data_governance_tracker.get_privacy_summary()

        # Include bias analyses if available
        if self.bias_analyses:
            metadata["bias_analyses"] = self.bias_analyses
            # Calculate overall bias summary
            if self.bias_analyses:
                risk_levels = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3, 'unknown': -1}
                max_risk = max(risk_levels.get(b.get("risk_level", "unknown"), -1) for b in self.bias_analyses)
                overall_risk = next((k for k, v in risk_levels.items() if v == max_risk), 'unknown')
                avg_fairness = sum(b.get("overall_fairness_score", 0) for b in self.bias_analyses) / len(self.bias_analyses)
                metadata["bias_summary"] = {
                    "total_analyses": len(self.bias_analyses),
                    "overall_risk_level": overall_risk,
                    "average_fairness_score": round(avg_fairness, 3)
                }

        return metadata

    def get_audit_trail(self) -> Optional[AuditTrail]:
        """Get the audit trail instance."""
        return self.audit_trail

    def get_version_control(self) -> Optional[VersionControl]:
        """Get the version control instance."""
        return self.version_control

    def get_data_governance_tracker(self) -> Optional[DataGovernanceTracker]:
        """Get the data governance tracker instance."""
        return self.data_governance_tracker

    def set_data_governance_tracker(self, tracker: DataGovernanceTracker):
        """
        Set or replace the data governance tracker.

        Args:
            tracker: DataGovernanceTracker instance
        """
        self.data_governance_tracker = tracker

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

    def save_to_file(self, filepath: str, save_audit: bool = True, save_versions: bool = True, save_data_governance: bool = True):
        """
        Save metadata to a JSON file.

        Args:
            filepath: Path to save the JSON file
            save_audit: Also save audit trail to separate file
            save_versions: Also save version history to separate file
            save_data_governance: Also save data governance to separate file
        """
        metadata = self.get_all_metadata()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Save audit trail to separate file
        if save_audit and self.audit_trail:
            from pathlib import Path
            audit_path = Path(filepath).with_suffix('.audit.json')
            self.audit_trail.save_to_file(str(audit_path))

        # Save version history to separate file
        if save_versions and self.version_control:
            from pathlib import Path
            version_path = Path(filepath).with_suffix('.versions.json')
            self.version_control.save_to_file(str(version_path))

        # Save data governance to separate file
        if save_data_governance and self.data_governance_tracker:
            from pathlib import Path
            governance_path = Path(filepath).with_suffix('.governance.json')
            self.data_governance_tracker.save_to_file(str(governance_path))

    def load_from_file(self, filepath: str, load_audit: bool = True, load_versions: bool = True, load_data_governance: bool = True):
        """
        Load metadata from a JSON file.

        Args:
            filepath: Path to the JSON file
            load_audit: Also load audit trail from separate file if it exists
            load_versions: Also load version history from separate file if it exists
            load_data_governance: Also load data governance from separate file if it exists
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.system_name = data.get("system_name", "unnamed_system")
            self.models = data.get("models", [])
            self.components = data.get("components", [])
            self.data_sources = data.get("data_sources", [])
            self.risk_assessment = data.get("risk_assessment", {})
            self.operational_metrics = data.get("operational_metrics", {})
            self.bias_analyses = data.get("bias_analyses", [])
            self.created_at = data.get("created_at", datetime.now().isoformat())

        # Load audit trail from separate file if it exists
        if load_audit and self.audit_trail:
            from pathlib import Path
            audit_path = Path(filepath).with_suffix('.audit.json')
            if audit_path.exists():
                self.audit_trail.load_from_file(str(audit_path))

        # Load version history from separate file if it exists
        if load_versions and self.version_control:
            from pathlib import Path
            version_path = Path(filepath).with_suffix('.versions.json')
            if version_path.exists():
                self.version_control.load_from_file(str(version_path))

        # Load data governance from separate file if it exists
        if load_data_governance and self.data_governance_tracker:
            from pathlib import Path
            governance_path = Path(filepath).with_suffix('.governance.json')
            if governance_path.exists():
                self.data_governance_tracker = DataGovernanceTracker.load_from_file(str(governance_path))

    def clear(self):
        """Clear all stored metadata."""
        self.models.clear()
        self.components.clear()
        self.data_sources.clear()
        self.risk_assessment = {}
        self.operational_metrics = {}
        self.bias_analyses.clear()
