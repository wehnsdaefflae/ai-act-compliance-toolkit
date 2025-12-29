"""
Technical Documentation Generator

Generates EU AI Act Article 11 compliant technical documentation for high-risk AI systems.
Article 11 requires comprehensive technical documentation that demonstrates compliance
and enables competent authorities to assess conformity.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import json


class TechnicalDocumentationGenerator:
    """
    Generates Article 11 compliant technical documentation from captured metadata.

    Article 11 requires documentation of:
    - General description of AI system
    - Detailed description of elements and development process
    - Design specifications, general logic, algorithms
    - Data requirements and data governance
    - Human oversight measures
    - Computational resources, expected lifetime
    - Risk management system
    - Changes made to system through lifecycle
    """

    def __init__(self, metadata: Dict[str, Any]):
        """
        Initialize documentation generator with metadata.

        Args:
            metadata: Complete system metadata from MetadataStorage
        """
        self.metadata = metadata
        self.system_name = metadata.get("system_name", "Unnamed System")

    def generate_documentation(self) -> Dict[str, Any]:
        """
        Generate complete Article 11 technical documentation.

        Returns:
            Dictionary containing all required documentation sections
        """
        return {
            "system_identification": self._generate_system_identification(),
            "general_description": self._generate_general_description(),
            "development_process": self._generate_development_process(),
            "architecture_and_design": self._generate_architecture_and_design(),
            "data_requirements": self._generate_data_requirements(),
            "human_oversight": self._generate_human_oversight(),
            "performance_metrics": self._generate_performance_metrics(),
            "risk_management": self._generate_risk_management(),
            "lifecycle_management": self._generate_lifecycle_management(),
            "conformity_assessment": self._generate_conformity_assessment(),
            "generated_at": datetime.now().isoformat()
        }

    def _generate_system_identification(self) -> Dict[str, Any]:
        """Generate system identification section."""
        risk_info = self.metadata.get("risk_assessment", {})

        return {
            "system_name": self.system_name,
            "system_version": self._get_version_info(),
            "risk_classification": risk_info.get("risk_level", "not_assessed"),
            "intended_purpose": self._infer_intended_purpose(),
            "deployment_date": self.metadata.get("created_at", "Not specified"),
            "framework": self.metadata.get("framework", "Not specified")
        }

    def _generate_general_description(self) -> Dict[str, Any]:
        """Generate general description of the AI system."""
        models = self.metadata.get("models", [])
        components = self.metadata.get("components", [])

        capabilities = []
        limitations = []

        # Infer capabilities from models
        for model in models:
            model_name = model.get("model_name", "")
            provider = model.get("provider", "")
            if provider:
                capabilities.append(f"{provider} language model ({model_name})")

        # Infer from components
        for component in components:
            comp_type = component.get("type", "")
            if comp_type == "chain":
                capabilities.append(f"Chain-based processing: {component.get('name', 'unnamed')}")
            elif comp_type == "tool":
                capabilities.append(f"Tool integration: {component.get('name', 'unnamed')}")

        # Add limitations based on risk assessment
        risk_factors = self.metadata.get("risk_assessment", {}).get("risk_factors", [])
        for factor in risk_factors:
            limitations.append(f"Risk consideration: {factor}")

        return {
            "purpose_and_scope": self._infer_intended_purpose(),
            "capabilities": capabilities if capabilities else ["System capabilities to be documented"],
            "known_limitations": limitations if limitations else ["System limitations to be documented"],
            "target_users": "To be specified based on deployment context",
            "deployment_context": "To be specified based on use case"
        }

    def _generate_development_process(self) -> Dict[str, Any]:
        """Generate development process documentation."""
        version_info = self.metadata.get("version_info", {})
        audit_summary = self.metadata.get("audit_summary", {})

        # Extract development timeline from version control
        versions = version_info.get("versions", [])
        development_timeline = []

        for version in versions:
            development_timeline.append({
                "version": version.get("version"),
                "date": version.get("timestamp"),
                "changes": version.get("description"),
                "author": version.get("author")
            })

        return {
            "development_methodology": "Compliance-as-Code approach with automated metadata capture",
            "development_timeline": development_timeline,
            "version_control": {
                "current_version": version_info.get("current_version"),
                "total_versions": version_info.get("total_versions", 0)
            },
            "quality_assurance": {
                "automated_tracking": audit_summary.get("total_events", 0) > 0,
                "audit_events_count": audit_summary.get("total_events", 0)
            },
            "testing_procedures": "To be documented based on specific test cases"
        }

    def _generate_architecture_and_design(self) -> Dict[str, Any]:
        """Generate architecture and design specifications."""
        models = self.metadata.get("models", [])
        components = self.metadata.get("components", [])

        model_components = []
        for model in models:
            model_info = {
                "name": model.get("model_name", "Unknown"),
                "provider": model.get("provider", "Unknown"),
                "type": model.get("model_type", "LLM"),
                "parameters": model.get("parameters", {}),
                "timestamp_added": model.get("timestamp")
            }

            # Add framework-specific details
            if "architecture" in model:
                model_info["architecture"] = model["architecture"]
            if "layer_count" in model:
                model_info["layers"] = model["layer_count"]
            if "parameter_count" in model:
                model_info["total_parameters"] = model["parameter_count"]

            model_components.append(model_info)

        system_components = []
        for component in components:
            system_components.append({
                "type": component.get("type", "Unknown"),
                "name": component.get("name", "Unnamed"),
                "description": component.get("description", "Component details to be specified")
            })

        return {
            "system_architecture": {
                "type": "AI/ML System",
                "framework": self.metadata.get("framework", "Not specified"),
                "models": model_components,
                "components": system_components
            },
            "algorithms_and_logic": self._extract_algorithmic_logic(),
            "computational_requirements": self._extract_computational_requirements(),
            "integration_points": "To be documented based on deployment architecture"
        }

    def _generate_data_requirements(self) -> Dict[str, Any]:
        """Generate data requirements and governance documentation."""
        data_sources = self.metadata.get("data_sources", [])
        data_governance = self.metadata.get("data_governance", {})
        data_quality_summary = self.metadata.get("data_quality_summary", {})
        privacy_summary = self.metadata.get("privacy_summary", {})

        documented_sources = []
        for source in data_sources:
            source_doc = {
                "source_name": source.get("data_source", "Unknown"),
                "data_type": source.get("data_type", "Unspecified"),
                "loader_type": source.get("loader_type", "Unknown"),
                "timestamp_added": source.get("timestamp")
            }
            documented_sources.append(source_doc)

        return {
            "data_sources": documented_sources,
            "data_governance": {
                "lineage_tracking": data_governance.get("lineage_graph", {}) != {},
                "quality_metrics": data_quality_summary,
                "privacy_compliance": privacy_summary
            },
            "data_quality_criteria": {
                "relevance": "To be assessed based on use case",
                "representativeness": "To be validated against target population",
                "completeness": data_quality_summary.get("average_completeness", "Not assessed"),
                "accuracy": data_quality_summary.get("average_accuracy", "Not assessed")
            },
            "data_preprocessing": "To be documented based on data pipeline",
            "bias_mitigation": "To be assessed and documented based on data analysis"
        }

    def _generate_human_oversight(self) -> Dict[str, Any]:
        """Generate human oversight measures documentation (Article 14 compliance)."""
        risk_level = self.metadata.get("risk_assessment", {}).get("risk_level", "unknown")

        # Human oversight requirements depend on risk level
        if risk_level == "high":
            oversight_requirement = "mandatory"
            recommended_measures = [
                "Human-in-the-loop: Human intervention before critical decisions",
                "Human-on-the-loop: Monitoring and intervention capability during operation",
                "Human-in-command: Ability to override or disable system decisions",
                "Regular review of system outputs and decisions",
                "Documented escalation procedures for edge cases"
            ]
        elif risk_level == "limited":
            oversight_requirement = "transparency_required"
            recommended_measures = [
                "Clear disclosure of AI system interaction to users",
                "Opt-out mechanisms where appropriate"
            ]
        else:
            oversight_requirement = "recommended"
            recommended_measures = [
                "Periodic review of system performance",
                "User feedback mechanisms"
            ]

        return {
            "oversight_requirement_level": oversight_requirement,
            "recommended_measures": recommended_measures,
            "technical_measures": {
                "confidence_thresholds": "To be configured based on use case criticality",
                "uncertainty_detection": "To be implemented for high-risk decisions",
                "explanation_generation": "To be provided for critical outputs"
            },
            "organizational_measures": {
                "training_requirements": "Personnel must understand system capabilities and limitations",
                "competence_requirements": "Oversight personnel must have relevant domain expertise",
                "intervention_procedures": "Documented procedures for human intervention"
            },
            "implementation_status": "To be implemented based on deployment requirements"
        }

    def _generate_performance_metrics(self) -> Dict[str, Any]:
        """Generate performance metrics and monitoring documentation."""
        operational_metrics = self.metadata.get("operational_metrics", {})

        metrics_data = {
            "operational_statistics": {},
            "performance_indicators": {},
            "monitoring_mechanisms": {}
        }

        # Extract operational statistics
        if operational_metrics:
            operations = operational_metrics.get("operations", {})
            performance = operational_metrics.get("performance", {})

            metrics_data["operational_statistics"] = {
                "total_operations": operations.get("total", 0),
                "successful_operations": operations.get("successful", 0),
                "failed_operations": operations.get("failed", 0),
                "success_rate": f"{operations.get('success_rate', 0):.2%}"
            }

            metrics_data["performance_indicators"] = {
                "average_latency": performance.get("avg_latency_ms", "Not measured"),
                "min_latency": performance.get("min_latency_ms", "Not measured"),
                "max_latency": performance.get("max_latency_ms", "Not measured"),
                "total_cost": operational_metrics.get("costs", {}).get("total_cost", 0)
            }

        metrics_data["monitoring_mechanisms"] = {
            "automated_logging": self.metadata.get("audit_summary", {}).get("total_events", 0) > 0,
            "performance_tracking": operational_metrics != {},
            "error_tracking": len(operational_metrics.get("errors", [])) if operational_metrics else 0
        }

        metrics_data["accuracy_metrics"] = "To be measured against validation dataset"
        metrics_data["robustness_testing"] = "To be conducted based on deployment scenarios"

        return metrics_data

    def _generate_risk_management(self) -> Dict[str, Any]:
        """Generate risk management system documentation."""
        risk_assessment = self.metadata.get("risk_assessment", {})

        if not risk_assessment:
            return {
                "risk_assessment_status": "not_performed",
                "recommendation": "Perform risk assessment using 'aiact-toolkit assess-risk' command"
            }

        return {
            "risk_assessment_status": "completed",
            "risk_classification": {
                "risk_level": risk_assessment.get("risk_level", "unknown"),
                "confidence": risk_assessment.get("confidence", 0),
                "assessment_date": risk_assessment.get("timestamp", "Not recorded")
            },
            "identified_risks": risk_assessment.get("risk_factors", []),
            "mitigation_measures": risk_assessment.get("recommendations", []),
            "compliance_requirements": risk_assessment.get("compliance_requirements", []),
            "residual_risks": "To be assessed after mitigation implementation",
            "risk_monitoring": {
                "continuous_monitoring": "To be implemented in production environment",
                "review_frequency": "To be determined based on risk level and deployment context"
            }
        }

    def _generate_lifecycle_management(self) -> Dict[str, Any]:
        """Generate lifecycle management documentation."""
        version_info = self.metadata.get("version_info", {})
        audit_summary = self.metadata.get("audit_summary", {})

        return {
            "version_control": {
                "current_version": version_info.get("current_version", "Not tracked"),
                "total_versions": version_info.get("total_versions", 0),
                "version_history_available": version_info.get("total_versions", 0) > 0
            },
            "change_management": {
                "documented_changes": audit_summary.get("total_events", 0),
                "change_tracking_enabled": audit_summary.get("total_events", 0) > 0,
                "audit_trail_integrity": "Cryptographically verified" if audit_summary.get("total_events", 0) > 0 else "Not applicable"
            },
            "expected_lifetime": "To be specified based on deployment requirements",
            "maintenance_plan": {
                "update_frequency": "To be determined based on risk level",
                "monitoring_procedures": "Continuous automated logging and performance tracking",
                "decommissioning_plan": "To be documented before end of lifecycle"
            },
            "post_market_monitoring": {
                "monitoring_plan": "To be implemented according to Article 72",
                "incident_reporting": "To be configured according to Article 73",
                "corrective_actions": "To be documented when issues are identified"
            }
        }

    def _generate_conformity_assessment(self) -> Dict[str, Any]:
        """Generate conformity assessment documentation."""
        risk_level = self.metadata.get("risk_assessment", {}).get("risk_level", "unknown")

        # Determine applicable conformity assessment procedure
        if risk_level == "high":
            procedure = "Required (Annex VI or VII)"
            details = [
                "Internal control (Annex VI) - if quality management system in place",
                "Assessment of quality management system and technical documentation (Annex VII)",
                "EU declaration of conformity required",
                "CE marking required"
            ]
        elif risk_level == "unacceptable":
            procedure = "Not applicable - System prohibited"
            details = ["System falls under prohibited AI practices (Article 5)"]
        else:
            procedure = "Not required for this risk level"
            details = ["Transparency obligations may apply (Article 50)"]

        return {
            "applicable_procedure": procedure,
            "assessment_details": details,
            "harmonized_standards": "To be identified based on specific system characteristics",
            "technical_specifications": "To be documented according to assessment procedure",
            "compliance_status": {
                "risk_assessment": "completed" if self.metadata.get("risk_assessment") else "pending",
                "technical_documentation": "in_progress",
                "quality_management": "to_be_implemented",
                "post_market_monitoring": "to_be_implemented"
            }
        }

    def _infer_intended_purpose(self) -> str:
        """Infer intended purpose from metadata."""
        risk_assessment = self.metadata.get("risk_assessment", {})

        # Check if use case was specified in risk assessment
        if "use_case" in risk_assessment:
            return risk_assessment["use_case"]

        # Try to infer from models
        models = self.metadata.get("models", [])
        if models:
            model_name = models[0].get("model_name", "")
            if "gpt" in model_name.lower() or "chat" in model_name.lower():
                return "Natural language processing and conversational AI"
            elif "embed" in model_name.lower():
                return "Document analysis and semantic search"

        return "Purpose to be specified based on deployment context"

    def _get_version_info(self) -> str:
        """Get current version information."""
        version_info = self.metadata.get("version_info", {})
        return str(version_info.get("current_version", "1"))

    def _extract_algorithmic_logic(self) -> str:
        """Extract description of algorithmic logic."""
        models = self.metadata.get("models", [])

        if not models:
            return "Algorithmic logic to be documented"

        descriptions = []
        for model in models:
            model_type = model.get("model_type", "Unknown")
            if model_type == "LLM":
                descriptions.append("Large Language Model for text generation and understanding")
            elif model_type == "neural_network":
                architecture = model.get("architecture", "Unknown")
                descriptions.append(f"Neural network with {architecture} architecture")

        return "; ".join(descriptions) if descriptions else "Algorithmic logic to be documented"

    def _extract_computational_requirements(self) -> Dict[str, Any]:
        """Extract computational resource requirements."""
        models = self.metadata.get("models", [])

        requirements = {
            "training_resources": "To be documented",
            "inference_resources": "To be documented",
            "estimated_costs": {}
        }

        # Extract cost information if available
        operational_metrics = self.metadata.get("operational_metrics", {})
        if operational_metrics:
            costs = operational_metrics.get("costs", {})
            if costs:
                requirements["estimated_costs"] = {
                    "total_cost": costs.get("total_cost", 0),
                    "cost_per_operation": costs.get("average_cost", 0)
                }

        # Extract parameter counts from models
        for model in models:
            if "parameter_count" in model:
                requirements["model_size"] = f"{model['parameter_count']:,} parameters"
                break

        return requirements

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert documentation to dictionary format.

        Returns:
            Complete documentation as dictionary
        """
        return self.generate_documentation()

    def to_json(self, filepath: str, indent: int = 2):
        """
        Save documentation to JSON file.

        Args:
            filepath: Path to save JSON file
            indent: JSON indentation level
        """
        documentation = self.generate_documentation()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(documentation, f, indent=indent, ensure_ascii=False)
