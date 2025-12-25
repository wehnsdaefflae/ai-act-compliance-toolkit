"""
Model Card Generator for AI Act Compliance Toolkit

Implements model card generation following industry standards (Google's Model Cards,
Hugging Face Model Cards) and EU AI Act transparency requirements (Article 13).

Model cards provide standardized documentation of ML models including:
- Model details and intended use
- Performance metrics and limitations
- Ethical considerations and biases
- Training data and procedures
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import json


class ModelCardVersion(str, Enum):
    """Model card specification version."""
    V1_0 = "1.0"
    V2_0 = "2.0"


@dataclass
class ModelDetails:
    """Model identification and ownership information."""
    name: str
    version: str
    description: str
    model_type: str  # e.g., "language_model", "classifier", "regression"
    architecture: Optional[str] = None
    developers: Optional[str] = None
    organization: Optional[str] = None
    license: Optional[str] = None
    citation: Optional[str] = None
    contact: Optional[str] = None
    model_date: Optional[str] = None
    framework: Optional[str] = None  # e.g., "langchain", "pytorch", "tensorflow"


@dataclass
class IntendedUse:
    """Intended use cases and out-of-scope uses."""
    primary_uses: List[str] = field(default_factory=list)
    primary_users: List[str] = field(default_factory=list)
    out_of_scope: List[str] = field(default_factory=list)
    use_case_description: Optional[str] = None
    application_domain: Optional[str] = None


@dataclass
class PerformanceMetric:
    """Performance metric information."""
    metric_name: str
    metric_value: Any
    description: Optional[str] = None
    dataset: Optional[str] = None
    threshold: Optional[float] = None


@dataclass
class TrainingData:
    """Training data information."""
    description: str
    data_sources: List[str] = field(default_factory=list)
    preprocessing: List[str] = field(default_factory=list)
    size: Optional[str] = None
    time_period: Optional[str] = None
    data_type: Optional[str] = None
    geographic_coverage: Optional[str] = None
    language: Optional[str] = None
    personal_data: bool = False
    sensitive_data: bool = False


@dataclass
class EthicalConsiderations:
    """Ethical considerations and bias information."""
    risks: List[str] = field(default_factory=list)
    biases: List[str] = field(default_factory=list)
    fairness_assessment: Optional[str] = None
    human_oversight: Optional[str] = None
    privacy_measures: List[str] = field(default_factory=list)
    environmental_impact: Optional[str] = None


@dataclass
class Limitations:
    """Model limitations and recommendations."""
    technical_limitations: List[str] = field(default_factory=list)
    known_biases: List[str] = field(default_factory=list)
    edge_cases: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class RegulatoryCompliance:
    """Regulatory compliance information for EU AI Act."""
    risk_level: Optional[str] = None
    eu_ai_act_category: Optional[str] = None
    gdpr_compliant: Optional[bool] = None
    article_13_transparency: bool = False
    article_14_human_oversight: bool = False
    compliance_documentation: List[str] = field(default_factory=list)


@dataclass
class ModelCard:
    """
    Complete model card following industry standards and EU AI Act requirements.

    Based on:
    - Model Cards for Model Reporting (Mitchell et al., 2019)
    - Hugging Face Model Card specification
    - EU AI Act Article 13 (Transparency requirements)
    """
    model_details: ModelDetails
    intended_use: IntendedUse
    performance: List[PerformanceMetric] = field(default_factory=list)
    training_data: Optional[TrainingData] = None
    ethical_considerations: Optional[EthicalConsiderations] = None
    limitations: Optional[Limitations] = None
    regulatory_compliance: Optional[RegulatoryCompliance] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)
    card_version: str = ModelCardVersion.V2_0.value
    generated_date: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert model card to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert model card to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save_json(self, filepath: str) -> None:
        """Save model card to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())


class ModelCardGenerator:
    """
    Generates model cards from toolkit metadata.

    Extracts relevant information from LangChain, PyTorch, TensorFlow metadata
    and structures it into a standardized model card format.
    """

    def __init__(self):
        self.default_developers = "Development Team"

    def generate_from_metadata(
        self,
        metadata: Dict[str, Any],
        model_name: Optional[str] = None,
        include_optional: bool = True
    ) -> ModelCard:
        """
        Generate a model card from toolkit metadata.

        Args:
            metadata: Metadata dictionary from LangChainMonitor, PyTorchMonitor, etc.
            model_name: Specific model to generate card for (if multiple models exist)
            include_optional: Include optional sections even if data is limited

        Returns:
            ModelCard object
        """
        # Extract model information
        model_info = self._extract_model_info(metadata, model_name)

        # Build model card sections
        model_details = self._build_model_details(metadata, model_info)
        intended_use = self._build_intended_use(metadata)
        performance = self._build_performance_metrics(metadata, model_info)
        training_data = self._build_training_data(metadata)
        ethical = self._build_ethical_considerations(metadata) if include_optional else None
        limitations = self._build_limitations(metadata, model_info) if include_optional else None
        regulatory = self._build_regulatory_compliance(metadata)

        return ModelCard(
            model_details=model_details,
            intended_use=intended_use,
            performance=performance,
            training_data=training_data,
            ethical_considerations=ethical,
            limitations=limitations,
            regulatory_compliance=regulatory,
            additional_info=self._build_additional_info(metadata)
        )

    def _extract_model_info(self, metadata: Dict[str, Any], model_name: Optional[str]) -> Dict[str, Any]:
        """Extract specific model information from metadata."""
        models = metadata.get("models", [])

        if not models:
            # Return empty model info
            return {}

        if model_name:
            # Find specific model
            for model in models:
                if model.get("model_name") == model_name or model.get("name") == model_name:
                    return model
            raise ValueError(f"Model '{model_name}' not found in metadata")

        # Return first model if not specified
        return models[0]

    def _build_model_details(self, metadata: Dict[str, Any], model_info: Dict[str, Any]) -> ModelDetails:
        """Build ModelDetails section."""
        system_name = metadata.get("system_name", "AI System")
        model_name = model_info.get("model_name") or model_info.get("name", "Unknown Model")

        # Determine framework
        framework = None
        if metadata.get("framework") == "langchain":
            framework = "LangChain"
        elif metadata.get("framework") == "pytorch":
            framework = "PyTorch"
        elif metadata.get("framework") == "tensorflow":
            framework = "TensorFlow/Keras"

        # Extract architecture info
        architecture = None
        if "architecture" in model_info:
            architecture = model_info["architecture"]
        elif "model_type" in model_info:
            architecture = model_info["model_type"]
        elif "layers" in model_info:
            layer_count = len(model_info["layers"])
            architecture = f"{layer_count}-layer neural network"

        return ModelDetails(
            name=model_name,
            version=metadata.get("version", "1.0"),
            description=metadata.get("description", f"AI model for {system_name}"),
            model_type=model_info.get("type", "machine_learning_model"),
            architecture=architecture,
            developers=metadata.get("developers", self.default_developers),
            organization=metadata.get("organization"),
            license=metadata.get("license"),
            model_date=metadata.get("created_at", datetime.now().isoformat()),
            framework=framework
        )

    def _build_intended_use(self, metadata: Dict[str, Any]) -> IntendedUse:
        """Build IntendedUse section."""
        primary_uses = []

        # Extract use case from metadata
        if "use_case" in metadata:
            primary_uses.append(metadata["use_case"])

        # Extract from description
        description = metadata.get("description", "")

        return IntendedUse(
            primary_uses=primary_uses if primary_uses else ["General AI/ML application"],
            primary_users=metadata.get("target_users", ["Developers", "Data Scientists"]),
            out_of_scope=metadata.get("out_of_scope_uses", []),
            use_case_description=metadata.get("use_case"),
            application_domain=metadata.get("application_domain")
        )

    def _build_performance_metrics(
        self,
        metadata: Dict[str, Any],
        model_info: Dict[str, Any]
    ) -> List[PerformanceMetric]:
        """Build performance metrics section."""
        metrics = []

        # Extract from operational metrics
        if "operational_metrics" in metadata:
            op_metrics = metadata["operational_metrics"]

            if "performance" in op_metrics:
                perf = op_metrics["performance"]
                if "avg_execution_time_ms" in perf:
                    metrics.append(PerformanceMetric(
                        metric_name="Average Execution Time",
                        metric_value=f"{perf['avg_execution_time_ms']:.2f}ms",
                        description="Average time to process a request"
                    ))

            if "operations" in op_metrics:
                ops = op_metrics["operations"]
                if "error_rate_percent" in ops:
                    metrics.append(PerformanceMetric(
                        metric_name="Error Rate",
                        metric_value=f"{ops['error_rate_percent']}%",
                        description="Percentage of failed operations",
                        threshold=5.0  # 5% threshold
                    ))

        # Extract model-specific metrics
        if "metrics" in model_info:
            for metric_name, metric_value in model_info["metrics"].items():
                metrics.append(PerformanceMetric(
                    metric_name=metric_name,
                    metric_value=metric_value
                ))

        return metrics

    def _build_training_data(self, metadata: Dict[str, Any]) -> Optional[TrainingData]:
        """Build TrainingData section."""
        data_sources = metadata.get("data_sources", [])

        if not data_sources:
            return None

        # Extract data source information
        source_list = []
        preprocessing_steps = []
        has_personal_data = False
        has_sensitive_data = False

        for source in data_sources:
            source_name = source.get("source") or source.get("path", "Unknown source")
            source_list.append(source_name)

            # Check for personal/sensitive data
            if source.get("personal_data"):
                has_personal_data = True
            if source.get("sensitive_data"):
                has_sensitive_data = True

            # Extract preprocessing
            if "preprocessing" in source:
                preprocessing_steps.extend(source["preprocessing"])

        # Build description
        description = f"Training data from {len(data_sources)} source(s)"
        if metadata.get("data_governance"):
            description = "Tracked and governed training data sources with lineage information"

        return TrainingData(
            description=description,
            data_sources=source_list,
            preprocessing=list(set(preprocessing_steps)),  # Remove duplicates
            size=metadata.get("data_size"),
            data_type=data_sources[0].get("data_type") if data_sources else None,
            personal_data=has_personal_data,
            sensitive_data=has_sensitive_data
        )

    def _build_ethical_considerations(self, metadata: Dict[str, Any]) -> EthicalConsiderations:
        """Build EthicalConsiderations section."""
        risks = []
        privacy_measures = []

        # Extract from risk assessment
        if "risk_assessment" in metadata:
            risk_info = metadata["risk_assessment"]
            risks.extend(risk_info.get("risk_factors", []))

        # Extract from data governance
        if "data_governance" in metadata:
            privacy_measures.append("Data lineage tracking")
            privacy_measures.append("Data quality monitoring")

        # Check for personal data
        data_sources = metadata.get("data_sources", [])
        if any(source.get("personal_data") for source in data_sources):
            privacy_measures.append("GDPR compliance measures")

        return EthicalConsiderations(
            risks=risks,
            biases=metadata.get("known_biases", []),
            fairness_assessment=metadata.get("fairness_assessment"),
            human_oversight=metadata.get("human_oversight_description"),
            privacy_measures=privacy_measures,
            environmental_impact=metadata.get("environmental_impact")
        )

    def _build_limitations(
        self,
        metadata: Dict[str, Any],
        model_info: Dict[str, Any]
    ) -> Limitations:
        """Build Limitations section."""
        technical = []
        recommendations = []

        # Extract from risk assessment
        if "risk_assessment" in metadata:
            risk_info = metadata["risk_assessment"]
            recommendations.extend(risk_info.get("recommendations", []))

        # Add general limitations based on model type
        if model_info:
            technical.append("Performance may vary on out-of-distribution data")
            technical.append("Model outputs should be validated before use in production")

        return Limitations(
            technical_limitations=technical,
            known_biases=metadata.get("known_biases", []),
            edge_cases=metadata.get("edge_cases", []),
            recommendations=recommendations
        )

    def _build_regulatory_compliance(self, metadata: Dict[str, Any]) -> RegulatoryCompliance:
        """Build RegulatoryCompliance section."""
        compliance_docs = []

        # Check for various compliance artifacts
        if "audit_trail" in metadata:
            compliance_docs.append("Audit trail (Article 12)")

        if "risk_assessment" in metadata:
            risk_level = metadata["risk_assessment"].get("risk_level", "unknown")
            compliance_docs.append(f"Risk assessment report (Risk level: {risk_level})")
        else:
            risk_level = None

        if "data_governance" in metadata:
            compliance_docs.append("Data governance documentation (Article 10)")

        # Determine EU AI Act category
        category = None
        if risk_level:
            if risk_level == "high":
                category = "High-Risk AI System"
            elif risk_level == "limited":
                category = "Limited Risk AI System"
            elif risk_level == "minimal":
                category = "Minimal Risk AI System"

        return RegulatoryCompliance(
            risk_level=risk_level,
            eu_ai_act_category=category,
            gdpr_compliant=metadata.get("gdpr_compliant"),
            article_13_transparency=True,  # Model card itself satisfies Article 13
            article_14_human_oversight=metadata.get("human_oversight_enabled", False),
            compliance_documentation=compliance_docs
        )

    def _build_additional_info(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Build additional information section."""
        additional = {}

        # Add operational metrics summary
        if "operational_metrics" in metadata:
            additional["operational_monitoring"] = "Enabled"

        # Add versioning info
        if "version_control" in metadata:
            additional["version_control"] = "Enabled"

        # Add framework-specific info
        if "components" in metadata:
            additional["components_count"] = len(metadata["components"])

        return additional


def generate_model_cards_for_all_models(metadata: Dict[str, Any]) -> List[ModelCard]:
    """
    Generate model cards for all models in metadata.

    Args:
        metadata: Metadata dictionary from toolkit

    Returns:
        List of ModelCard objects, one per model
    """
    generator = ModelCardGenerator()
    cards = []

    models = metadata.get("models", [])

    if not models:
        # Generate single card for the system
        cards.append(generator.generate_from_metadata(metadata))
    else:
        # Generate card for each model
        for model in models:
            model_name = model.get("model_name") or model.get("name")
            card = generator.generate_from_metadata(metadata, model_name=model_name)
            cards.append(card)

    return cards
