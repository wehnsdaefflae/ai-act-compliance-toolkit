"""
Risk Assessment Module

Provides automated risk classification for AI systems according to EU AI Act requirements.
Analyzes system metadata to determine risk level and provides compliance recommendations.
"""

from typing import Dict, Any, List, Optional
from enum import Enum
from datetime import datetime


class RiskLevel(Enum):
    """EU AI Act risk classification levels."""
    UNACCEPTABLE = "unacceptable"
    HIGH = "high"
    LIMITED = "limited"
    MINIMAL = "minimal"
    UNKNOWN = "unknown"


class AIActRiskAssessor:
    """
    Assesses AI system risk levels according to EU AI Act requirements.

    The EU AI Act classifies AI systems into four risk categories:
    - Unacceptable Risk: Prohibited AI systems (e.g., social scoring, subliminal manipulation)
    - High Risk: AI systems in critical areas (health, education, employment, law enforcement, etc.)
    - Limited Risk: AI systems with transparency obligations (chatbots, deepfakes)
    - Minimal Risk: All other AI systems
    """

    # High-risk application domains per EU AI Act Annex III
    HIGH_RISK_DOMAINS = {
        "biometric_identification": [
            "biometric", "facial_recognition", "fingerprint", "iris_scan",
            "voice_recognition", "gait_analysis"
        ],
        "critical_infrastructure": [
            "traffic", "water_supply", "gas_supply", "electricity",
            "heating", "critical_infrastructure"
        ],
        "education": [
            "education", "training", "student", "exam", "grading",
            "admission", "learning"
        ],
        "employment": [
            "recruitment", "hiring", "cv_screening", "performance_evaluation",
            "promotion", "employment", "termination", "workforce"
        ],
        "essential_services": [
            "credit_scoring", "creditworthiness", "loan", "insurance",
            "emergency_response", "emergency_services"
        ],
        "law_enforcement": [
            "crime", "criminal", "law_enforcement", "police", "risk_assessment",
            "recidivism", "polygraph", "evidence_evaluation"
        ],
        "migration": [
            "asylum", "visa", "immigration", "border_control", "migration"
        ],
        "justice": [
            "court", "judicial", "legal", "justice", "dispute_resolution"
        ],
        "healthcare": [
            "medical", "health", "diagnosis", "treatment", "patient",
            "clinical", "healthcare", "disease", "symptom"
        ]
    }

    # Unacceptable risk indicators
    UNACCEPTABLE_INDICATORS = [
        "social_scoring", "subliminal_manipulation", "exploit_vulnerabilities",
        "real_time_biometric_identification"
    ]

    # Limited risk indicators (transparency requirements)
    LIMITED_RISK_INDICATORS = [
        "chatbot", "conversational", "deepfake", "synthetic_media",
        "emotion_recognition", "biometric_categorization"
    ]

    def __init__(self):
        """Initialize risk assessor."""
        pass

    def assess_risk(
        self,
        metadata: Dict[str, Any],
        use_case: Optional[str] = None,
        application_domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Assess risk level of an AI system based on metadata and use case.

        Args:
            metadata: System metadata from LangChain monitor
            use_case: Optional description of the system's use case
            application_domain: Optional specification of application domain

        Returns:
            Dictionary containing:
            - risk_level: RiskLevel enum value
            - risk_factors: List of identified risk factors
            - compliance_requirements: List of applicable compliance requirements
            - recommendations: List of recommended actions
            - confidence: Confidence score (0-1) in the assessment
        """
        risk_factors = []
        compliance_requirements = []
        recommendations = []

        # Analyze use case and domain
        use_case_lower = (use_case or "").lower()
        domain_lower = (application_domain or "").lower()

        # Check for unacceptable risk
        if self._check_unacceptable_risk(use_case_lower, domain_lower):
            return {
                "risk_level": RiskLevel.UNACCEPTABLE.value,
                "risk_factors": ["System may fall under prohibited AI practices"],
                "compliance_requirements": ["PROHIBITED - System cannot be deployed under EU AI Act"],
                "recommendations": [
                    "Redesign system to comply with EU AI Act",
                    "Consult legal counsel before deployment"
                ],
                "confidence": 0.7,
                "timestamp": datetime.now().isoformat()
            }

        # Check for high-risk domains
        high_risk_domain = self._identify_high_risk_domain(use_case_lower, domain_lower)

        # Analyze metadata for risk factors
        metadata_risks = self._analyze_metadata_risks(metadata)
        risk_factors.extend(metadata_risks)

        # Determine risk level
        if high_risk_domain:
            risk_level = RiskLevel.HIGH
            risk_factors.append(f"Application in high-risk domain: {high_risk_domain}")
            compliance_requirements = self._get_high_risk_requirements()
            recommendations = self._get_high_risk_recommendations()
            confidence = 0.8

        elif self._check_limited_risk(use_case_lower, metadata):
            risk_level = RiskLevel.LIMITED
            risk_factors.append("System requires transparency obligations")
            compliance_requirements = self._get_limited_risk_requirements()
            recommendations = self._get_limited_risk_recommendations()
            confidence = 0.75

        else:
            risk_level = RiskLevel.MINIMAL
            risk_factors.append("No high-risk or limited-risk indicators detected")
            compliance_requirements = self._get_minimal_risk_requirements()
            recommendations = self._get_minimal_risk_recommendations()
            confidence = 0.6

        # Add general recommendations based on metadata
        if not metadata.get("data_sources"):
            recommendations.append("Document all data sources used for training/operation")

        if not metadata.get("models"):
            recommendations.append("Ensure all AI models are properly documented")
            confidence *= 0.8  # Lower confidence if metadata is incomplete

        return {
            "risk_level": risk_level.value,
            "risk_factors": risk_factors,
            "compliance_requirements": compliance_requirements,
            "recommendations": recommendations,
            "confidence": round(confidence, 2),
            "timestamp": datetime.now().isoformat(),
            "high_risk_domain": high_risk_domain
        }

    def _check_unacceptable_risk(self, use_case: str, domain: str) -> bool:
        """Check if system falls under unacceptable risk category."""
        combined = f"{use_case} {domain}"
        return any(indicator in combined for indicator in self.UNACCEPTABLE_INDICATORS)

    def _identify_high_risk_domain(self, use_case: str, domain: str) -> Optional[str]:
        """Identify if system operates in a high-risk domain."""
        combined = f"{use_case} {domain}"

        for domain_name, keywords in self.HIGH_RISK_DOMAINS.items():
            if any(keyword in combined for keyword in keywords):
                return domain_name.replace("_", " ").title()

        return None

    def _check_limited_risk(self, use_case: str, metadata: Dict[str, Any]) -> bool:
        """Check if system falls under limited risk category."""
        # Check use case
        if any(indicator in use_case for indicator in self.LIMITED_RISK_INDICATORS):
            return True

        # Check if it's a conversational system based on metadata
        if metadata.get("models"):
            for model in metadata["models"]:
                model_name = str(model.get("model_name", "")).lower()
                if "chat" in model_name or "gpt" in model_name:
                    return True

        return False

    def _analyze_metadata_risks(self, metadata: Dict[str, Any]) -> List[str]:
        """Analyze metadata for additional risk factors."""
        risks = []

        # Check for sensitive data sources
        if metadata.get("data_sources"):
            for source in metadata["data_sources"]:
                source_path = str(source.get("data_source", "")).lower()
                if any(term in source_path for term in ["personal", "medical", "patient", "health"]):
                    risks.append("Processing potentially sensitive data")
                    break

        # Check for high-capability models
        if metadata.get("models"):
            for model in metadata["models"]:
                model_name = str(model.get("model_name", "")).lower()
                # Large language models with high capabilities
                if any(name in model_name for name in ["gpt-4", "claude-3", "opus", "gemini-ultra"]):
                    risks.append("Using high-capability foundation model")
                    break

        return risks

    def _get_high_risk_requirements(self) -> List[str]:
        """Get compliance requirements for high-risk AI systems."""
        return [
            "Implement risk management system (Article 9)",
            "Ensure high-quality training data (Article 10)",
            "Maintain technical documentation (Article 11, Annex IV)",
            "Design for automatic logging (Article 12)",
            "Ensure transparency and user information (Article 13)",
            "Implement human oversight measures (Article 14)",
            "Ensure accuracy, robustness, and cybersecurity (Article 15)",
            "Establish quality management system (Article 17)",
            "Conduct conformity assessment (Article 43)",
            "Register system in EU database (Article 71)"
        ]

    def _get_limited_risk_requirements(self) -> List[str]:
        """Get compliance requirements for limited-risk AI systems."""
        return [
            "Ensure users are informed they are interacting with AI (Article 52)",
            "Provide clear disclosure of AI-generated content",
            "Mark synthetic content (deepfakes, synthetic media)",
            "Implement basic transparency measures"
        ]

    def _get_minimal_risk_requirements(self) -> List[str]:
        """Get compliance requirements for minimal-risk AI systems."""
        return [
            "No specific EU AI Act obligations",
            "Consider voluntary compliance with AI Pact",
            "Follow general data protection requirements (GDPR)"
        ]

    def _get_high_risk_recommendations(self) -> List[str]:
        """Get recommendations for high-risk AI systems."""
        return [
            "Engage legal counsel specialized in EU AI Act compliance",
            "Establish comprehensive risk management process",
            "Prepare for conformity assessment procedure",
            "Set up post-market monitoring system",
            "Create detailed technical documentation",
            "Implement human oversight mechanisms",
            "Conduct regular testing and validation",
            "Prepare incident response procedures"
        ]

    def _get_limited_risk_recommendations(self) -> List[str]:
        """Get recommendations for limited-risk AI systems."""
        return [
            "Implement clear AI disclosure in user interface",
            "Add transparency notice in terms of service",
            "Consider implementing voluntary transparency measures",
            "Document AI decision-making processes"
        ]

    def _get_minimal_risk_recommendations(self) -> List[str]:
        """Get recommendations for minimal-risk AI systems."""
        return [
            "Maintain basic documentation of AI system",
            "Follow GDPR requirements if processing personal data",
            "Consider adopting voluntary codes of conduct",
            "Monitor for changes in use case that might affect risk level"
        ]

    def generate_risk_report(
        self,
        metadata: Dict[str, Any],
        assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive risk assessment report.

        Args:
            metadata: System metadata
            assessment: Risk assessment results

        Returns:
            Complete report dictionary for template rendering
        """
        return {
            "system_name": metadata.get("system_name", "Unknown System"),
            "assessment_date": datetime.now().isoformat(),
            "risk_level": assessment["risk_level"],
            "confidence": assessment["confidence"],
            "risk_factors": assessment["risk_factors"],
            "compliance_requirements": assessment["compliance_requirements"],
            "recommendations": assessment["recommendations"],
            "high_risk_domain": assessment.get("high_risk_domain"),
            "models": metadata.get("models", []),
            "data_sources": metadata.get("data_sources", []),
            "components": metadata.get("components", []),
            "metadata_summary": metadata.get("summary", {})
        }
