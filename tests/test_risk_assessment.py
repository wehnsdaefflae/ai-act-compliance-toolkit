"""Tests for the Risk Assessment module."""

import pytest
from aiact_toolkit.risk_assessment import AIActRiskAssessor, RiskLevel


class TestRiskLevel:
    """Test RiskLevel enum."""

    def test_risk_levels_exist(self):
        assert RiskLevel.UNACCEPTABLE.value == "unacceptable"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.LIMITED.value == "limited"
        assert RiskLevel.MINIMAL.value == "minimal"
        assert RiskLevel.UNKNOWN.value == "unknown"


class TestAIActRiskAssessor:
    """Test AIActRiskAssessor class."""

    def setup_method(self):
        self.assessor = AIActRiskAssessor()
        self.base_metadata = {
            "system_name": "Test System",
            "models": [{"model_name": "test-model", "provider": "TestProvider"}],
            "data_sources": [{"data_source": "test.csv"}],
        }

    def test_healthcare_classified_high_risk(self):
        result = self.assessor.assess_risk(
            self.base_metadata,
            use_case="Medical diagnosis chatbot",
            application_domain="healthcare"
        )
        assert result["risk_level"] == "high"
        assert any("high-risk domain" in f.lower() for f in result["risk_factors"])

    def test_education_classified_high_risk(self):
        result = self.assessor.assess_risk(
            self.base_metadata,
            use_case="Student exam grading system"
        )
        assert result["risk_level"] == "high"

    def test_law_enforcement_classified_high_risk(self):
        result = self.assessor.assess_risk(
            self.base_metadata,
            application_domain="crime risk_assessment"
        )
        assert result["risk_level"] == "high"

    def test_social_scoring_classified_unacceptable(self):
        result = self.assessor.assess_risk(
            self.base_metadata,
            use_case="social_scoring system for citizens"
        )
        assert result["risk_level"] == "unacceptable"

    def test_subliminal_manipulation_classified_unacceptable(self):
        result = self.assessor.assess_risk(
            self.base_metadata,
            use_case="subliminal_manipulation advertising"
        )
        assert result["risk_level"] == "unacceptable"

    def test_chatbot_classified_limited_risk(self):
        result = self.assessor.assess_risk(
            self.base_metadata,
            use_case="General chatbot for customer service"
        )
        assert result["risk_level"] == "limited"

    def test_chat_model_classified_limited_risk(self):
        metadata = {
            "system_name": "Chat System",
            "models": [{"model_name": "gpt-4", "provider": "OpenAI"}],
            "data_sources": [],
        }
        result = self.assessor.assess_risk(metadata)
        assert result["risk_level"] == "limited"

    def test_generic_system_classified_minimal(self):
        metadata = {
            "system_name": "Generic System",
            "models": [{"model_name": "custom-model", "provider": "internal"}],
            "data_sources": [],
        }
        result = self.assessor.assess_risk(metadata, use_case="internal data processing")
        assert result["risk_level"] == "minimal"

    def test_high_risk_returns_article_requirements(self):
        result = self.assessor.assess_risk(
            self.base_metadata,
            application_domain="healthcare"
        )
        assert len(result["compliance_requirements"]) > 0
        assert any("Article 9" in r for r in result["compliance_requirements"])
        assert any("Article 11" in r for r in result["compliance_requirements"])

    def test_result_contains_timestamp(self):
        result = self.assessor.assess_risk(self.base_metadata)
        assert "timestamp" in result

    def test_result_contains_confidence(self):
        result = self.assessor.assess_risk(
            self.base_metadata, application_domain="healthcare"
        )
        assert 0 < result["confidence"] <= 1.0

    def test_missing_models_lowers_confidence(self):
        empty_metadata = {"system_name": "Empty", "models": [], "data_sources": []}
        result = self.assessor.assess_risk(empty_metadata)
        full_result = self.assessor.assess_risk(self.base_metadata)
        assert result["confidence"] < full_result["confidence"]

    def test_sensitive_data_flagged_as_risk_factor(self):
        metadata = {
            "system_name": "Test",
            "models": [{"model_name": "m", "provider": "p"}],
            "data_sources": [{"data_source": "patient_medical_records.csv"}],
        }
        result = self.assessor.assess_risk(metadata)
        assert any("sensitive" in f.lower() for f in result["risk_factors"])

    def test_high_capability_model_flagged(self):
        metadata = {
            "system_name": "Test",
            "models": [{"model_name": "gpt-4-turbo", "provider": "OpenAI"}],
            "data_sources": [],
        }
        result = self.assessor.assess_risk(metadata)
        assert any("high-capability" in f.lower() for f in result["risk_factors"])

    def test_generate_risk_report(self):
        assessment = self.assessor.assess_risk(
            self.base_metadata, application_domain="healthcare"
        )
        report = self.assessor.generate_risk_report(self.base_metadata, assessment)
        assert report["system_name"] == "Test System"
        assert report["risk_level"] == "high"
        assert "models" in report
        assert "data_sources" in report

    def test_no_use_case_defaults_to_metadata_analysis(self):
        result = self.assessor.assess_risk(self.base_metadata)
        assert result["risk_level"] in ["minimal", "limited", "high", "unacceptable"]
        assert len(result["recommendations"]) > 0
