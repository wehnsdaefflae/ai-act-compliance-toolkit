"""
Tests for Model Card Generator Module
"""

import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aiact_toolkit.model_card import (
    ModelCard,
    ModelCardGenerator,
    ModelDetails,
    IntendedUse,
    PerformanceMetric,
    TrainingData,
    RegulatoryCompliance,
    generate_model_cards_for_all_models,
)


def _sample_metadata():
    """Return metadata dict for tests."""
    return {
        "system_name": "Test Medical Chatbot",
        "created_at": "2025-01-01T00:00:00",
        "models": [
            {
                "model_name": "gpt-4",
                "provider": "OpenAI",
                "parameters": {"temperature": 0.3, "max_tokens": 1000},
            },
            {
                "model_name": "sentence-transformers",
                "provider": "HuggingFace",
                "parameters": {"device": "cpu"},
            },
        ],
        "data_sources": [
            {"source": "medical_qa.csv", "data_type": "csv", "personal_data": True},
        ],
        "risk_assessment": {
            "risk_level": "high",
            "risk_factors": ["Healthcare application"],
            "recommendations": ["Human oversight required"],
        },
        "data_governance": {"data_sources": [{"name": "medical_qa"}]},
        "operational_metrics": {
            "performance": {"avg_execution_time_ms": 250.5},
            "operations": {"error_rate_percent": 1.2},
        },
    }


def test_generate_from_metadata():
    """Test basic model card generation from metadata."""
    generator = ModelCardGenerator()
    card = generator.generate_from_metadata(_sample_metadata())

    assert isinstance(card, ModelCard)
    assert card.model_details.name == "gpt-4"
    assert card.model_details.model_type == "machine_learning_model"
    assert card.intended_use is not None
    assert card.regulatory_compliance is not None
    assert card.regulatory_compliance.risk_level == "high"
    assert card.regulatory_compliance.eu_ai_act_category == "High-Risk AI System"
    print("✓ test_generate_from_metadata passed")


def test_generate_specific_model():
    """Test generating a card for a specific model by name."""
    generator = ModelCardGenerator()
    card = generator.generate_from_metadata(_sample_metadata(), model_name="sentence-transformers")

    assert card.model_details.name == "sentence-transformers"
    print("✓ test_generate_specific_model passed")


def test_generate_model_not_found():
    """Test error when requesting a nonexistent model."""
    generator = ModelCardGenerator()
    try:
        generator.generate_from_metadata(_sample_metadata(), model_name="nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "nonexistent" in str(e)
    print("✓ test_generate_model_not_found passed")


def test_generate_empty_metadata():
    """Test generation with minimal metadata."""
    generator = ModelCardGenerator()
    card = generator.generate_from_metadata({"system_name": "Empty"})

    assert card.model_details.name == "Unknown Model"
    assert card.training_data is None
    assert card.performance == []
    print("✓ test_generate_empty_metadata passed")


def test_to_dict_and_json():
    """Test serialization to dict and JSON."""
    generator = ModelCardGenerator()
    card = generator.generate_from_metadata(_sample_metadata())

    d = card.to_dict()
    assert isinstance(d, dict)
    assert d["model_details"]["name"] == "gpt-4"

    j = card.to_json()
    parsed = json.loads(j)
    assert parsed["model_details"]["name"] == "gpt-4"
    print("✓ test_to_dict_and_json passed")


def test_save_json():
    """Test saving model card to a JSON file."""
    generator = ModelCardGenerator()
    card = generator.generate_from_metadata(_sample_metadata())

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        card.save_json(path)
        with open(path, "r") as f:
            data = json.load(f)
        assert data["model_details"]["name"] == "gpt-4"
    finally:
        os.unlink(path)
    print("✓ test_save_json passed")


def test_generate_all_models():
    """Test generating cards for all models in metadata."""
    cards = generate_model_cards_for_all_models(_sample_metadata())
    assert len(cards) == 2
    names = {c.model_details.name for c in cards}
    assert "gpt-4" in names
    assert "sentence-transformers" in names
    print("✓ test_generate_all_models passed")


def test_generate_all_models_empty():
    """Test generating cards when no models exist."""
    cards = generate_model_cards_for_all_models({"system_name": "Empty"})
    assert len(cards) == 1  # Generates one card for the system
    print("✓ test_generate_all_models_empty passed")


def test_performance_metrics_extraction():
    """Test that operational metrics are extracted into performance section."""
    generator = ModelCardGenerator()
    card = generator.generate_from_metadata(_sample_metadata())

    assert len(card.performance) > 0
    metric_names = [m.metric_name for m in card.performance]
    assert "Average Execution Time" in metric_names
    print("✓ test_performance_metrics_extraction passed")


def test_training_data_extraction():
    """Test that data sources are captured in training data section."""
    generator = ModelCardGenerator()
    card = generator.generate_from_metadata(_sample_metadata())

    assert card.training_data is not None
    assert card.training_data.personal_data is True
    assert len(card.training_data.data_sources) == 1
    print("✓ test_training_data_extraction passed")


def test_regulatory_compliance_section():
    """Test regulatory compliance section population."""
    generator = ModelCardGenerator()
    card = generator.generate_from_metadata(_sample_metadata())

    reg = card.regulatory_compliance
    assert reg is not None
    assert reg.risk_level == "high"
    assert reg.article_13_transparency is True
    assert len(reg.compliance_documentation) > 0
    print("✓ test_regulatory_compliance_section passed")


def test_ethical_considerations():
    """Test ethical considerations section."""
    generator = ModelCardGenerator()
    card = generator.generate_from_metadata(_sample_metadata())

    assert card.ethical_considerations is not None
    assert len(card.ethical_considerations.risks) > 0
    assert len(card.ethical_considerations.privacy_measures) > 0
    print("✓ test_ethical_considerations passed")


def test_limitations_section():
    """Test limitations section includes risk recommendations."""
    generator = ModelCardGenerator()
    card = generator.generate_from_metadata(_sample_metadata())

    assert card.limitations is not None
    assert len(card.limitations.recommendations) > 0
    print("✓ test_limitations_section passed")
