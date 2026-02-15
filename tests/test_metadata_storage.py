"""Tests for the Metadata Storage module."""

import json
import os
import tempfile
import pytest
from aiact_toolkit.metadata_storage import MetadataStorage


class TestMetadataStorage:
    """Test MetadataStorage class."""

    def setup_method(self):
        self.storage = MetadataStorage(system_name="test_system")

    def test_initialization(self):
        assert self.storage.system_name == "test_system"
        assert self.storage.models == []
        assert self.storage.components == []
        assert self.storage.data_sources == []
        assert self.storage.audit_trail is not None

    def test_initialization_without_auditing(self):
        storage = MetadataStorage(system_name="test", enable_auditing=False)
        assert storage.audit_trail is None

    def test_add_model(self):
        model_info = {"model_name": "gpt-4", "provider": "OpenAI", "parameters": {}}
        self.storage.add_model(model_info)
        assert len(self.storage.models) == 1
        assert self.storage.models[0]["model_name"] == "gpt-4"

    def test_add_duplicate_model_ignored(self):
        model_info = {"model_name": "gpt-4", "provider": "OpenAI", "parameters": {}}
        self.storage.add_model(model_info)
        self.storage.add_model(model_info)
        assert len(self.storage.models) == 1

    def test_add_different_models(self):
        self.storage.add_model({"model_name": "gpt-4", "provider": "OpenAI", "parameters": {}})
        self.storage.add_model({"model_name": "claude", "provider": "Anthropic", "parameters": {}})
        assert len(self.storage.models) == 2

    def test_add_component(self):
        comp = {"chain_type": "LLMChain", "name": "test_chain"}
        self.storage.add_component(comp)
        assert len(self.storage.components) == 1

    def test_add_data_source(self):
        source = {"data_source": "data.csv", "loader_type": "CSVLoader"}
        self.storage.add_data_source(source)
        assert len(self.storage.data_sources) == 1

    def test_add_duplicate_data_source_ignored(self):
        source = {"data_source": "data.csv"}
        self.storage.add_data_source(source)
        self.storage.add_data_source(source)
        assert len(self.storage.data_sources) == 1

    def test_set_risk_assessment(self):
        risk = {"risk_level": "high", "confidence": 0.8}
        self.storage.set_risk_assessment(risk)
        assert self.storage.risk_assessment["risk_level"] == "high"

    def test_set_operational_metrics(self):
        metrics = {"operations": {"total": 100}}
        self.storage.set_operational_metrics(metrics)
        assert self.storage.operational_metrics["operations"]["total"] == 100

    def test_add_bias_analysis(self):
        analysis = {"risk_level": "low", "fairness_score": 0.95}
        self.storage.add_bias_analysis(analysis)
        assert len(self.storage.bias_analyses) == 1

    def test_get_all_metadata(self):
        self.storage.add_model({"model_name": "test", "provider": "p", "parameters": {}})
        self.storage.add_data_source({"data_source": "data.csv"})
        metadata = self.storage.get_all_metadata()
        assert metadata["system_name"] == "test_system"
        assert len(metadata["models"]) == 1
        assert len(metadata["data_sources"]) == 1
        assert "audit_summary" in metadata
        assert "created_at" in metadata

    def test_get_all_metadata_includes_risk(self):
        self.storage.set_risk_assessment({"risk_level": "high"})
        metadata = self.storage.get_all_metadata()
        assert "risk_assessment" in metadata

    def test_get_all_metadata_includes_bias(self):
        self.storage.add_bias_analysis({"risk_level": "low"})
        metadata = self.storage.get_all_metadata()
        assert "bias_analyses" in metadata

    def test_clear(self):
        self.storage.add_model({"model_name": "test", "provider": "p", "parameters": {}})
        self.storage.add_data_source({"data_source": "data.csv"})
        self.storage.set_risk_assessment({"risk_level": "high"})
        self.storage.clear()
        assert self.storage.models == []
        assert self.storage.data_sources == []
        assert self.storage.risk_assessment == {}

    def test_save_and_load(self):
        self.storage.add_model({"model_name": "gpt-4", "provider": "OpenAI", "parameters": {"temperature": 0.7}})
        self.storage.add_data_source({"data_source": "test.csv"})
        self.storage.set_risk_assessment({"risk_level": "high"})

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            self.storage.save_to_file(filepath)

            loaded = MetadataStorage(system_name="different")
            loaded.load_from_file(filepath)

            assert loaded.system_name == "test_system"
            assert len(loaded.models) == 1
            assert loaded.models[0]["model_name"] == "gpt-4"
            assert loaded.risk_assessment["risk_level"] == "high"
        finally:
            os.unlink(filepath)

    def test_deduplicate_models(self):
        self.storage.models = [
            {"model_name": "gpt-4", "provider": "OpenAI", "parameters": {}},
            {"model_name": "gpt-4", "provider": "OpenAI", "parameters": {}},
            {"model_name": "claude", "provider": "Anthropic", "parameters": {}},
        ]
        deduplicated = self.storage._deduplicate_models()
        assert len(deduplicated) == 2

    def test_deduplicate_components(self):
        self.storage.components = [
            {"chain_type": "LLMChain"},
            {"chain_type": "LLMChain"},
            {"chain_type": "RetrievalQA"},
        ]
        deduplicated = self.storage._deduplicate_components()
        assert len(deduplicated) == 2

    def test_get_audit_trail(self):
        trail = self.storage.get_audit_trail()
        assert trail is not None
        assert len(trail.events) > 0  # system_created event

    def test_get_data_governance_tracker(self):
        tracker = self.storage.get_data_governance_tracker()
        assert tracker is not None
        assert tracker.system_name == "test_system"

    def test_extra_metadata_preserved_on_load(self):
        self.storage.metadata["framework"] = "pytorch"
        self.storage.metadata["training_history"] = [{"epoch": 1}]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            self.storage.save_to_file(filepath)

            loaded = MetadataStorage()
            loaded.load_from_file(filepath)
            assert loaded.metadata["framework"] == "pytorch"
            assert loaded.metadata["training_history"] == [{"epoch": 1}]
        finally:
            os.unlink(filepath)

    def test_audit_events_recorded_for_model_add(self):
        initial_count = len(self.storage.audit_trail.events)
        self.storage.add_model({"model_name": "test", "provider": "p", "parameters": {}})
        assert len(self.storage.audit_trail.events) == initial_count + 1

    def test_audit_events_recorded_for_risk_assessment(self):
        initial_count = len(self.storage.audit_trail.events)
        self.storage.set_risk_assessment({"risk_level": "high"})
        assert len(self.storage.audit_trail.events) == initial_count + 1
