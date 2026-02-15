"""Tests for the Data Governance module."""

import pytest
from aiact_toolkit.data_governance import (
    DataType, DataQualityStatus, TransformationType,
    DataSource, DataTransformation, DataGovernanceTracker
)


class TestDataSource:
    """Test DataSource class."""

    def test_creation(self):
        source = DataSource(
            source_id="src-001",
            name="Training Data",
            description="Main training dataset",
            data_type=DataType.TRAINING,
        )
        assert source.source_id == "src-001"
        assert source.data_type == DataType.TRAINING
        assert source.personal_data is False
        assert source.quality_status == DataQualityStatus.UNKNOWN

    def test_personal_data_flag(self):
        source = DataSource(
            source_id="src-002",
            name="User Data",
            description="User records",
            data_type=DataType.PRODUCTION,
            personal_data=True,
            sensitive_data=True,
        )
        assert source.personal_data is True
        assert source.sensitive_data is True

    def test_set_quality_status(self):
        source = DataSource("s1", "test", "desc", DataType.TRAINING)
        source.set_quality_status(DataQualityStatus.GOOD)
        assert source.quality_status == DataQualityStatus.GOOD

    def test_to_dict(self):
        source = DataSource("s1", "test", "desc", DataType.TRAINING, size_records=1000)
        d = source.to_dict()
        assert d["source_id"] == "s1"
        assert d["data_type"] == "training"
        assert d["size_records"] == 1000
        assert "registered_at" in d


class TestDataTransformation:
    """Test DataTransformation class."""

    def test_creation(self):
        t = DataTransformation(
            transformation_id="t-001",
            transformation_type=TransformationType.CLEANING,
            description="Remove duplicates",
            input_source_id="src-001",
            output_source_id="src-002",
        )
        assert t.transformation_type == TransformationType.CLEANING

    def test_to_dict(self):
        t = DataTransformation("t1", TransformationType.ANONYMIZATION, "Anonymize PII", "s1", "s2")
        d = t.to_dict()
        assert d["transformation_type"] == "anonymization"
        assert d["input_source_id"] == "s1"


class TestDataGovernanceTracker:
    """Test DataGovernanceTracker class."""

    def setup_method(self):
        self.tracker = DataGovernanceTracker("test_system")

    def test_register_data_source(self):
        source = self.tracker.register_data_source(
            "src-001", "Training Data", "Main dataset", DataType.TRAINING
        )
        assert source.source_id == "src-001"
        assert "src-001" in self.tracker.sources

    def test_register_transformation(self):
        self.tracker.register_data_source("s1", "Raw", "Raw data", DataType.TRAINING)
        self.tracker.register_data_source("s2", "Clean", "Cleaned data", DataType.TRAINING)
        t = self.tracker.register_transformation(
            "t1", TransformationType.CLEANING, "Remove nulls", "s1", "s2"
        )
        assert len(self.tracker.transformations) == 1
        assert t.input_source_id == "s1"

    def test_quality_summary_empty(self):
        summary = self.tracker.get_data_quality_summary()
        assert summary["total_sources"] == 0

    def test_quality_summary_with_sources(self):
        src = self.tracker.register_data_source("s1", "Data", "desc", DataType.TRAINING)
        src.set_quality_status(DataQualityStatus.GOOD)
        self.tracker.register_data_source("s2", "Data2", "desc", DataType.VALIDATION)

        summary = self.tracker.get_data_quality_summary()
        assert summary["total_sources"] == 2
        assert summary["sources_with_quality_metrics"] == 1
        assert summary["quality_distribution"]["good"] == 1

    def test_privacy_summary(self):
        self.tracker.register_data_source(
            "s1", "User Data", "desc", DataType.TRAINING,
            personal_data=True, sensitive_data=True
        )
        self.tracker.register_data_source(
            "s2", "Public Data", "desc", DataType.TRAINING
        )

        summary = self.tracker.get_privacy_summary()
        assert summary["personal_data_sources"] == 1
        assert summary["sensitive_data_sources"] == 1
        assert summary["total_sources"] == 2

    def test_article10_report(self):
        self.tracker.register_data_source("s1", "Data", "desc", DataType.TRAINING)
        self.tracker.register_transformation(
            "t1", TransformationType.NORMALIZATION, "Normalize", "s1", "s2"
        )

        report = self.tracker.generate_article10_report()
        assert report["system_name"] == "test_system"
        assert report["data_sources"]["total"] == 1
        assert report["transformations"]["total"] == 1
        assert "compliance_checks" in report
        assert report["compliance_checks"]["total"] == 3

    def test_to_dict(self):
        self.tracker.register_data_source("s1", "Data", "desc", DataType.TRAINING)
        d = self.tracker.to_dict()
        assert d["system_name"] == "test_system"
        assert "s1" in d["sources"]
        assert "created_at" in d
