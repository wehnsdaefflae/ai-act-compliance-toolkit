"""
Data Governance Tracking

Basic data tracking and quality monitoring to support EU AI Act Article 10
(Data and data governance) requirements.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum
import json


class DataType(Enum):
    """Types of data used in AI systems."""
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"
    PRODUCTION = "production"


class DataQualityStatus(Enum):
    """Data quality assessment status."""
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNKNOWN = "unknown"


class TransformationType(Enum):
    """Types of data transformations."""
    CLEANING = "cleaning"
    NORMALIZATION = "normalization"
    AUGMENTATION = "augmentation"
    FILTERING = "filtering"
    ANONYMIZATION = "anonymization"


class DataSource:
    """Represents a data source for EU AI Act Article 10 compliance."""

    def __init__(
        self,
        source_id: str,
        name: str,
        description: str,
        data_type: DataType,
        location: Optional[str] = None,
        size_records: Optional[int] = None,
        personal_data: bool = False,
        sensitive_data: bool = False,
    ):
        """Initialize a data source."""
        self.source_id = source_id
        self.name = name
        self.description = description
        self.data_type = data_type
        self.location = location
        self.size_records = size_records
        self.personal_data = personal_data
        self.sensitive_data = sensitive_data
        self.registered_at = datetime.now().isoformat()
        self.quality_status: DataQualityStatus = DataQualityStatus.UNKNOWN

    def set_quality_status(self, status: DataQualityStatus):
        """Set overall quality assessment status."""
        self.quality_status = status

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_id": self.source_id,
            "name": self.name,
            "description": self.description,
            "data_type": self.data_type.value,
            "location": self.location,
            "size_records": self.size_records,
            "personal_data": self.personal_data,
            "sensitive_data": self.sensitive_data,
            "registered_at": self.registered_at,
            "quality_status": self.quality_status.value
        }


class DataTransformation:
    """Represents a transformation applied to data."""

    def __init__(
        self,
        transformation_id: str,
        transformation_type: TransformationType,
        description: str,
        input_source_id: str,
        output_source_id: str
    ):
        """Initialize a data transformation."""
        self.transformation_id = transformation_id
        self.transformation_type = transformation_type
        self.description = description
        self.input_source_id = input_source_id
        self.output_source_id = output_source_id
        self.performed_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "transformation_id": self.transformation_id,
            "transformation_type": self.transformation_type.value,
            "description": self.description,
            "input_source_id": self.input_source_id,
            "output_source_id": self.output_source_id,
            "performed_at": self.performed_at
        }


class DataGovernanceTracker:
    """Tracks data governance for EU AI Act Article 10 compliance."""

    def __init__(self, system_name: str):
        """Initialize data governance tracker."""
        self.system_name = system_name
        self.sources: Dict[str, DataSource] = {}
        self.transformations: List[DataTransformation] = []
        self.created_at = datetime.now().isoformat()

    def register_data_source(
        self,
        source_id: str,
        name: str,
        description: str,
        data_type: DataType,
        **kwargs
    ) -> DataSource:
        """Register a new data source."""
        source = DataSource(source_id, name, description, data_type, **kwargs)
        self.sources[source_id] = source
        return source

    def register_transformation(
        self,
        transformation_id: str,
        transformation_type: TransformationType,
        description: str,
        input_source_id: str,
        output_source_id: str
    ) -> DataTransformation:
        """Register a data transformation."""
        transformation = DataTransformation(
            transformation_id, transformation_type, description,
            input_source_id, output_source_id
        )
        self.transformations.append(transformation)
        return transformation

    def get_data_quality_summary(self) -> Dict[str, Any]:
        """Get summary of data quality across all sources."""
        if not self.sources:
            return {"total_sources": 0, "sources_with_quality_metrics": 0, "quality_distribution": {}}

        quality_counts = {status.value: 0 for status in DataQualityStatus}
        assessed_count = 0
        for source in self.sources.values():
            quality_counts[source.quality_status.value] += 1
            if source.quality_status != DataQualityStatus.UNKNOWN:
                assessed_count += 1

        return {
            "total_sources": len(self.sources),
            "sources_with_quality_metrics": assessed_count,
            "quality_distribution": quality_counts
        }

    def get_privacy_summary(self) -> Dict[str, Any]:
        """Get summary of privacy-related data."""
        sources = list(self.sources.values())
        return {
            "total_sources": len(sources),
            "personal_data_sources": sum(1 for s in sources if s.personal_data),
            "sensitive_data_sources": sum(1 for s in sources if s.sensitive_data),
            "sources_with_license": 0,  # Not tracked in proof-of-concept
            "sources_with_copyright": 0  # Not tracked in proof-of-concept
        }

    def generate_article10_report(self) -> Dict[str, Any]:
        """Generate compliance report for EU AI Act Article 10."""
        # Count sources by type
        sources_by_type = {dt.value: 0 for dt in DataType}
        for src in self.sources.values():
            sources_by_type[src.data_type.value] = sources_by_type.get(src.data_type.value, 0) + 1

        # Count transformations by type
        transforms_by_type = {tt.value: 0 for tt in TransformationType}
        for t in self.transformations:
            transforms_by_type[t.transformation_type.value] = transforms_by_type.get(t.transformation_type.value, 0) + 1

        # Basic compliance checks
        quality_summary = self.get_data_quality_summary()
        checks_passed = sum([
            len(self.sources) > 0,
            quality_summary["sources_with_quality_metrics"] > 0,
            any(s.personal_data for s in self.sources.values()) is False or len(self.sources) > 0,
        ])
        checks_total = 3

        return {
            "system_name": self.system_name,
            "report_generated": datetime.now().isoformat(),
            "article": "EU AI Act Article 10 - Data and Data Governance",
            "data_sources": {
                "total": len(self.sources),
                "by_type": sources_by_type,
                "sources": [s.to_dict() for s in self.sources.values()]
            },
            "transformations": {
                "total": len(self.transformations),
                "by_type": transforms_by_type,
                "transformations": [t.to_dict() for t in self.transformations]
            },
            "data_quality": quality_summary,
            "privacy_compliance": self.get_privacy_summary(),
            "compliance_checks": {
                "total": checks_total,
                "passed": checks_passed,
                "failed": checks_total - checks_passed
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "system_name": self.system_name,
            "created_at": self.created_at,
            "sources": {sid: src.to_dict() for sid, src in self.sources.items()},
            "transformations": [t.to_dict() for t in self.transformations]
        }
