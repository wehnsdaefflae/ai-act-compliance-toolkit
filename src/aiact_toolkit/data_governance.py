"""
Data Governance and Provenance Tracking

Implements data lineage tracking, quality monitoring, and governance
documentation to support EU AI Act Article 10 (Data and data governance)
requirements for high-risk AI systems.
"""

from typing import Any, Dict, List, Optional, Set
from datetime import datetime
from enum import Enum
import hashlib
import json


class DataType(Enum):
    """Types of data used in AI systems."""
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"
    PRODUCTION = "production"
    REFERENCE = "reference"


class DataQualityStatus(Enum):
    """Data quality assessment status."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNKNOWN = "unknown"


class TransformationType(Enum):
    """Types of data transformations."""
    CLEANING = "cleaning"
    NORMALIZATION = "normalization"
    AUGMENTATION = "augmentation"
    FEATURE_EXTRACTION = "feature_extraction"
    SAMPLING = "sampling"
    FILTERING = "filtering"
    ANONYMIZATION = "anonymization"
    AGGREGATION = "aggregation"
    TOKENIZATION = "tokenization"
    EMBEDDING = "embedding"


class DataSource:
    """
    Represents a data source with full provenance information.
    Supports EU AI Act Article 10 requirements for data documentation.
    """

    def __init__(
        self,
        source_id: str,
        name: str,
        description: str,
        data_type: DataType,
        location: Optional[str] = None,
        format: Optional[str] = None,
        size_records: Optional[int] = None,
        size_bytes: Optional[int] = None,
        collection_date: Optional[str] = None,
        source_origin: Optional[str] = None,
        license: Optional[str] = None,
        copyright_info: Optional[str] = None,
        personal_data: bool = False,
        sensitive_data: bool = False,
    ):
        """
        Initialize a data source.

        Args:
            source_id: Unique identifier for the data source
            name: Human-readable name
            description: Detailed description of the data
            data_type: Type of data (training, validation, etc.)
            location: File path, URL, or database location
            format: Data format (CSV, JSON, Parquet, etc.)
            size_records: Number of records/samples
            size_bytes: Size in bytes
            collection_date: When data was collected
            source_origin: Original source of the data
            license: Data license
            copyright_info: Copyright information
            personal_data: Whether contains personal data (GDPR)
            sensitive_data: Whether contains sensitive data
        """
        self.source_id = source_id
        self.name = name
        self.description = description
        self.data_type = data_type
        self.location = location
        self.format = format
        self.size_records = size_records
        self.size_bytes = size_bytes
        self.collection_date = collection_date or datetime.now().isoformat()
        self.source_origin = source_origin
        self.license = license
        self.copyright_info = copyright_info
        self.personal_data = personal_data
        self.sensitive_data = sensitive_data
        self.registered_at = datetime.now().isoformat()

        # Quality metrics
        self.quality_metrics: Dict[str, Any] = {}
        self.quality_status: DataQualityStatus = DataQualityStatus.UNKNOWN

        # Lineage tracking
        self.parent_sources: List[str] = []
        self.transformations: List['DataTransformation'] = []

    def add_quality_metric(self, metric_name: str, value: Any, description: Optional[str] = None):
        """Add a quality metric for this data source."""
        self.quality_metrics[metric_name] = {
            "value": value,
            "description": description,
            "assessed_at": datetime.now().isoformat()
        }

    def set_quality_status(self, status: DataQualityStatus):
        """Set overall quality assessment status."""
        self.quality_status = status

    def add_parent_source(self, parent_id: str):
        """Add a parent data source (for derived datasets)."""
        if parent_id not in self.parent_sources:
            self.parent_sources.append(parent_id)

    def add_transformation(self, transformation: 'DataTransformation'):
        """Add a transformation applied to this data."""
        self.transformations.append(transformation)

    def compute_checksum(self) -> str:
        """Compute checksum for data integrity verification."""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_id": self.source_id,
            "name": self.name,
            "description": self.description,
            "data_type": self.data_type.value,
            "location": self.location,
            "format": self.format,
            "size_records": self.size_records,
            "size_bytes": self.size_bytes,
            "collection_date": self.collection_date,
            "source_origin": self.source_origin,
            "license": self.license,
            "copyright_info": self.copyright_info,
            "personal_data": self.personal_data,
            "sensitive_data": self.sensitive_data,
            "registered_at": self.registered_at,
            "quality_metrics": self.quality_metrics,
            "quality_status": self.quality_status.value,
            "parent_sources": self.parent_sources,
            "transformations": [t.to_dict() for t in self.transformations]
        }


class DataTransformation:
    """
    Represents a transformation applied to data.
    Tracks lineage and maintains audit trail.
    """

    def __init__(
        self,
        transformation_id: str,
        transformation_type: TransformationType,
        description: str,
        input_source_ids: List[str],
        output_source_id: str,
        parameters: Optional[Dict[str, Any]] = None,
        performed_by: Optional[str] = None,
        tool_used: Optional[str] = None,
    ):
        """
        Initialize a data transformation.

        Args:
            transformation_id: Unique identifier
            transformation_type: Type of transformation
            description: What was done
            input_source_ids: Input data source IDs
            output_source_id: Output data source ID
            parameters: Transformation parameters
            performed_by: Who performed the transformation
            tool_used: Tool/library used
        """
        self.transformation_id = transformation_id
        self.transformation_type = transformation_type
        self.description = description
        self.input_source_ids = input_source_ids
        self.output_source_id = output_source_id
        self.parameters = parameters or {}
        self.performed_by = performed_by or "system"
        self.tool_used = tool_used
        self.performed_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "transformation_id": self.transformation_id,
            "transformation_type": self.transformation_type.value,
            "description": self.description,
            "input_source_ids": self.input_source_ids,
            "output_source_id": self.output_source_id,
            "parameters": self.parameters,
            "performed_by": self.performed_by,
            "tool_used": self.tool_used,
            "performed_at": self.performed_at
        }


class DataLineageGraph:
    """
    Tracks complete data lineage from source to model.
    Maintains a directed acyclic graph of data transformations.
    """

    def __init__(self):
        """Initialize lineage graph."""
        self.sources: Dict[str, DataSource] = {}
        self.transformations: Dict[str, DataTransformation] = {}

    def add_source(self, source: DataSource):
        """Add a data source to the lineage graph."""
        self.sources[source.source_id] = source

    def add_transformation(self, transformation: DataTransformation):
        """Add a transformation to the lineage graph."""
        self.transformations[transformation.transformation_id] = transformation

        # Update output source with transformation and parent references
        if transformation.output_source_id in self.sources:
            output_source = self.sources[transformation.output_source_id]
            output_source.add_transformation(transformation)
            for input_id in transformation.input_source_ids:
                output_source.add_parent_source(input_id)

    def get_source_lineage(self, source_id: str) -> List[DataSource]:
        """
        Get complete lineage for a data source (all ancestors).

        Args:
            source_id: Source to trace

        Returns:
            List of ancestor data sources in chronological order
        """
        if source_id not in self.sources:
            return []

        lineage = []
        visited = set()

        def trace_lineage(sid: str):
            if sid in visited or sid not in self.sources:
                return
            visited.add(sid)
            source = self.sources[sid]

            # Trace parents first (depth-first)
            for parent_id in source.parent_sources:
                trace_lineage(parent_id)

            lineage.append(source)

        trace_lineage(source_id)
        return lineage

    def get_transformation_chain(self, source_id: str) -> List[DataTransformation]:
        """Get all transformations that led to this data source."""
        lineage = self.get_source_lineage(source_id)
        transformations = []

        for source in lineage:
            transformations.extend(source.transformations)

        return transformations

    def to_dict(self) -> Dict[str, Any]:
        """Convert lineage graph to dictionary."""
        return {
            "sources": {sid: src.to_dict() for sid, src in self.sources.items()},
            "transformations": {tid: t.to_dict() for tid, t in self.transformations.items()}
        }


class DataGovernanceTracker:
    """
    Main class for tracking data governance and compliance.
    Implements EU AI Act Article 10 requirements.
    """

    def __init__(self, system_name: str):
        """
        Initialize data governance tracker.

        Args:
            system_name: Name of the AI system
        """
        self.system_name = system_name
        self.lineage_graph = DataLineageGraph()
        self.governance_policies: Dict[str, Any] = {}
        self.compliance_checks: List[Dict[str, Any]] = []
        self.created_at = datetime.now().isoformat()

    def register_data_source(
        self,
        source_id: str,
        name: str,
        description: str,
        data_type: DataType,
        **kwargs
    ) -> DataSource:
        """
        Register a new data source.

        Args:
            source_id: Unique identifier
            name: Human-readable name
            description: Description
            data_type: Type of data
            **kwargs: Additional DataSource parameters

        Returns:
            Created DataSource object
        """
        source = DataSource(
            source_id=source_id,
            name=name,
            description=description,
            data_type=data_type,
            **kwargs
        )
        self.lineage_graph.add_source(source)
        return source

    def register_transformation(
        self,
        transformation_id: str,
        transformation_type: TransformationType,
        description: str,
        input_source_ids: List[str],
        output_source_id: str,
        **kwargs
    ) -> DataTransformation:
        """
        Register a data transformation.

        Args:
            transformation_id: Unique identifier
            transformation_type: Type of transformation
            description: What was done
            input_source_ids: Input data source IDs
            output_source_id: Output data source ID
            **kwargs: Additional DataTransformation parameters

        Returns:
            Created DataTransformation object
        """
        transformation = DataTransformation(
            transformation_id=transformation_id,
            transformation_type=transformation_type,
            description=description,
            input_source_ids=input_source_ids,
            output_source_id=output_source_id,
            **kwargs
        )
        self.lineage_graph.add_transformation(transformation)
        return transformation

    def add_governance_policy(self, policy_name: str, policy_details: Dict[str, Any]):
        """Add a data governance policy."""
        self.governance_policies[policy_name] = {
            "details": policy_details,
            "added_at": datetime.now().isoformat()
        }

    def run_compliance_check(
        self,
        check_name: str,
        check_type: str,
        passed: bool,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Record a compliance check result.

        Args:
            check_name: Name of the check
            check_type: Type (e.g., "data_quality", "privacy", "bias")
            passed: Whether check passed
            details: Additional details
        """
        self.compliance_checks.append({
            "check_name": check_name,
            "check_type": check_type,
            "passed": passed,
            "details": details or {},
            "performed_at": datetime.now().isoformat()
        })

    def get_data_quality_summary(self) -> Dict[str, Any]:
        """Get summary of data quality across all sources."""
        total_sources = len(self.lineage_graph.sources)
        if total_sources == 0:
            return {"total_sources": 0, "quality_distribution": {}}

        quality_counts = {status.value: 0 for status in DataQualityStatus}

        for source in self.lineage_graph.sources.values():
            quality_counts[source.quality_status.value] += 1

        return {
            "total_sources": total_sources,
            "quality_distribution": quality_counts,
            "sources_with_quality_metrics": sum(
                1 for s in self.lineage_graph.sources.values()
                if s.quality_metrics
            )
        }

    def get_privacy_summary(self) -> Dict[str, Any]:
        """Get summary of privacy-related data."""
        sources = list(self.lineage_graph.sources.values())
        return {
            "total_sources": len(sources),
            "personal_data_sources": sum(1 for s in sources if s.personal_data),
            "sensitive_data_sources": sum(1 for s in sources if s.sensitive_data),
            "sources_with_license": sum(1 for s in sources if s.license),
            "sources_with_copyright": sum(1 for s in sources if s.copyright_info)
        }

    def get_lineage_report(self, source_id: str) -> Dict[str, Any]:
        """
        Generate a complete lineage report for a data source.

        Args:
            source_id: Source to report on

        Returns:
            Lineage report dictionary
        """
        if source_id not in self.lineage_graph.sources:
            return {"error": "Source not found"}

        source = self.lineage_graph.sources[source_id]
        lineage = self.lineage_graph.get_source_lineage(source_id)
        transformations = self.lineage_graph.get_transformation_chain(source_id)

        return {
            "source": source.to_dict(),
            "lineage_depth": len(lineage),
            "ancestor_sources": [s.to_dict() for s in lineage if s.source_id != source_id],
            "transformations": [t.to_dict() for t in transformations],
            "total_transformations": len(transformations)
        }

    def generate_article10_report(self) -> Dict[str, Any]:
        """
        Generate compliance report for EU AI Act Article 10
        (Data and data governance).

        Returns:
            Article 10 compliance report
        """
        return {
            "system_name": self.system_name,
            "report_generated": datetime.now().isoformat(),
            "article": "EU AI Act Article 10 - Data and Data Governance",
            "data_sources": {
                "total": len(self.lineage_graph.sources),
                "by_type": self._count_sources_by_type(),
                "sources": [s.to_dict() for s in self.lineage_graph.sources.values()]
            },
            "transformations": {
                "total": len(self.lineage_graph.transformations),
                "by_type": self._count_transformations_by_type(),
                "transformations": [t.to_dict() for t in self.lineage_graph.transformations.values()]
            },
            "data_quality": self.get_data_quality_summary(),
            "privacy_compliance": self.get_privacy_summary(),
            "governance_policies": self.governance_policies,
            "compliance_checks": {
                "total": len(self.compliance_checks),
                "passed": sum(1 for c in self.compliance_checks if c["passed"]),
                "failed": sum(1 for c in self.compliance_checks if not c["passed"]),
                "checks": self.compliance_checks
            }
        }

    def _count_sources_by_type(self) -> Dict[str, int]:
        """Count data sources by type."""
        counts = {dt.value: 0 for dt in DataType}
        for source in self.lineage_graph.sources.values():
            counts[source.data_type.value] += 1
        return counts

    def _count_transformations_by_type(self) -> Dict[str, int]:
        """Count transformations by type."""
        counts = {tt.value: 0 for tt in TransformationType}
        for transformation in self.lineage_graph.transformations.values():
            counts[transformation.transformation_type.value] += 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "system_name": self.system_name,
            "created_at": self.created_at,
            "lineage_graph": self.lineage_graph.to_dict(),
            "governance_policies": self.governance_policies,
            "compliance_checks": self.compliance_checks
        }

    def save_to_file(self, filepath: str):
        """Save data governance information to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'DataGovernanceTracker':
        """Load data governance information from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tracker = cls(system_name=data["system_name"])
        tracker.created_at = data["created_at"]

        # Restore lineage graph
        for source_data in data["lineage_graph"]["sources"].values():
            source = DataSource(
                source_id=source_data["source_id"],
                name=source_data["name"],
                description=source_data["description"],
                data_type=DataType(source_data["data_type"]),
                location=source_data.get("location"),
                format=source_data.get("format"),
                size_records=source_data.get("size_records"),
                size_bytes=source_data.get("size_bytes"),
                collection_date=source_data.get("collection_date"),
                source_origin=source_data.get("source_origin"),
                license=source_data.get("license"),
                copyright_info=source_data.get("copyright_info"),
                personal_data=source_data.get("personal_data", False),
                sensitive_data=source_data.get("sensitive_data", False)
            )
            source.quality_metrics = source_data.get("quality_metrics", {})
            source.quality_status = DataQualityStatus(source_data.get("quality_status", "unknown"))
            source.parent_sources = source_data.get("parent_sources", [])
            tracker.lineage_graph.add_source(source)

        # Restore transformations
        for trans_data in data["lineage_graph"]["transformations"].values():
            transformation = DataTransformation(
                transformation_id=trans_data["transformation_id"],
                transformation_type=TransformationType(trans_data["transformation_type"]),
                description=trans_data["description"],
                input_source_ids=trans_data["input_source_ids"],
                output_source_id=trans_data["output_source_id"],
                parameters=trans_data.get("parameters"),
                performed_by=trans_data.get("performed_by"),
                tool_used=trans_data.get("tool_used")
            )
            tracker.lineage_graph.add_transformation(transformation)

        tracker.governance_policies = data.get("governance_policies", {})
        tracker.compliance_checks = data.get("compliance_checks", [])

        return tracker
