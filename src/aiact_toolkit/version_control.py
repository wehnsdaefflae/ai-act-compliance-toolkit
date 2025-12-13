"""
Version Control Module

Provides version control for AI system metadata to track changes over time.
Essential for EU AI Act compliance to maintain historical records of system
configurations, models, and risk assessments.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import copy
from pathlib import Path


class MetadataVersion:
    """
    Represents a specific version of system metadata.

    Each version captures the complete state of the system at a point in time,
    allowing for comparison and rollback capabilities.
    """

    def __init__(
        self,
        version_number: int,
        metadata: Dict[str, Any],
        change_description: str = "",
        changed_by: str = "system"
    ):
        """
        Initialize a metadata version.

        Args:
            version_number: Sequential version number
            metadata: Complete metadata snapshot
            change_description: Description of what changed in this version
            changed_by: Who/what made the change
        """
        self.version_number = version_number
        self.timestamp = datetime.now().isoformat()
        self.metadata = copy.deepcopy(metadata)
        self.change_description = change_description
        self.changed_by = changed_by

    def to_dict(self) -> Dict[str, Any]:
        """Convert version to dictionary for serialization."""
        return {
            "version_number": self.version_number,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "change_description": self.change_description,
            "changed_by": self.changed_by
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetadataVersion':
        """Reconstruct version from dictionary."""
        version = cls(
            version_number=data["version_number"],
            metadata=data["metadata"],
            change_description=data.get("change_description", ""),
            changed_by=data.get("changed_by", "system")
        )
        version.timestamp = data["timestamp"]
        return version


class VersionControl:
    """
    Manages version history of AI system metadata.

    Provides Git-like version control for compliance metadata,
    enabling tracking of all changes over time.
    """

    def __init__(self, system_name: str):
        """
        Initialize version control for a system.

        Args:
            system_name: Name of the AI system
        """
        self.system_name = system_name
        self.versions: List[MetadataVersion] = []
        self.current_version = 0
        self.created_at = datetime.now().isoformat()

    def commit(
        self,
        metadata: Dict[str, Any],
        description: str = "",
        author: str = "system"
    ) -> MetadataVersion:
        """
        Create a new version of the metadata.

        Args:
            metadata: Current metadata to version
            description: Description of changes
            author: Who made the changes

        Returns:
            The created MetadataVersion
        """
        self.current_version += 1
        version = MetadataVersion(
            version_number=self.current_version,
            metadata=metadata,
            change_description=description,
            changed_by=author
        )
        self.versions.append(version)
        return version

    def get_version(self, version_number: int) -> Optional[MetadataVersion]:
        """
        Get a specific version by number.

        Args:
            version_number: Version number to retrieve

        Returns:
            MetadataVersion if found, None otherwise
        """
        for version in self.versions:
            if version.version_number == version_number:
                return version
        return None

    def get_latest_version(self) -> Optional[MetadataVersion]:
        """Get the most recent version."""
        if not self.versions:
            return None
        return self.versions[-1]

    def get_version_history(self) -> List[Dict[str, Any]]:
        """
        Get summary of all versions.

        Returns:
            List of version summaries
        """
        return [
            {
                "version": v.version_number,
                "timestamp": v.timestamp,
                "description": v.change_description,
                "changed_by": v.changed_by
            }
            for v in self.versions
        ]

    def compare_versions(
        self,
        version1: int,
        version2: int
    ) -> Dict[str, Any]:
        """
        Compare two versions and identify differences.

        Args:
            version1: First version number
            version2: Second version number

        Returns:
            Dictionary describing differences between versions
        """
        v1 = self.get_version(version1)
        v2 = self.get_version(version2)

        if not v1 or not v2:
            return {
                "error": "One or both versions not found",
                "version1_exists": v1 is not None,
                "version2_exists": v2 is not None
            }

        differences = {
            "version1": version1,
            "version2": version2,
            "timestamp1": v1.timestamp,
            "timestamp2": v2.timestamp,
            "changes": []
        }

        # Compare models
        models1 = {m.get("model_name"): m for m in v1.metadata.get("models", [])}
        models2 = {m.get("model_name"): m for m in v2.metadata.get("models", [])}

        for model_name in set(models1.keys()) | set(models2.keys()):
            if model_name not in models1:
                differences["changes"].append({
                    "type": "model_added",
                    "model_name": model_name,
                    "details": models2[model_name]
                })
            elif model_name not in models2:
                differences["changes"].append({
                    "type": "model_removed",
                    "model_name": model_name,
                    "details": models1[model_name]
                })
            elif models1[model_name] != models2[model_name]:
                differences["changes"].append({
                    "type": "model_modified",
                    "model_name": model_name,
                    "old": models1[model_name],
                    "new": models2[model_name]
                })

        # Compare data sources
        sources1 = set(str(s.get("data_source")) for s in v1.metadata.get("data_sources", []))
        sources2 = set(str(s.get("data_source")) for s in v2.metadata.get("data_sources", []))

        for source in sources2 - sources1:
            differences["changes"].append({
                "type": "data_source_added",
                "data_source": source
            })

        for source in sources1 - sources2:
            differences["changes"].append({
                "type": "data_source_removed",
                "data_source": source
            })

        # Compare risk assessments
        risk1 = v1.metadata.get("risk_assessment", {})
        risk2 = v2.metadata.get("risk_assessment", {})

        if risk1.get("risk_level") != risk2.get("risk_level"):
            differences["changes"].append({
                "type": "risk_level_changed",
                "old_level": risk1.get("risk_level"),
                "new_level": risk2.get("risk_level")
            })

        differences["total_changes"] = len(differences["changes"])
        return differences

    def get_changes_since_version(self, since_version: int) -> Dict[str, Any]:
        """
        Get all changes since a specific version.

        Args:
            since_version: Version number to compare from

        Returns:
            Summary of all changes since that version
        """
        if not self.versions or since_version >= self.current_version:
            return {
                "since_version": since_version,
                "current_version": self.current_version,
                "changes": []
            }

        # Get all versions after the specified version
        subsequent_versions = [v for v in self.versions if v.version_number > since_version]

        changes = []
        for version in subsequent_versions:
            changes.append({
                "version": version.version_number,
                "timestamp": version.timestamp,
                "description": version.change_description,
                "changed_by": version.changed_by
            })

        # Get detailed diff between since_version and current
        detailed_diff = self.compare_versions(since_version, self.current_version)

        return {
            "since_version": since_version,
            "current_version": self.current_version,
            "versions_changed": changes,
            "total_versions": len(changes),
            "detailed_changes": detailed_diff.get("changes", [])
        }

    def rollback_to_version(self, version_number: int) -> Optional[Dict[str, Any]]:
        """
        Get metadata from a specific version (for rollback purposes).

        Note: This doesn't actually change the version history,
        it just returns the metadata from the specified version
        so it can be used to restore the system state.

        Args:
            version_number: Version to rollback to

        Returns:
            Metadata from that version, or None if version not found
        """
        version = self.get_version(version_number)
        if version:
            return copy.deepcopy(version.metadata)
        return None

    def save_to_file(self, filepath: str):
        """
        Save version history to JSON file.

        Args:
            filepath: Path to save the version history
        """
        version_data = {
            "system_name": self.system_name,
            "created_at": self.created_at,
            "current_version": self.current_version,
            "versions": [v.to_dict() for v in self.versions]
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(version_data, f, indent=2, ensure_ascii=False)

    def load_from_file(self, filepath: str):
        """
        Load version history from JSON file.

        Args:
            filepath: Path to the version history file
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.system_name = data.get("system_name", self.system_name)
            self.created_at = data.get("created_at", self.created_at)
            self.current_version = data.get("current_version", 0)
            self.versions = [
                MetadataVersion.from_dict(v_data)
                for v_data in data.get("versions", [])
            ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert version control to dictionary."""
        return {
            "system_name": self.system_name,
            "created_at": self.created_at,
            "current_version": self.current_version,
            "total_versions": len(self.versions),
            "version_history": self.get_version_history()
        }


class VersionControlIntegration:
    """
    Helper class to integrate version control with metadata storage.

    Provides convenient methods for tracking metadata changes automatically.
    """

    def __init__(self, system_name: str):
        """Initialize version control integration."""
        self.version_control = VersionControl(system_name)

    def track_change(
        self,
        metadata: Dict[str, Any],
        description: str,
        author: str = "system"
    ) -> MetadataVersion:
        """
        Track a metadata change.

        Args:
            metadata: Updated metadata
            description: What changed
            author: Who made the change

        Returns:
            The created version
        """
        return self.version_control.commit(metadata, description, author)

    def get_history(self) -> List[Dict[str, Any]]:
        """Get version history."""
        return self.version_control.get_version_history()

    def compare_with_previous(self) -> Optional[Dict[str, Any]]:
        """Compare current version with previous version."""
        if self.version_control.current_version < 2:
            return None

        return self.version_control.compare_versions(
            self.version_control.current_version - 1,
            self.version_control.current_version
        )
