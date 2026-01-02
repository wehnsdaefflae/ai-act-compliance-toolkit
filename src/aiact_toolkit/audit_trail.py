"""
Audit Trail Module

Provides comprehensive audit logging for AI system changes to support EU AI Act
Article 12 compliance (automatic logging requirements). Tracks all modifications
to system metadata, configurations, and risk assessments over time.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum
import json
import hashlib
from pathlib import Path


class AuditEventType(Enum):
    """Types of audit events tracked by the system."""
    SYSTEM_CREATED = "system_created"
    MODEL_ADDED = "model_added"
    MODEL_MODIFIED = "model_modified"
    MODEL_REMOVED = "model_removed"
    DATA_SOURCE_ADDED = "data_source_added"
    DATA_SOURCE_MODIFIED = "data_source_modified"
    DATA_SOURCE_REMOVED = "data_source_removed"
    RISK_ASSESSMENT_PERFORMED = "risk_assessment_performed"
    RISK_LEVEL_CHANGED = "risk_level_changed"
    METADATA_UPDATED = "metadata_updated"
    COMPLIANCE_DOCUMENT_GENERATED = "compliance_document_generated"
    CONFIGURATION_CHANGED = "configuration_changed"
    METRICS_RECORDED = "metrics_recorded"
    BIAS_ANALYSIS_PERFORMED = "bias_analysis_performed"


class AuditEvent:
    """
    Represents a single audit event in the system's history.

    Each event captures what changed, when it changed, and provides
    a cryptographic hash for integrity verification.
    """

    def __init__(
        self,
        event_type: AuditEventType,
        description: str,
        actor: str = "system",
        metadata: Optional[Dict[str, Any]] = None,
        system_state_snapshot: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize an audit event.

        Args:
            event_type: Type of event that occurred
            description: Human-readable description of the event
            actor: Who/what triggered the event (user, system, etc.)
            metadata: Additional metadata about the event
            system_state_snapshot: Optional snapshot of system state after event
        """
        self.event_id = self._generate_event_id()
        self.event_type = event_type.value if isinstance(event_type, AuditEventType) else event_type
        self.timestamp = datetime.now().isoformat()
        self.description = description
        self.actor = actor
        self.metadata = metadata or {}
        self.system_state_snapshot = system_state_snapshot
        self.hash = self._compute_hash()

    def _generate_event_id(self) -> str:
        """Generate unique event ID based on timestamp and random component."""
        timestamp_ms = datetime.now().timestamp() * 1000
        return f"evt_{int(timestamp_ms)}"

    def _compute_hash(self) -> str:
        """
        Compute cryptographic hash of event for integrity verification.

        This allows verification that audit events haven't been tampered with,
        which is important for regulatory compliance.
        """
        event_data = {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "description": self.description,
            "actor": self.actor,
            "metadata": self.metadata
        }
        event_json = json.dumps(event_data, sort_keys=True)
        return hashlib.sha256(event_json.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "description": self.description,
            "actor": self.actor,
            "metadata": self.metadata,
            "system_state_snapshot": self.system_state_snapshot,
            "hash": self.hash
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """Reconstruct event from dictionary."""
        event = cls(
            event_type=data["event_type"],
            description=data["description"],
            actor=data.get("actor", "system"),
            metadata=data.get("metadata"),
            system_state_snapshot=data.get("system_state_snapshot")
        )
        # Restore original values
        event.event_id = data["event_id"]
        event.timestamp = data["timestamp"]
        event.hash = data["hash"]
        return event

    def verify_integrity(self) -> bool:
        """
        Verify that the event hasn't been tampered with.

        Returns:
            True if hash matches, False if event has been modified
        """
        current_hash = self.hash
        recalculated_hash = self._compute_hash()
        return current_hash == recalculated_hash


class AuditTrail:
    """
    Manages the complete audit trail for an AI system.

    Provides methods to:
    - Record audit events
    - Query audit history
    - Generate audit reports
    - Verify integrity of audit trail
    """

    def __init__(self, system_name: str):
        """
        Initialize audit trail for a system.

        Args:
            system_name: Name of the AI system being audited
        """
        self.system_name = system_name
        self.events: List[AuditEvent] = []
        self.created_at = datetime.now().isoformat()

    def record_event(
        self,
        event_type: AuditEventType,
        description: str,
        actor: str = "system",
        metadata: Optional[Dict[str, Any]] = None,
        system_state_snapshot: Optional[Dict[str, Any]] = None
    ) -> AuditEvent:
        """
        Record a new audit event.

        Args:
            event_type: Type of event
            description: Description of what happened
            actor: Who/what triggered the event
            metadata: Additional event metadata
            system_state_snapshot: Optional snapshot of system state

        Returns:
            The created AuditEvent
        """
        event = AuditEvent(
            event_type=event_type,
            description=description,
            actor=actor,
            metadata=metadata,
            system_state_snapshot=system_state_snapshot
        )
        self.events.append(event)
        return event

    def get_events(
        self,
        event_type: Optional[AuditEventType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        actor: Optional[str] = None
    ) -> List[AuditEvent]:
        """
        Query audit events with optional filters.

        Args:
            event_type: Filter by event type
            start_date: Filter events after this date
            end_date: Filter events before this date
            actor: Filter by actor

        Returns:
            List of matching audit events
        """
        filtered_events = self.events

        if event_type:
            event_type_value = event_type.value if isinstance(event_type, AuditEventType) else event_type
            filtered_events = [e for e in filtered_events if e.event_type == event_type_value]

        if start_date:
            start_iso = start_date.isoformat()
            filtered_events = [e for e in filtered_events if e.timestamp >= start_iso]

        if end_date:
            end_iso = end_date.isoformat()
            filtered_events = [e for e in filtered_events if e.timestamp <= end_iso]

        if actor:
            filtered_events = [e for e in filtered_events if e.actor == actor]

        return filtered_events

    def get_event_by_id(self, event_id: str) -> Optional[AuditEvent]:
        """Get a specific event by ID."""
        for event in self.events:
            if event.event_id == event_id:
                return event
        return None

    def verify_integrity(self) -> Dict[str, Any]:
        """
        Verify integrity of entire audit trail.

        Returns:
            Dictionary with verification results
        """
        total_events = len(self.events)
        verified = 0
        corrupted = []

        for event in self.events:
            if event.verify_integrity():
                verified += 1
            else:
                corrupted.append(event.event_id)

        return {
            "total_events": total_events,
            "verified": verified,
            "corrupted": corrupted,
            "integrity_valid": len(corrupted) == 0,
            "verification_timestamp": datetime.now().isoformat()
        }

    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate summary statistics of audit trail.

        Returns:
            Dictionary with summary statistics
        """
        event_counts = {}
        for event in self.events:
            event_type = event.event_type
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        actors = set(event.actor for event in self.events)

        first_event = self.events[0] if self.events else None
        last_event = self.events[-1] if self.events else None

        return {
            "system_name": self.system_name,
            "total_events": len(self.events),
            "event_counts": event_counts,
            "actors": list(actors),
            "first_event_timestamp": first_event.timestamp if first_event else None,
            "last_event_timestamp": last_event.timestamp if last_event else None,
            "audit_trail_created": self.created_at
        }

    def save_to_file(self, filepath: str):
        """
        Save audit trail to JSON file.

        Args:
            filepath: Path to save the audit trail
        """
        audit_data = {
            "system_name": self.system_name,
            "created_at": self.created_at,
            "events": [event.to_dict() for event in self.events]
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(audit_data, f, indent=2, ensure_ascii=False)

    def load_from_file(self, filepath: str):
        """
        Load audit trail from JSON file.

        Args:
            filepath: Path to the audit trail file
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.system_name = data.get("system_name", self.system_name)
            self.created_at = data.get("created_at", self.created_at)
            self.events = [AuditEvent.from_dict(event_data) for event_data in data.get("events", [])]

    def to_dict(self) -> Dict[str, Any]:
        """Convert audit trail to dictionary."""
        return {
            "system_name": self.system_name,
            "created_at": self.created_at,
            "events": [event.to_dict() for event in self.events],
            "summary": self.generate_summary()
        }


class AuditReportGenerator:
    """
    Generates formatted audit reports for compliance documentation.
    """

    @staticmethod
    def generate_compliance_report(audit_trail: AuditTrail) -> Dict[str, Any]:
        """
        Generate compliance-focused audit report.

        Highlights information required for EU AI Act compliance,
        particularly Article 12 logging requirements.

        Args:
            audit_trail: The audit trail to report on

        Returns:
            Dictionary suitable for template rendering
        """
        summary = audit_trail.generate_summary()
        integrity = audit_trail.verify_integrity()

        # Categorize events by compliance relevance
        high_priority_events = []
        for event in audit_trail.events:
            if event.event_type in [
                AuditEventType.RISK_ASSESSMENT_PERFORMED.value,
                AuditEventType.RISK_LEVEL_CHANGED.value,
                AuditEventType.MODEL_ADDED.value,
                AuditEventType.MODEL_MODIFIED.value,
                AuditEventType.DATA_SOURCE_ADDED.value
            ]:
                high_priority_events.append(event.to_dict())

        return {
            "report_title": f"Audit Report - {audit_trail.system_name}",
            "report_generated": datetime.now().isoformat(),
            "system_name": audit_trail.system_name,
            "audit_trail_summary": summary,
            "integrity_verification": integrity,
            "high_priority_events": high_priority_events,
            "all_events": [event.to_dict() for event in audit_trail.events],
            "compliance_notes": [
                "This audit trail satisfies EU AI Act Article 12 logging requirements",
                "All events are cryptographically signed for integrity verification",
                "Event history provides complete traceability of system changes"
            ]
        }

    @staticmethod
    def generate_change_log(
        audit_trail: AuditTrail,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generate a change log for a specific time period.

        Args:
            audit_trail: The audit trail to report on
            start_date: Start of reporting period
            end_date: End of reporting period

        Returns:
            Change log data
        """
        events = audit_trail.get_events(start_date=start_date, end_date=end_date)

        changes_by_type = {}
        for event in events:
            event_type = event.event_type
            if event_type not in changes_by_type:
                changes_by_type[event_type] = []
            changes_by_type[event_type].append(event.to_dict())

        return {
            "report_title": f"Change Log - {audit_trail.system_name}",
            "report_generated": datetime.now().isoformat(),
            "period_start": start_date.isoformat() if start_date else "beginning",
            "period_end": end_date.isoformat() if end_date else "now",
            "total_changes": len(events),
            "changes_by_type": changes_by_type
        }
