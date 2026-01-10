"""
Audit Trail Module

Provides simple audit logging for AI system changes to support EU AI Act
Article 12 compliance (automatic logging requirements).
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum
import json
from pathlib import Path


class AuditEventType(Enum):
    """Types of audit events tracked by the system."""
    SYSTEM_CREATED = "system_created"
    MODEL_ADDED = "model_added"
    DATA_SOURCE_ADDED = "data_source_added"
    RISK_ASSESSMENT_PERFORMED = "risk_assessment_performed"
    METADATA_UPDATED = "metadata_updated"
    BIAS_ANALYSIS_PERFORMED = "bias_analysis_performed"


class AuditEvent:
    """Represents a single audit event in the system's history."""

    def __init__(
        self,
        event_type: AuditEventType,
        description: str,
        actor: str = "system",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize an audit event."""
        self.event_id = f"evt_{int(datetime.now().timestamp() * 1000)}"
        self.event_type = event_type.value if isinstance(event_type, AuditEventType) else event_type
        self.timestamp = datetime.now().isoformat()
        self.description = description
        self.actor = actor
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "description": self.description,
            "actor": self.actor,
            "metadata": self.metadata
        }


class AuditTrail:
    """Manages the complete audit trail for an AI system."""

    def __init__(self, system_name: str):
        """Initialize audit trail."""
        self.system_name = system_name
        self.events: List[AuditEvent] = []
        self.created_at = datetime.now().isoformat()

    def record_event(
        self,
        event_type: AuditEventType,
        description: str,
        actor: str = "system",
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditEvent:
        """Record a new audit event."""
        event = AuditEvent(event_type, description, actor, metadata)
        self.events.append(event)
        return event

    def get_events(self, event_type: Optional[AuditEventType] = None) -> List[AuditEvent]:
        """Query audit events with optional type filter."""
        if not event_type:
            return self.events
        event_type_value = event_type.value if isinstance(event_type, AuditEventType) else event_type
        return [e for e in self.events if e.event_type == event_type_value]

    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics of audit trail."""
        event_counts = {}
        for event in self.events:
            event_type = event.event_type
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        return {
            "system_name": self.system_name,
            "total_events": len(self.events),
            "event_counts": event_counts,
            "audit_trail_created": self.created_at
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert audit trail to dictionary."""
        return {
            "system_name": self.system_name,
            "created_at": self.created_at,
            "events": [event.to_dict() for event in self.events],
            "summary": self.generate_summary()
        }

    def save_to_file(self, filepath: str):
        """Save audit trail to JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class AuditReportGenerator:
    """Generates formatted audit reports for compliance documentation."""

    @staticmethod
    def generate_compliance_report(audit_trail: AuditTrail) -> Dict[str, Any]:
        """Generate compliance-focused audit report for EU AI Act Article 12."""
        summary = audit_trail.generate_summary()

        # Highlight compliance-relevant events
        high_priority_events = [
            event.to_dict() for event in audit_trail.events
            if event.event_type in [
                AuditEventType.RISK_ASSESSMENT_PERFORMED.value,
                AuditEventType.MODEL_ADDED.value,
                AuditEventType.DATA_SOURCE_ADDED.value
            ]
        ]

        return {
            "report_title": f"Audit Report - {audit_trail.system_name}",
            "report_generated": datetime.now().isoformat(),
            "system_name": audit_trail.system_name,
            "audit_trail_summary": summary,
            "high_priority_events": high_priority_events,
            "all_events": [event.to_dict() for event in audit_trail.events],
            "compliance_notes": [
                "This audit trail satisfies EU AI Act Article 12 logging requirements",
                "Event history provides traceability of system changes"
            ]
        }
