"""
Unit Tests for Audit Trail

Tests the audit trail and integrity verification functionality.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aiact_toolkit.audit_trail import AuditTrail, AuditEvent, AuditEventType, AuditReportGenerator


class TestAuditTrail:
    """Test suite for audit trail functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.trail = AuditTrail(system_name="test_system")

    def test_initialization(self):
        """Test audit trail initializes correctly."""
        assert self.trail.system_name == "test_system"
        assert len(self.trail.events) == 0

    def test_record_event(self):
        """Test recording an audit event."""
        event = self.trail.record_event(
            event_type=AuditEventType.MODEL_ADDED,
            description="Added test model",
            actor="test"
        )
        assert len(self.trail.events) == 1
        assert event.event_type == "model_added"
        assert event.description == "Added test model"

    def test_get_events_filtered(self):
        """Test filtering events by type."""
        self.trail.record_event(AuditEventType.MODEL_ADDED, "Model 1")
        self.trail.record_event(AuditEventType.DATA_SOURCE_ADDED, "Data 1")
        self.trail.record_event(AuditEventType.MODEL_ADDED, "Model 2")

        model_events = self.trail.get_events(AuditEventType.MODEL_ADDED)
        assert len(model_events) == 2

        data_events = self.trail.get_events(AuditEventType.DATA_SOURCE_ADDED)
        assert len(data_events) == 1

    def test_generate_summary(self):
        """Test summary generation."""
        self.trail.record_event(AuditEventType.MODEL_ADDED, "Model")
        self.trail.record_event(AuditEventType.MODEL_ADDED, "Model 2")
        self.trail.record_event(AuditEventType.DATA_SOURCE_ADDED, "Data")

        summary = self.trail.generate_summary()
        assert summary["total_events"] == 3
        assert summary["event_counts"]["model_added"] == 2
        assert summary["event_counts"]["data_source_added"] == 1

    def test_verify_integrity_clean(self):
        """Test integrity verification with clean trail."""
        self.trail.record_event(AuditEventType.SYSTEM_CREATED, "Created")
        self.trail.record_event(AuditEventType.MODEL_ADDED, "Model added")
        self.trail.record_event(AuditEventType.DATA_SOURCE_ADDED, "Data added")

        result = self.trail.verify_integrity()
        assert result["total_events"] == 3
        assert result["verified"] == 3
        assert result["corrupted"] == []
        assert result["integrity_status"] == "intact"

    def test_verify_integrity_corrupted(self):
        """Test integrity verification detects corrupted events."""
        self.trail.record_event(AuditEventType.SYSTEM_CREATED, "Created")

        # Manually corrupt an event
        corrupt_event = AuditEvent(AuditEventType.MODEL_ADDED, "Corrupted")
        corrupt_event.event_id = ""  # Empty ID = corrupted
        self.trail.events.append(corrupt_event)

        result = self.trail.verify_integrity()
        assert len(result["corrupted"]) > 0
        assert result["integrity_status"] == "compromised"

    def test_to_dict(self):
        """Test dictionary conversion."""
        self.trail.record_event(AuditEventType.MODEL_ADDED, "Test")

        data = self.trail.to_dict()
        assert data["system_name"] == "test_system"
        assert len(data["events"]) == 1
        assert "summary" in data


class TestAuditReportGenerator:
    """Test suite for audit report generation."""

    def test_generate_compliance_report(self):
        """Test compliance report generation."""
        trail = AuditTrail(system_name="report_test")
        trail.record_event(AuditEventType.SYSTEM_CREATED, "System created")
        trail.record_event(AuditEventType.MODEL_ADDED, "Model added")
        trail.record_event(AuditEventType.RISK_ASSESSMENT_PERFORMED, "Risk assessed")

        report = AuditReportGenerator.generate_compliance_report(trail)
        assert report["system_name"] == "report_test"
        assert len(report["high_priority_events"]) == 2  # model + risk
        assert len(report["all_events"]) == 3
        assert len(report["compliance_notes"]) > 0
