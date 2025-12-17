"""
AI Act Compliance Toolkit

A Python package for automatically extracting compliance metadata from AI/ML applications
to support EU AI Act and GDPR compliance documentation.

Supports multiple frameworks:
- LangChain
- PyTorch
- TensorFlow/Keras
"""

from .langchain_monitor import LangChainMonitor
from .pytorch_monitor import PyTorchMonitor
from .tensorflow_monitor import TensorFlowMonitor
from .metadata_storage import MetadataStorage
from .document_generator import DocumentGenerator
from .risk_assessment import AIActRiskAssessor, RiskLevel
from .operational_metrics import OperationalMetricsTracker, MetricsAnalyzer
from .audit_trail import AuditTrail, AuditEvent, AuditEventType, AuditReportGenerator
from .version_control import VersionControl, MetadataVersion, VersionControlIntegration

__version__ = "0.1.0"
__all__ = [
    "LangChainMonitor",
    "PyTorchMonitor",
    "TensorFlowMonitor",
    "MetadataStorage",
    "DocumentGenerator",
    "AIActRiskAssessor",
    "RiskLevel",
    "OperationalMetricsTracker",
    "MetricsAnalyzer",
    "AuditTrail",
    "AuditEvent",
    "AuditEventType",
    "AuditReportGenerator",
    "VersionControl",
    "MetadataVersion",
    "VersionControlIntegration"
]
