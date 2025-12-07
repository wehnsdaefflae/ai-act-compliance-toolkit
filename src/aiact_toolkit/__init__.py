"""
AI Act Compliance Toolkit

A Python package for automatically extracting compliance metadata from LangChain applications
to support EU AI Act and GDPR compliance documentation.
"""

from .langchain_monitor import LangChainMonitor
from .metadata_storage import MetadataStorage
from .document_generator import DocumentGenerator
from .risk_assessment import AIActRiskAssessor, RiskLevel

__version__ = "0.1.0"
__all__ = [
    "LangChainMonitor",
    "MetadataStorage",
    "DocumentGenerator",
    "AIActRiskAssessor",
    "RiskLevel"
]
