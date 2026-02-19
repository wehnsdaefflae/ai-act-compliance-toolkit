"""
Conformity Assessment Module for EU AI Act Compliance

Implements automated conformity assessment as per EU AI Act Articles 43-46.
Validates compliance status by checking required documentation, processes, and controls.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class ComplianceStatus(Enum):
    """Compliance status levels"""
    COMPLIANT = "compliant"
    PARTIAL = "partial"
    NON_COMPLIANT = "non_compliant"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class RequirementCategory(Enum):
    """EU AI Act requirement categories"""
    RISK_MANAGEMENT = "risk_management"
    DATA_GOVERNANCE = "data_governance"
    TECHNICAL_DOCUMENTATION = "technical_documentation"
    RECORD_KEEPING = "record_keeping"
    TRANSPARENCY = "transparency"
    HUMAN_OVERSIGHT = "human_oversight"
    ACCURACY_ROBUSTNESS = "accuracy_robustness"
    CYBERSECURITY = "cybersecurity"


@dataclass
class ComplianceRequirement:
    """Individual compliance requirement"""
    requirement_id: str
    category: RequirementCategory
    article: str
    description: str
    risk_levels: List[str]  # Which risk levels this applies to
    mandatory: bool
    verification_method: str
    status: ComplianceStatus = ComplianceStatus.UNKNOWN
    findings: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)


@dataclass
class ConformityAssessmentResult:
    """Results of a conformity assessment"""
    system_name: str
    assessment_date: str
    risk_level: str
    overall_status: ComplianceStatus
    requirements_checked: int
    requirements_passed: int
    requirements_failed: int
    requirements_partial: int
    requirements_na: int
    category_results: Dict[str, Dict[str, Any]]
    detailed_results: List[ComplianceRequirement]
    recommendations: List[str]
    critical_gaps: List[str]

    @property
    def compliance_score(self) -> float:
        """Calculate weighted compliance score (0-100).

        Fully met requirements count as 1.0, partial as 0.5, N/A excluded.
        """
        evaluated = self.requirements_checked - self.requirements_na
        if evaluated == 0:
            return 0.0
        weighted = self.requirements_passed + self.requirements_partial * 0.5
        return round(weighted / evaluated * 100, 1)


class ConformityAssessor:
    """
    Automated conformity assessment system for EU AI Act compliance.

    Checks compliance status by validating:
    - Required documentation completeness
    - Data governance processes
    - Technical controls implementation
    - Record-keeping systems
    - Transparency measures
    """

    def __init__(self):
        self.requirements = self._initialize_requirements()

    def _initialize_requirements(self) -> List[ComplianceRequirement]:
        """Initialize compliance requirements based on EU AI Act"""
        requirements = []

        # Article 9 - Risk Management System
        requirements.append(ComplianceRequirement(
            requirement_id="REQ-001",
            category=RequirementCategory.RISK_MANAGEMENT,
            article="Article 9",
            description="Risk management system established and documented",
            risk_levels=["high", "unacceptable"],
            mandatory=True,
            verification_method="Check for risk assessment documentation"
        ))

        # Article 10 - Data Governance
        requirements.append(ComplianceRequirement(
            requirement_id="REQ-002",
            category=RequirementCategory.DATA_GOVERNANCE,
            article="Article 10.2",
            description="Training data governance and quality measures documented",
            risk_levels=["high"],
            mandatory=True,
            verification_method="Check data governance documentation and lineage tracking"
        ))

        requirements.append(ComplianceRequirement(
            requirement_id="REQ-003",
            category=RequirementCategory.DATA_GOVERNANCE,
            article="Article 10.3",
            description="Data quality metrics and validation processes established",
            risk_levels=["high"],
            mandatory=True,
            verification_method="Check for data quality reports and validation procedures"
        ))

        requirements.append(ComplianceRequirement(
            requirement_id="REQ-004",
            category=RequirementCategory.DATA_GOVERNANCE,
            article="Article 10.5",
            description="Personal data processing documented (GDPR compliance)",
            risk_levels=["high", "limited"],
            mandatory=True,
            verification_method="Check for personal data flags and GDPR documentation"
        ))

        # Article 11 - Technical Documentation
        requirements.append(ComplianceRequirement(
            requirement_id="REQ-005",
            category=RequirementCategory.TECHNICAL_DOCUMENTATION,
            article="Article 11.1",
            description="Complete technical documentation available",
            risk_levels=["high"],
            mandatory=True,
            verification_method="Check for Article 11 technical documentation"
        ))

        requirements.append(ComplianceRequirement(
            requirement_id="REQ-006",
            category=RequirementCategory.TECHNICAL_DOCUMENTATION,
            article="Article 11.1(a)",
            description="System description and intended purpose documented",
            risk_levels=["high"],
            mandatory=True,
            verification_method="Check metadata for system identification and use case"
        ))

        # Article 12 - Record-Keeping (Logging)
        requirements.append(ComplianceRequirement(
            requirement_id="REQ-007",
            category=RequirementCategory.RECORD_KEEPING,
            article="Article 12.1",
            description="Automatic logging capabilities enabled",
            risk_levels=["high"],
            mandatory=True,
            verification_method="Check for audit trail system"
        ))

        requirements.append(ComplianceRequirement(
            requirement_id="REQ-008",
            category=RequirementCategory.RECORD_KEEPING,
            article="Article 12.2",
            description="Logs protected against tampering (integrity verification)",
            risk_levels=["high"],
            mandatory=True,
            verification_method="Check audit trail integrity verification"
        ))

        # Article 13 - Transparency and Information
        requirements.append(ComplianceRequirement(
            requirement_id="REQ-009",
            category=RequirementCategory.TRANSPARENCY,
            article="Article 13.1",
            description="Model Cards or documentation for transparency",
            risk_levels=["high", "limited"],
            mandatory=True,
            verification_method="Check for model card documentation"
        ))

        requirements.append(ComplianceRequirement(
            requirement_id="REQ-010",
            category=RequirementCategory.TRANSPARENCY,
            article="Article 13.3(b)",
            description="System capabilities and limitations documented",
            risk_levels=["high"],
            mandatory=True,
            verification_method="Check technical documentation for capabilities/limitations"
        ))

        # Article 14 - Human Oversight
        requirements.append(ComplianceRequirement(
            requirement_id="REQ-011",
            category=RequirementCategory.HUMAN_OVERSIGHT,
            article="Article 14.1",
            description="Human oversight measures documented and implemented",
            risk_levels=["high"],
            mandatory=True,
            verification_method="Check technical documentation for human oversight requirements"
        ))

        # Article 15 - Accuracy, Robustness, Cybersecurity
        requirements.append(ComplianceRequirement(
            requirement_id="REQ-012",
            category=RequirementCategory.ACCURACY_ROBUSTNESS,
            article="Article 15.1",
            description="Accuracy and robustness requirements defined",
            risk_levels=["high"],
            mandatory=True,
            verification_method="Check for performance metrics and validation data"
        ))

        requirements.append(ComplianceRequirement(
            requirement_id="REQ-013",
            category=RequirementCategory.CYBERSECURITY,
            article="Article 15.1",
            description="Cybersecurity measures documented",
            risk_levels=["high"],
            mandatory=True,
            verification_method="Check for security controls documentation"
        ))

        # Bias Detection and Fairness (Article 10.2f, Recital 44)
        requirements.append(ComplianceRequirement(
            requirement_id="REQ-014",
            category=RequirementCategory.DATA_GOVERNANCE,
            article="Article 10.2(f)",
            description="Bias detection and mitigation measures implemented",
            risk_levels=["high"],
            mandatory=True,
            verification_method="Check for bias detection reports and fairness metrics"
        ))

        # Minimal Risk - Limited Requirements
        requirements.append(ComplianceRequirement(
            requirement_id="REQ-015",
            category=RequirementCategory.TRANSPARENCY,
            article="Article 52",
            description="Basic transparency obligations (minimal risk systems)",
            risk_levels=["minimal"],
            mandatory=False,
            verification_method="Check for basic documentation"
        ))

        return requirements

    def assess_compliance(self, metadata: Dict[str, Any]) -> ConformityAssessmentResult:
        """
        Perform comprehensive conformity assessment on system metadata.

        Args:
            metadata: System metadata from MetadataStorage

        Returns:
            ConformityAssessmentResult with detailed compliance status
        """
        system_name = metadata.get("system_name", "Unknown System")
        risk_level = metadata.get("risk_assessment", {}).get("risk_level", "unknown")

        # Filter requirements applicable to this risk level
        applicable_requirements = [
            req for req in self.requirements
            if risk_level in req.risk_levels or req.risk_levels == ["all"]
        ]

        # Assess each requirement
        for req in applicable_requirements:
            self._assess_requirement(req, metadata, risk_level)

        # Calculate statistics
        passed = sum(1 for r in applicable_requirements if r.status == ComplianceStatus.COMPLIANT)
        failed = sum(1 for r in applicable_requirements if r.status == ComplianceStatus.NON_COMPLIANT)
        partial = sum(1 for r in applicable_requirements if r.status == ComplianceStatus.PARTIAL)
        na = sum(1 for r in applicable_requirements if r.status == ComplianceStatus.NOT_APPLICABLE)

        # Determine overall status
        if failed == 0 and partial == 0:
            overall_status = ComplianceStatus.COMPLIANT
        elif failed > 0 and any(r.mandatory and r.status == ComplianceStatus.NON_COMPLIANT
                                for r in applicable_requirements):
            overall_status = ComplianceStatus.NON_COMPLIANT
        else:
            overall_status = ComplianceStatus.PARTIAL

        # Generate category results
        category_results = self._generate_category_results(applicable_requirements)

        # Generate recommendations and critical gaps
        recommendations = self._generate_recommendations(applicable_requirements, risk_level)
        critical_gaps = self._identify_critical_gaps(applicable_requirements)

        return ConformityAssessmentResult(
            system_name=system_name,
            assessment_date=datetime.now().isoformat(),
            risk_level=risk_level,
            overall_status=overall_status,
            requirements_checked=len(applicable_requirements),
            requirements_passed=passed,
            requirements_failed=failed,
            requirements_partial=partial,
            requirements_na=na,
            category_results=category_results,
            detailed_results=applicable_requirements,
            recommendations=recommendations,
            critical_gaps=critical_gaps
        )

    def _assess_requirement(self, req: ComplianceRequirement, metadata: Dict[str, Any],
                           risk_level: str) -> None:
        """Assess a single compliance requirement"""

        if req.requirement_id == "REQ-001":
            # Risk management system
            if metadata.get("risk_assessment"):
                risk_data = metadata["risk_assessment"]
                if risk_data.get("risk_level") and risk_data.get("compliance_requirements"):
                    req.status = ComplianceStatus.COMPLIANT
                    req.evidence.append("Risk assessment documented with level and requirements")
                else:
                    req.status = ComplianceStatus.PARTIAL
                    req.findings.append("Risk assessment incomplete - missing requirements")
            else:
                req.status = ComplianceStatus.NON_COMPLIANT
                req.findings.append("No risk assessment found")

        elif req.requirement_id == "REQ-002":
            # Data governance
            if "data_governance" in metadata:
                dg = metadata["data_governance"]
                if dg.get("data_sources") and len(dg["data_sources"]) > 0:
                    req.status = ComplianceStatus.COMPLIANT
                    req.evidence.append(f"{len(dg['data_sources'])} data sources documented with governance")
                else:
                    req.status = ComplianceStatus.PARTIAL
                    req.findings.append("Data governance enabled but no sources documented")
            else:
                req.status = ComplianceStatus.NON_COMPLIANT
                req.findings.append("Data governance not implemented")

        elif req.requirement_id == "REQ-003":
            # Data quality metrics
            if "data_governance" in metadata:
                dg = metadata["data_governance"]
                sources_with_quality = [s for s in dg.get("data_sources", [])
                                       if s.get("quality_metrics")]
                if sources_with_quality:
                    req.status = ComplianceStatus.COMPLIANT
                    req.evidence.append(f"{len(sources_with_quality)} data sources with quality metrics")
                else:
                    req.status = ComplianceStatus.PARTIAL
                    req.findings.append("Data sources documented but quality metrics missing")
            else:
                req.status = ComplianceStatus.NON_COMPLIANT
                req.findings.append("No data quality documentation")

        elif req.requirement_id == "REQ-004":
            # Personal data processing
            if "data_governance" in metadata:
                dg = metadata["data_governance"]
                sources = dg.get("data_sources", [])
                personal_data_sources = [s for s in sources if s.get("personal_data")]
                if personal_data_sources:
                    req.status = ComplianceStatus.COMPLIANT
                    req.evidence.append(f"{len(personal_data_sources)} sources with personal data flags")
                else:
                    req.status = ComplianceStatus.COMPLIANT
                    req.evidence.append("No personal data processing detected")
            else:
                req.status = ComplianceStatus.PARTIAL
                req.findings.append("Cannot verify personal data handling - governance not enabled")

        elif req.requirement_id == "REQ-005":
            # Technical documentation
            if metadata.get("models") and metadata.get("data_sources"):
                req.status = ComplianceStatus.COMPLIANT
                req.evidence.append("Models and data sources documented")
            else:
                req.status = ComplianceStatus.PARTIAL
                req.findings.append("Technical documentation incomplete")

        elif req.requirement_id == "REQ-006":
            # System description
            if metadata.get("system_name") and metadata.get("risk_assessment", {}).get("use_case"):
                req.status = ComplianceStatus.COMPLIANT
                req.evidence.append("System identification and use case documented")
            else:
                req.status = ComplianceStatus.PARTIAL
                req.findings.append("System description incomplete - add use case to risk assessment")

        elif req.requirement_id == "REQ-007":
            # Automatic logging
            if "audit_trail" in metadata and metadata["audit_trail"].get("events"):
                req.status = ComplianceStatus.COMPLIANT
                req.evidence.append(f"{len(metadata['audit_trail']['events'])} audit events logged")
            else:
                req.status = ComplianceStatus.NON_COMPLIANT
                req.findings.append("Audit trail not enabled or no events logged")

        elif req.requirement_id == "REQ-008":
            # Log integrity - Note: cryptographic hashing is planned for future versions
            if "audit_trail" in metadata or "audit_summary" in metadata:
                req.status = ComplianceStatus.PARTIAL
                req.evidence.append("Audit trail system operational")
                req.findings.append("Cryptographic integrity verification planned for production release")
            else:
                req.status = ComplianceStatus.NON_COMPLIANT
                req.findings.append("No audit trail system")

        elif req.requirement_id == "REQ-009":
            # Model cards / transparency documentation
            if metadata.get("models") and len(metadata["models"]) > 0:
                models_with_params = [m for m in metadata["models"] if m.get("parameters")]
                if models_with_params:
                    req.status = ComplianceStatus.COMPLIANT
                    req.evidence.append(f"{len(models_with_params)} models with parameter documentation")
                else:
                    req.status = ComplianceStatus.PARTIAL
                    req.findings.append("Models documented but parameter details missing")
            else:
                req.status = ComplianceStatus.NON_COMPLIANT
                req.findings.append("No model documentation found")

        elif req.requirement_id == "REQ-010":
            # System capabilities and limitations
            if metadata.get("risk_assessment", {}).get("recommendations"):
                req.status = ComplianceStatus.COMPLIANT
                req.evidence.append("System limitations documented in risk assessment")
            else:
                req.status = ComplianceStatus.PARTIAL
                req.findings.append("Capabilities/limitations not explicitly documented")

        elif req.requirement_id == "REQ-011":
            # Human oversight
            if risk_level == "high":
                # For high-risk systems, human oversight is mandatory
                # Check if documented in technical docs or risk assessment
                if metadata.get("risk_assessment", {}).get("compliance_requirements"):
                    reqs = metadata["risk_assessment"]["compliance_requirements"]
                    has_oversight = any("oversight" in r.lower() or "human" in r.lower()
                                      for r in reqs)
                    if has_oversight:
                        req.status = ComplianceStatus.COMPLIANT
                        req.evidence.append("Human oversight requirements documented")
                    else:
                        req.status = ComplianceStatus.PARTIAL
                        req.findings.append("Human oversight requirements not explicitly stated")
                else:
                    req.status = ComplianceStatus.NON_COMPLIANT
                    req.findings.append("Human oversight not documented")
            else:
                req.status = ComplianceStatus.NOT_APPLICABLE

        elif req.requirement_id == "REQ-012":
            # Accuracy and robustness
            if metadata.get("operational_metrics"):
                metrics = metadata["operational_metrics"]
                if metrics.get("performance"):
                    req.status = ComplianceStatus.COMPLIANT
                    req.evidence.append("Performance metrics tracked")
                else:
                    req.status = ComplianceStatus.PARTIAL
                    req.findings.append("Operational metrics enabled but performance data limited")
            else:
                req.status = ComplianceStatus.NON_COMPLIANT
                req.findings.append("No accuracy/robustness metrics tracked")

        elif req.requirement_id == "REQ-013":
            # Cybersecurity
            # This is a placeholder - actual implementation would check for security controls
            req.status = ComplianceStatus.PARTIAL
            req.findings.append("Cybersecurity controls not yet captured in metadata")

        elif req.requirement_id == "REQ-014":
            # Bias detection - check both possible keys for compatibility
            bias_data = metadata.get("bias_analyses") or metadata.get("bias_detection", {}).get("reports", [])
            if bias_data and len(bias_data) > 0:
                req.status = ComplianceStatus.COMPLIANT
                req.evidence.append(f"{len(bias_data)} bias analysis report(s) available")
            else:
                req.status = ComplianceStatus.NON_COMPLIANT
                req.findings.append("Bias detection not performed - run bias analysis for high-risk systems")

        elif req.requirement_id == "REQ-015":
            # Minimal risk transparency
            if metadata.get("models") or metadata.get("system_name"):
                req.status = ComplianceStatus.COMPLIANT
                req.evidence.append("Basic system documentation present")
            else:
                req.status = ComplianceStatus.NON_COMPLIANT
                req.findings.append("No basic documentation")

    def _generate_category_results(self, requirements: List[ComplianceRequirement]) -> Dict[str, Dict[str, Any]]:
        """Generate per-category compliance summary"""
        category_results = {}
        for category in RequirementCategory:
            cat_reqs = [r for r in requirements if r.category == category]
            if not cat_reqs:
                continue
            passed = sum(1 for r in cat_reqs if r.status == ComplianceStatus.COMPLIANT)
            category_results[category.value] = {
                "total_requirements": len(cat_reqs),
                "passed": passed,
                "compliance_percentage": passed / len(cat_reqs) * 100,
                "critical_gaps": [r.requirement_id for r in cat_reqs
                                 if r.mandatory and r.status == ComplianceStatus.NON_COMPLIANT]
            }
        return category_results

    @staticmethod
    def _is_req_compliant(requirements: List[ComplianceRequirement], req_id: str) -> bool:
        """Check if a specific requirement is compliant."""
        return any(r.requirement_id == req_id and r.status == ComplianceStatus.COMPLIANT
                   for r in requirements)

    def _generate_recommendations(self, requirements: List[ComplianceRequirement],
                                 risk_level: str) -> List[str]:
        """Generate actionable recommendations for improvement"""
        recommendations = []

        failed_mandatory = [r for r in requirements
                           if r.mandatory and r.status == ComplianceStatus.NON_COMPLIANT]

        if failed_mandatory:
            recommendations.append(
                f"KRITISCH: {len(failed_mandatory)} verpflichtende Anforderungen nicht erfüllt. "
                "Diese müssen vor dem Inverkehrbringen implementiert werden."
            )

        if not self._is_req_compliant(requirements, "REQ-002"):
            recommendations.append(
                "Aktivieren Sie Data Governance Tracking mit enable_data_governance=True "
                "beim Monitor-Start für Artikel 10 Compliance."
            )

        if not self._is_req_compliant(requirements, "REQ-007"):
            recommendations.append(
                "Aktivieren Sie Audit Trail mit enable_audit_trail=True "
                "für Artikel 12 Compliance (automatische Protokollierung)."
            )

        if risk_level == "high" and not self._is_req_compliant(requirements, "REQ-014"):
            recommendations.append(
                "Implementieren Sie Bias Detection für Hochrisiko-Systeme "
                "(Artikel 10.2f - Vermeidung diskriminierender Verzerrungen)."
            )

        # Partial compliance issues
        partial_reqs = [r for r in requirements if r.status == ComplianceStatus.PARTIAL]
        if partial_reqs:
            recommendations.append(
                f"{len(partial_reqs)} Anforderungen sind teilweise erfüllt. "
                "Vervollständigen Sie die Dokumentation für volle Compliance."
            )

        return recommendations

    def _identify_critical_gaps(self, requirements: List[ComplianceRequirement]) -> List[str]:
        """Identify critical compliance gaps"""
        gaps = []

        for req in requirements:
            if req.mandatory and req.status in [ComplianceStatus.NON_COMPLIANT, ComplianceStatus.PARTIAL]:
                gap_msg = f"{req.requirement_id} ({req.article}): {req.description}"
                if req.findings:
                    gap_msg += f" - {'; '.join(req.findings)}"
                gaps.append(gap_msg)

        return gaps


def generate_conformity_report(result: ConformityAssessmentResult) -> str:
    """Generate a text summary of conformity assessment results"""
    lines = []
    lines.append("=" * 80)
    lines.append("KONFORMITÄTSBEWERTUNG - EU AI ACT")
    lines.append("=" * 80)
    lines.append(f"System: {result.system_name}")
    lines.append(f"Bewertungsdatum: {result.assessment_date}")
    lines.append(f"Risikolevel: {result.risk_level.upper()}")
    lines.append(f"Gesamtstatus: {result.overall_status.value.upper()}")
    lines.append("-" * 80)

    lines.append(f"\nÜBERBLICK:")
    lines.append(f"  Geprüfte Anforderungen: {result.requirements_checked}")
    lines.append(f"  ✓ Erfüllt: {result.requirements_passed}")
    lines.append(f"  ✗ Nicht erfüllt: {result.requirements_failed}")
    lines.append(f"  ~ Teilweise: {result.requirements_partial}")
    lines.append(f"  - Nicht anwendbar: {result.requirements_na}")

    lines.append(f"\nCOMPLIANCE-SCORE: {result.compliance_score}%")

    lines.append("\nKATEGORIE-ERGEBNISSE:")
    for category, data in result.category_results.items():
        lines.append(f"  {category}: {data['passed']}/{data['total_requirements']} "
                    f"({data['compliance_percentage']:.0f}%)")
        if data['critical_gaps']:
            lines.append(f"    Kritische Lücken: {', '.join(data['critical_gaps'])}")

    if result.critical_gaps:
        lines.append("\nKRITISCHE LÜCKEN:")
        for gap in result.critical_gaps:
            lines.append(f"  ✗ {gap}")

    if result.recommendations:
        lines.append("\nEMPFEHLUNGEN:")
        for i, rec in enumerate(result.recommendations, 1):
            lines.append(f"  {i}. {rec}")

    lines.append("\n" + "=" * 80)

    return "\n".join(lines)
