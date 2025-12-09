"""
Document Generator

Handles generation of compliance documents from metadata using Jinja2 templates.
"""

import os
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, Template, TemplateNotFound


class DocumentGenerator:
    """
    Generator for compliance documents from captured metadata.
    """

    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initialize document generator.

        Args:
            templates_dir: Path to templates directory. If None, uses package templates.
        """
        if templates_dir is None:
            # Default to package templates directory
            package_dir = Path(__file__).parent.parent.parent
            templates_dir = package_dir / "templates"

        self.templates_dir = Path(templates_dir)

        if not self.templates_dir.exists():
            raise ValueError(f"Templates directory not found: {self.templates_dir}")

        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )

    def list_templates(self) -> List[str]:
        """
        List all available templates.

        Returns:
            List of template filenames
        """
        templates = []
        for file in self.templates_dir.glob("*.jinja2"):
            templates.append(file.name)
        return sorted(templates)

    def load_metadata(self, metadata_path: str) -> Dict[str, Any]:
        """
        Load metadata from JSON file.

        Args:
            metadata_path: Path to metadata JSON file

        Returns:
            Metadata dictionary

        Raises:
            FileNotFoundError: If metadata file doesn't exist
            ValueError: If JSON is invalid
        """
        metadata_file = Path(metadata_path)
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_file, 'r', encoding='utf-8') as f:
            try:
                metadata = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in metadata file: {e}")

        return metadata

    def generate_document(
        self,
        template_name: str,
        metadata: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate compliance document from template and metadata.

        Args:
            template_name: Name of the template file (e.g., 'dsgvo_dsfa.md.jinja2')
            metadata: Metadata dictionary to render template with
            output_path: Optional path to save generated document

        Returns:
            Generated document content as string

        Raises:
            TemplateNotFound: If template doesn't exist
        """
        try:
            template = self.env.get_template(template_name)
        except TemplateNotFound:
            raise TemplateNotFound(
                f"Template '{template_name}' not found in {self.templates_dir}"
            )

        # Render document
        document = template.render(**metadata)

        # Save to file if path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(document)

        return document

    def validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate metadata completeness and structure.

        Args:
            metadata: Metadata dictionary to validate

        Returns:
            Dictionary with validation results:
            {
                "valid": bool,
                "warnings": List[str],
                "missing_fields": List[str],
                "recommendations": List[str]
            }
        """
        warnings = []
        missing_fields = []
        recommendations = []

        # Check required top-level fields
        required_fields = ["system_name", "models", "data_sources"]
        for field in required_fields:
            if field not in metadata:
                missing_fields.append(field)

        # Check if models list is empty
        if "models" in metadata and not metadata["models"]:
            warnings.append("No models captured - ensure LangChain components were created after monitor.start()")

        # Check if data sources list is empty
        if "data_sources" in metadata and not metadata["data_sources"]:
            warnings.append("No data sources captured - consider documenting data loaders")

        # Check for minimal model information
        if "models" in metadata and metadata["models"]:
            for i, model in enumerate(metadata["models"]):
                if "model_name" not in model or model["model_name"] == "unknown":
                    warnings.append(f"Model {i+1}: model_name is unknown or missing")
                if "provider" not in model or model["provider"] == "unknown":
                    warnings.append(f"Model {i+1}: provider is unknown or missing")

        # Check risk assessment
        if "risk_assessment" not in metadata or not metadata["risk_assessment"]:
            recommendations.append("Run risk assessment to classify system under EU AI Act")
        else:
            risk_level = metadata["risk_assessment"].get("risk_level")
            if risk_level == "high":
                warnings.append("HIGH RISK system - extensive compliance requirements apply")
            elif risk_level == "unacceptable":
                warnings.append("UNACCEPTABLE RISK - system deployment is prohibited under EU AI Act")

        # Provide recommendations
        if not metadata.get("components"):
            recommendations.append("Consider documenting chains, tools, and prompts used")

        if metadata.get("models") and len(metadata["models"]) > 0:
            has_temp = any("temperature" in m.get("parameters", {}) for m in metadata["models"])
            if not has_temp:
                recommendations.append("Document temperature and other model parameters for transparency")

        # Check operational metrics
        if "operational_metrics" not in metadata or not metadata["operational_metrics"]:
            recommendations.append("Enable operational metrics tracking for transparency and accountability")
        else:
            ops_metrics = metadata["operational_metrics"]
            # Check if there's actual operation data
            if ops_metrics.get("operations", {}).get("total", 0) == 0:
                warnings.append("No operations recorded in metrics - ensure monitoring is active during execution")

            # Check error rate
            error_rate = ops_metrics.get("operations", {}).get("error_rate_percent", 0)
            if error_rate > 5:
                warnings.append(f"High error rate in operations: {error_rate}% - investigate system reliability")

            # Check for cost data
            if "costs" not in ops_metrics or ops_metrics.get("costs", {}).get("total_estimated_usd", 0) == 0:
                recommendations.append("Cost tracking not available - verify model compatibility with cost estimation")

        return {
            "valid": len(missing_fields) == 0,
            "warnings": warnings,
            "missing_fields": missing_fields,
            "recommendations": recommendations
        }

    def generate_all_documents(
        self,
        metadata: Dict[str, Any],
        output_dir: str
    ) -> List[str]:
        """
        Generate all available compliance documents.

        Args:
            metadata: Metadata dictionary
            output_dir: Directory to save generated documents

        Returns:
            List of generated file paths
        """
        generated_files = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for template_name in self.list_templates():
            # Derive output filename from template name
            output_filename = template_name.replace('.jinja2', '')
            output_file = output_path / output_filename

            try:
                self.generate_document(
                    template_name=template_name,
                    metadata=metadata,
                    output_path=str(output_file)
                )
                generated_files.append(str(output_file))
            except Exception as e:
                print(f"Warning: Failed to generate {template_name}: {e}")

        return generated_files
