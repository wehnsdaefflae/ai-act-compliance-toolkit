"""
Unit tests for Technical Documentation Generator (Article 11)
"""

import sys
import os
import json
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aiact_toolkit.technical_documentation import TechnicalDocumentationGenerator


class TestTechnicalDocumentationGenerator:
    """Test suite for TechnicalDocumentationGenerator class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create temp directory for test outputs
        self.temp_dir = tempfile.mkdtemp()

        # Sample metadata for testing
        self.sample_metadata = {
            "system_name": "test_medical_chatbot",
            "framework": "langchain",
            "created_at": "2024-01-01T00:00:00",
            "timestamp": "2024-01-15T10:30:00",
            "models": [
                {
                    "model_name": "gpt-4",
                    "provider": "OpenAI",
                    "model_type": "LLM",
                    "parameters": {
                        "temperature": 0.7,
                        "max_tokens": 500
                    },
                    "timestamp": "2024-01-01T00:00:00"
                }
            ],
            "data_sources": [
                {
                    "data_source": "./medical_documents.pdf",
                    "loader_type": "PDFLoader",
                    "data_type": "medical_texts",
                    "timestamp": "2024-01-01T00:00:00"
                }
            ],
            "components": [
                {
                    "type": "chain",
                    "name": "medical_qa_chain",
                    "timestamp": "2024-01-01T00:00:00"
                }
            ],
            "risk_assessment": {
                "risk_level": "high",
                "confidence": 0.85,
                "risk_factors": [
                    "Healthcare application",
                    "Medical advice generation"
                ],
                "compliance_requirements": [
                    "Article 11: Technical documentation required",
                    "Article 14: Human oversight mandatory"
                ],
                "recommendations": [
                    "Implement human-in-the-loop for all medical recommendations",
                    "Regular validation against medical guidelines"
                ],
                "timestamp": "2024-01-02T00:00:00"
            },
            "operational_metrics": {
                "operations": {
                    "total": 150,
                    "successful": 145,
                    "failed": 5,
                    "success_rate": 0.9667
                },
                "performance": {
                    "avg_latency_ms": 1250.5,
                    "min_latency_ms": 800.0,
                    "max_latency_ms": 3200.0
                },
                "costs": {
                    "total_cost": 12.50,
                    "average_cost": 0.083
                },
                "errors": []
            },
            "audit_summary": {
                "total_events": 25,
                "event_types": {
                    "SYSTEM_CREATED": 1,
                    "MODEL_ADDED": 1,
                    "DATA_SOURCE_ADDED": 1,
                    "RISK_ASSESSMENT_PERFORMED": 1
                }
            },
            "version_info": {
                "current_version": 3,
                "total_versions": 3,
                "versions": [
                    {
                        "version": 1,
                        "timestamp": "2024-01-01T00:00:00",
                        "description": "Initial system creation",
                        "author": "system"
                    },
                    {
                        "version": 2,
                        "timestamp": "2024-01-02T00:00:00",
                        "description": "Added model: gpt-4",
                        "author": "system"
                    },
                    {
                        "version": 3,
                        "timestamp": "2024-01-02T01:00:00",
                        "description": "Risk assessment: high",
                        "author": "system"
                    }
                ]
            },
            "data_governance": {
                "lineage_graph": {
                    "sources": {},
                    "transformations": []
                }
            },
            "data_quality_summary": {
                "total_sources": 1,
                "sources_with_quality_metrics": 0,
                "average_completeness": "Not assessed",
                "average_accuracy": "Not assessed"
            },
            "privacy_summary": {
                "personal_data_sources": 1,
                "sensitive_data_sources": 1,
                "sources_with_license": 0,
                "sources_with_copyright": 0
            },
            "summary": {
                "total_models": 1,
                "total_components": 1,
                "total_data_sources": 1
            }
        }

    def test_init(self):
        """Test initialization."""
        generator = TechnicalDocumentationGenerator(self.sample_metadata)
        assert generator.metadata == self.sample_metadata
        assert generator.system_name == "test_medical_chatbot"

    def test_generate_documentation(self):
        """Test complete documentation generation."""
        generator = TechnicalDocumentationGenerator(self.sample_metadata)
        documentation = generator.generate_documentation()

        # Verify all required sections are present
        assert "system_identification" in documentation
        assert "general_description" in documentation
        assert "development_process" in documentation
        assert "architecture_and_design" in documentation
        assert "data_requirements" in documentation
        assert "human_oversight" in documentation
        assert "performance_metrics" in documentation
        assert "risk_management" in documentation
        assert "lifecycle_management" in documentation
        assert "conformity_assessment" in documentation
        assert "generated_at" in documentation

    def test_system_identification(self):
        """Test system identification section."""
        generator = TechnicalDocumentationGenerator(self.sample_metadata)
        documentation = generator.generate_documentation()

        system_id = documentation["system_identification"]
        assert system_id["system_name"] == "test_medical_chatbot"
        assert system_id["risk_classification"] == "high"
        assert system_id["framework"] == "langchain"
        assert "system_version" in system_id

    def test_general_description(self):
        """Test general description section."""
        generator = TechnicalDocumentationGenerator(self.sample_metadata)
        documentation = generator.generate_documentation()

        general = documentation["general_description"]
        assert "purpose_and_scope" in general
        assert "capabilities" in general
        assert "known_limitations" in general
        assert isinstance(general["capabilities"], list)
        assert isinstance(general["known_limitations"], list)

    def test_architecture_and_design(self):
        """Test architecture and design section."""
        generator = TechnicalDocumentationGenerator(self.sample_metadata)
        documentation = generator.generate_documentation()

        architecture = documentation["architecture_and_design"]
        assert "system_architecture" in architecture
        assert "algorithms_and_logic" in architecture
        assert "computational_requirements" in architecture

        # Verify model components are extracted
        models = architecture["system_architecture"]["models"]
        assert len(models) == 1
        assert models[0]["name"] == "gpt-4"
        assert models[0]["provider"] == "OpenAI"

    def test_data_requirements(self):
        """Test data requirements section."""
        generator = TechnicalDocumentationGenerator(self.sample_metadata)
        documentation = generator.generate_documentation()

        data_req = documentation["data_requirements"]
        assert "data_sources" in data_req
        assert "data_governance" in data_req
        assert "data_quality_criteria" in data_req

        # Verify data sources are extracted
        sources = data_req["data_sources"]
        assert len(sources) == 1
        assert sources[0]["source_name"] == "./medical_documents.pdf"

    def test_human_oversight_high_risk(self):
        """Test human oversight requirements for high-risk system."""
        generator = TechnicalDocumentationGenerator(self.sample_metadata)
        documentation = generator.generate_documentation()

        oversight = documentation["human_oversight"]
        assert oversight["oversight_requirement_level"] == "mandatory"
        assert "recommended_measures" in oversight
        assert len(oversight["recommended_measures"]) > 0
        assert "technical_measures" in oversight
        assert "organizational_measures" in oversight

    def test_human_oversight_limited_risk(self):
        """Test human oversight requirements for limited-risk system."""
        # Modify metadata to limited risk
        metadata = self.sample_metadata.copy()
        metadata["risk_assessment"] = {
            "risk_level": "limited",
            "confidence": 0.9
        }

        generator = TechnicalDocumentationGenerator(metadata)
        documentation = generator.generate_documentation()

        oversight = documentation["human_oversight"]
        assert oversight["oversight_requirement_level"] == "transparency_required"

    def test_performance_metrics(self):
        """Test performance metrics section."""
        generator = TechnicalDocumentationGenerator(self.sample_metadata)
        documentation = generator.generate_documentation()

        metrics = documentation["performance_metrics"]
        assert "operational_statistics" in metrics
        assert "performance_indicators" in metrics
        assert "monitoring_mechanisms" in metrics

        # Verify operational statistics
        ops = metrics["operational_statistics"]
        assert ops["total_operations"] == 150
        assert ops["successful_operations"] == 145
        assert "96.67%" in ops["success_rate"]

    def test_risk_management_completed(self):
        """Test risk management section with completed assessment."""
        generator = TechnicalDocumentationGenerator(self.sample_metadata)
        documentation = generator.generate_documentation()

        risk_mgmt = documentation["risk_management"]
        assert risk_mgmt["risk_assessment_status"] == "completed"
        assert "risk_classification" in risk_mgmt
        assert risk_mgmt["risk_classification"]["risk_level"] == "high"
        assert "identified_risks" in risk_mgmt
        assert "mitigation_measures" in risk_mgmt

    def test_risk_management_not_performed(self):
        """Test risk management section without assessment."""
        # Remove risk assessment
        metadata = self.sample_metadata.copy()
        metadata.pop("risk_assessment", None)

        generator = TechnicalDocumentationGenerator(metadata)
        documentation = generator.generate_documentation()

        risk_mgmt = documentation["risk_management"]
        assert risk_mgmt["risk_assessment_status"] == "not_performed"
        assert "recommendation" in risk_mgmt

    def test_lifecycle_management(self):
        """Test lifecycle management section."""
        generator = TechnicalDocumentationGenerator(self.sample_metadata)
        documentation = generator.generate_documentation()

        lifecycle = documentation["lifecycle_management"]
        assert "version_control" in lifecycle
        assert "change_management" in lifecycle
        assert "maintenance_plan" in lifecycle
        assert "post_market_monitoring" in lifecycle

        # Verify version control info
        version_ctrl = lifecycle["version_control"]
        assert version_ctrl["current_version"] == "3"
        assert version_ctrl["total_versions"] == 3

    def test_conformity_assessment_high_risk(self):
        """Test conformity assessment for high-risk system."""
        generator = TechnicalDocumentationGenerator(self.sample_metadata)
        documentation = generator.generate_documentation()

        conformity = documentation["conformity_assessment"]
        assert "Required" in conformity["applicable_procedure"]
        assert "assessment_details" in conformity
        assert len(conformity["assessment_details"]) > 0
        assert "compliance_status" in conformity

    def test_to_dict(self):
        """Test to_dict method."""
        generator = TechnicalDocumentationGenerator(self.sample_metadata)
        doc_dict = generator.to_dict()

        assert isinstance(doc_dict, dict)
        assert "system_identification" in doc_dict
        assert "generated_at" in doc_dict

    def test_to_json(self):
        """Test JSON export."""
        generator = TechnicalDocumentationGenerator(self.sample_metadata)

        output_path = Path(self.temp_dir) / "tech_doc.json"
        generator.to_json(str(output_path))

        assert output_path.exists()

        # Verify JSON is valid
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            assert "system_identification" in data
            assert data["system_identification"]["system_name"] == "test_medical_chatbot"

    def test_minimal_metadata(self):
        """Test with minimal metadata."""
        minimal_metadata = {
            "system_name": "minimal_system",
            "models": [],
            "data_sources": [],
            "components": []
        }

        generator = TechnicalDocumentationGenerator(minimal_metadata)
        documentation = generator.generate_documentation()

        # Should still generate all sections without errors
        assert "system_identification" in documentation
        assert "general_description" in documentation
        assert documentation["system_identification"]["system_name"] == "minimal_system"


def run_tests():
    """Run all tests."""
    test_suite = TestTechnicalDocumentationGenerator()

    tests = [
        ("test_init", test_suite.test_init),
        ("test_generate_documentation", test_suite.test_generate_documentation),
        ("test_system_identification", test_suite.test_system_identification),
        ("test_general_description", test_suite.test_general_description),
        ("test_architecture_and_design", test_suite.test_architecture_and_design),
        ("test_data_requirements", test_suite.test_data_requirements),
        ("test_human_oversight_high_risk", test_suite.test_human_oversight_high_risk),
        ("test_human_oversight_limited_risk", test_suite.test_human_oversight_limited_risk),
        ("test_performance_metrics", test_suite.test_performance_metrics),
        ("test_risk_management_completed", test_suite.test_risk_management_completed),
        ("test_risk_management_not_performed", test_suite.test_risk_management_not_performed),
        ("test_lifecycle_management", test_suite.test_lifecycle_management),
        ("test_conformity_assessment_high_risk", test_suite.test_conformity_assessment_high_risk),
        ("test_to_dict", test_suite.test_to_dict),
        ("test_to_json", test_suite.test_to_json),
        ("test_minimal_metadata", test_suite.test_minimal_metadata),
    ]

    passed = 0
    failed = 0

    print("Running Technical Documentation Generator Tests...\n")

    for test_name, test_func in tests:
        try:
            test_suite.setup_method()
            test_func()
            print(f"✓ {test_name}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_name}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_name}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Tests passed: {passed}/{len(tests)}")
    print(f"Tests failed: {failed}/{len(tests)}")
    print(f"{'='*60}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_tests())
