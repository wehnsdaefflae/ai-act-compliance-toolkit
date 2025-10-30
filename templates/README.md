# Compliance Document Templates

This directory contains Jinja2 templates for generating EU AI Act and GDPR compliance documentation.

## Available Templates

### 1. DSGVO-DSFA Template (`dsgvo_dsfa.md.jinja2`)

**Purpose:** Generate a Data Protection Impact Assessment (DPIA) document as required by GDPR (DSGVO in German).

**Use Case:** Required for high-risk AI systems that process personal data, particularly in healthcare, education, employment, and law enforcement.

**Automatically Populated Fields:**
- AI system name and version
- Model information (name, provider, parameters)
- Framework components used
- Data sources and loading methods
- Timestamps

**Manual Input Required:**
- System purpose and scope
- Legal basis for data processing
- Risk assessment
- Technical and organizational measures
- Affected persons' rights implementation

### 2. Article 53 Summary Template (`article_53_summary.md.jinja2`)

**Purpose:** Generate documentation for General Purpose AI (GPAI) systems as required by Article 53 of the EU AI Act.

**Use Case:** Mandatory for providers of GPAI systems to document training data sources and copyright compliance.

**Automatically Populated Fields:**
- System identification
- Model information and parameters
- Data sources and loading methods
- Framework components
- Timestamps and metadata summary

**Manual Input Required:**
- Provider information
- Training data volume and composition
- Copyright and licensing information
- Data provenance
- Opt-out mechanisms for rights holders
- Compliance attestation

## Using the Templates

### Basic Usage

```python
from jinja2 import Template
import json

# Load captured metadata
with open("aiact_metadata.json") as f:
    metadata = json.load(f)

# Load template
with open("templates/dsgvo_dsfa.md.jinja2") as f:
    template = Template(f.read())

# Generate document
output = template.render(**metadata)

# Save
with open("compliance_dpia.md", "w") as f:
    f.write(output)
```

### With the Document Generator

See `examples/generate_compliance_docs.py` for a complete example.

## Template Variables

Both templates expect a metadata dictionary with the following structure:

```python
{
    "system_name": "My AI System",
    "timestamp": "2025-10-30T12:00:00",
    "created_at": "2025-10-30T12:00:00",
    "models": [
        {
            "model_name": "gpt-4",
            "provider": "OpenAI",
            "framework_component": "ChatOpenAI",
            "temperature": 0.7,
            "max_tokens": 500,
            "parameters": {
                "temperature": 0.7,
                "max_tokens": 500
            },
            "timestamp": "2025-10-30T12:00:00"
        }
    ],
    "components": [
        {
            "chain_type": "LLMChain",
            "timestamp": "2025-10-30T12:00:00"
        }
    ],
    "data_sources": [
        {
            "data_source": "./data/training.txt",
            "data_type": "text",
            "loader_type": "TextLoader",
            "timestamp": "2025-10-30T12:00:00"
        }
    ],
    "summary": {
        "total_models": 1,
        "total_components": 1,
        "total_data_sources": 1
    }
}
```

## Customizing Templates

The templates use Jinja2 syntax and can be easily customized:

- Add new sections by editing the `.jinja2` files
- Modify existing sections to match your organization's requirements
- Add conditional blocks for specific use cases
- Include custom styling or formatting

## Compliance Notes

### DSGVO-DSFA Template
- Covers Articles 35-36 of GDPR (Data Protection Impact Assessment)
- Required before deploying high-risk AI systems
- Must be reviewed by Data Protection Officer if applicable
- Should be updated when system or data processing changes

### Article 53 Template
- Fulfills Article 53(1)(d) of EU AI Act
- Must be publicly available for GPAI systems
- Should include sufficiently detailed training data summary
- Must document copyright and licensing information
- Should be updated when training data changes significantly

## Language Support

- **DSGVO-DSFA Template:** German (DSGVO is the German GDPR)
- **Article 53 Template:** English (can be translated as needed)

Both templates include clear markers for fields requiring manual input: `[MANUAL INPUT - description]`

## Further Documentation

- See `docs/USAGE_GUIDE.md` for detailed usage instructions
- See `examples/` for working examples
- See `docs/COVERAGE_ANALYSIS.md` for information on automatic vs. manual fields
