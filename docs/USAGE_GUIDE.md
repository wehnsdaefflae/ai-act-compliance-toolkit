# Usage Guide - AI Act Compliance Toolkit

This guide provides detailed instructions for using the AI Act Compliance Toolkit to automatically extract compliance metadata from your LangChain applications.

## Installation

```bash
pip install -r requirements.txt
```

## Basic Workflow

### 1. Initialize the Monitor

```python
from aiact_toolkit import LangChainMonitor

monitor = LangChainMonitor(system_name="my_ai_application")
monitor.start()
```

The `system_name` parameter identifies your AI system in the generated compliance documents.

### 2. Use LangChain Normally

Once the monitor is active, all LangChain operations are automatically tracked:

```python
from langchain_openai import ChatOpenAI
from langchain.document_loaders import TextLoader

# Model instantiation is automatically captured
llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.7,
    max_tokens=500
)

# Data loading operations are captured
loader = TextLoader("./data/training_data.txt")
documents = loader.load()

# LLM calls are logged
response = llm.invoke("What are the compliance requirements?")
```

### 3. Retrieve Metadata

Access the captured metadata at any time:

```python
# Get metadata as dictionary
metadata = monitor.get_metadata()

# Save to JSON file
monitor.save_to_file("compliance_metadata.json")

# Stop monitoring
monitor.stop()
```

### 4. Generate Compliance Documents

Use the captured metadata with provided templates:

```python
from jinja2 import Environment, FileSystemLoader

# Load template environment
env = Environment(loader=FileSystemLoader('templates'))

# Generate GDPR DPIA document
dpia_template = env.get_template('dsgvo_dsfa.md.jinja2')
dpia_document = dpia_template.render(**metadata)

with open('dpia_document.md', 'w') as f:
    f.write(dpia_document)

# Generate AI Act Article 53 summary
article53_template = env.get_template('article_53_summary.md.jinja2')
article53_document = article53_template.render(**metadata)

with open('article53_summary.md', 'w') as f:
    f.write(article53_document)
```

## Captured Metadata Fields

The toolkit automatically extracts:

### Model Information
- `model_name`: Name of the language model (e.g., "gpt-4", "llama2-7b-chat")
- `provider`: Model provider (OpenAI, Anthropic, HuggingFace, etc.)
- `model_type`: Type of model (chat, completion, embeddings)

### Model Parameters
- `temperature`: Randomness control (0.0-1.0)
- `max_tokens`: Maximum output length
- `top_p`: Nucleus sampling parameter
- Additional provider-specific parameters

### Data Sources
- `file_paths`: List of loaded data files
- `loader_types`: Types of data loaders used
- `data_types`: Formats of loaded data (text, CSV, JSON, etc.)

### Framework Components
- `chains`: LangChain chains used
- `tools`: External tools integrated
- `prompts`: Prompt templates utilized

## Complete Example

See `examples/basic_usage.py` for a complete working example:

```bash
python examples/basic_usage.py
```

## High-Risk AI Systems

For high-risk AI systems (healthcare, employment, education), additional manual documentation is required:

1. Risk assessment and mitigation strategies
2. Human oversight mechanisms
3. Performance metrics and validation results
4. Post-market monitoring procedures

The toolkit provides the technical foundation; you must supplement with risk and governance documentation.

## GPAI Systems (Article 53)

For General Purpose AI systems, ensure you document:

1. Copyright and licensing information for training data
2. Data provenance and sources
3. Opt-out mechanisms for content creators
4. Model evaluation and testing procedures

## Troubleshooting

### Monitor Not Capturing Data

Ensure the monitor is started before creating LangChain components:

```python
monitor.start()  # Must be called first
llm = ChatOpenAI(...)  # Then create components
```

### Missing Metadata Fields

Some metadata requires specific LangChain versions or integrations. Check that:
- LangChain is version 0.1.0 or higher
- Required integration packages are installed (langchain-openai, langchain-anthropic, etc.)

### Template Rendering Errors

Verify that all required template variables are present:

```python
# Check what metadata was captured
print(json.dumps(metadata, indent=2))

# Templates may require manual fields
metadata['legal_basis'] = 'GDPR Article 6(1)(b)'
metadata['risk_level'] = 'High'
```

## Next Steps

1. Review generated compliance documents
2. Fill in manual sections marked with "TODO" or "MANUAL INPUT REQUIRED"
3. Conduct formal risk assessment
4. Validate technical documentation with legal team
5. Establish monitoring and audit procedures

## Support

For issues or questions:
- Check existing examples in `examples/`
- Run tests to verify installation: `python tests/test_langchain_monitor.py`
- Open an issue on GitHub
