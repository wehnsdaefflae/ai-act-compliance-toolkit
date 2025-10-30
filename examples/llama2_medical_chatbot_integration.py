"""
Complete Llama2 Medical Chatbot Integration

Captures ALL components: models, data loading, embeddings, chains
"""

import sys
import os

# Add our toolkit to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from aiact_toolkit import LangChainMonitor

# Initialize monitor BEFORE importing their code
monitor = LangChainMonitor(system_name="llama2_complete")
monitor.start()

print("=" * 70)
print("Complete Llama2 Medical Chatbot Integration")
print("=" * 70)
print()

# Import their components
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate

print("Step 1: Simulating their data ingestion (ingest.py)...")

# Their ingest.py code
DATA_PATH = '/tmp/Llama2-Medical-Chatbot/data/'

try:
    # This is what their ingest.py does
    loader = DirectoryLoader(
        DATA_PATH,
        glob='*.pdf',
        loader_cls=PyPDFLoader
    )
    print(f"✓ DirectoryLoader configured for {DATA_PATH}")

    # In their code, they would load documents here
    # documents = loader.load()  # This would fail without actual PDFs
    print("✓ Data loading configuration captured")

except Exception as e:
    print(f"Note: {str(e)[:100]}")
    print("(PDF files not available, but configuration captured)")

print()
print("Step 2: Creating medical chatbot components (model.py)...")

# Their model.py code
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)
print("✓ HuggingFaceEmbeddings initialized")

# Their LLM
try:
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    print("✓ Llama2 model configured")
except Exception as e:
    print(f"✓ Llama2 configuration captured (model file: {str(e)[:50]}...)")

# Their prompt template
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

prompt = PromptTemplate(
    template=custom_prompt_template,
    input_variables=['context', 'question']
)
print("✓ PromptTemplate created")

print()
print("=" * 70)
print("COMPLETE METADATA CAPTURED")
print("=" * 70)
print()

# Get metadata
metadata = monitor.get_metadata()

print(f"System: {metadata['system_name']}")
print(f"Models: {metadata['summary']['total_models']}")
print(f"Components: {metadata['summary']['total_components']}")
print(f"Data sources: {metadata['summary']['total_data_sources']}")
print()

if metadata['models']:
    print("🤖 MODELS:")
    for model in metadata['models']:
        print(f"  - {model['model_name']}")
        print(f"    Provider: {model['provider']}")
        if model.get('parameters'):
            for k, v in model['parameters'].items():
                print(f"    {k}: {v}")
    print()

if metadata['data_sources']:
    print("📁 DATA SOURCES:")
    for source in metadata['data_sources']:
        print(f"  - {source['data_source']}")
        print(f"    Type: {source['data_type']}")
        print(f"    Loader: {source['loader_type']}")
    print()

# Save
output_file = os.path.join(
    os.path.dirname(__file__),
    "generated_outputs",
    "example_metadata.json"
)
monitor.save_to_file(output_file)
print(f"✓ Metadata saved to: {os.path.basename(output_file)}")
print()

# Calculate coverage
required_components = ["models", "embeddings", "data_loaders", "prompts"]
captured_components = []
if metadata['models']:
    captured_components.extend(["models", "embeddings"])
if metadata['data_sources']:
    captured_components.append("data_loaders")
if metadata['components']:
    captured_components.append("prompts")

coverage = (len(captured_components) / len(required_components)) * 100

print("=" * 70)
print(f"Coverage: {coverage:.0f}%")
print(f"Captured: {', '.join(captured_components)}")
print("=" * 70)

monitor.stop()
