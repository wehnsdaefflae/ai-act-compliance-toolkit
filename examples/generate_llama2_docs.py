"""
Generate Compliance Documents from Real Llama2 Medical Chatbot Metadata
"""

import sys
import os
import json
from jinja2 import Environment, FileSystemLoader

def main():
    # Load real Llama2 metadata
    metadata_file = os.path.join(
        os.path.dirname(__file__),
        "generated_outputs",
        "example_metadata.json"
    )

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    print("Generating compliance documents from REAL Llama2 Medical Chatbot...")
    print()

    # Setup templates
    template_dir = os.path.join(os.path.dirname(__file__), '..', 'templates')
    env = Environment(loader=FileSystemLoader(template_dir))

    # Generate GDPR DPIA
    dpia_template = env.get_template('dsgvo_dsfa.md.jinja2')
    dpia_output = dpia_template.render(**metadata)

    dpia_file = os.path.join(
        os.path.dirname(__file__),
        "generated_outputs",
        "example_dpia.md"
    )
    with open(dpia_file, 'w', encoding='utf-8') as f:
        f.write(dpia_output)

    print(f"✓ GDPR DPIA generated from real Llama2 chatbot")

    # Generate Article 53
    article53_template = env.get_template('article_53_summary.md.jinja2')
    article53_output = article53_template.render(**metadata)

    article53_file = os.path.join(
        os.path.dirname(__file__),
        "generated_outputs",
        "example_article53.md"
    )
    with open(article53_file, 'w', encoding='utf-8') as f:
        f.write(article53_output)

    print(f"✓ Article 53 Summary generated from real Llama2 chatbot")
    print()
    print("Documents contain REAL captured metadata from:")
    print("  - Llama2-7B-Chat model (temperature: 0.5, tokens: 512)")
    print("  - sentence-transformers embeddings (all-MiniLM-L6-v2)")
    print()

if __name__ == "__main__":
    main()
