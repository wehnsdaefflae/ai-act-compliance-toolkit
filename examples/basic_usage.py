"""
Basic Usage Example for AI Act Compliance Toolkit

Simple demonstration of plugin integration and metadata capture.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aiact_toolkit import LangChainMonitor

def main():
    """Basic usage example."""
    print("=" * 60)
    print("AI Act Compliance Toolkit - Basic Usage")
    print("=" * 60)
    print()

    # Step 1: Initialize monitor
    print("Step 1: Initialize monitor")
    monitor = LangChainMonitor(system_name="basic_example")
    monitor.start()
    print("✓ Monitor started")
    print()

    # Step 2: Use LangChain components
    print("Step 2: Create LangChain components")
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.7,
        max_tokens=500
    )
    print("✓ ChatOpenAI created")
    print()

    # Step 3: Get captured metadata
    print("Step 3: Get captured metadata")
    metadata = monitor.get_metadata()
    print(f"✓ Captured {metadata['summary']['total_models']} model(s)")
    print()

    # Step 4: Save metadata
    output_file = os.path.join(
        os.path.dirname(__file__),
        "generated_outputs",
        "basic_metadata.json"
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    monitor.save_to_file(output_file)
    print(f"✓ Saved to: {os.path.basename(output_file)}")
    print()

    # Display captured data
    if metadata['models']:
        print("Captured Model:")
        model = metadata['models'][0]
        print(f"  Name: {model['model_name']}")
        print(f"  Provider: {model['provider']}")
        print(f"  Parameters: {model['parameters']}")

    monitor.stop()
    print()
    print("=" * 60)
    print("Example complete! Metadata captured automatically.")
    print("=" * 60)

if __name__ == "__main__":
    main()
