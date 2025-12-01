"""
Setup configuration for AI Act Compliance Toolkit
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [
            line.strip() for line in f
            if line.strip() and not line.startswith('#')
        ]
else:
    requirements = []

setup(
    name="aiact-toolkit",
    version="0.1.0",
    author="AI Act Compliance Toolkit Contributors",
    description="Automated extraction of compliance metadata for EU AI Act and GDPR",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wehnsdaefflae/ai-act-compliance-toolkit",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "aiact_toolkit": [],
    },
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "aiact-toolkit=aiact_toolkit.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="ai-act compliance gdpr langchain metadata automation documentation",
    project_urls={
        "Bug Reports": "https://github.com/wehnsdaefflae/ai-act-compliance-toolkit/issues",
        "Source": "https://github.com/wehnsdaefflae/ai-act-compliance-toolkit",
    },
)
