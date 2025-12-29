# AI Act Compliance Toolkit

> Automatisierte Extraktion von Compliance-Metadaten für EU AI Act und DSGVO

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Übersicht

Das **AI Act Compliance Toolkit** ist ein Python-Paket zur automatisierten Extraktion von Compliance-Metadaten aus AI/ML-Anwendungen gemäß EU AI Act und DSGVO-Anforderungen. Es implementiert einen "Compliance-as-Code"-Ansatz, der compliance-relevante Informationen direkt während der Entwicklung erfasst.

### Unterstützte Frameworks

- **LangChain**: Vollständige Integration für LLM-Anwendungen
- **PyTorch**: Monitoring für PyTorch-Modelle, Training und Datasets
- **TensorFlow/Keras**: Monitoring für TensorFlow/Keras-Modelle mit automatischem Callback-System

### Lösungsansatz

Alle Compliance-Informationen existieren bereits im Code:

```python
# Diese Zeile enthält: Modelltyp, Anbieter, Parameter
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# Diese Zeile enthält: Datenquelle, Datentyp
loader = TextLoader("./data/training_data.txt")
```

Das Toolkit erfasst diese Metadaten automatisch während der Entwicklung und verwendet sie zur Befüllung von Compliance-Dokumentvorlagen.

### Funktionsumfang

- **Automatische Metadaten-Extraktion**: Grundlegende Erfassung wichtiger Compliance-Daten
- **Minimale Integration**: 2-3 Codezeilen erforderlich
- **Compliance-Vorlagen**: Erste Prototypen für DSGVO-DSFA und AI Act Artikel 53
- **Hochrisiko-KI-Unterstützung**: Proof-of-Concept mit medizinischem Chatbot (Gesundheitswesen)
- **Framework-Integration**: Initiale Unterstützung für LangChain, PyTorch und TensorFlow

## Schnellstart

### Grundlegende Verwendung

#### LangChain

```python
from aiact_toolkit import LangChainMonitor

# 1. Monitor initialisieren
monitor = LangChainMonitor(system_name="my_ai_app")
monitor.start()

# 2. LangChain normal verwenden - Metadaten werden automatisch erfasst
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7, max_tokens=500)

# 3. Erfasste Metadaten abrufen
metadata = monitor.get_metadata()
monitor.save_to_file("compliance_metadata.json")
```

Vollständiges Beispiel: [examples/basic_usage.py](examples/basic_usage.py)

#### PyTorch

```python
from aiact_toolkit import PyTorchMonitor

# 1. Monitor initialisieren
monitor = PyTorchMonitor(system_name="my_pytorch_model")
monitor.start()

# 2. Modell und Komponenten registrieren
monitor.register_model(model, name="ResNet50")
monitor.register_optimizer(optimizer)
monitor.register_dataset(train_dataset, name="training_data")

# 3. Training-Konfiguration erfassen
monitor.log_training_config(epochs=100, batch_size=32)
monitor.save_to_file("pytorch_metadata.json")
```

Vollständiges Beispiel: [examples/pytorch_example.py](examples/pytorch_example.py)

#### TensorFlow/Keras

```python
from aiact_toolkit import TensorFlowMonitor

# 1. Monitor initialisieren
monitor = TensorFlowMonitor(system_name="my_keras_model")
monitor.start()

# 2. Modell automatisch erfassen
monitor.register_model(model, name="MobileNetV2")

# 3. Keras Callback für automatisches Tracking verwenden
callback = monitor.create_keras_callback()
model.fit(train_data, callbacks=[callback])

monitor.save_to_file("tensorflow_metadata.json")
```

Vollständiges Beispiel: [examples/tensorflow_example.py](examples/tensorflow_example.py)

#### Model Cards generieren

```python
from aiact_toolkit import ModelCardGenerator

# Model Card aus Metadaten generieren
generator = ModelCardGenerator()
model_card = generator.generate_from_metadata(metadata)

# Als Markdown speichern
from aiact_toolkit import DocumentGenerator
doc_gen = DocumentGenerator()
doc_gen.generate_document(
    template_name="model_card.md.jinja2",
    metadata=model_card.to_dict(),
    output_path="model_card.md"
)

# Als JSON speichern
model_card.save_json("model_card.json")
```

#### Compliance-Dokumente generieren

```python
from jinja2 import Environment, FileSystemLoader
env = Environment(loader=FileSystemLoader('templates'))
template = env.get_template('dsgvo_dsfa.md.jinja2')
document = template.render(**metadata)
```

## Erfasste Metadaten

Das Toolkit extrahiert automatisch:

### LangChain

| Metadatentyp | Beispiele | Abdeckung |
|--------------|-----------|-----------|
| **Modellinformationen** | Modellname (gpt-4, llama2), Anbieter (OpenAI, Anthropic) | 45% |
| **Modellparameter** | Temperature, max_tokens, top_p | 50% |
| **Datenquellen** | Dateipfade, Dataset-Identifikatoren | 35% |
| **Datenladen** | Loader-Typen (TextLoader, CSVLoader), Datentypen | 40% |
| **Framework-Komponenten** | Chains, Tools, Prompts | 25% |
| **Anbieterinformationen** | OpenAI, Anthropic, HuggingFace, etc. | 30% |

### PyTorch

| Metadatentyp | Beispiele | Abdeckung |
|--------------|-----------|-----------|
| **Modellarchitektur** | Layer-Typen, Parameter-Anzahl, Modellstruktur | 40% |
| **Training-Konfiguration** | Optimizer, Learning Rate, Batch Size | 35% |
| **Datasets** | Dataset-Typ, Größe, Split (train/val/test) | 30% |
| **Hardware** | GPU/CPU, Device-Platzierung | 50% |
| **Training-Metriken** | Loss, Accuracy pro Epoch | 20% |

### TensorFlow/Keras

| Metadatentyp | Beispiele | Abdeckung |
|--------------|-----------|-----------|
| **Modellarchitektur** | Layer-Konfiguration, Input/Output Shapes | 35% |
| **Training-Konfiguration** | Optimizer, Loss Function, Metrics | 40% |
| **Datasets** | tf.data.Dataset Info, Batch-Konfiguration | 25% |
| **Hardware** | GPU/TPU Verfügbarkeit | 45% |
| **Training-Metriken** | Automatisch via Keras Callback | 30% |

## Compliance-Vorlagen

### 1. Model Cards (EU AI Act Artikel 13 - Transparenz)

Model Cards sind standardisierte Dokumentationen von ML-Modellen, die Transparenz und verantwortungsvolle KI-Praktiken unterstützen.

**Automatisch befüllte Abschnitte:**
- Modelldetails (Name, Version, Architektur, Framework)
- Vorgesehene Verwendungszwecke
- Performance-Metriken
- Trainingsdaten-Informationen
- Ethische Überlegungen
- Einschränkungen und Empfehlungen
- EU AI Act Compliance-Status

**Format-Optionen:**
- Markdown (für Dokumentation)
- JSON (für maschinelle Verarbeitung)

**Verwendung:**
```bash
# Einzelne Model Card generieren
aiact-toolkit generate-model-card metadata.json -o model_card.md

# Model Cards für alle Modelle generieren
aiact-toolkit generate-model-card metadata.json --all -o model_cards/

# Als JSON exportieren
aiact-toolkit generate-model-card metadata.json --format json -o model_card.json
```

### 2. Technische Dokumentation (EU AI Act Artikel 11)

Erforderlich für Hochrisiko-KI-Systeme gemäß EU AI Act Artikel 11. Umfassende technische Dokumentation zur Konformitätsbewertung.

**CLI-Verwendung:**
```bash
# Markdown-Dokumentation generieren
aiact-toolkit generate-technical-doc metadata.json -o technical_documentation.md

# JSON-Format generieren
aiact-toolkit generate-technical-doc metadata.json --format json -o tech_doc.json
```

**Automatisch befüllte Abschnitte:**
- Systemidentifikation und Risikoeinstufung
- Allgemeine Systembeschreibung mit Fähigkeiten und Einschränkungen
- Entwicklungsprozess mit Versionskontrolle und Änderungshistorie
- Architektur und Design (Modelle, Komponenten, Algorithmen)
- Datenanforderungen und Data Governance (Artikel 10)
- Menschliche Aufsichtsmaßnahmen (Artikel 14)
- Leistungsmetriken und Monitoring
- Risikomanagementsystem
- Lifecycle-Management und Post-Market Monitoring
- Konformitätsbewertungsverfahren

**Manuelle Eingabe erforderlich:**
- Deployment-Kontext und Zielgruppe
- Detaillierte Testverfahren
- Bias-Mitigation-Strategien
- Harmonisierte Normen

### 3. DSGVO-DSFA (Datenschutz-Folgenabschätzung)

Erforderlich für Hochrisiko-KI-Systeme, die personenbezogene Daten verarbeiten (DSGVO Artikel 35).

**Automatisch befüllte Abschnitte:**
- Systembeschreibung und Modellinformationen
- Datenquellen und Verarbeitungsvorgänge
- Technische Parameter

**Manuelle Eingabe erforderlich:**
- Rechtsgrundlage der Verarbeitung
- Risikobewertung
- Abhilfemaßnahmen

### 4. Artikel 53 Zusammenfassung (AI Act GPAI-Transparenz)

Erforderlich für General Purpose AI-Systeme (AI Act Artikel 53).

**Automatisch befüllte Abschnitte:**
- Modellidentifikation und Parameter
- Trainingsdatenquellen
- Framework-Komponenten

**Manuelle Eingabe erforderlich:**
- Urheberrechts- und Lizenzinformationen
- Datenherkunft
- Opt-out-Mechanismen

Details: [templates/README.md](templates/README.md)

## Repository-Struktur

```
ai-act-compliance-toolkit/
├── README.md                   # Diese Datei
├── LICENSE                     # MIT-Lizenz
├── requirements.txt            # Python-Abhängigkeiten
│
├── src/aiact_toolkit/
│   ├── __init__.py
│   ├── langchain_monitor.py         # LangChain Integration
│   ├── pytorch_monitor.py           # PyTorch Integration
│   ├── tensorflow_monitor.py        # TensorFlow Integration
│   ├── metadata_storage.py          # Speichermechanismus
│   ├── model_card.py                # Model Card Generator
│   ├── technical_documentation.py   # Technische Dokumentation (Artikel 11)
│   ├── risk_assessment.py           # Risikobewertung
│   ├── audit_trail.py               # Audit Trail (Artikel 12)
│   ├── version_control.py           # Versionskontrolle
│   ├── data_governance.py           # Data Governance (Artikel 10)
│   ├── operational_metrics.py       # Metriken-Tracking
│   ├── document_generator.py        # Dokumentengenerierung
│   └── cli.py                       # CLI-Tool
│
├── templates/
│   ├── model_card.md.jinja2                        # Model Card-Vorlage (Artikel 13)
│   ├── article11_technical_documentation.md.jinja2 # Technische Dokumentation (Artikel 11)
│   ├── dsgvo_dsfa.md.jinja2                        # DSGVO-DSFA-Vorlage
│   ├── article_53_summary.md.jinja2                # Artikel 53-Vorlage
│   ├── risk_assessment_report.md.jinja2            # Risikobewertungs-Bericht
│   ├── audit_report.md.jinja2                      # Audit-Bericht
│   ├── operational_report.md.jinja2                # Operational-Bericht
│   ├── article10_data_governance.md.jinja2         # Artikel 10-Bericht
│   └── README.md                                   # Vorlagen-Dokumentation
│
├── examples/
│   ├── basic_usage.py                           # LangChain Beispiel
│   ├── pytorch_example.py                       # PyTorch Beispiel
│   ├── tensorflow_example.py                    # TensorFlow Beispiel
│   ├── llama2_medical_chatbot_integration.py    # Praxistest
│   ├── generate_llama2_docs.py                  # Dokumentengenerierung
│   └── generated_outputs/                       # Beispielausgaben
│
├── tests/
│   ├── test_langchain_monitor.py           # Unit-Tests
│   ├── test_technical_documentation.py     # Artikel 11 Tests
│   └── test_llama2_medical_chatbot.py      # Integrationstests
│
└── docs/
    ├── USAGE_GUIDE.md              # Detaillierte Anleitung
    ├── COVERAGE_ANALYSIS.md        # Abdeckungsanalyse
    └── TESTING_RESULTS.md          # Testergebnisse
```

## Tests

### Unit-Tests ausführen

```bash
python tests/test_langchain_monitor.py
```

### Integrationstests ausführen

```bash
python tests/test_llama2_medical_chatbot.py
```

### Beispiele ausführen

```bash
# LangChain Beispiel
python examples/basic_usage.py

# PyTorch Beispiel
python examples/pytorch_example.py

# TensorFlow Beispiel
python examples/tensorflow_example.py

# Llama2 Medical Chatbot Integration
python examples/llama2_medical_chatbot_integration.py

# Compliance-Dokumente aus realen Daten generieren
python examples/generate_llama2_docs.py

# Erfasste Metadaten anzeigen
cat examples/generated_outputs/example_metadata.json
```

## Praxisvalidierung

Erfolgreich integriert mit [Llama2 Medical Chatbot](https://github.com/AIAnytime/Llama2-Medical-Chatbot), einem realen medizinischen KI-System im Gesundheitswesen.

**Erfasste Daten aus realem Code:**
- Llama2-7B-Chat-Modell (CTransformers)
  - Temperature: 0.5
  - Max tokens: 512
  - Modelltyp: llama
- Sentence Transformers Embeddings (all-MiniLM-L6-v2)
  - Device: CPU

**Nachweis:** `examples/generated_outputs/example_metadata.json`

**Generierte Dokumente:**
- `example_dpia.md` - DSGVO-Compliance-Dokument mit automatisch befüllten Modelldaten
- `example_article53.md` - AI Act Artikel 53 Zusammenfassung mit realen Konfigurationen

## EU AI Act Compliance

### Hochrisiko-KI-Systeme

Hochrisiko-KI-Systeme (Gesundheitswesen, Bildung, Beschäftigung, kritische Infrastruktur) erfordern umfassende Dokumentation gemäß AI Act Anhänge III, XI und XII.

**Vom Toolkit bereitgestellt:**
- **Model Cards (Artikel 13):** Standardisierte Modelldokumentation für Transparenz
- **Automatische Erfassung** von über 10 erforderlichen Metadatenfeldern
- **Risikobewertungsdokumentation:** Automatische Risikoeinstufung und -berichte
- **DSGVO-DSFA-Integration** für personenbezogene Datenverarbeitung
- **Audit-Trail (Artikel 12):** Vollständige Nachverfolgbarkeit von Modell- und Datennutzung
- **Data Governance (Artikel 10):** Datenherkunft und Qualitätsverfolgung

**Manuell zu ergänzen:**
- Risikobewertung und Abhilfestrategien
- Mechanismen zur menschlichen Aufsicht
- Leistungsmetriken und Validierungsergebnisse
- Post-Market-Monitoring-Verfahren

### GPAI-Systeme (Artikel 53)

Anbieter von General Purpose AI-Systemen müssen Trainingsdaten und Urheberrechts-Compliance dokumentieren.

**Vom Toolkit bereitgestellt:**
- Dokumentation der Trainingsdatenquellen
- Protokollierung der Modellparameter
- Tracking von Framework-Komponenten
- Strukturierte Vorlage für Artikel 53 Zusammenfassung

## Einschränkungen und zukünftige Entwicklungen

Dieses Projekt stellt einen Proof-of-Concept dar, der die grundlegende Machbarkeit des Ansatzes demonstriert. Folgende wesentliche Entwicklungen sind erforderlich, um ein produktionsreifes Tool zu erstellen:

**Kritische Erweiterungen:**
- **Vollständige Framework-Abdeckung**: Erhöhung der Metadaten-Erfassungsrate auf >85% für alle unterstützten Frameworks
- **Erweiterte Framework-Integrationen**: JAX, Scikit-learn, HuggingFace Transformers
- **Validierte Compliance-Vorlagen**: Überprüfung durch Rechtsexperten und Regulierungsbehörden
- **Automatisierte Risikobewertung**: KI-gestützte Analyse zur Bestimmung der Risikoklasse
- **Produktionsinfrastruktur**: Skalierbare Datenbankarchitektur, API-Server, Web-Interface

**Zusätzliche Features:**
- PyPI-Paketverteilung für einfache Installation
- Automatisierte Compliance-Reports mit Visualisierungen und Dashboards
- Integration mit MLOps-Plattformen (MLflow, Weights & Biases, etc.)
- Echtzeit-Monitoring und Alerting für Compliance-Verstöße
- Multi-Sprachen-Unterstützung (Englisch, weitere EU-Sprachen)

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert - siehe [LICENSE](LICENSE) für Details.

## Kontakt

Für Fragen, Issues oder Feedback:
- Issue auf GitHub öffnen
- Repository: <https://github.com/wehnsdaefflae/ai-act-compliance-toolkit>
