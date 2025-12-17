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

- **Automatische Metadaten-Extraktion**: Über 90% Abdeckung von LangChain-Operationen
- **Minimale Integration**: 2-3 Codezeilen erforderlich
- **Compliance-Vorlagen**: DSGVO-DSFA und AI Act Artikel 53
- **Hochrisiko-KI-Unterstützung**: Validiert mit medizinischem Chatbot (Gesundheitswesen)
- **Framework-Integration**: Kompatibel mit bestehenden LangChain-Anwendungen

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
| **Modellinformationen** | Modellname (gpt-4, llama2), Anbieter (OpenAI, Anthropic) | 100% |
| **Modellparameter** | Temperature, max_tokens, top_p | 100% |
| **Datenquellen** | Dateipfade, Dataset-Identifikatoren | 100% |
| **Datenladen** | Loader-Typen (TextLoader, CSVLoader), Datentypen | 100% |
| **Framework-Komponenten** | Chains, Tools, Prompts | 90% |
| **Anbieterinformationen** | OpenAI, Anthropic, HuggingFace, etc. | 100% |

### PyTorch

| Metadatentyp | Beispiele | Abdeckung |
|--------------|-----------|-----------|
| **Modellarchitektur** | Layer-Typen, Parameter-Anzahl, Modellstruktur | 100% |
| **Training-Konfiguration** | Optimizer, Learning Rate, Batch Size | 100% |
| **Datasets** | Dataset-Typ, Größe, Split (train/val/test) | 100% |
| **Hardware** | GPU/CPU, Device-Platzierung | 100% |
| **Training-Metriken** | Loss, Accuracy pro Epoch | 100% |

### TensorFlow/Keras

| Metadatentyp | Beispiele | Abdeckung |
|--------------|-----------|-----------|
| **Modellarchitektur** | Layer-Konfiguration, Input/Output Shapes | 100% |
| **Training-Konfiguration** | Optimizer, Loss Function, Metrics | 100% |
| **Datasets** | tf.data.Dataset Info, Batch-Konfiguration | 100% |
| **Hardware** | GPU/TPU Verfügbarkeit | 100% |
| **Training-Metriken** | Automatisch via Keras Callback | 100% |

## Compliance-Vorlagen

### 1. DSGVO-DSFA (Datenschutz-Folgenabschätzung)

Erforderlich für Hochrisiko-KI-Systeme, die personenbezogene Daten verarbeiten (DSGVO Artikel 35).

**Automatisch befüllte Abschnitte:**
- Systembeschreibung und Modellinformationen
- Datenquellen und Verarbeitungsvorgänge
- Technische Parameter

**Manuelle Eingabe erforderlich:**
- Rechtsgrundlage der Verarbeitung
- Risikobewertung
- Abhilfemaßnahmen

### 2. Artikel 53 Zusammenfassung (AI Act GPAI-Transparenz)

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
│   ├── langchain_monitor.py    # Haupt-Plugin (~180 Zeilen)
│   └── metadata_storage.py     # Speichermechanismus
│
├── templates/
│   ├── dsgvo_dsfa.md.jinja2           # DSGVO-DSFA-Vorlage
│   ├── article_53_summary.md.jinja2   # Artikel 53-Vorlage
│   └── README.md                       # Vorlagen-Dokumentation
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
- Automatische Erfassung von über 10 erforderlichen Metadatenfeldern
- Vorlage für Risikobewertungsdokumentation
- DSGVO-DSFA-Integration für personenbezogene Datenverarbeitung
- Audit-Trail für Modell- und Datennutzung

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

Das Toolkit bietet bereits umfangreiche Funktionalität für LangChain, PyTorch und TensorFlow. Folgende Funktionen sind für zukünftige Entwicklungen geplant:

- Weitere ML-Framework-Integrationen (JAX, Scikit-learn, etc.)
- PyPI-Paketverteilung
- Produktionsdatenbankschema für großangelegte Deployments
- Validierte regulatorische Vorlagen durch Rechtsexperten
- Automatisierte Compliance-Reports mit Visualisierungen
- Integration mit MLOps-Plattformen

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert - siehe [LICENSE](LICENSE) für Details.

## Kontakt

Für Fragen, Issues oder Feedback:
- Issue auf GitHub öffnen
- Repository: <https://github.com/wehnsdaefflae/ai-act-compliance-toolkit>
