# Datenschutz-Folgenabschätzung (DPIA)
# Data Protection Impact Assessment - GDPR Compliance

**Dokument erstellt am:** 2025-10-30T12:44:45.018687
**System Name:** llama2_complete

---

## 1. Systembeschreibung

### 1.1 Name des KI-Systems
**System:** llama2_complete

### 1.2 Zweck und Einsatzgebiet
**Hauptzweck:** [MANUELL AUSFÜLLEN - Beschreiben Sie den Hauptzweck des Systems]

**Einsatzbereich:** [MANUELL AUSFÜLLEN - z.B. Medizin, Bildung, Beschäftigung, kritische Infrastruktur]

**Zielgruppe:** [MANUELL AUSFÜLLEN - Wer nutzt das System?]

### 1.3 Verwendete Technologien

**KI-Modelle:**


- **sentence-transformers/all-MiniLM-L6-v2** (Anbieter: HuggingFaceEmbeddings)
  - Framework-Komponente: HuggingFaceEmbeddings

  - Parameter:

    - device: cpu



- **TheBloke/Llama-2-7B-Chat-GGML** (Anbieter: CTransformers)
  - Framework-Komponente: CTransformers

  - Parameter:

    - temperature: 0.5

    - max_new_tokens: 512

    - model_type: llama





**Framework-Komponenten:**

[Keine Komponenten erfasst]


---

## 2. Verarbeitete Daten

### 2.1 Art der verarbeiteten Daten
**Datentypen:** [MANUELL AUSFÜLLEN - z.B. personenbezogene Daten, Gesundheitsdaten, biometrische Daten]

**Sensible Daten:** [MANUELL AUSFÜLLEN - Werden besondere Kategorien personenbezogener Daten verarbeitet? (Art. 9 DSGVO)]

### 2.2 Datenquellen und Speicherorte

**Erfasste Datenquellen:**


- **/tmp/Llama2-Medical-Chatbot/data/**
  - Datentyp: directory
  - Loader: DirectoryLoader
  - Erfasst am: 2025-10-30T12:44:40.305525



**Weitere Datenquellen:** [MANUELL AUSFÜLLEN - Zusätzliche nicht automatisch erfasste Datenquellen]

**Speicherorte:** [MANUELL AUSFÜLLEN - Wo werden die Daten gespeichert? Server-Standorte?]

### 2.3 Datenvolumen
**Geschätztes Datenvolumen:** [MANUELL AUSFÜLLEN - Anzahl der Datensätze, Größe in GB]

**Aufbewahrungsdauer:** [MANUELL AUSFÜLLEN - Wie lange werden die Daten gespeichert?]

---

## 3. Zwecke der Datenverarbeitung

### 3.1 Verarbeitungszwecke
**Hauptzweck:** [MANUELL AUSFÜLLEN - Detaillierte Beschreibung des Verarbeitungszwecks]

**Sekundärzwecke:** [MANUELL AUSFÜLLEN - Weitere Verarbeitungszwecke, falls vorhanden]

### 3.2 Rechtsgrundlagen
**Rechtsgrundlage gemäß DSGVO:**
- [ ] Art. 6 Abs. 1 lit. a DSGVO (Einwilligung)
- [ ] Art. 6 Abs. 1 lit. b DSGVO (Vertragserfüllung)
- [ ] Art. 6 Abs. 1 lit. c DSGVO (Rechtliche Verpflichtung)
- [ ] Art. 6 Abs. 1 lit. d DSGVO (Schutz vitaler Interessen)
- [ ] Art. 6 Abs. 1 lit. e DSGVO (Öffentliches Interesse)
- [ ] Art. 6 Abs. 1 lit. f DSGVO (Berechtigtes Interesse)

**Erläuterung:** [MANUELL AUSFÜLLEN - Begründung der Rechtsgrundlage]

---

## 4. Technische und organisatorische Maßnahmen

### 4.1 Zugriffskontrolle
**Authentifizierung:** [MANUELL AUSFÜLLEN - Wie wird der Zugriff kontrolliert?]

**Autorisierung:** [MANUELL AUSFÜLLEN - Rollenbasierte Zugriffskontrolle?]

**Audit-Logs:** [MANUELL AUSFÜLLEN - Werden Zugriffe protokolliert?]

### 4.2 Datensicherheit
**Verschlüsselung:**
- Übertragung: [MANUELL AUSFÜLLEN - TLS/SSL?]
- Speicherung: [MANUELL AUSFÜLLEN - Verschlüsselung im Ruhezustand?]

**Backup-Strategie:** [MANUELL AUSFÜLLEN - Wie werden Backups erstellt und geschützt?]

**Incident Response:** [MANUELL AUSFÜLLEN - Prozess für Datenschutzverletzungen]

### 4.3 Modelparameter-Sicherheit

Folgende Modellparameter wurden für Compliance erfasst:

- **sentence-transformers/all-MiniLM-L6-v2**: Temperatur nicht angegeben, Max Tokens nicht angegeben

- **TheBloke/Llama-2-7B-Chat-GGML**: Temperatur 0.5, Max Tokens nicht angegeben


**Begründung der Parameter:** [MANUELL AUSFÜLLEN - Warum wurden diese Parameter gewählt? Risikominimierung?]


---

## 5. Risikobewertung

### 5.1 Identifizierte Risiken

**Risiko 1:** [MANUELL AUSFÜLLEN]
- **Beschreibung:** [Detaillierte Beschreibung des Risikos]
- **Betroffene Personen:** [Wer ist betroffen?]
- **Eintrittswahrscheinlichkeit:** ☐ Niedrig ☐ Mittel ☐ Hoch
- **Schadenshöhe:** ☐ Niedrig ☐ Mittel ☐ Hoch
- **Gesamtrisiko:** ☐ Niedrig ☐ Mittel ☐ Hoch

**Risiko 2:** [MANUELL AUSFÜLLEN]
- **Beschreibung:** [Detaillierte Beschreibung des Risikos]
- **Betroffene Personen:** [Wer ist betroffen?]
- **Eintrittswahrscheinlichkeit:** ☐ Niedrig ☐ Mittel ☐ Hoch
- **Schadenshöhe:** ☐ Niedrig ☐ Mittel ☐ Hoch
- **Gesamtrisiko:** ☐ Niedrig ☐ Mittel ☐ Hoch

### 5.2 Spezifische KI-Risiken

**Bias und Diskriminierung:**

Verwendete Modelle: sentence-transformers/all-MiniLM-L6-v2, TheBloke/Llama-2-7B-Chat-GGML

- **Risiko:** [MANUELL AUSFÜLLEN - Können die Modelle diskriminierende Entscheidungen treffen?]
- **Maßnahmen:** [MANUELL AUSFÜLLEN - Wie wird Bias erkannt und verhindert?]

**Transparenz und Erklärbarkeit:**
- **Risiko:** [MANUELL AUSFÜLLEN - Sind die Entscheidungen des Systems nachvollziehbar?]
- **Maßnahmen:** [MANUELL AUSFÜLLEN - Wie wird Transparenz gewährleistet?]

**Datenqualität:**

Erfasste 1 Datenquelle(n).

- **Risiko:** [MANUELL AUSFÜLLEN - Ist die Qualität der Trainingsdaten ausreichend?]
- **Maßnahmen:** [MANUELL AUSFÜLLEN - Wie wird die Datenqualität sichergestellt?]

---

## 6. Maßnahmen zur Risikominimierung

### 6.1 Technische Maßnahmen
1. [MANUELL AUSFÜLLEN - z.B. Implementierung von Fairness-Metriken]
2. [MANUELL AUSFÜLLEN - z.B. Explainable AI-Techniken]
3. [MANUELL AUSFÜLLEN - z.B. Kontinuierliches Monitoring]

### 6.2 Organisatorische Maßnahmen
1. [MANUELL AUSFÜLLEN - z.B. Schulung der Mitarbeiter]
2. [MANUELL AUSFÜLLEN - z.B. Regelmäßige Audits]
3. [MANUELL AUSFÜLLEN - z.B. Dokumentation und Versionierung]

### 6.3 Menschliche Aufsicht
**Human-in-the-loop:** [MANUELL AUSFÜLLEN - Wie ist menschliche Kontrolle implementiert?]

**Eskalationsprozesse:** [MANUELL AUSFÜLLEN - Wann und wie werden Menschen eingebunden?]

---

## 7. Betroffenenrechte

### 7.1 Informationspflichten
**Transparenz:** [MANUELL AUSFÜLLEN - Wie werden Betroffene informiert?]

**Dokumentation:** [MANUELL AUSFÜLLEN - Welche Informationen werden bereitgestellt?]

### 7.2 Auskunftsrecht
**Prozess:** [MANUELL AUSFÜLLEN - Wie können Betroffene Auskunft verlangen?]

### 7.3 Weitere Rechte
- Recht auf Berichtigung
- Recht auf Löschung
- Recht auf Einschränkung der Verarbeitung
- Recht auf Widerspruch
- Recht auf Datenübertragbarkeit

**Umsetzung:** [MANUELL AUSFÜLLEN - Wie werden diese Rechte technisch umgesetzt?]

---

## 8. Dokumentation und Versionierung

**Dokument-Version:** 1.0
**Erstellt am:** 2025-10-30T12:44:45.018687
**Nächste Überprüfung:** [MANUELL AUSFÜLLEN - Datum]

**Verantwortliche Person:**
- Name: [MANUELL AUSFÜLLEN]
- Position: [MANUELL AUSFÜLLEN]
- Kontakt: [MANUELL AUSFÜLLEN]

**Datenschutzbeauftragter:**
- Name: [MANUELL AUSFÜLLEN]
- Kontakt: [MANUELL AUSFÜLLEN]

---

## 9. Zusammenfassung und Empfehlung

**Gesamtbewertung des Risikos:** ☐ Niedrig ☐ Mittel ☐ Hoch

**Empfehlung:** [MANUELL AUSFÜLLEN - Kann das System in Betrieb genommen werden? Welche Auflagen gibt es?]

**Unterschrift:** ________________________________
**Datum:** ________________________________

---

*Dieses Dokument wurde teilweise automatisch generiert durch das AI Act Compliance Toolkit.*
*Automatisch erfasste Daten: 2 Modell(e), 0 Komponente(n), 1 Datenquelle(n).*
*Alle mit [MANUELL AUSFÜLLEN] markierten Felder müssen von einem Verantwortlichen ausgefüllt werden.*