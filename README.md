Powered by Georg Plessl
Version 1.1
Date: 2026-01-16

# Transmissionsmessung Pipeline (Bachelorarbeit)

Dieses Repository verarbeitet SOLOS/JOIA-LM XML-Transmissionsmessungen fuer
Brillenglaeser und erzeugt ISO 8980-3 Kennwerte, Kategoriezuordnungen und
Plots. Die Pipeline folgt DIN EN ISO 8980-3:2022 (+A1:2025) und ordnet die
Kategorien nach EN ISO 12312-1 zu.

## Funktionen
- SOLOS XML-Exporte parsen und UV/VIS-Transmissionskennwerte berechnen
- Glaser in ISO 12312-1 Kategorien einordnen
- Summary-Tabellen, Linsen-Aggregate und Plots generieren
- Optionale ANOVA/Tukey-Analyse, wenn statsmodels installiert ist

## Repository-Struktur
- `quellcode/scripts/process_transmission.py`: Haupt-Pipeline
- `data/raw_xml/`: Eingangsdaten (XML-Messungen)
- `outputs/`: Standard-Ausgaben (Summaries, Spektren, Plots)
- `outputs_real/`: separater Lauf fuer den realen Messdatensatz
- `Normen/`: normative PDF-Quellen
- `tests/`: pytest Suite

## Schnellstart
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 quellcode/scripts/process_transmission.py --in_dir data/raw_xml --out_dir outputs
```

Optionale Statistik-Abhaengigkeiten:
```bash
python3 -m pip install -r requirements-dev.txt
```

## Outputs
- `outputs/summary.csv`: eine Zeile pro XML-Messung
- `outputs/aggregated/by_glass.csv`: Linsen-Aggregation fuer Analysen
- `outputs/plots/`: Plots pro Messung und aggregierte Plots

## Tests
```bash
pytest
```

## Hinweise
- `outputs_real/` fuer den separaten Analyse-Lauf in diesem Repository verwenden.
- XML-Eingaben in `data/raw_xml/` belassen und keine Rohdaten mit generierten Outputs mischen.
