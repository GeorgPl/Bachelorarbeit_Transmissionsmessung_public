# Georg Project

This repository processes SOLOS XML transmission measurements, produces CSV
summaries, plots, and a PDF report, and includes a static HTML reference page.

## Repository Layout
- `data/`: input XML files and other data assets.
- `outputs/`: generated spectra CSVs, plots, and aggregated reports.
- `scripts/`: processing and report generation scripts.
- `georg.html`: static reference page.
- `AGENTS.md`: engineering and quality gate requirements.

## Dependencies (Current)
Python 3.x with:
- `numpy`
- `pandas`
- `matplotlib`
- `statsmodels` (optional, only for ANOVA helper)

Note: dependencies are not pinned yet (see Findings).

## Usage
Process all XML files:
```bash
python3 scripts/process_transmission.py --in_dir data/raw_xml --out_dir outputs
```

Generate sample reference filter XMLs:
```bash
python3 scripts/generate_ref_filters.py
```

Build PDF report from aggregated outputs:
```bash
python3 scripts/export_report.py \
  --by-glass outputs/aggregated/by_glass.csv \
  --qc outputs/aggregated/qc_filters.csv \
  --summary outputs/summary.csv \
  --out-pdf outputs/aggregated/report.pdf \
  --author "Your Name"
```

## Outputs
- `outputs/summary.csv`: per-file metrics.
- `outputs/aggregated/by_glass.csv`: per-glass aggregates.
- `outputs/aggregated/qc_filters.csv`: reference filter QC (if detected).
- `outputs/spectra/`: per-file spectra CSVs.
- `outputs/plots/`: per-file spectra plots.
- `outputs/aggregated/plots/`: per-glass boxplots.

## Findings (Current Review)
- HIGH: Reference filter detection ignores underscores (e.g., `NDUVW_10B`), so QC
  is skipped and those entries are treated as normal glasses. Affects
  `scripts/process_transmission.py` and files like
  `data/raw_xml/NDUVW_10B_04_20251218_120040_TOPCON_SOLOS_420001417.xml`.
- HIGH: Per-file processing exceptions are caught and only logged; CSV/PDF
  generation can complete with partial data, and report CSV read failures are
  suppressed in `scripts/export_report.py`.
- MEDIUM: Outputs are not deterministic due to random jitter in boxplots and a
  timestamp embedded in the PDF title page.
- MEDIUM: Spectral coverage checks only warn; results continue without being
  flagged invalid.
- LOW: Report text claims strict separation of glasses vs reference filters,
  but aggregation does not exclude filters before `by_glass.csv`.
- MEDIUM: Reproducibility/testing gates are missing: no dependency pin/lockfile
  (`requirements.txt`/`pyproject.toml`) and no automated tests.

## Copyright
Copyright: Mr. Georg Plessl Version 1.01 29.12.2025
