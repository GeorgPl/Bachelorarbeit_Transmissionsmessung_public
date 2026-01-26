#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SOLOS Transmission Processor
DIN EN ISO 8980-3:2022 (+ A1:2025 Wellenlängengrenze) + 10x-Auswertung + Referenzfilter-QC + Boxplots
----------------------------------------------------------------------------------------------

Dieses Skript wertet spektrale Transmissionsmessungen von Brillengläsern aus
und berechnet insbesondere:

- photopische Transmission τ_v (D65·V-gewichtet, 380–780 nm),
- τV nach DIN EN ISO 8980-3 (identische Berechnung, explizit als Normgröße),
- UV-gewichtete Transmission τ_SUVB (280–315 nm) und τ_SUVA (315–380 nm) nach Anhang B,
- solarer Blaulicht-Transmissionsgrad τ_SB (380–500 nm) nach Anhang B,
- optionale Anzeigegrößen (einfache Mittelwerte in definierten Spektralbereichen),
- Kategorie-Zuordnung (0–4) nach EN ISO 12312-1 auf Basis τV,
- einfache QC-Auswertung für ND-Referenzfilter,
- 10x-Statistik je Glas (Mittelwert, SD, 95%-CI, Repeatability-Limit),
- Boxplots je Glas und Kennwert,
- Spektren-CSV und Spektrenplots.

Hinweis:
Die Änderung EN ISO 8980-3/A1:2025 führt Anforderungen an die deklarierte
Wellenlängengrenze ein; diese wird hier anhand der Messdaten geprüft/abgeleitet.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import re
from typing import Tuple, Optional, List, Dict

# Matplotlib-Cache in ein schreibbares Verzeichnis legen, um Warnungen zu vermeiden.
if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path.cwd() / ".mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

import matplotlib
# Agg-Backend erzwingt headless Rendering und vermeidet GUI-Abhaengigkeiten.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==============================================================
# Normative Wellenlängenbereiche (DIN EN ISO 8980-3:2022)
# ==============================================================

WL_MIN_NORM = 280.0  # untere Grenze für UVB-Bereich
WL_MAX_NORM = 780.0  # obere Grenze für sichtbaren Bereich (D65·V)

PHOTOPIC_RANGE = (380.0, 780.0)
UVB_RANGE = (280.0, 315.0)
UVA_RANGE = (315.0, 380.0)
SOLAR_BLUE_RANGE = (380.0, 500.0)

# Numpy-Kompatibilitaet: trapezoid ist erst in neueren Versionen vorhanden.
_TRAPZ = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

# ==============================================================
# Gewichtungsdaten
# ==============================================================

# Normative Hinweise (ISO 8980-3:2022 / A1:2025, Normen/):
# - τV nach Gl. (E.4): Summe τ(λ)*S_D65(λ)*V(λ)*Δλ / Summe S_D65(λ)*V(λ)*Δλ, 380–780 nm.
# - UV-Transmissionen: W(λ)=E_S(λ)*S(λ) (Tabelle B.1), UVB 280–315 nm, UVA 315–380 nm;
#   E_S(λ) kann zwischen 280–290 nm zu 0 gesetzt werden; lineare Interpolation zwischen Stützstellen.
# - τ_SB: Gewichtung E_S(λ)*B(λ) (Tabelle B.1), 380–500 nm.
# - A1:2025 Wellenlängengrenze: T(λ_decl)<5% und T<1% für 280..λ_decl-10; Auflösung <=1 nm,
#   Messbereich 280–780 nm, Bandbreite <=2 nm (Bandbreite im XML nicht prüfbar).

# Tabelle A.2: Produkt S_D65(λ)·V(λ), 5-nm-Raster (DIN EN ISO 8980-3:2022).
D65V_A5 = {
    380: 0.0001, 385: 0.0002, 390: 0.0003, 395: 0.0007, 400: 0.0016,
    405: 0.0026, 410: 0.0052, 415: 0.0095, 420: 0.0177, 425: 0.0311,
    430: 0.0476, 435: 0.0763, 440: 0.1141, 445: 0.1564, 450: 0.2104,
    455: 0.2667, 460: 0.3345, 465: 0.4068, 470: 0.4945, 475: 0.6148,
    480: 0.7625, 485: 0.9001, 490: 1.0710, 495: 1.3347, 500: 1.6713,
    505: 2.0925, 510: 2.5657, 515: 3.0589, 520: 3.5203, 525: 3.9873,
    530: 4.3922, 535: 4.5905, 540: 4.7128, 545: 4.8343, 550: 4.8981,
    555: 4.8272, 560: 4.7078, 565: 4.5455, 570: 4.3393, 575: 4.1607,
    580: 3.9431, 585: 3.5626, 590: 3.1766, 595: 2.9377, 600: 2.6873,
    605: 2.4084, 610: 2.1324, 615: 1.8506, 620: 1.5810, 625: 1.2985,
    630: 1.0443, 635: 0.8573, 640: 0.6931, 645: 0.5353, 650: 0.4052,
    655: 0.3093, 660: 0.2315, 665: 0.1714, 670: 0.1246, 675: 0.0881,
    680: 0.0630, 685: 0.0417, 690: 0.0271, 695: 0.0191, 700: 0.0139,
    705: 0.0101, 710: 0.0074, 715: 0.0048, 720: 0.0031, 725: 0.0023,
    730: 0.0017, 735: 0.0012, 740: 0.0009, 745: 0.0006, 750: 0.0004,
    755: 0.0002, 760: 0.0001, 765: 0.0001, 770: 0.0001, 775: 0.0001, 780: 0.0000,
}

# Tabelle B.1: Solare spektrale Bestrahlungsstaerke E_S(λ), 5-nm-Raster (340–500 nm).
E_SOLAR_340_500 = {
    340: 151, 345: 170, 350: 188, 355: 210, 360: 233, 365: 253, 370: 279,
    375: 306, 380: 336, 385: 365, 390: 397, 395: 432, 400: 470, 405: 562,
    410: 672, 415: 705, 420: 733, 425: 760, 430: 787, 435: 849, 440: 911,
    445: 959, 450: 1006, 455: 1037, 460: 1080, 465: 1109, 470: 1138,
    475: 1161, 480: 1183, 485: 1197, 490: 1210, 495: 1213, 500: 1215,
}

# Tabelle B.1: Funktion fuer Gefaehrdung durch blaues Licht B(λ), 5-nm-Raster (380–500 nm).
B_BLUE_HAZARD_B5 = {
    380: 0.006, 385: 0.012, 390: 0.025, 395: 0.050, 400: 0.100,
    405: 0.200, 410: 0.400, 415: 0.800, 420: 0.900, 425: 0.950,
    430: 0.980, 435: 1.000, 440: 1.000, 445: 0.970, 450: 0.940,
    455: 0.900, 460: 0.800, 465: 0.700, 470: 0.620, 475: 0.550,
    480: 0.450, 485: 0.400, 490: 0.220, 495: 0.160, 500: 0.100,
}

# ISO 8980-3 Anhang B: W(λ) fuer UV (280–380 nm, 5-nm-Raster).
W_UV_B5 = {
    280: 0.0, 285: 0.0, 290: 0.0, 295: 0.00011, 300: 0.0243,
    305: 0.115, 310: 0.165, 315: 0.090, 320: 0.054, 325: 0.040,
    330: 0.041, 335: 0.044, 340: 0.042, 345: 0.041, 350: 0.038,
    355: 0.034, 360: 0.030, 365: 0.028, 370: 0.026, 375: 0.024, 380: 0.022,
}

_WEIGHT_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}


def _interp_weights(table: Dict[int, float], wl_nm: np.ndarray) -> np.ndarray:
    """Lineare Interpolation einer Gewichtungstabelle auf ein Wellenlängenraster."""
    # Cache spart wiederholtes Sortieren/Vectorisieren der fixen Tabellen.
    cache_key = id(table)
    cached = _WEIGHT_CACHE.get(cache_key)
    if cached is None:
        xs = np.array(sorted(table.keys()), dtype=float)
        ys = np.array([table[k] for k in xs], dtype=float)
        _WEIGHT_CACHE[cache_key] = (xs, ys)
    else:
        xs, ys = cached
    # Ausserhalb des Tabellenbereichs wird die Gewichtung auf 0 gesetzt.
    return np.interp(wl_nm, xs, ys, left=0.0, right=0.0)


# ==============================================================
# Kategorie-Schwellen (EN ISO 12312-1, Tabelle 2)
# ==============================================================


def categorize_by_tauV(tauV_percent: float) -> Tuple[str, Optional[int], bool]:
    """
    Kategorie-Zuordnung nach EN ISO 12312-1 anhand τV [%].

    Rückgabe: (status, category_or_None, driving_ok)
      status: "ok" | "out_of_range"
      category: 0..4 oder None
      driving_ok: False bei Kategorie 4, sonst True
    """
    if not np.isfinite(tauV_percent):
        return ("out_of_range", None, False)
    # Grenzen nach Tabelle 2: Cat0 >80; Cat1: 80≥τv>43; Cat2: 43≥τv>18; Cat3: 18≥τv>8; Cat4: 8≥τv>3.
    if tauV_percent > 80.0:
        return ("ok", 0, True)
    if tauV_percent > 43.0:
        return ("ok", 1, True)
    if tauV_percent > 18.0:
        return ("ok", 2, True)
    if tauV_percent > 8.0:
        return ("ok", 3, True)
    if tauV_percent > 3.0:
        return ("ok", 4, False)
    return ("out_of_range", None, False)


# ==============================================================
# Referenzfilter-Erkennung (QC)
# ==============================================================

REF_FILTERS = {
    "NDUVW01B": {"nominal_tauV_percent": 79.0, "tol_percent": 5.0},   # OD 0.1 → ~79 %
    "NDUVW05B": {"nominal_tauV_percent": 31.6, "tol_percent": 5.0},   # OD 0.5 → ~31.6 %
    "NDUVW10B": {"nominal_tauV_percent": 10.0, "tol_percent": 5.0},   # OD 1.0 → ~10 %
}

# Erlaubt sowohl NDUVW01B als auch NDUVW_01B etc. (verschiedene Export-Varianten).
REF_REGEX = re.compile(
    r"(NDUVW)(?:_)?(01B|03B|10B)|(^REF_)",
    re.IGNORECASE,
)
# Vorcompilierter Zahlen-Regex fuer die XML-Werte.
NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")



def detect_ref_id(stem: str) -> Optional[str]:
    m = REF_REGEX.search(stem)
    if not m:
        return None
    if m.group(1) and m.group(2):
        return f"{m.group(1).upper()}{m.group(2).upper()}"
    # Kein konkretes Filterkuerzel, aber "REF_" im Namen.
    return "REF_GENERIC"


# ==============================================================
# ID-Hilfsfunktionen: Hersteller & Glas-ID aus Dateinamen
# ==============================================================

def manufacturer_from_filename(stem: str) -> str:
    """
    Versucht, den Hersteller aus dem Dateinamen abzuleiten.

    Erwartete Konvention (Beispiele):
      "ZEISS_G01_rep01" -> "ZEISS"
      "HOYA_G05_03"     -> "HOYA"

    Fallback: erstes Segment vor dem Unterstrich.
    """
    return stem.split("_")[0]


def glass_id_from_filename(stem: str) -> str:
    """
    Glas-ID als Kombination aus Hersteller und Glasnummer.

    Beispiel:
      "ZEISS_G01_rep01" -> "ZEISS_G01"
      "HOYA_G02_rep03"  -> "HOYA_G02"

    Fallback: alles vor dem ersten Unterstrich.
    """
    parts = stem.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return parts[0]


# ==============================================================
# XML-Reader (robust für SOLOS/JOIA-LM)
# ==============================================================

def read_solos_xml(xml_path: Path) -> pd.DataFrame:
    """
    Liest ein SOLOS/JOIA-LM XML-Spektrum mit einzelnen <WaveLength>/<Transmittance>-Knoten.

    Erwartet:
      - Wellenlängen-Elemente mit numerischem Text
      - Transmittance-Elemente (ohne UV-Summaries) mit numerischem Text

    Rückgabe:
      DataFrame mit Spalten ['wavelength_nm', 'T_percent'] (aufsteigend nach Wellenlänge).
    """
    root = ET.parse(xml_path).getroot()
    wl: List[float] = []
    tr: List[float] = []

    # XML enthaelt oft viele Knoten; wir sammeln nur numerische WL/Transmission.
    for elem in root.iter():
        tag = elem.tag.lower()
        txt = (elem.text or "").strip()
        if not txt:
            continue
        # NUM_RE stellt sicher, dass nur reine Zahlwerte eingelesen werden.
        if "wavelength" in tag and NUM_RE.fullmatch(txt):
            wl.append(float(txt))
        elif "transmitt" in tag and ("uv" not in tag) and NUM_RE.fullmatch(txt):
            # UV-Zusammenfassungen werden uebersprungen, nur Spektralwerte.
            tr.append(float(txt))

    # Robustheit: gelegentlich ist ein Transmittance-Wert zu viel vorhanden
    if len(tr) == len(wl) + 1:
        # Bei beobachteten Geraete-Exports: letztes Spektrum korrekt alignen.
        tr = tr[-len(wl):]

    if len(wl) != len(tr) or len(wl) < 3:
        raise ValueError(
            f"Unerwartetes XML-Format in {xml_path.name}: {len(wl)} wl vs {len(tr)} tr"
        )

    df = pd.DataFrame({"wavelength_nm": wl, "T_percent": tr})
    # Doppelte Wellenlaengen werden gemittelt.
    df = df.groupby("wavelength_nm", as_index=False)["T_percent"].mean()
    return df.sort_values("wavelength_nm").reset_index(drop=True)


# ==============================================================
# Kennwerte – Anzeige-/Arbeitsgrößen (nicht direkt normativ)
# ==============================================================
def tau_band_avg(
    wavelength_nm: np.ndarray, T_percent: np.ndarray, lo: float, hi: float
) -> float:
    """Einfacher Mittelwert der Transmission [%] im Wellenlängenband [lo, hi]."""
    wl = np.asarray(wavelength_nm)
    T = np.asarray(T_percent)
    m = (wl >= lo) & (wl <= hi)
    # Hinweis: ungewichteter Mittelwert, setzt gleichmaessige Schrittweite voraus.
    return float(np.mean(T[m])) if np.any(m) else float("nan")


def tau_SB_8980(wavelength_nm: np.ndarray, T_percent: np.ndarray) -> float:
    """
    Solarer Blaulicht-Transmissionsgrad τ_SB in % (ISO 8980-3, Anhang B).

    Gewichtung: E_S(λ) * B(λ), 380–500 nm (Tabelle B.1), linear interpoliert.
    Diese Funktion verwendet zwei gewichtete Funktionen multipliziert miteinander:
    - E_S(λ): Spektrale Leistungsdichte der Sonne (D65)
    - B(λ): Blaulicht-Gefährdungsfunktion (ISO 8995-1)
    """
    wl = np.asarray(wavelength_nm)
    T = np.clip(np.asarray(T_percent) / 100.0, 0.0, 1.0)
    B = _interp_weights(B_BLUE_HAZARD_B5, wl)
    E = _interp_weights(E_SOLAR_340_500, wl)
    W = B * E
    lo, hi = SOLAR_BLUE_RANGE
    m = (wl >= lo) & (wl <= hi) & (W > 0)
    if not np.any(m):
        return float("nan")
    # Normierung ueber das Integral der Bewertungsfunktion.
    num = _TRAPZ(T[m] * W[m], wl[m])
    den = _TRAPZ(W[m], wl[m])
    return float(100.0 * num / den)


# ==============================================================
# Kennwerte – DIN EN ISO 8980-3
# ==============================================================

def tau_V_norm_8980(wl_nm: np.ndarray, T_percent: np.ndarray) -> float:
    """
    τV nach DIN EN ISO 8980-3:2022 (S_D65·V, 380–780 nm).
    """
    wl = np.asarray(wl_nm)
    # Prozent -> 0..1, negative oder >100 Werte werden gekappt.
    T = np.clip(np.asarray(T_percent) / 100.0, 0.0, 1.0)
    W = _interp_weights(D65V_A5, wl)
    lo, hi = PHOTOPIC_RANGE
    m = (wl >= lo) & (wl <= hi) & (W > 0)
    num = _TRAPZ(T[m] * W[m], wl[m])
    den = _TRAPZ(W[m], wl[m])
    return float(100.0 * num / den) if den > 0 else float("nan")


def _tau_SUV(wl_nm: np.ndarray, T_percent: np.ndarray, lo: float, hi: float) -> float:
    """
    τ_SUVB (280–315) bzw. τ_SUVA (315–380) nach Anhang B von DIN EN ISO 8980-3.

    Robust:
    - benötigt >= 2 Stützstellen mit W(λ) > 0,
    - Nenner > 0, sonst NaN.
    """
    wl = np.asarray(wl_nm)
    T = np.clip(np.asarray(T_percent) / 100.0, 0.0, 1.0)
    W = _interp_weights(W_UV_B5, wl)
    m = (wl >= lo) & (wl <= hi) & (W > 0)
    if np.count_nonzero(m) < 2:
        # Mindestens zwei Stuetzpunkte mit Gewichtung noetig.
        return float("nan")
    num = _TRAPZ(T[m] * W[m], wl[m])
    den = _TRAPZ(W[m], wl[m])
    if den <= 0 or not np.isfinite(den):
        return float("nan")
    return float(100.0 * num / den)


def tau_SUVB_8980(wl_nm: np.ndarray, T_percent: np.ndarray) -> float:
    """UVB-gewichtete Transmission τ_SUVB (280–315 nm) nach DIN EN ISO 8980-3."""
    lo, hi = UVB_RANGE
    return _tau_SUV(wl_nm, T_percent, lo, hi)


def tau_SUVA_8980(wl_nm: np.ndarray, T_percent: np.ndarray) -> float:
    """UVA-gewichtete Transmission τ_SUVA (315–380 nm) nach DIN EN ISO 8980-3."""
    lo, hi = UVA_RANGE
    return _tau_SUV(wl_nm, T_percent, lo, hi)


# ==============================================================
# Deklarierte Wellenlängengrenze (A1:2025)
# ==============================================================

def compute_declared_wavelength_limit_A1_2025(
    wl_nm: np.ndarray, T_percent: np.ndarray
) -> float:
    """
    Deklarierte Wellenlaengengrenze nach EN ISO 8980-3/A1:2025, 6.7.2.

    Regel:
      - T(λ_decl) < 5,0 %,
      - fuer alle λ in [280 nm, λ_decl - 10 nm] gilt T(λ) < 1,0 %.

    Rueckgabe: groesste λ_decl (nm) auf 1-nm-Gitter, die diese Bedingungen erfuellt.
    """
    wl = np.asarray(wl_nm, dtype=float)
    T = np.asarray(T_percent, dtype=float)
    mask = np.isfinite(wl) & np.isfinite(T)
    wl = wl[mask]
    T = T[mask]
    if wl.size < 2:
        return float("nan")

    order = np.argsort(wl)
    wl = wl[order]
    T = T[order]

    if wl.size > 1:
        max_step = float(np.max(np.diff(np.unique(wl))))
        if max_step > 1.0:
            print(
                f"WARNUNG: Wellenlaengenraster groesser als 1 nm (max Δλ={max_step:.1f} nm); "
                "A1:2025 fordert <=1 nm fuer die Deklaration der Wellenlaengengrenze."
            )

    wl_min = float(wl.min())
    wl_max = float(wl.max())
    if wl_min > 280.0 or wl_max < 290.0:
        return float("nan")

    # 1-nm-Gitter fuer die Grenzpruefung (A1 fordert <= 1 nm Aufloesung).
    grid_lo = max(280, int(np.ceil(wl_min)))
    grid_hi = int(np.floor(wl_max))
    if grid_hi - grid_lo < 10:
        return float("nan")
    grid = np.arange(grid_lo, grid_hi + 1, 1, dtype=float)
    T_grid = np.interp(grid, wl, T)

    # Kumulatives Maximum fuer schnelle 1%-Pruefung bis λ-10.
    cummax = np.maximum.accumulate(T_grid)
    idx_shift = 10  # 10 nm bei 1-nm-Gitter

    candidate = float("nan")
    for i in range(idx_shift, grid.size):
        if T_grid[i] < 5.0 and cummax[i - idx_shift] < 1.0:
            candidate = float(grid[i])

    return candidate


# ==============================================================
# Hilfen: Ausreißer, CI, Repeatability
# ==============================================================

T_VALUES_95 = {  # t_(0.975, n-1) für n=2..30 (gerundet)
    2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365,
    9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145,
    16: 2.131, 17: 2.120, 18: 2.110, 19: 2.101, 20: 2.093, 21: 2.086,
    22: 2.080, 23: 2.074, 24: 2.069, 25: 2.064, 26: 2.060, 27: 2.056,
    28: 2.052, 29: 2.048, 30: 2.045,
}


def iqr_outlier_mask(values: np.ndarray, k: float = 1.5) -> np.ndarray:
    """IQR-basiertes Ausreißerkriterium (Tukey), Rückgabe: Bool-Maske für Ausreißer."""
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return (values < lo) | (values > hi)


def mean_sd_ci95(series: pd.Series) -> Tuple[float, float, float, float]:
    """Berechnet Mittelwert, Standardabweichung und 95%-Vertrauensintervall."""
    s = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    n = len(s)
    if n == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    m = float(np.mean(s))
    sd = float(np.std(s, ddof=1)) if n > 1 else 0.0
    t = T_VALUES_95.get(max(2, min(30, n)), 1.96)
    ci = float(t * (sd / np.sqrt(n))) if n > 1 else 0.0
    return (m, sd, m - ci, m + ci)


def repeatability_limit(sd_within: float) -> float:
    """Repeatability-Limit r ≈ 2,8 · s_r nach ISO 5725 (Faustwert).
    Formel: r = 2.8 × s_r (Faustformel)
    Interpretation: Zwei Messungen an demselben Objekt liegen mit 95%iger Wahrscheinlichkeit
    innerhalb des Bereichs ±r um den wahren Wert.
    """
    return 2.8 * sd_within


# ==============================================================
# Boxplots je Glas (10× Wiederholungen)
# ==============================================================

def _clean_series_for_metric(df: pd.DataFrame, glass_id: str, col: str) -> np.ndarray:
    s = pd.to_numeric(df.loc[df["glass_id"] == glass_id, col], errors="coerce").dropna()
    return s.to_numpy()


def _plot_box_with_points(values: np.ndarray, title: str, ylabel: str, out_png: Path):
    if values.size == 0:
        return
    fig = plt.figure()
    plt.boxplot(values, vert=True, showmeans=False, whis=1.5)
    if values.size == 1:
        x = np.array([1.0])
    else:
        x = np.linspace(0.9, 1.1, num=values.size)
    plt.scatter(x, values, alpha=0.75)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_boxplots_per_glass(df_files: pd.DataFrame, out_dir: Path) -> None:
    """Erzeugt Boxplots pro Glas für τV_ISO, τ_SUVA_ISO, τ_SUVB_ISO (falls vorhanden)."""
    metrics = [
        ("tauV_percent_ISO8980", "τV [%] (ISO 8980-3)"),
        ("tau_SUVA_315_380_percent_ISO8980", "τ_SUVA 315–380 [%] (ISO 8980-3)"),
        ("tau_SUVB_280_315_percent_ISO8980", "τ_SUVB 280–315 [%] (ISO 8980-3)"),
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    for gid, _g in df_files.groupby("glass_id", dropna=False):
        if pd.isna(gid):
            continue
        for col, label in metrics:
            vals = _clean_series_for_metric(df_files, gid, col)
            if vals.size == 0:
                continue
            title = f"{gid} – {label} (n={vals.size})"
            out_png = out_dir / f"{gid}_{col}_box.png"
            _plot_box_with_points(vals, title, ylabel=label, out_png=out_png)


# ==============================================================
# Plot eines Spektrums
# ==============================================================

def plot_spectrum(df: pd.DataFrame, title: str, out_png: Path):
    plt.figure()
    plt.plot(df["wavelength_nm"], df["T_percent"])
    plt.xlabel("Wellenlänge [nm]")
    plt.ylabel("Transmission [%]")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()


# ==============================================================
# Spektrale Abdeckungs-Prüfung (Plausibilität)
# ==============================================================

def check_spectral_coverage(wl_nm: np.ndarray) -> None:
    """
    Plausibilitätsprüfung der spektralen Abdeckung.

    Warnung in der Konsole, falls das Spektrum die Bereiche 280–780 nm
    nicht vollständig abdeckt.
    """
    wl_sorted = np.unique(np.asarray(wl_nm, dtype=float))
    if wl_sorted.size == 0:
        return
    wl_min = float(np.min(wl_sorted))
    wl_max = float(np.max(wl_sorted))
    # Prueft nur min/max; interne Luecken werden nicht erkannt.
    if wl_min > WL_MIN_NORM or wl_max < WL_MAX_NORM:
        print(
            f"WARNUNG: Spektrum deckt normative Bereiche {WL_MIN_NORM:.0f}–"
            f"{WL_MAX_NORM:.0f} nm nicht vollständig ab "
            f"(min={wl_min:.1f} nm, max={wl_max:.1f} nm)."
        )
    if wl_sorted.size > 1:
        max_step = float(np.max(np.diff(wl_sorted)))
        if max_step > 5.0:
            print(
                f"WARNUNG: Wellenlaengenraster groesser als 5 nm (max Δλ={max_step:.1f} nm); "
                "ISO 8980-3 sieht <=5 nm fuer Messdaten vor."
            )


# ==============================================================
# Pipeline
# ==============================================================

def _round_or_nan(value: Optional[float], digits: int = 3) -> float:
    if value is None:
        return np.nan
    try:
        if not np.isfinite(value):
            return np.nan
    except TypeError:
        return np.nan
    return round(float(value), digits)


def process_folder(in_dir: Path, out_dir: Path) -> pd.DataFrame:
    if not in_dir.exists():
        raise FileNotFoundError(f"Eingabeverzeichnis nicht gefunden: {in_dir}")
    # Ausgabeordner fuer Einzelspektren, Plots und aggregierte Tabellen.
    spectra_dir = out_dir / "spectra"
    plots_dir = out_dir / "plots"
    agg_dir = out_dir / "aggregated"
    spectra_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    agg_dir.mkdir(parents=True, exist_ok=True)

    per_file_rows: List[Dict[str, object]] = []
    filter_rows: List[Dict[str, object]] = []

    for xml_path in sorted(in_dir.glob("*.xml")):
        try:
            df = read_solos_xml(xml_path)
            wl = df["wavelength_nm"].to_numpy()
            T = df["T_percent"].to_numpy()

            # Plausibilitätsprüfung der Abdeckung
            check_spectral_coverage(wl)

            # Anzeige-/Arbeitsgroessen (tau_v entspricht tauV_iso, daher nur einmal berechnet)
            tauV_iso = tau_V_norm_8980(wl, T)
            tau_v = tauV_iso
            lo_uva, hi_uva = UVA_RANGE
            tau_uva = tau_band_avg(wl, T, lo_uva, hi_uva)
            tau_sb = tau_SB_8980(wl, T)

            # DIN EN ISO 8980-3 Kennwerte
            tauSvb_iso = tau_SUVB_8980(wl, T)
            tauSua_iso = tau_SUVA_8980(wl, T)

            # Kategorie nach EN ISO 12312-1 aus τV_ISO bestimmen
            status_cat, cat, driving_ok = categorize_by_tauV(tauV_iso)

            # IDs
            stem = xml_path.stem
            manufacturer = manufacturer_from_filename(stem)
            glass_id = glass_id_from_filename(stem)
            ref_id = detect_ref_id(stem)
            if ref_id or stem.upper().startswith("NDUVW"):
                # Alle NDUVW-Referenzfilter stammen von ThorLabs.
                manufacturer = "ThorLabs"

            # Spektrum & Plot
            spec_csv = spectra_dir / f"{stem}.csv"
            df.to_csv(spec_csv, index=False)
            plot_spectrum(df, title=stem, out_png=plots_dir / f"{stem}.png")

            # Sammelstruktur fuer die Spaetere Summary-Ausgabe.
            row: Dict[str, object] = {
                "file": xml_path.name,
                "glass_id": glass_id,
                "manufacturer": manufacturer,
                "ref_id": ref_id or "",
                # Anzeigegrößen
                "tau_v_percent": _round_or_nan(tau_v),
                "tau_UVA_315_380_percent": _round_or_nan(tau_uva),
                "tau_SB_380_500_percent_ISO8980": _round_or_nan(tau_sb),
                # DIN EN ISO 8980-3
                "tauV_percent_ISO8980": _round_or_nan(tauV_iso),
                "tau_SUVB_280_315_percent_ISO8980": _round_or_nan(tauSvb_iso),
                "tau_SUVA_315_380_percent_ISO8980": _round_or_nan(tauSua_iso),
                "n_points": len(df),
                # Kategorie-Reporting
                "tauV_category_ISO12312_1": (int(cat) if status_cat == "ok" else pd.NA),
                "tauV_category_status": status_cat,  # "ok" | "out_of_range"
                "tauV_driving_ok": (
                    bool(driving_ok) if status_cat == "ok" else pd.NA
                ),
                # Deklarierte Wellenlaengengrenze (A1:2025)
                "lambda_declared_nm_A1_2025": compute_declared_wavelength_limit_A1_2025(
                    wl, T
                ),
            }

            per_file_rows.append(row)

            # QC: Referenzfilter auflisten
            if ref_id:
                # Vergleich gegen nominale ND-Filterwerte (Toleranz in %-Punkten).
                nominal = REF_FILTERS.get(ref_id, {}).get("nominal_tauV_percent")
                tol = REF_FILTERS.get(ref_id, {}).get("tol_percent")
                delta = (tauV_iso - nominal) if nominal is not None else np.nan
                within = (
                    abs(delta) <= tol
                    if (
                        nominal is not None
                        and tol is not None
                        and np.isfinite(delta)
                    )
                    else None
                )
                filter_rows.append(
                    {
                        "file": xml_path.name,
                        "ref_id": ref_id,
                        "tauV_percent_ISO8980": _round_or_nan(tauV_iso),
                        "nominal_tauV_percent": nominal,
                        "tol_percent": tol,
                        "delta_percent": _round_or_nan(delta),
                        "within_tol": within,
                    }
                )

            # Labels für die Konsole (Kategorie + Verkehrstauglichkeit)
            cat_str = f"Cat={cat}" if status_cat == "ok" else "Cat=out_of_range"
            drive_str = (
                "Driving=ok"
                if (status_cat == "ok" and driving_ok)
                else "Driving=not_permitted"
                if status_cat == "ok"
                else "Driving=n/a"
            )

            print(
                f"OK: {xml_path.name} -> τV_ISO={tauV_iso:.2f}% | "
                f"τ_SUVA={tauSua_iso:.2f}% | "
                f"τ_SUVB={(tauSvb_iso if pd.notna(tauSvb_iso) else float('nan')):.2f}% | "
                f"{cat_str} | {drive_str}"
            )

        except Exception as e:
            print(f"FEHLER in {xml_path.name}: {e}")

    # ------ pro Datei schreiben ------
    df_files = pd.DataFrame(per_file_rows)
    # Direktes Schreiben spart das Zwischen-String-Objekt bei grossen Datensaetzen.
    summary_path = out_dir / "summary.csv"
    df_files.to_csv(summary_path, index=False, encoding="utf-8")

    # ------ Referenzfilter-QC schreiben ------
    if filter_rows:
        pd.DataFrame(filter_rows).to_csv(agg_dir / "qc_filters.csv", index=False)

    # ------ 10x-Glasaggregation ------
    if not df_files.empty:
        df_glass = aggregate_by_glass(df_files)
        df_glass.to_csv(agg_dir / "by_glass.csv", index=False)
        # Boxplots je Glas erzeugen
        make_boxplots_per_glass(df_files, agg_dir / "plots")
        print(f"\nAggregat geschrieben: {agg_dir/'by_glass.csv'}")
        if filter_rows:
            print(f"QC-Filter geschrieben: {agg_dir/'qc_filters.csv'}")

    print(f"\nSummary geschrieben: {out_dir/'summary.csv'}")
    return df_files


# ==============================================================
# Aggregation (10x) pro Glas
# ==============================================================

def aggregate_by_glass(df_files: pd.DataFrame) -> pd.DataFrame:
    """
    Je glass_id: n, mean, sd, 95%-CI und Repeatability-Limit für
    tauV_percent_ISO8980, tau_SUVA_315_380_percent_ISO8980,
    tau_SUVB_280_315_percent_ISO8980.
    IQR-Ausreißer werden entfernt (n_eff wird ausgewiesen).
    Hersteller wird pro Glas aus df_files abgeleitet.
    """
    target_cols = [
        "tauV_percent_ISO8980",
        "tau_SUVA_315_380_percent_ISO8980",
        "tau_SUVB_280_315_percent_ISO8980",
    ]
    records: List[Dict[str, object]] = []

    for gid, g in df_files.groupby("glass_id", dropna=False):
        # n_total ist die Rohanzahl pro Glas (vor Ausreisser-Filter).
        rec: Dict[str, object] = {"glass_id": gid, "n_total": int(len(g))}

        # Hersteller pro Glas (über Mode bzw. ersten nicht-null Wert)
        manuf_series = g["manufacturer"].dropna()
        # Mode ist robuster als "erstes Vorkommen" bei gemischten Eingaben.
        rec["manufacturer"] = (
            str(manuf_series.mode().iat[0]) if not manuf_series.empty else ""
        )

        for col in target_cols:
            vals = (
                pd.to_numeric(g[col], errors="coerce")
                .dropna()
                .to_numpy()
            )
            if vals.size == 0:
                rec.update(
                    {
                        f"{col}_n": 0,
                        f"{col}_mean": np.nan,
                        f"{col}_sd": np.nan,
                        f"{col}_ci95_lo": np.nan,
                        f"{col}_ci95_hi": np.nan,
                        f"{col}_repeatability_limit": np.nan,
                        f"{col}_removed_outliers": 0,
                    }
                )
                continue
            # IQR-Ausreisser entfernen, bevor Kennwerte berechnet werden.
            mask_out = iqr_outlier_mask(vals)
            vals_clean = vals[~mask_out]
            removed = int(mask_out.sum())
            n = vals_clean.size
            m, sd, lo, hi = mean_sd_ci95(pd.Series(vals_clean))
            rec.update(
                {
                    f"{col}_n": int(n),
                    f"{col}_mean": round(m, 3) if np.isfinite(m) else np.nan,
                    f"{col}_sd": round(sd, 3) if np.isfinite(sd) else np.nan,
                    f"{col}_ci95_lo": round(lo, 3) if np.isfinite(lo) else np.nan,
                    f"{col}_ci95_hi": round(hi, 3) if np.isfinite(hi) else np.nan,
                    f"{col}_repeatability_limit": (
                        round(repeatability_limit(sd), 3)
                        if np.isfinite(sd)
                        else np.nan
                    ),
                    f"{col}_removed_outliers": removed,
                }
            )
            if col == "tauV_percent_ISO8980":
                # Kategorie/Driving-Status wird auch aus Mittelwert/CI abgeleitet.
                st_mean, cat_mean, drv_mean = (
                    categorize_by_tauV(m)
                    if np.isfinite(m)
                    else ("out_of_range", None, False)
                )
                st_lo, cat_lo, drv_lo = (
                    categorize_by_tauV(lo)
                    if np.isfinite(lo)
                    else ("out_of_range", None, False)
                )

                rec.update(
                    {
                        "tauV_category_ISO12312_1_from_mean": (
                            int(cat_mean) if st_mean == "ok" else pd.NA
                        ),
                        "tauV_driving_ok_from_mean": (
                            bool(drv_mean) if st_mean == "ok" else pd.NA
                        ),
                        "tauV_category_ISO12312_1_from_CI95_lo": (
                            int(cat_lo) if st_lo == "ok" else pd.NA
                        ),
                        "tauV_driving_ok_from_CI95_lo": (
                            bool(drv_lo) if st_lo == "ok" else pd.NA
                        ),
                    }
                )

        records.append(rec)

    df = (
        pd.DataFrame.from_records(records)
        .sort_values("glass_id")
        .reset_index(drop=True)
    )
    return df

# ==============================================================
# Command Line Interface (CLI)
# ==============================================================

def main():
    ap = argparse.ArgumentParser(
        description=(
            "SOLOS Transmission – DIN EN ISO 8980-3:2022 (+ A1:2025 Wellenlängengrenze), "
            "10x-Auswertung, Referenzfilter, Boxplots"
        )
    )
    ap.add_argument(
        "--in_dir",
        type=Path,
        default=Path("/Users/georgplessl/Desktop/BA_TM_review/Bachelorarbeit_Transmissionsmessung_public/data/raw_xml"),
        help="Ordner mit SOLOS-XML",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/Users/georgplessl/Desktop/BA_TM_review/Bachelorarbeit_Transmissionsmessung_public/outputs"),
        help="Zielordner für CSV/Plots",
    )
    args = ap.parse_args()
    process_folder(args.in_dir, args.out_dir)


if __name__ == "__main__":
    main()
