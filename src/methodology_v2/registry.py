"""Dataset registry for methodology_v2 Part 1.

Single source of truth for: local paths, authoritative sources, valid
original labels, and *documented* acquisition facts. Anything not documented
by the original source (or by a frozen audit in this repository) must be
represented as None / "unknown" downstream — never inferred silently.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "methodology_v2" / "part1_audit"

# ---------------------------------------------------------------------------
# CWRU
# ---------------------------------------------------------------------------
# Local copies are renamed official files; the internal MATLAB variable names
# (X097, X105, ...) carry the canonical CWRU file numbers.
#
# Sampling rates follow the frozen finding of
# docs/cwru_legacy_rate_impact_note.md (2026-08-02):
#   - fault files in data/raw (*_DE12k)      -> genuinely 12 kHz
#   - fault files in data/raw_cwru_48k       -> genuinely 48 kHz
#   - the four Normal_* baseline files       -> genuinely 48 kHz
#     (byte-identical official 97-100.mat, present in BOTH local dirs)
CWRU = {
    "name": "CWRU",
    "source_url": "https://engineering.case.edu/bearingdatacenter",
    "reference": (
        "CWRU Bearing Data Center; rates per frozen audit "
        "docs/cwru_legacy_rate_impact_note.md and "
        "metadata/vibrationclip_v1/cwru_48k_enumeration.json"
    ),
    "paths": {
        "12k_de": DATA_ROOT / "raw",
        "48k_de": DATA_ROOT / "raw_cwru_48k",
    },
    # Documented load->nominal shaft speed table (CWRU website).
    "nominal_rpm_by_load_hp": {0: 1797, 1: 1772, 2: 1750, 3: 1730},
    "valid_labels": {
        "Normal",
        "IR007", "IR014", "IR021", "IR028",
        "B007", "B014", "B021", "B028",
        "OR007@3", "OR007@6", "OR007@12",
        "OR014@6",
        "OR021@3", "OR021@6", "OR021@12",
    },
    "machinery": "2 hp motor test stand, seeded-fault drive-end bearing "
                 "(SKF 6205-2RS JEM; NTN equivalent for 28 mil)",
    "sensor": "accelerometer (DE / FE / BA where present)",
}

# ---------------------------------------------------------------------------
# JNU
# ---------------------------------------------------------------------------
JNU = {
    "name": "JNU",
    "source_url": "https://github.com/ClarkGableWang/JNU-Bearing-Dataset",
    "reference": (
        "Jiangnan University bearing dataset; repo readme.md: PCB MA352A60 "
        "accelerometer, vertical direction, 50 kHz, speeds 600/800/1000 rpm, "
        "wire-cut dents 0.3 mm x 0.05 mm on outer ring, inner ring, roller"
    ),
    "paths": {"root": DATA_ROOT / "raw_jnu" / "JNU-Bearing-Dataset"},
    "clone_commit": "75b33611b51649d1da8ff5999397899420753e5b",
    "sampling_rate_hz": 50_000,
    "speeds_rpm": (600, 800, 1000),
    # file prefix -> documented condition
    "valid_labels": {"n", "ib", "ob", "tb"},
    "label_meaning": {
        "n": "healthy",
        "ib": "inner-race fault",
        "ob": "outer-race fault",
        "tb": "rolling-element (roller) fault",
    },
    "machinery": "rotating machinery test rig, rolling bearing",
    "sensor": "PCB MA352A60 accelerometer, vertical direction, 1 channel",
}

# ---------------------------------------------------------------------------
# HIT
# ---------------------------------------------------------------------------
# Hou et al., "An Inter-Shaft Bearing Fault Diagnosis Dataset from an
# Aero-Engine System", J. Dynamics, Monitoring and Diagnostics 2(4), 2023,
# doi:10.37965/jdmd.2023.314. Facts below are from the paper (Tables III-VI).
HIT = {
    "name": "HIT",
    "source_url": "https://github.com/HouLeiHIT/HIT-dataset",
    "reference": "Hou et al. 2023, doi:10.37965/jdmd.2023.314",
    "paths": {
        "github_release": DATA_ROOT / "raw_hit" / "HIT-dataset",
        "gdrive_full": DATA_ROOT / "raw_hit" / "gdrive_full" / "HIT-dataset",
    },
    "clone_commit": "ef17655977519eda9463a3d060da9fb8b47fab6f",
    "sampling_rate_hz": 25_000,
    "series_len": 20_480,
    "n_speed_groups_planned": 28,
    "channels": {
        1: "displacement, LP rotor (KISTLER 8776A50M1)",
        2: "displacement, LP rotor (KISTLER 8776A50M1)",
        3: "acceleration, casing (K9000XL)",
        4: "acceleration, casing (K9000XL)",
        5: "acceleration, casing (K9000XL)",
        6: "acceleration, casing (K9000XL)",
    },
    # session -> documented bearing state (paper Table III + VI)
    "sessions": {
        "data1": {"label": 0, "condition": "healthy",
                  "fault_depth_mm": 0.0, "fault_len_mm": 0.0},
        "data2": {"label": 0, "condition": "healthy",
                  "fault_depth_mm": 0.0, "fault_len_mm": 0.0},
        "data3": {"label": 1, "condition": "inner-ring fault",
                  "fault_depth_mm": 0.5, "fault_len_mm": 0.5},
        "data4": {"label": 1, "condition": "inner-ring fault",
                  "fault_depth_mm": 0.5, "fault_len_mm": 1.0},
        "data5": {"label": 2, "condition": "outer-ring fault",
                  "fault_depth_mm": 0.5, "fault_len_mm": 0.5},
    },
    "valid_labels": {0, 1, 2},
    "label_meaning": {0: "healthy", 1: "inner-ring fault",
                      2: "outer-ring fault"},
    "machinery": "modified real aero-engine (dual rotor), inter-shaft bearing",
    "sensor": "2x displacement + 4x acceleration, 6 channels",
}

# ---------------------------------------------------------------------------
# MaFaulDa
# ---------------------------------------------------------------------------
MAFAULDA = {
    "name": "MAFAULDA",
    "source_url": "https://www02.smt.ufrj.br/~offshore/mfs/page_01.html",
    "reference": (
        "UFRJ Machinery Fault Database; SpectraQuest MFS ABVT; 1951 sequences, "
        "50 kHz, 5 s, 8 columns (tachometer; underhang acc ax/rad/tan; "
        "overhang acc ax/rad/tan; microphone)"
    ),
    "paths": {"root": DATA_ROOT / "raw_mafaulda" / "full"},
    "archive": DATA_ROOT / "raw_mafaulda" / "full.zip",
    "sampling_rate_hz": 50_000,
    "expected_n_samples": 250_000,
    "expected_sequences": 1951,
    # Documented per-category sequence counts (website section 1.4).
    "expected_counts": {
        "normal": 49,
        "horizontal-misalignment": 197,
        "vertical-misalignment": 301,
        "imbalance": 333,
        "underhang": 558,
        "overhang": 513,
    },
    # Operational labels are the archive's own folder taxonomy; validated
    # after extraction. The website's prose (sec 1.4.5) permutes fault-type
    # names relative to its own summary table - flagged in the integrity
    # report, folder names are treated as operational ground truth.
    "valid_top_labels": {
        "normal", "horizontal-misalignment", "vertical-misalignment",
        "imbalance", "underhang", "overhang",
    },
    "machinery": "SpectraQuest Machinery Fault Simulator ABVT (single rig)",
    "sensor": "2x triaxial-equivalent accelerometer sets + tachometer + mic",
}

DATASETS = {d["name"]: d for d in (CWRU, JNU, HIT, MAFAULDA)}
