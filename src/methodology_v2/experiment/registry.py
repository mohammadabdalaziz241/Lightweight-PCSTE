"""Immutable Part-5D experiment registry builder + master freeze hash.

Deterministic run IDs (no UUIDs):
  SSL:        ssl_f{fold}_s{seed}
  Downstream: {arm}_f{fold}_s{seed}_l{fraction_pct:03d}
Matrix: 9 SSL runs + 90 downstream runs (2 arms x 5 fractions x 3 folds
x 3 seeds) = 99 main runs. Ablations are registered separately and NOT
multiplied across fractions.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from ..integrity import sha256_file
from ..part3b_windows import PART3B_DIR
from ..registry import REPO_ROOT
from .heads import CLASS_ORDERS
from .label_subsets import FOLDS, FRACTIONS, SEEDS
from .trainers import (DOWNSTREAM_EPOCHS, EFFECTIVE_BATCH, OPTIMIZER_SPEC,
                       SSL_EPOCHS, steps_per_epoch)

PART5D_DIR = REPO_ROOT / "methodology_v2" / "part5_experiment_registry"
SUBSET_DIR = PART5D_DIR / "label_subsets"

UPSTREAM = {
    "part2": "527ccc1d449b223a37ecf109ed27be5279e17d85ba4e881abb9c68f4035e69c6",
    "part3b": "99ffde7e5c0e2cb9b05713801aedcb10b11ccc229d4c2d10a58a1506db10bb51",
    "part4c": "ee9414e8988c36b8a1ecad7d2622a54439a4bcc180a6a4e6a50b2f256160064f",
    "part5b_architecture": "962bb1de520a941a1c3a67c63c97d28983783327f7e263a73c9ec0bad0b6711f",
}


def part5c_spec_hash() -> str:
    return sha256_file(REPO_ROOT / "methodology_v2" / "part5_ssl_design"
                       / "proposed_ssl_spec.yaml")


def train_counts() -> dict[int, int]:
    out = {}
    for f in FOLDS:
        man = pd.read_csv(PART3B_DIR / f"window_manifest_fold_{f}.csv",
                          usecols=["split"])
        out[f] = int((man["split"] == "train").sum())
    return out


def build_registries(subset_hashes: dict[tuple[int, int], str]
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = train_counts()
    spec5c = part5c_spec_hash()
    common = {
        "architecture_hash": UPSTREAM["part5b_architecture"],
        "part2_hash": UPSTREAM["part2"],
        "part3b_hash": UPSTREAM["part3b"],
        "part4c_hash": UPSTREAM["part4c"],
        "part5c_spec_hash": spec5c,
        "effective_batch": EFFECTIVE_BATCH,
        "optimizer": json.dumps(OPTIMIZER_SPEC, sort_keys=True),
        "status": "REGISTERED",
    }
    ssl_rows = []
    for f in FOLDS:
        for s in SEEDS:
            ssl_rows.append({
                "run_id": f"ssl_f{f}_s{s}", "fold": f, "seed": s,
                "sampler": "ssl_dataset_group_window_balanced_16x4",
                "mask": "M1_random@0.60 learned-token "
                        "sha256(seed|epoch|window)",
                "validation_mask": "FIXED per seed "
                                   "(sha256('valmask|'+seed), epoch-free)",
                "max_epochs": SSL_EPOCHS,
                "steps_per_epoch": steps_per_epoch(counts[f]),
                "checkpoint_metric": "MacroDomainReconMSE (minimize; "
                                     "per-dataset window-mean MSE)",
                "output_location": f"results/methodology_v2/ssl/"
                                   f"ssl_f{f}_s{s}/",
                **common})
    main_rows = []
    for arm in ("s0", "s1"):
        for frac in FRACTIONS:
            for f in FOLDS:
                for s in SEEDS:
                    pct = int(frac * 100)
                    main_rows.append({
                        "run_id": f"{arm}_f{f}_s{s}_l{pct:03d}",
                        "arm": arm.upper(), "fold": f, "seed": s,
                        "label_fraction": frac,
                        "ssl_checkpoint_dependency": (
                            f"ssl_f{f}_s{s}" if arm == "s1" else "none"),
                        "label_subset_hash": subset_hashes[(f, s)],
                        "sampler": "sup_dataset_class_group_window_16x4",
                        "head_config": json.dumps(
                            {ds: len(c) for ds, c in
                             CLASS_ORDERS.items()}, sort_keys=True),
                        "head_init_seed_rule": "sha256('heads|'+fold+'|'"
                                               "+seed)[:4] shared S0/S1",
                        "max_epochs": DOWNSTREAM_EPOCHS,
                        "steps_per_epoch": steps_per_epoch(counts[f]),
                        "checkpoint_metric": "MacroDomainF1_val "
                            "(maximize; exact tie -> earlier epoch)",
                        "primary_test_metric": "MacroDomainF1_test",
                        "output_location": f"results/methodology_v2/"
                            f"downstream/{arm}_f{f}_s{s}_l{pct:03d}/",
                        **common})
    return pd.DataFrame(ssl_rows), pd.DataFrame(main_rows)


def master_hash(files: list[Path]) -> str:
    entries = sorted((p.name, sha256_file(p)) for p in files)
    src = "".join(f"{n}:{h}\n" for n, h in entries)
    return hashlib.sha256(src.encode()).hexdigest()


def verify_part5d_hash(base: Path | None = None) -> None:
    """FAIL CLOSED if any sealed Part-5D registry artifact changed."""
    base = Path(base) if base else PART5D_DIR
    rec = pd.read_csv(base / "part5d_hashes.csv")
    stored = {r["file"]: r["sha256"] for _, r in rec.iterrows()}
    entries = []
    for name, expect in stored.items():
        if name == "PART5D_MASTER_HASH":
            continue
        got = sha256_file(base / name)
        if got != expect:
            raise AssertionError(
                f"FROZEN PART-5D ARTIFACT CHANGED: {name} (fail closed)")
        entries.append((Path(name).name, got))
    src = "".join(f"{n}:{h}\n" for n, h in sorted(entries))
    if hashlib.sha256(src.encode()).hexdigest() \
            != stored["PART5D_MASTER_HASH"]:
        raise AssertionError("Part-5D master hash mismatch (fail closed)")
