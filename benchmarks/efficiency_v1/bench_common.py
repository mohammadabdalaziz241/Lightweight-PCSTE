"""Shared setup for the frozen Full-S1 vs K1 efficiency benchmark (efficiency_v1).

Loads the deterministically fixed GF1/seed-42 checkpoint pair, verifies both
checkpoint hashes against the frozen publication model table, and prepares the
deterministic VALIDATION inputs.

TEST is never read: every selected row is asserted to be a `validation` row and
the manifest is filtered to `split == "validation"` before selection.

This module does not edit or import-and-mutate any historical benchmark source.
`src/methodology_v2/compression/benchmark.py` is left untouched; see
`EFFICIENCY_BENCHMARK_PLAN.md` for why its input conventions could not be
reused verbatim.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PUB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PUB))

# The frozen checkpoints and normalizers are excluded from Git by the artifact
# policy; they are read from the canonical scientific repository.
CANON = Path(os.environ.get(
    "PCSTE_CANONICAL_REPO",
    "/scratch/ma06314/Standard Project/rotating_machinery_fault_diagnosis"))

DATASETS = ("CWRU", "JNU", "HIT", "MAFAULDA")
FOLD = 1
SEED = 42
PROTOCOL = "global_v2_final_s0s1_v1"

MANIFEST = PUB / f"protocols/splits/{PROTOCOL}/global_v2_fold_{FOLD}.csv"
MODEL_TABLE = PUB / "configs/lightweight_k1/final_test_v1/PUBLICATION_FROZEN_MODEL_TABLE.csv"
PROTOCOL_DIR = CANON / "pcste_v2/protocol" / PROTOCOL

# Analytic selective-scan FLOP constant, preserved from the historical Part-6
# harness (`analytic_scan_flops`): approximately this many elementary flops per
# (d_inner x d_state) state element per sequential step.
SCAN_FLOPS_PER_STATE_ELEMENT_PER_STEP = 6


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# frozen model pair — fixed by the deterministic rule, not by results
# ---------------------------------------------------------------------------

def frozen_pair_row() -> pd.Series:
    """The deterministic representative cell: lowest fold, then lowest seed."""
    tab = pd.read_csv(MODEL_TABLE)
    tab = tab.sort_values(["fold", "seed"], kind="mergesort")
    row = tab[(tab.fold == FOLD) & (tab.seed == SEED)].iloc[0]
    assert int(row["fold"]) == int(tab.iloc[0]["fold"]) == FOLD
    return row


def verify_checkpoints() -> dict:
    row = frozen_pair_row()
    out = {}
    for fam, path_col, sha_col in (("Full-S1", "full_s1_checkpoint", "full_s1_sha256"),
                                   ("K1", "k1_checkpoint", "k1_sha256")):
        path = CANON / str(row[path_col])
        if not path.is_file():
            raise FileNotFoundError(f"{fam} checkpoint not found: {path}")
        got = sha256_file(path)
        want = str(row[sha_col])
        if got != want:
            raise RuntimeError(f"{fam} checkpoint hash mismatch\n  expected {want}\n  got      {got}")
        out[fam] = {"path": str(row[path_col]), "sha256": got,
                    "bytes": path.stat().st_size, "run_id": str(row[f"{'full_s1' if fam=='Full-S1' else 'k1'}_run_id"])}
    return out


def load_models(device="cpu"):
    """Return {'Full-S1': (enc, heads), 'K1': (enc, heads)} strict-loaded."""
    from src.methodology_v2.encoder import PCSTE
    from src.methodology_v2.experiment.heads import DatasetHeads
    from src.methodology_v2.compression.student import half_4x1_spec, build_encoder

    row = frozen_pair_row()
    verify_checkpoints()

    fck = torch.load(CANON / str(row["full_s1_checkpoint"]), map_location="cpu", weights_only=False)
    if any(not k.startswith("core.") for k in fck["model"]):
        raise RuntimeError("unexpected final-v1 model tensor outside core.*")
    fe = PCSTE()
    r = fe.load_state_dict({k[len("core."):]: v for k, v in fck["model"].items()}, strict=True)
    assert not r.missing_keys and not r.unexpected_keys
    fh = DatasetHeads()
    r = fh.load_state_dict(fck["heads"], strict=True)
    assert not r.missing_keys and not r.unexpected_keys

    kck = torch.load(CANON / str(row["k1_checkpoint"]), map_location="cpu", weights_only=False)
    ke = build_encoder(half_4x1_spec("mean_of_remaining"), seed=0)
    r = ke.load_state_dict(kck["encoder"], strict=True)
    assert not r.missing_keys and not r.unexpected_keys
    kh = DatasetHeads()
    r = kh.load_state_dict(kck["heads"], strict=True)
    assert not r.missing_keys and not r.unexpected_keys

    n_k1 = sum(p.numel() for p in ke.parameters())
    if n_k1 != 1_375_953:
        raise RuntimeError(f"K1 encoder parameter count changed: {n_k1} != 1375953")

    models = {"Full-S1": (fe, fh), "K1": (ke, kh)}
    for enc, hd in models.values():
        enc.to(device).eval()
        hd.to(device).eval()
    return models


# ---------------------------------------------------------------------------
# deterministic VALIDATION inputs — never TEST
# ---------------------------------------------------------------------------

def _builder():
    from src.pcste_v2 import representation as REPR
    from src.pcste_v2.protocol_cwru_native12 import load_allowlist
    REPR.NATIVE12_ALLOWLIST = load_allowlist(PUB / "protocols/datasets/cwru_native12_load_v1")
    return REPR, REPR.ItemBuilder(PROTOCOL_DIR, FOLD, grids=("v1",), channels={}, envelope=False)


def select_rows() -> dict:
    """First VALIDATION window per dataset in frozen manifest order."""
    man = pd.read_csv(MANIFEST)
    sel = {}
    for ds in DATASETS:
        sub = man[(man["dataset"] == ds) & (man["split"] == "validation")]
        if sub.empty:
            raise RuntimeError(f"no validation rows for {ds}")
        row = sub.iloc[0]
        if row["split"] != "validation":
            raise RuntimeError("TEST GUARD: selected a non-validation row")
        sel[ds] = row
    return sel


def prepare_inputs(device="cpu") -> dict:
    """Per dataset: resident raw waveform, resident collated representation,
    and the metadata needed to reproduce the selection. No TEST, no timed I/O."""
    from src.methodology_v2.encoder import collate_representations
    from src.methodology_v2.encoder.patchify import patchify
    from src.methodology_v2.experiment.heads import CLASS_ORDERS
    REPR, builder = _builder()

    rows = select_rows()
    out = {}
    for ds, row in rows.items():
        key = REPR.rate_key(ds, int(row["native_sampling_rate_hz"]))
        n_fft, hop = REPR.STFT_GRIDS["v1"][key]
        npath = str(REPR.norm_path(PROTOCOL_DIR, FOLD, "v1", key, 0))
        norm = REPR._load_norm(npath)          # warm the lru_cache: no disk I/O when timed
        raw = np.asarray(REPR.read_raw(row, 0), dtype=np.float64)   # resident in memory

        z, f, t = builder.build(row)["streams"]["g0_c0"]
        batch = collate_representations([(z, f, t)])
        batch = {k: v.to(device) for k, v in batch.items()}
        _, _, tokm, _ = patchify(batch["spec"].cpu(), batch["cell_mask"].cpu())
        fb, tp = int(tokm.shape[1]), int(tokm.shape[2])

        out[ds] = {
            "row": row, "raw": raw, "batch": batch,
            "stft_key": key, "n_fft": int(n_fft), "hop": int(hop),
            "norm_path": npath, "norm_sha256": sha256_file(Path(npath)),
            "meta": {
                "dataset": ds,
                "window_id": str(row["window_id"]),
                "recording_id": str(row["recording_id"]),
                "source_file": str(row["source_file"]),
                "split": str(row["split"]),
                "start_sample": int(row["start_sample"]),
                "end_sample": int(row["end_sample"]),
                "native_sampling_rate_hz": int(row["native_sampling_rate_hz"]),
                "raw_samples": int(raw.shape[0]),
                "raw_seconds": round(float(raw.shape[0]) / int(row["native_sampling_rate_hz"]), 6),
                "stft_key": key, "stft_n_fft": int(n_fft), "stft_hop": int(hop),
                "spec_shape_bins_frames": [int(batch["spec"].shape[1]), int(batch["spec"].shape[2])],
                "patch_grid_bands_timepatches": [fb, tp],
                "head_dataset": ds,
                "head_n_classes": len(CLASS_ORDERS[ds]),
                "normalizer_sha256": sha256_file(Path(npath)),
            },
        }
    return out


def end_to_end_fn(ds: str, prep: dict, enc, hd, device: str):
    """raw (resident) -> STFT -> normalise -> collate -> model -> logits.
    Disk I/O excluded: the waveform and the normaliser are already in memory."""
    from src.pcste_v2 import representation as REPR
    from src.methodology_v2.encoder import collate_representations
    p = prep[ds]
    raw, key, npath = p["raw"], p["stft_key"], p["norm_path"]
    hop, n_fft = p["hop"], p["n_fft"]
    rate = REPR.RATES[key]

    def fn():
        rep = REPR.stft_rep(raw, key, "v1")
        nm = REPR._load_norm(npath)                      # cached: no disk access
        z = (rep - nm["mean"]) / nm["std_denominator"]
        item = (np.ascontiguousarray(z.T, dtype=np.float32),
                nm["frequency_hz"].astype(np.float32),
                (np.arange(rep.shape[0]) * hop / rate).astype(np.float32))
        b = collate_representations([item])
        if device != "cpu":
            b = {k: v.to(device, non_blocking=False) for k, v in b.items()}
        return hd(enc(**b)["global_embedding"], ds)
    return fn


def model_only_fn(ds: str, prep: dict, enc, hd):
    b = prep[ds]["batch"]

    def fn():
        return hd(enc(**b)["global_embedding"], ds)
    return fn


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------

def sh(cmd: str):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return r.stdout.strip() or None
    except Exception:
        return None


def environment() -> dict:
    import scipy
    import sklearn
    return {
        "hostname": socket.getfqdn(),
        "kernel": platform.release(),
        "cpu_model": (sh("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-") or "").strip(),
        "cpu_physical_cores": sh("lscpu | awk -F: '/^Core\\(s\\) per socket/{c=$2} /^Socket\\(s\\)/{s=$2} END{print c*s}'"),
        "cpu_logical_cpus": os.cpu_count(),
        "cpu_affinity_cpus": len(os.sched_getaffinity(0)),
        "ram_total_gib": round(int(sh("grep MemTotal /proc/meminfo | awk '{print $2}'") or 0) / 2**20, 1),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "gpu_query": sh("nvidia-smi --query-gpu=name,driver_version,memory.total,pstate,persistence_mode,clocks.max.sm,power.limit --format=csv,noheader"),
        "gpu_utilization_pct": sh("nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader"),
        "gpu_compute_processes": sh("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader") or "(none)",
        "cuda_version_torch": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "torch_default_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "env_vars": {k: os.environ.get(k) for k in
                     ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                      "CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF")},
        "loadavg": sh("cat /proc/loadavg"),
    }
