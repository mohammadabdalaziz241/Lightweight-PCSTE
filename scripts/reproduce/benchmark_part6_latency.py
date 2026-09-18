#!/usr/bin/env python
"""Validation-only latency benchmark for the final Part-6 Macro-3 study.

This intentionally reuses the sealed Part-6 constructors/loaders and Q8
implementations.  It never requests an allow_test token and fails if any
selected window is not explicitly labelled ``validation``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from scripts.methodology_v2.part6_compression import (  # noqa: E402
    RepStore, load_model_for)
from src.methodology_v2.compression import quantization as Q  # noqa: E402
from src.methodology_v2.compression.guards import (  # noqa: E402
    Part6GuardError, assert_no_test_windows, load_fold_manifest)
from src.methodology_v2.compression.student import count_params  # noqa: E402
from src.methodology_v2.encoder import collate_representations  # noqa: E402
from src.methodology_v2.experiment.heads import CLASS_ORDERS  # noqa: E402
from src.methodology_v2.integrity import sha256_file  # noqa: E402

FOLDS = (1, 2, 3)
SEEDS = (42, 1337, 2026)
DATASETS = ("JNU", "HIT", "MAFAULDA")
MODELS = ("Full S1", "K1", "Q8(K1)")
EXPECTED_PARAMS = {"Full S1": 2_382_033, "K1": 1_375_953,
                   "Q8(K1)": 1_375_953}
EXPECTED_LOGITS = {"JNU": 4, "HIT": 3, "MAFAULDA": 10}
OUT_DIR = REPO / "results/methodology_v2/part6_compression/latency"
AUDIT_PATH = OUT_DIR / "lightweight_protocol_audit.md"


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          check=True, capture_output=True, text=True).stdout.strip()


def cpu_info() -> dict:
    fields: dict[str, list[str]] = {}
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if ":" in line:
            key, value = (x.strip() for x in line.split(":", 1))
            fields.setdefault(key, []).append(value)
    physical = None
    if fields.get("physical id") and fields.get("core id"):
        physical = len(set(zip(fields["physical id"], fields["core id"])))
    if not physical:
        physical = os.cpu_count()
    return {"model": fields.get("model name", [platform.processor()])[0],
            "physical_cores": physical, "logical_cores": os.cpu_count()}


def metadata(args) -> dict:
    ci = cpu_info()
    cuda = torch.cuda.is_available() and "cuda" in args.devices
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(), "cpu": ci,
        "cpu_threads": torch.get_num_threads(),
        "cpu_interop_threads": torch.get_num_interop_threads(),
        "quantized_engine": torch.backends.quantized.engine,
        "quantized_supported_engines": list(torch.backends.quantized.supported_engines),
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda, "torch_version": torch.__version__,
        "python_version": platform.python_version(), "git_commit": git_head(),
        "warmup_iterations": args.warmup, "timed_iterations": args.iterations,
        "batch_size": 1, "datasets": list(DATASETS), "folds": list(FOLDS),
        "seeds": list(SEEDS), "devices_requested": list(args.devices),
        "timing_cpu": "time.perf_counter_ns",
        "timing_cuda": "CUDA Events with synchronization around each forward",
        "q8_gpu": "N/A — genuine INT8 CUDA runtime not implemented",
    }


def checkpoint_paths() -> dict:
    full, k1, ptq = {}, {}, {}
    for fold in FOLDS:
        for seed in SEEDS:
            cell = f"f{fold}_s{seed}"
            full[cell] = str(REPO / "results/methodology_v2/downstream" /
                             f"s1_f{fold}_s{seed}_l100/best.pt")
            k1[cell] = str(REPO / "results/methodology_v2/part6_compression/runs" /
                           f"k1_f{fold}_s{seed}/best.pt")
            ptq[cell] = str(REPO / "results/methodology_v2/part6_compression/ptq" /
                            f"ptq_k1_f{fold}_s{seed}.json")
    return {"Full S1": full, "K1": k1, "Q8 records": ptq}


def audit_artifacts(paths: dict) -> dict:
    records = {}
    for fold in FOLDS:
        for seed in SEEDS:
            cell = f"f{fold}_s{seed}"
            kp = Path(paths["K1"][cell])
            cp = kp.parent / "completion.json"
            stp = kp.parent / "state.json"
            qp = Path(paths["Q8 records"][cell])
            for p in (Path(paths["Full S1"][cell]), kp, cp, stp, qp):
                if not p.is_file():
                    raise Part6GuardError(f"required artifact missing: {p}")
            completion = json.loads(cp.read_text())
            state = json.loads(stp.read_text())
            qrec = json.loads(qp.read_text())
            k_hash = sha256_file(kp)
            if completion["best_checkpoint_sha256"] != k_hash:
                raise Part6GuardError(f"{cell}: K1 completion hash mismatch")
            if qrec["checkpoint_sha256"] != k_hash or qrec["model_id"] != f"k1_{cell}":
                raise Part6GuardError(f"{cell}: Q8 record does not identify final K1")
            ac = state["arm_config"]
            if (state["status"] != "COMPLETE" or ac["arm"] != "k1" or
                    ac["fold"] != fold or ac["seed"] != seed or
                    ac["architecture"]["name"] != "half_4x1" or
                    ac["teacher_set"] != "s1" or
                    ac["loss"]["kind"] != "kd_ensemble+relational"):
                raise Part6GuardError(f"{cell}: unexpected final K1 protocol identity")
            records[cell] = {"k1_sha256": k_hash,
                             "full_s1_sha256": sha256_file(Path(paths["Full S1"][cell])),
                             "q8_record": qrec}
    return records


def choose_validation_inputs() -> tuple[dict, dict]:
    batches, provenance = {}, {}
    for fold in FOLDS:
        man = load_fold_manifest(fold)  # structurally TRAIN+VALIDATION only
        selected = []
        store = RepStore(fold, man)
        for ds in DATASETS:
            candidates = sorted(man.index[(man["split"] == "validation") &
                                          (man["dataset"] == ds)])
            if not candidates:
                raise Part6GuardError(f"fold {fold} {ds}: no validation window")
            wid = candidates[0]
            if man.loc[wid, "split"] != "validation":
                raise Part6GuardError(f"non-validation selection: {wid}")
            selected.append(wid)
        assert_no_test_windows(selected, "latency benchmark selection")
        store.preload(selected)
        for ds, wid in zip(DATASETS, selected):
            batch = collate_representations([store.rep(wid)])
            if not bool(batch["cell_mask"].any()):
                raise Part6GuardError(f"{wid}: normal cell validity mask is empty")
            batches[(fold, ds)] = batch
            provenance[f"f{fold}_{ds}"] = {
                "window_id": wid, "split": "validation",
                "input_shapes": {k: list(v.shape) for k, v in batch.items()},
                "valid_input_cells": int(batch["cell_mask"].sum()),
                "temporal_patch_positions": int(batch["spec"].shape[-1] // 8),
            }
    return batches, provenance


def load_fp32(model: str, fold: int, seed: int, device: str):
    model_id = (f"s1_f{fold}_s{seed}_l100" if model == "Full S1"
                else f"k1_f{fold}_s{seed}")
    enc, heads, ck_hash, spec = load_model_for(model_id, "fp32", device)
    n = count_params(enc)
    if n != EXPECTED_PARAMS[model]:
        raise Part6GuardError(f"{model_id}: encoder params {n} != {EXPECTED_PARAMS[model]}")
    return enc, heads, ck_hash, spec


def load_q8_pair(fold: int, seed: int):
    """Return registered predictive simulation and real packed CPU runtime."""
    enc, heads, ck_hash, spec = load_fp32("K1", fold, seed, "cpu")
    sim_enc = Q.apply_q8_simulated(enc).model.eval()
    sim_heads = Q.apply_q8_simulated(heads).model.eval()
    packed_enc, enc_info = Q.cpu_dynamic_quantize(enc)
    packed_heads, head_info = Q.cpu_dynamic_quantize(heads)
    packed_enc.eval(); packed_heads.eval()
    if count_params(sim_enc) != EXPECTED_PARAMS["Q8(K1)"]:
        raise Part6GuardError("Q8 simulation logical topology changed")
    if enc_info["n_dynamic_linears"] != enc_info["n_planned"] or \
            head_info["n_dynamic_linears"] != head_info["n_planned"]:
        raise Part6GuardError("not every planned Q8 Linear became packed")
    return (sim_enc, sim_heads), (packed_enc, packed_heads), ck_hash, {
        "encoder": enc_info, "heads": head_info}


def forward(enc, heads, batch: dict, dataset: str):
    return heads(enc(**batch)["global_embedding"], dataset)


def validate_logits(enc, heads, batch, ds, context: str):
    with torch.inference_mode():
        logits = forward(enc, heads, batch, ds)
    expected = (1, EXPECTED_LOGITS[ds])
    if tuple(logits.shape) != expected:
        raise Part6GuardError(f"{context}: logits {tuple(logits.shape)} != {expected}")
    if not bool(torch.isfinite(logits).all()):
        raise Part6GuardError(f"{context}: non-finite logits")
    return logits


def q8_parity(batches: dict, args) -> tuple[list[dict], dict]:
    rows, runtime = [], {}
    for fold in FOLDS:
        man = load_fold_manifest(fold)
        store = RepStore(fold, man)
        chosen = {}
        for ds in DATASETS:
            ids = sorted(man.index[(man["split"] == "validation") &
                                   (man["dataset"] == ds)])[:args.parity_windows]
            assert_no_test_windows(ids, "Q8 packed parity")
            chosen[ds] = ids
        store.preload([w for ids in chosen.values() for w in ids])
        for seed in SEEDS:
            sim, packed, ck_hash, info = load_q8_pair(fold, seed)
            runtime[f"f{fold}_s{seed}"] = {"checkpoint_sha256": ck_hash, **info}
            for ds in DATASETS:
                total_abs = 0.0
                n_logits = agree = 0
                max_abs = 0.0
                for wid in chosen[ds]:
                    b = collate_representations([store.rep(wid)])
                    ls = validate_logits(*sim, b, ds, f"Q8 simulation {wid}")
                    lp = validate_logits(*packed, b, ds, f"Q8 packed {wid}")
                    d = (lp.float() - ls.float()).abs()
                    max_abs = max(max_abs, float(d.max()))
                    total_abs += float(d.sum()); n_logits += d.numel()
                    agree += int(lp.argmax(-1).item() == ls.argmax(-1).item())
                rows.append({"fold": fold, "seed": seed, "dataset": ds,
                             "n_validation_windows": len(chosen[ds]),
                             "prediction_agreement_percent": 100 * agree / len(chosen[ds]),
                             "max_abs_logit_difference": max_abs,
                             "mean_abs_logit_difference": total_abs / n_logits})
    return rows, runtime


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_cell(values: list[float]) -> dict:
    mean = statistics.fmean(values)
    return {"mean_ms": mean, "sd_ms": statistics.stdev(values),
            "median_ms": statistics.median(values), "p95_ms": percentile(values, 95),
            "windows_per_second": 1000.0 / mean}


def cpu_times(enc, heads, batch, ds, warmup, iterations):
    with torch.inference_mode():
        for _ in range(warmup):
            forward(enc, heads, batch, ds)
        values = []
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            forward(enc, heads, batch, ds)
            values.append((time.perf_counter_ns() - t0) / 1e6)
    return values


def gpu_times(enc, heads, batch, ds, warmup, iterations):
    with torch.inference_mode():
        for _ in range(warmup):
            forward(enc, heads, batch, ds)
        torch.cuda.synchronize()
        values = []
        for _ in range(iterations):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            forward(enc, heads, batch, ds)
            end.record()
            torch.cuda.synchronize()
            values.append(float(start.elapsed_time(end)))
    return values


def run_benchmarks(batches, args):
    raw, cells = [], []
    for device in args.devices:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device_models = MODELS if device == "cpu" else ("Full S1", "K1")
        for fold in FOLDS:
            for seed in SEEDS:
                for model in device_models:
                    if model == "Q8(K1)":
                        _, (enc, heads), _, _ = load_q8_pair(fold, seed)
                    else:
                        enc, heads, _, _ = load_fp32(model, fold, seed, device)
                    enc.eval(); heads.eval()
                    for ds in DATASETS:
                        batch = {k: v.to(device) for k, v in batches[(fold, ds)].items()}
                        validate_logits(enc, heads, batch, ds,
                                        f"{model} f{fold} s{seed} {ds} {device}")
                        peak = None
                        if device == "cuda":
                            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                            values = gpu_times(enc, heads, batch, ds,
                                               args.warmup, args.iterations)
                            peak = torch.cuda.max_memory_allocated() / 2**20
                        else:
                            values = cpu_times(enc, heads, batch, ds,
                                               args.warmup, args.iterations)
                        for iteration, value in enumerate(values, 1):
                            raw.append({"model": model, "fold": fold, "seed": seed,
                                        "dataset": ds, "device": device,
                                        "iteration": iteration, "latency_ms": value})
                        cells.append({"model": model, "fold": fold, "seed": seed,
                                      "dataset": ds, "device": device,
                                      **summarize_cell(values),
                                      "peak_cuda_memory_mb": peak})
                        print(f"BENCH {device:4s} {model:8s} f{fold} s{seed} {ds:8s} "
                              f"{cells[-1]['mean_ms']:.4f} ms", flush=True)
                    del enc, heads
                    if device == "cuda":
                        torch.cuda.empty_cache()
    return raw, cells


def make_summary(cells: list[dict]) -> list[dict]:
    rows = []
    combos = [(m, d) for d in ("cpu", "cuda") for m in MODELS
              if any(x["model"] == m and x["device"] == d for x in cells)]
    full_by_device = {}
    for model, device in combos:
        ds_means = {ds: statistics.fmean(x["mean_ms"] for x in cells
                                         if x["model"] == model and
                                         x["device"] == device and x["dataset"] == ds)
                    for ds in DATASETS}
        cell_equal = []
        for fold in FOLDS:
            for seed in SEEDS:
                vals = [x["mean_ms"] for x in cells if x["model"] == model and
                        x["device"] == device and x["fold"] == fold and
                        x["seed"] == seed]
                if len(vals) == len(DATASETS):
                    cell_equal.append(statistics.fmean(vals))
        eq = statistics.fmean(ds_means.values())
        row = {"model": model, "device": device,
               "jnu_mean_ms": ds_means["JNU"], "hit_mean_ms": ds_means["HIT"],
               "mafaulda_mean_ms": ds_means["MAFAULDA"],
               "equal_domain_mean_ms": eq,
               "equal_domain_sd_ms": statistics.stdev(cell_equal),
               "median_ms": statistics.median(cell_equal),
               "p95_ms": percentile(cell_equal, 95),
               "windows_per_second": 1000.0 / eq}
        rows.append(row)
        if model == "Full S1":
            full_by_device[device] = eq
    for row in rows:
        baseline = full_by_device[row["device"]]
        row["speedup_vs_full"] = baseline / row["equal_domain_mean_ms"]
        row["latency_reduction_percent"] = 100 * (1 - row["equal_domain_mean_ms"] / baseline)
        if row["model"] == "Q8(K1)":
            k1 = next(x["equal_domain_mean_ms"] for x in rows
                      if x["model"] == "K1" and x["device"] == "cpu")
            row["speedup_vs_k1"] = k1 / row["equal_domain_mean_ms"]
            row["latency_reduction_vs_k1_percent"] = 100 * (1 - row["equal_domain_mean_ms"] / k1)
        else:
            row["speedup_vs_k1"] = None
            row["latency_reduction_vs_k1_percent"] = None
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def md_table(rows, cols, labels=None, digits=4):
    labels = labels or cols
    out = ["| " + " | ".join(labels) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for row in rows:
        vals = []
        for c in cols:
            v = row.get(c)
            vals.append("N/A" if v is None else
                        (f"{v:.{digits}f}" if isinstance(v, float) else str(v)))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def audit_markdown(paths: dict, records: dict) -> str:
    full_paths = "\n".join(f"- `{p}`" for p in paths["Full S1"].values())
    k1_paths = "\n".join(f"- `{p}`" for p in paths["K1"].values())
    ptq_paths = "\n".join(f"- `{p}`" for p in paths["Q8 records"].values())
    first = records["f1_s42"]["q8_record"]
    return f"""# Lightweight protocol audit

This audit identifies the actual artifacts behind the dissertation's final
Part-6 Macro-3 presentation. It does not reinterpret the experiment as a
three-domain training run.

## A. Full S1 reference

The Full S1 reference is the nine frozen 100%-label primary downstream
checkpoints below. The primary supervised sampler defines the four training datasets as
`CWRU`, `JNU`, `HIT`, and `MAFAULDA` and gives each present dataset equal loss
weight (`src/methodology_v2/experiment/samplers.py:17-84` and `src/methodology_v2/experiment/trainers.py:190-231`). Therefore these
references and the downstream protocol contain four classification heads and
CWRU contributed supervised classification loss.

{full_paths}

## B. K1

K1 uses the nine `half_4x1` checkpoints below (four temporal layers, forward
direction retained), with 1,375,953 encoder parameters
(`methodology_v2/part6_compression/student_spec.yaml:2-35`). K1 training used
all four dataset buckets (`src/methodology_v2/compression/trainer.py:22-29,151-181`).
Its registered `kd_ensemble+relational` loss includes CE and KD for every
dataset, including CWRU (`src/methodology_v2/compression/losses.py:1-22,137-190`).
The K1 teacher is the same-fold ensemble `S1(f,42)+S1(f,1337)+S1(f,2026)`,
combined as mean softmax probability at T=4
(`methodology_v2/part6_compression/kd_spec.yaml:69-78` and
`src/methodology_v2/compression/teachers.py:1-17,100-128`). The relational
target is the same-cell S1 teacher's valid-band attention.

{k1_paths}

The repository explicitly describes Macro-3 as a post-hoc reaggregation of
saved JNU, HIT, and MaFaulDa results from the frozen four-domain experiment;
CWRU is excluded only from that aggregate
(`results/lightweight_macro3_reaggregation/README.md:1-8` and
`results/lightweight_macro3_reaggregation/provenance/metric_definition.md:1-4`).

## C. Q8(K1)

No standalone Q8 checkpoint is stored. Each PTQ JSON below binds the registered
Q8 derivation to the SHA-256 of its final K1 `best.pt`; this benchmark verified
all nine bindings. The predictive Q8 results use `apply_q8_simulated`, which
per-output-channel quantizes allowlisted Linear weights and installs their
dequantized FP32 values for cross-device inference
(`scripts/methodology_v2/part6_compression.py:545-578` and
`src/methodology_v2/compression/quantization.py:86-144`).

{ptq_paths}

The reported compact serialized size is generated by `q8_report` from the INT8
tensors, FP32 per-channel scales, biases, and untouched FP32 tensors; for
f1/s42 it is {first['int8_total_bytes']:,} bytes (1.600486 decimal MB), not the
packed runtime state (`src/methodology_v2/compression/quantization.py:186-209`
and `scripts/methodology_v2/part6_compression.py:582-620`).

The real packed CPU deployment path is `cpu_dynamic_quantize`: restricted
allowlisted `torch.ao.quantization.quantize_dynamic` with per-channel dynamic
qconfig and packed INT8 Linear weights/quantized activations
(`src/methodology_v2/compression/quantization.py:166-184`). It is a distinct
deployment representation and its numerical relationship to the dissertation
prediction representation is measured on validation data below. The CUDA
simulation is the same `apply_q8_simulated` FP32-compute representation; it is
not a packed INT8 CUDA runtime and is not latency-benchmarked here.
"""


def report_markdown(meta, paths, records, parity, cells, summary) -> str:
    cpu_ds = []
    gpu_ds = []
    for model in MODELS:
        for device, target in (("cpu", cpu_ds), ("cuda", gpu_ds)):
            for ds in DATASETS:
                vals = [x for x in cells if x["model"] == model and
                        x["device"] == device and x["dataset"] == ds]
                if vals:
                    target.append({"model": model, "dataset": ds,
                                   "mean_ms": statistics.fmean(x["mean_ms"] for x in vals),
                                   "sd_cell_ms": statistics.stdev(x["mean_ms"] for x in vals),
                                   "peak_mb": (max(x["peak_cuda_memory_mb"] for x in vals)
                                               if device == "cuda" else None)})
    parity_summary = []
    for ds in DATASETS:
        rr = [x for x in parity if x["dataset"] == ds]
        parity_summary.append({"dataset": ds,
            "n": sum(x["n_validation_windows"] for x in rr),
            "agreement": statistics.fmean(x["prediction_agreement_percent"] for x in rr),
            "max_delta": max(x["max_abs_logit_difference"] for x in rr),
            "mean_delta": statistics.fmean(x["mean_abs_logit_difference"] for x in rr)})
    q8_na = {"model": "Q8(K1)", "device": "cuda", "jnu_mean_ms": None,
             "hit_mean_ms": None, "mafaulda_mean_ms": None,
             "equal_domain_mean_ms": None, "speedup_vs_full": None,
             "latency_reduction_percent": None}
    return f"""# Part-6 inference latency report

## Scope and protocol audit

Model forward-pass latency only was measured at batch size 1: encoder plus the
correct dataset-specific head through logits. Disk I/O, checkpoint loading,
dataset loading, STFT, N2 preprocessing, transfers, and serialization were
outside timing. There were {meta['warmup_iterations']} warm-ups and
{meta['timed_iterations']} individually timed forwards per cell. No significance
testing was performed on repeated timings.

Full details are in [lightweight_protocol_audit.md](lightweight_protocol_audit.md).
Full S1 and K1 were trained under the frozen four-domain CWRU+JNU+HIT+MaFaulDa
protocol. K1's CWRU windows contributed CE, KD, and relational loss. The final
dissertation Macro-3 values—and this equal-domain latency summary—exclude CWRU
post hoc and average JNU, HIT, and MaFaulDa equally.

Only deterministic N2-normalised **validation** representations were loaded.
Every selected ID was structurally checked as validation. **TEST data and sealed
TEST artifacts were not opened or modified by this benchmark.**

## Hardware and runtime

- Hostname: `{meta['hostname']}`
- CPU: {meta['cpu']['model']}
- CPU cores: {meta['cpu']['physical_cores']} physical / {meta['cpu']['logical_cores']} logical
- PyTorch CPU threads: {meta['cpu_threads']} intra-op / {meta['cpu_interop_threads']} inter-op
- Packed Q8 engine: `{meta['quantized_engine']}`
- GPU: {meta['gpu'] or 'not benchmarked'}
- CUDA: {meta['cuda_version']}
- PyTorch: {meta['torch_version']}
- Python: {meta['python_version']}
- Git commit: `{meta['git_commit']}`

FP32 execution uses the repository's `build_encoder`/`build_heads`, strict
checkpoint loads, `PCSTE.forward`, and `DatasetHeads.forward`. No AMP/autocast
was used. Packed Q8 CPU uses the registered `cpu_dynamic_quantize` path on both
encoder and heads. Exact per-cell paths and hashes are recorded in metadata.

## Q8 runtime/parity audit

The dissertation predictive representation is per-output-channel weight-only
Q8 followed by FP32 compute. The packed deployment model additionally performs
dynamic activation quantization, so it is neither assumed nor described as
numerically identical. Comparison against the dissertation representation:

{md_table(parity_summary, ['dataset','n','agreement','max_delta','mean_delta'], ['Dataset','N validation windows','Prediction agreement %','Max absolute logit delta','Mean absolute logit delta'], 6)}

The 1.600486 MB result is the compact serializable tensor representation for
the f1/s42 example; it is not the packed `torch.ao` runtime state size.

## CPU latency by dataset

Values are means across the nine matched fold-seed cell means; SD is across
those nine model cells.

{md_table(cpu_ds, ['model','dataset','mean_ms','sd_cell_ms'], ['Model','Dataset','Mean ms/window','Cell SD ms'])}

## Equal-domain CPU results and speedups

{md_table([x for x in summary if x['device']=='cpu'], ['model','equal_domain_mean_ms','equal_domain_sd_ms','windows_per_second','speedup_vs_full','latency_reduction_percent','speedup_vs_k1','latency_reduction_vs_k1_percent'], ['Model','Equal-domain ms','Cell SD ms','windows/s','Speedup vs Full','Reduction vs Full %','Speedup vs K1','Reduction vs K1 %'])}

## GPU latency by dataset

{md_table(gpu_ds, ['model','dataset','mean_ms','sd_cell_ms','peak_mb'], ['Model','Dataset','Mean ms/window','Cell SD ms','Peak allocated MiB'])}

## Equal-domain GPU results

{md_table([x for x in summary if x['device']=='cuda'] + [q8_na], ['model','device','jnu_mean_ms','hit_mean_ms','mafaulda_mean_ms','equal_domain_mean_ms','speedup_vs_full','latency_reduction_percent'], ['Model','Device','JNU ms','HIT ms','MaFaulDa ms','Equal-domain ms','Speedup vs Full','Reduction vs Full %'])}

Peak CUDA allocated memory is reported per dataset/cell in
`latency_by_cell.csv`; the table above reports the maximum over nine cells.

**A genuine INT8 Q8(K1) GPU latency is not reported because the current
implementation has no packed INT8 CUDA execution path.**

## Warnings and interpretation

- K1/Q8 are four-domain-trained artifacts; Macro-3 is a post-hoc reporting scope.
- Packed Q8 CPU and the compact predictive Q8 representation are distinct, as
  quantified by the validation parity audit.
- The pure-PyTorch reference selective scan is the active model backend.
- Timing iterations estimate runtime variability and are not independent
  experimental replicates. A later paired comparison should use matched
  fold-seed/domain model cells.
"""


def print_terminal(meta, parity, cells, summary):
    print("\nA. LIGHTWEIGHT PROTOCOL AUDIT")
    print("Four-domain training (CWRU+JNU+HIT+MaFaulDa); Macro-3 is post-hoc. See", AUDIT_PATH)
    print("\nB. Q8 RUNTIME/PARITY AUDIT")
    print(md_table(parity, ["fold","seed","dataset","n_validation_windows",
                           "prediction_agreement_percent","max_abs_logit_difference",
                           "mean_abs_logit_difference"]))
    for device, title in (("cpu", "C. CPU RESULTS"), ("cuda", "D. GPU RESULTS")):
        print("\n" + title)
        dsrows = []
        for m in MODELS:
            for ds in DATASETS:
                rr = [x for x in cells if x["device"] == device and
                      x["model"] == m and x["dataset"] == ds]
                if rr:
                    dsrows.append({"model": m, "dataset": ds,
                                   "mean_ms": statistics.fmean(x["mean_ms"] for x in rr)})
        print(md_table(dsrows, ["model","dataset","mean_ms"]))
        if device == "cuda":
            print("Q8(K1): N/A — genuine INT8 CUDA runtime not implemented")
    print("\nE. EQUAL-DOMAIN SUMMARY")
    print(md_table(summary, ["model","device","equal_domain_mean_ms",
                             "windows_per_second","speedup_vs_full"]))
    print("\nF. SPEEDUPS AND LATENCY REDUCTIONS")
    print(md_table(summary, ["model","device","speedup_vs_full",
                             "latency_reduction_percent","speedup_vs_k1",
                             "latency_reduction_vs_k1_percent"]))
    print("\nG. OUTPUT PATHS")
    for p in (AUDIT_PATH, OUT_DIR / "latency_raw.csv", OUT_DIR / "latency_by_cell.csv",
              OUT_DIR / "latency_summary.csv", OUT_DIR / "latency_metadata.json",
              OUT_DIR / "LATENCY_REPORT.md"):
        print(p)
    print("\nH. TEST-ISOLATION CONFIRMATION")
    print("Validation representations only; TEST data and sealed TEST artifacts were not opened or modified.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", nargs="+", choices=("cpu", "cuda"),
                    default=("cpu", "cuda"))
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iterations", type=int, default=500)
    ap.add_argument("--parity-windows", type=int, default=16,
                    help="deterministic validation windows per fold/dataset")
    args = ap.parse_args()
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as e:
        print(f"WARNING: could not set inter-op threads before work: {e}", file=sys.stderr)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = checkpoint_paths()
    records = audit_artifacts(paths)
    AUDIT_PATH.write_text(audit_markdown(paths, records))
    batches, input_provenance = choose_validation_inputs()
    parity, packed_runtime = q8_parity(batches, args)
    raw, cells = run_benchmarks(batches, args)
    summary = make_summary(cells)
    meta = metadata(args)
    meta.update({"artifact_paths": paths,
                 "artifact_hashes_and_q8_records": records,
                 "validation_inputs": input_provenance,
                 "q8_packed_runtime": packed_runtime,
                 "q8_parity": parity,
                 "test_isolation": "load_fold_manifest called without allow_test; selected IDs explicitly validation; no TEST artifact path opened"})
    write_csv(OUT_DIR / "latency_raw.csv", raw,
              ["model","fold","seed","dataset","device","iteration","latency_ms"])
    write_csv(OUT_DIR / "latency_by_cell.csv", cells,
              ["model","fold","seed","dataset","device","mean_ms","sd_ms",
               "median_ms","p95_ms","windows_per_second","peak_cuda_memory_mb"])
    write_csv(OUT_DIR / "latency_summary.csv", summary,
              ["model","device","jnu_mean_ms","hit_mean_ms","mafaulda_mean_ms",
               "equal_domain_mean_ms","equal_domain_sd_ms","median_ms","p95_ms",
               "windows_per_second","speedup_vs_full","latency_reduction_percent",
               "speedup_vs_k1","latency_reduction_vs_k1_percent"])
    (OUT_DIR / "latency_metadata.json").write_text(json.dumps(meta, indent=2,
                                                               sort_keys=True))
    (OUT_DIR / "LATENCY_REPORT.md").write_text(
        report_markdown(meta, paths, records, parity, cells, summary))
    print_terminal(meta, parity, cells, summary)


if __name__ == "__main__":
    main()
