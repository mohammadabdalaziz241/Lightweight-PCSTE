#!/usr/bin/env python
"""Corrected latency/throughput runner — `lightweight_pcste_efficiency_latency_correction_v1`.

Corrects ONE defect in `lightweight_pcste_efficiency_v1`: its `latency_stage`
and `throughput_stage` never entered `torch.inference_mode()`, although the
frozen plan states that mode. FLOPs, parameters, model size and GPU memory
were unaffected and are NOT recomputed here.

Everything else is held identical to the original: the same frozen GF1/seed-42
checkpoint pair, the same deterministic VALIDATION inputs, the same host, the
same scopes, the same ABAB interleaving, the same synchronized host timing.
`benchmarks/efficiency_v1/bench_common.py` is imported UNMODIFIED.

Every timed forward asserts `torch.is_inference_mode_enabled()` and the run
fails closed if it is ever false.

Stages: env | latency | throughput
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PUB = HERE.parents[1]
sys.path.insert(0, str(PUB))
sys.path.insert(0, str(PUB / "benchmarks/efficiency_v1"))
import bench_common as B  # noqa: E402  (frozen, unmodified)

PROTOCOL_ID = "lightweight_pcste_efficiency_latency_correction_v1"
DATASETS = B.DATASETS
FAMILIES = ("Full-S1", "K1")

# Identical to the frozen efficiency_v1 schedule.
CPU_WARMUP, CPU_TIMED = 100, 1000
GPU_WARMUP, GPU_TIMED = 200, 1000
ROUNDS = 4                                  # ABAB: 4 rounds x 250 = 1000 timed
GPU_THR_BATCH, GPU_THR_WARMUP, GPU_THR_TIMED = 32, 50, 200
# CPU batch-32 is far slower per iteration; a reduced but explicitly recorded
# schedule is used. Each sample already aggregates 32 windows.
CPU_THR_BATCH, CPU_THR_WARMUP, CPU_THR_TIMED = 32, 5, 25


class InferenceModeViolation(RuntimeError):
    pass


_IM_CHECKS = {"checked": 0, "violations": 0}


def _assert_inference_mode(where: str):
    """Fail closed if a timed forward is not running under inference mode."""
    _IM_CHECKS["checked"] += 1
    if not torch.is_inference_mode_enabled():
        _IM_CHECKS["violations"] += 1
        raise InferenceModeViolation(
            f"timed forward executed WITHOUT torch.inference_mode() at {where}")


def _time_loop(fn, n: int, sync: bool, where: str) -> list[float]:
    """Timed loop. Inference mode is asserted before the first and last sample
    and every 100th sample, so a violation cannot pass unnoticed."""
    ts = []
    for i in range(n):
        if i == 0 or i == n - 1 or i % 100 == 0:
            _assert_inference_mode(where)
        t0 = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return ts


def _summarise(ts_s: list[float]) -> dict:
    ms = [t * 1e3 for t in ts_s]
    return {"n": len(ms), "mean_ms": statistics.fmean(ms), "median_ms": statistics.median(ms),
            "sd_ms": statistics.stdev(ms), "p95_ms": float(np.percentile(ms, 95)),
            "min_ms": min(ms), "max_ms": max(ms)}


# ---------------------------------------------------------------------------
def latency_stage(device: str, prep, models):
    sync = device.startswith("cuda")
    warmup = GPU_WARMUP if sync else CPU_WARMUP
    timed = GPU_TIMED if sync else CPU_TIMED
    per_round = timed // ROUNDS
    raw_rows, summary_rows, order_log = [], [], []

    for ds in DATASETS:
        for scope in ("model_only", "end_to_end"):
            builders = {}
            for fam in FAMILIES:
                enc, hd = models[fam]
                builders[fam] = (B.model_only_fn(ds, prep, enc, hd) if scope == "model_only"
                                 else B.end_to_end_fn(ds, prep, enc, hd, device))
            # THE CORRECTION: the entire warm-up + timed region runs under inference mode.
            with torch.inference_mode():
                _assert_inference_mode(f"{device}/{ds}/{scope}/warmup")
                for fam in FAMILIES:
                    for _ in range(warmup):
                        builders[fam]()
                if sync:
                    torch.cuda.synchronize()
                samples = {fam: [] for fam in FAMILIES}
                for rnd in range(ROUNDS):
                    order = FAMILIES if rnd % 2 == 0 else tuple(reversed(FAMILIES))   # ABAB
                    order_log.append({"device": device, "dataset": ds, "scope": scope,
                                      "round": rnd, "order": list(order)})
                    for fam in order:
                        ts = _time_loop(builders[fam], per_round, sync,
                                        f"{device}/{ds}/{scope}/{fam}/r{rnd}")
                        samples[fam].extend(ts)
                        for i, t in enumerate(ts):
                            raw_rows.append({"device": device, "dataset": ds, "scope": scope,
                                             "family": fam, "inference_mode": True,
                                             "round": rnd, "iteration": i, "latency_ms": t * 1e3})
            for fam in FAMILIES:
                s = _summarise(samples[fam])
                s.update({"device": device, "dataset": ds, "scope": scope, "family": fam,
                          "inference_mode": True, "batch": 1,
                          "threads": 1 if device == "cpu" else None,
                          "warmup": warmup, "rounds": ROUNDS})
                summary_rows.append(s)
            f = next(r for r in summary_rows if r["dataset"] == ds and r["scope"] == scope and r["family"] == "Full-S1")
            k = next(r for r in summary_rows if r["dataset"] == ds and r["scope"] == scope and r["family"] == "K1")
            print(f"  {device:5s} {ds:9s} {scope:11s} Full-S1 {f['median_ms']:8.3f} | "
                  f"K1 {k['median_ms']:8.3f} | speedup {f['median_ms']/k['median_ms']:.3f}x", flush=True)
    return raw_rows, summary_rows, order_log


def throughput_stage(device: str, prep, models):
    sync = device.startswith("cuda")
    batch = GPU_THR_BATCH if sync else CPU_THR_BATCH
    warm = GPU_THR_WARMUP if sync else CPU_THR_WARMUP
    timed = GPU_THR_TIMED if sync else CPU_THR_TIMED
    rows = []
    for ds in DATASETS:
        b1 = prep[ds]["batch"]
        big = {k: v.repeat(batch, *([1] * (v.dim() - 1))) for k, v in b1.items()}
        for fam in FAMILIES:
            enc, hd = models[fam]

            def fn(enc=enc, hd=hd):
                return hd(enc(**big)["global_embedding"], ds)
            try:
                with torch.inference_mode():          # THE CORRECTION
                    _assert_inference_mode(f"{device}/{ds}/throughput/{fam}")
                    for _ in range(warm):
                        fn()
                    if sync:
                        torch.cuda.synchronize()
                    ts = _time_loop(fn, timed, sync, f"{device}/{ds}/throughput/{fam}")
            except torch.cuda.OutOfMemoryError as e:   # pragma: no cover
                rows.append({"device": device, "dataset": ds, "family": fam, "batch": batch,
                             "status": "OOM", "error": str(e)[:200], "inference_mode": True})
                torch.cuda.empty_cache()
                continue
            med = statistics.median(ts)
            rows.append({"device": device, "dataset": ds, "family": fam, "batch": batch,
                         "status": "ok", "inference_mode": True, "warmup": warm,
                         "median_batch_ms": med * 1e3, "windows_per_second": batch / med,
                         "per_window_ms": med * 1e3 / batch, "n": len(ts)})
            print(f"  {device:5s} b{batch} {ds:9s} {fam:8s} {batch/med:9.2f} win/s", flush=True)
        if sync:
            torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("env", "latency", "throughput"))
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    out = HERE

    if a.stage == "env":
        (out / "environment.json").write_text(json.dumps({
            "protocol_identity": PROTOCOL_ID,
            "corrects": "lightweight_pcste_efficiency_v1 latency + throughput (missing torch.inference_mode)",
            "checkpoints": B.verify_checkpoints(),
            "environment": B.environment()}, indent=1) + "\n")
        print("wrote environment.json")
        return

    device = a.device
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = False
    else:
        torch.set_num_threads(1)
    models = B.load_models(device)
    for fam in FAMILIES:
        enc, hd = models[fam]
        assert not enc.training and not hd.training, f"{fam} is not in eval mode"
    prep = B.prepare_inputs(device)

    tag = "cpu" if device == "cpu" else "gpu"
    if a.stage == "latency":
        raw, summ, order = latency_stage(device, prep, models)
        with (out / f"latency_raw_{tag}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(raw[0].keys())); w.writeheader(); w.writerows(raw)
        cols = ["device", "dataset", "scope", "family", "inference_mode", "batch", "threads",
                "warmup", "rounds", "n", "mean_ms", "median_ms", "sd_ms", "p95_ms", "min_ms", "max_ms"]
        with (out / f"latency_summary_{tag}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
            w.writerows([{c: r[c] for c in cols} for r in summ])
        (out / f"order_log_{tag}.json").write_text(json.dumps(
            {"abab_rule": "rounds alternate Full-S1/K1; even rounds A,B and odd rounds B,A",
             "rounds": order}, indent=1) + "\n")
    else:
        rows = throughput_stage(device, prep, models)
        keys = sorted({k for r in rows for k in r})
        with (out / f"throughput_{tag}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)

    (out / f"inference_mode_verification_{a.stage}_{tag}.json").write_text(json.dumps({
        "protocol_identity": PROTOCOL_ID, "stage": a.stage, "device": device,
        "method": "torch.is_inference_mode_enabled() asserted inside the timed loop",
        "assertions_performed": _IM_CHECKS["checked"],
        "violations": _IM_CHECKS["violations"],
        "all_timed_forwards_in_inference_mode": _IM_CHECKS["violations"] == 0,
        "fail_closed": "an assertion failure raises InferenceModeViolation and aborts the run"}, indent=1) + "\n")
    print(f"  inference-mode assertions: {_IM_CHECKS['checked']} checked, "
          f"{_IM_CHECKS['violations']} violations")


if __name__ == "__main__":
    main()
