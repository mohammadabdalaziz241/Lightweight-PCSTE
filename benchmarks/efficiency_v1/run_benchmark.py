#!/usr/bin/env python
"""Measurement driver for `lightweight_pcste_efficiency_v1`.

Stages: env | params | flops | latency | throughput | memory

Every stage follows `EFFICIENCY_BENCHMARK_PLAN.json`. No TEST data is read, no
model is trained, no Q8 path is executed, and no historical source is modified.

Usage:
    python run_benchmark.py --stage latency --device cpu
    python run_benchmark.py --stage memory --model K1     # internal, per-process
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench_common as B  # noqa: E402

DATASETS = B.DATASETS
FAMILIES = ("Full-S1", "K1")

CPU_WARMUP, CPU_TIMED = 100, 1000
GPU_WARMUP, GPU_TIMED = 200, 1000
THR_BATCH, THR_WARMUP, THR_TIMED = 32, 50, 200
ROUNDS = 4          # ABAB interleaving: 4 rounds x 250 samples = 1000 timed


# ---------------------------------------------------------------------------
def param_report(enc, heads) -> dict:
    def tally(m):
        tot = sum(p.numel() for p in m.parameters())
        tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
        raw = sum(p.numel() * p.element_size() for p in m.parameters())
        return {"total_params": tot, "trainable_params": tr, "raw_fp32_bytes": raw}

    enc_t, head_t = tally(enc), tally(heads)
    full = {"total_params": enc_t["total_params"] + head_t["total_params"],
            "trainable_params": enc_t["trainable_params"] + head_t["trainable_params"],
            "raw_fp32_bytes": enc_t["raw_fp32_bytes"] + head_t["raw_fp32_bytes"]}

    def serialized(state: dict) -> int:
        fd, tmp = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        try:
            torch.save(state, tmp)          # state_dict-only FP32, identical procedure
            return Path(tmp).stat().st_size
        finally:
            Path(tmp).unlink(missing_ok=True)   # never enters Git

    enc_state = {k: v.float().cpu() for k, v in enc.state_dict().items()}
    head_state = {k: v.float().cpu() for k, v in heads.state_dict().items()}
    full_state = {**{f"encoder.{k}": v for k, v in enc_state.items()},
                  **{f"heads.{k}": v for k, v in head_state.items()}}
    enc_t["serialized_state_dict_bytes"] = serialized(enc_state)
    full["serialized_state_dict_bytes"] = serialized(full_state)
    head_t["serialized_state_dict_bytes"] = serialized(head_state)
    for d in (enc_t, head_t, full):
        d["raw_fp32_mib"] = d["raw_fp32_bytes"] / 2**20
        d["serialized_state_dict_mib"] = d["serialized_state_dict_bytes"] / 2**20
    return {"encoder_only": enc_t, "heads_only": head_t, "complete_inference_model": full}


# ---------------------------------------------------------------------------
def flop_report(models, prep) -> dict:
    from torch.utils.flop_counter import FlopCounterMode
    from src.methodology_v2.encoder.ssm import BiMambaLayer

    out = {"convention": {
        "reported_quantity": "FLOPs (not MACs)",
        "dense_term": "torch.utils.flop_counter.FlopCounterMode (conv/addmm/mm/bmm), 2 x MAC for matrix products",
        "scan_term_formula": "batch * n_bands * sum_layers(directions) * T * d_inner * d_state * C",
        "scan_constant_C": B.SCAN_FLOPS_PER_STATE_ELEMENT_PER_STEP,
        "scan_constant_provenance": "preserved from the historical Part-6 analytic_scan_flops",
        "scan_constant_sensitivity": "a strict per-op tally gives ~8, scaling the scan term by 4/3; identical for both models so the ratio is unchanged",
        "total": "total = dense + scan (never the profiler alone)",
        "n_bands_and_T": "derived from the live patch grid, not hardcoded",
        "unit": "per one 1-second window",
    }, "per_model": {}}

    for fam in FAMILIES:
        enc, hd = models[fam]
        enc.eval(); hd.eval()
        blk = enc.temporal.layers[0].fwd
        dirs_total = sum(2 if isinstance(L, BiMambaLayer) else 1 for L in enc.temporal.layers)
        per_ds = {}
        for ds in DATASETS:
            b = prep[ds]["batch"]
            fb, tp = prep[ds]["meta"]["patch_grid_bands_timepatches"]
            with torch.inference_mode():
                fc = FlopCounterMode(display=False)
                with fc:
                    hd(enc(**b)["global_embedding"], ds)
            dense = float(fc.get_total_flops())
            scan = float(1 * fb * dirs_total * tp * blk.d_inner * blk.d_state
                         * B.SCAN_FLOPS_PER_STATE_ELEMENT_PER_STEP)
            per_ds[ds] = {
                "n_bands": fb, "time_patches": tp,
                "directions_summed_over_layers": dirs_total,
                "d_inner": blk.d_inner, "d_state": blk.d_state,
                "dense_flops": dense, "scan_flops": scan, "total_flops": dense + scan,
                "dense_gflops": dense / 1e9, "scan_gflops": scan / 1e9,
                "total_gflops": (dense + scan) / 1e9,
                "dense_op_breakdown": {str(k): int(v) for k, v in fc.get_flop_counts()["Global"].items()},
            }
        macro = {k: float(np.mean([per_ds[d][k] for d in DATASETS]))
                 for k in ("dense_gflops", "scan_gflops", "total_gflops")}
        out["per_model"][fam] = {"per_dataset": per_ds, "macro4_equal_domain_mean": macro}

    f, k = out["per_model"]["Full-S1"], out["per_model"]["K1"]
    out["comparison"] = {
        "per_dataset": {ds: {
            "full_s1_total_gflops": f["per_dataset"][ds]["total_gflops"],
            "k1_total_gflops": k["per_dataset"][ds]["total_gflops"],
            "ratio_full_over_k1": f["per_dataset"][ds]["total_gflops"] / k["per_dataset"][ds]["total_gflops"],
            "k1_reduction_fraction": 1 - k["per_dataset"][ds]["total_gflops"] / f["per_dataset"][ds]["total_gflops"],
        } for ds in DATASETS},
        "macro4": {
            "full_s1_total_gflops": f["macro4_equal_domain_mean"]["total_gflops"],
            "k1_total_gflops": k["macro4_equal_domain_mean"]["total_gflops"],
            "ratio_full_over_k1": f["macro4_equal_domain_mean"]["total_gflops"] / k["macro4_equal_domain_mean"]["total_gflops"],
            "k1_reduction_fraction": 1 - k["macro4_equal_domain_mean"]["total_gflops"] / f["macro4_equal_domain_mean"]["total_gflops"],
        }}
    return out


# ---------------------------------------------------------------------------
def _time_loop(fn, n: int, sync: bool) -> list[float]:
    ts = []
    for _ in range(n):
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


def latency_stage(device: str, prep, models) -> tuple[list[dict], list[dict]]:
    sync = device.startswith("cuda")
    warmup = GPU_WARMUP if sync else CPU_WARMUP
    timed = GPU_TIMED if sync else CPU_TIMED
    per_round = timed // ROUNDS

    raw_rows, summary_rows = [], []
    for ds in DATASETS:
        for scope in ("model_only", "end_to_end"):
            builders = {}
            for fam in FAMILIES:
                enc, hd = models[fam]
                builders[fam] = (B.model_only_fn(ds, prep, enc, hd) if scope == "model_only"
                                 else B.end_to_end_fn(ds, prep, enc, hd, device))
            # warm-up both before any timing
            for fam in FAMILIES:
                for _ in range(warmup):
                    builders[fam]()
            if sync:
                torch.cuda.synchronize()

            samples = {fam: [] for fam in FAMILIES}
            for rnd in range(ROUNDS):
                order = FAMILIES if rnd % 2 == 0 else tuple(reversed(FAMILIES))  # ABAB
                for fam in order:
                    ts = _time_loop(builders[fam], per_round, sync)
                    samples[fam].extend(ts)
                    for i, t in enumerate(ts):
                        raw_rows.append({"device": device, "dataset": ds, "scope": scope,
                                         "family": fam, "round": rnd, "iteration": i,
                                         "latency_ms": t * 1e3})
            for fam in FAMILIES:
                s = _summarise(samples[fam])
                s.update({"device": device, "dataset": ds, "scope": scope, "family": fam,
                          "batch": 1, "warmup": warmup, "rounds": ROUNDS})
                summary_rows.append(s)
            f = next(r for r in summary_rows if r["dataset"] == ds and r["scope"] == scope and r["family"] == "Full-S1")
            k = next(r for r in summary_rows if r["dataset"] == ds and r["scope"] == scope and r["family"] == "K1")
            print(f"  {device:5s} {ds:9s} {scope:11s} Full-S1 {f['median_ms']:8.3f} ms | "
                  f"K1 {k['median_ms']:8.3f} ms | speedup {f['median_ms']/k['median_ms']:.3f}x", flush=True)
    return raw_rows, summary_rows


# ---------------------------------------------------------------------------
def throughput_stage(prep, models, device="cuda") -> list[dict]:
    rows = []
    for ds in DATASETS:
        b1 = prep[ds]["batch"]
        big = {k: v.repeat(THR_BATCH, *([1] * (v.dim() - 1))) for k, v in b1.items()}
        for fam in FAMILIES:
            enc, hd = models[fam]

            def fn(enc=enc, hd=hd):
                return hd(enc(**big)["global_embedding"], ds)
            try:
                for _ in range(THR_WARMUP):
                    fn()
                torch.cuda.synchronize()
                ts = _time_loop(fn, THR_TIMED, True)
            except torch.cuda.OutOfMemoryError as e:  # pragma: no cover
                rows.append({"dataset": ds, "family": fam, "batch": THR_BATCH,
                             "status": "OOM", "error": str(e)[:200]})
                torch.cuda.empty_cache()
                continue
            med = statistics.median(ts)
            rows.append({"dataset": ds, "family": fam, "batch": THR_BATCH, "status": "ok",
                         "median_batch_ms": med * 1e3,
                         "windows_per_second": THR_BATCH / med,
                         "per_window_ms": med * 1e3 / THR_BATCH, "n": len(ts)})
        torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------------------
_MEM_ONE = """
import json, sys, torch
sys.path.insert(0, {here!r})
import bench_common as B
fam = {fam!r}
torch.cuda.init(); torch.cuda.synchronize()
base = torch.cuda.memory_allocated()
models = B.load_models("cuda")
enc, hd = models[fam]
del models[{other!r}]
torch.cuda.synchronize()
after_load = torch.cuda.memory_allocated()
prep = B.prepare_inputs("cuda")
out = {{}}
for ds in B.DATASETS:
    fn = B.model_only_fn(ds, prep, enc, hd)
    with torch.inference_mode():
        for _ in range(5):
            fn()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    with torch.inference_mode():
        fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    out[ds] = {{"allocated_before_forward_bytes": before,
               "peak_allocated_during_forward_bytes": peak,
               "forward_increment_bytes": peak - before}}
print("@@JSON@@" + json.dumps({{"family": fam, "baseline_bytes": base,
      "allocated_after_model_load_bytes": after_load,
      "model_resident_bytes": after_load - base, "per_dataset": out}}))
"""


def memory_stage(repeat: int = 2) -> dict:
    res = {"method": "clean subprocess per model; reset_peak_memory_stats + synchronize; batch 1 model-only",
           "repeats": repeat, "runs": []}
    for r in range(repeat):
        for fam in FAMILIES:
            other = "K1" if fam == "Full-S1" else "Full-S1"
            code = _MEM_ONE.format(here=str(HERE), fam=fam, other=other)
            p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                               cwd=str(B.PUB))
            if p.returncode != 0:
                res["runs"].append({"repeat": r, "family": fam, "status": "failed",
                                    "stderr": p.stderr[-800:]})
                continue
            line = [l for l in p.stdout.splitlines() if l.startswith("@@JSON@@")][-1]
            d = json.loads(line[len("@@JSON@@"):])
            d.update({"repeat": r, "status": "ok"})
            res["runs"].append(d)
            print(f"  memory repeat={r} {fam:8s} resident={d['model_resident_bytes']/2**20:.2f} MiB", flush=True)
    return res


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("env", "params", "flops", "latency", "throughput", "memory"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    out = HERE

    if args.stage == "env":
        (out / "environment.json").write_text(json.dumps(
            {"benchmark_identity": "lightweight_pcste_efficiency_v1",
             "checkpoints": B.verify_checkpoints(),
             "environment": B.environment()}, indent=1) + "\n")
        print("wrote environment.json"); return

    if args.stage == "memory":
        (out / "memory_results.json").write_text(json.dumps(memory_stage(), indent=1) + "\n")
        print("wrote memory_results.json"); return

    device = args.device
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = False
    else:
        torch.set_num_threads(1)
    models = B.load_models(device)
    prep = B.prepare_inputs(device)

    if args.stage == "params":
        rep = {"note": "device-independent; measured on CPU-resident FP32 parameters",
               "per_model": {fam: param_report(*models[fam]) for fam in FAMILIES}}
        f, k = rep["per_model"]["Full-S1"], rep["per_model"]["K1"]
        rep["comparison"] = {part: {
            "full_s1": f[part]["total_params"], "k1": k[part]["total_params"],
            "k1_reduction_fraction": 1 - k[part]["total_params"] / f[part]["total_params"],
            "ratio_full_over_k1": f[part]["total_params"] / k[part]["total_params"],
            "full_s1_serialized_bytes": f[part]["serialized_state_dict_bytes"],
            "k1_serialized_bytes": k[part]["serialized_state_dict_bytes"],
            "serialized_reduction_fraction": 1 - k[part]["serialized_state_dict_bytes"] / f[part]["serialized_state_dict_bytes"],
            "full_s1_raw_fp32_bytes": f[part]["raw_fp32_bytes"], "k1_raw_fp32_bytes": k[part]["raw_fp32_bytes"],
            "raw_fp32_reduction_fraction": 1 - k[part]["raw_fp32_bytes"] / f[part]["raw_fp32_bytes"],
        } for part in ("encoder_only", "complete_inference_model")}
        (out / "parameter_size_results.json").write_text(json.dumps(rep, indent=1) + "\n")
        print("wrote parameter_size_results.json"); return

    if args.stage == "flops":
        (out / "flops_results.json").write_text(json.dumps(flop_report(models, prep), indent=1) + "\n")
        print("wrote flops_results.json"); return

    if args.stage == "latency":
        raw, summ = latency_stage(device, prep, models)
        tag = "cpu" if device == "cpu" else "gpu"
        with open(out / f"latency_raw_{tag}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(raw[0].keys())); w.writeheader(); w.writerows(raw)
        with open(out / f"latency_summary_{tag}.csv", "w", newline="") as fh:
            cols = ["device", "dataset", "scope", "family", "batch", "warmup", "rounds", "n",
                    "mean_ms", "median_ms", "sd_ms", "p95_ms", "min_ms", "max_ms"]
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
            w.writerows([{c: r[c] for c in cols} for r in summ])
        print(f"wrote latency_raw_{tag}.csv and latency_summary_{tag}.csv"); return

    if args.stage == "throughput":
        rows = throughput_stage(prep, models, device)
        keys = sorted({k for r in rows for k in r})
        with open(out / "throughput_results.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
        print("wrote throughput_results.csv"); return


if __name__ == "__main__":
    main()
