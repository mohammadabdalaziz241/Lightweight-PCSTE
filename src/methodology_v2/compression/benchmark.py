"""Four-axis lightweight measurement harness (size / compute / latency /
memory) — always reported together, never collapsed into one number.

Size    : exact numel (encoder, heads, total); measured torch.save bytes
          of the fp32 state and of the real int8 state (quantization.py).
Compute : FlopCounterMode GFLOP per dataset input shape (+ analytic scan
          estimate), sequential scan steps per forward, recurrent work
          R = layers x dirs x T x d_inner x d_state.
Latency : CPU fp32 b1/b16 at 1 and 4 threads (5 warm-up, 20 timed,
          median + IQR), ABAB-interleaved against a baseline model in the
          same session (noise floor = baseline re-measured); GPU b16/b64
          with cuda.synchronize when idle. Every table carries the
          backend caveat (pure-PyTorch reference scan).
Memory  : peak RSS (VmHWM) of a fresh subprocess doing warm-up + one
          measured forward — the most reliable per-forward peak on this
          stack; torch.cuda.max_memory_allocated on GPU.
Energy  : NOT measured (RAPL root-only); CPU-seconds reported as a
          flagged proxy.

The harness refuses to run latency while the host is busy unless
`allow_busy=True` is passed explicitly (and then tags the result).
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ..encoder import collate_representations
from ..encoder.ssm import BiMambaLayer
from .quantization import apply_q8_simulated, serialized_bytes
from .student import count_params, scan_steps_per_forward, recurrent_work_R

DATASET_SHAPES = {"CWRU": (513, 184), "JNU": (513, 192), "HIT": (257, 192),
                  "MAFAULDA": (513, 192)}
BAND_SEQ_LEN = {"CWRU": 23, "JNU": 24, "HIT": 24, "MAFAULDA": 24}
BACKEND_CAVEAT = ("pure-PyTorch reference selective scan (fused kernels "
                  "unavailable on this stack); latency reflects this backend")


def synthetic_batch(dataset: str, batch: int, device="cpu") -> dict:
    """Shape-faithful synthetic input (values irrelevant for cost)."""
    bins, frames = DATASET_SHAPES[dataset]
    g = torch.Generator().manual_seed(0)
    x = torch.randn(batch, bins, frames, generator=g)
    rate = {"CWRU": 48000, "JNU": 50000, "HIT": 25000, "MAFAULDA": 50000}[dataset]
    n_fft = 512 if dataset == "HIT" else 1024
    hop = n_fft // 4
    freq = torch.arange(bins) * rate / n_fft
    t = torch.arange(frames) * hop / rate
    items = [(x[i].numpy(), freq.numpy(), t.numpy()) for i in range(batch)]
    b = collate_representations(items)
    return {k: v.to(device) for k, v in b.items()}


# ---------------------------------------------------------------------------
# size
# ---------------------------------------------------------------------------
def size_axis(encoder, heads) -> dict:
    enc_p, head_p = count_params(encoder), count_params(heads)
    full_state = {**{f"encoder.{k}": v for k, v in encoder.state_dict().items()},
                  **{f"heads.{k}": v for k, v in heads.state_dict().items()}}
    fp32_bytes = serialized_bytes(full_state)
    q8_enc = apply_q8_simulated(encoder)
    q8_heads = apply_q8_simulated(heads)
    int8_bytes = serialized_bytes(
        {**{f"encoder.{k}": v for k, v in q8_enc.int8_state.items()},
         **{f"heads.{k}": v for k, v in q8_heads.int8_state.items()}})
    return {"encoder_params": enc_p, "head_params": head_p,
            "total_params": enc_p + head_p,
            "fp32_state_bytes": fp32_bytes, "fp32_state_mb": fp32_bytes / 2**20,
            "int8_q8_state_bytes": int8_bytes, "int8_q8_state_mb": int8_bytes / 2**20,
            "int8_fraction_of_params": (q8_enc.n_int8_params + q8_heads.n_int8_params)
            / (enc_p + head_p)}


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------
def analytic_scan_flops(encoder, dataset: str, batch: int = 1) -> float:
    """Elementwise + einsum work of the scan per forward (not counted by
    FlopCounter): per step ~ 6 flops per (d_inner x d_state) element
    (exp, 2 mul, add, mul-add for C·h) x bands x steps."""
    layer0 = encoder.temporal.layers[0]
    blk = layer0.fwd
    n_bands = 33 if dataset != "HIT" else 17
    steps = scan_steps_per_forward(encoder, BAND_SEQ_LEN[dataset])
    return float(batch * n_bands * steps * blk.d_inner * blk.d_state * 6)


def compute_axis(encoder, heads, datasets=("CWRU", "JNU", "HIT", "MAFAULDA"),
                 batch: int = 1) -> dict:
    from torch.utils.flop_counter import FlopCounterMode
    encoder.eval()
    heads.eval()
    out = {"backend_caveat": BACKEND_CAVEAT, "per_dataset": {}}
    for ds in datasets:
        b = synthetic_batch(ds, batch)
        with torch.no_grad():
            fc = FlopCounterMode(display=False)
            with fc:
                z = encoder(**b)["global_embedding"]
                heads(z, ds)
        flops = fc.get_total_flops()
        t = BAND_SEQ_LEN[ds]
        out["per_dataset"][ds] = {
            "flopcounter_gflop_per_window": flops / batch / 1e9,
            "analytic_scan_gflop_per_window": analytic_scan_flops(encoder, ds) / 1e9,
            "scan_steps_per_forward": scan_steps_per_forward(encoder, t),
            "R_per_band": recurrent_work_R(encoder, t),
            "band_seq_len": t}
    return out


# ---------------------------------------------------------------------------
# latency
# ---------------------------------------------------------------------------
def host_busy(threshold_load: float = 1.5, gpu_util_threshold: int = 20) -> dict:
    load1 = os.getloadavg()[0]
    gpu_util = None
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        gpu_util = int(r.stdout.strip().split("\n")[0])
    except Exception:
        pass
    busy = load1 > threshold_load or (gpu_util is not None
                                      and gpu_util > gpu_util_threshold)
    return {"load1": load1, "gpu_util": gpu_util, "busy": busy}


def _time_forward(fn, warmup: int, timed: int, sync: bool) -> list[float]:
    for _ in range(warmup):
        fn()
    if sync:
        torch.cuda.synchronize()
    ts = []
    for _ in range(timed):
        t0 = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return ts


def _summ(ts: list[float]) -> dict:
    q1, med, q3 = np.percentile(ts, [25, 50, 75])
    return {"median_ms": med * 1e3, "iqr_ms": (q3 - q1) * 1e3,
            "min_ms": min(ts) * 1e3, "n": len(ts)}


@torch.no_grad()
def latency_axis(models: dict, dataset: str = "JNU", device: str = "cpu",
                 batches=(1, 16), threads=(1, 4), warmup: int = 5,
                 timed: int = 20, allow_busy: bool = False,
                 abab_rounds: int = 2) -> dict:
    """models: {'baseline': (encoder, heads), 'candidate': (...), ...}.
    ABAB interleaving: for each (batch, threads) setting, rounds alternate
    over the models; per model the timed samples are pooled. The baseline
    is measured in every round, so its spread is the noise floor."""
    hb = host_busy()
    if hb["busy"] and not allow_busy:
        raise RuntimeError(f"host busy ({hb}); refuse latency measurement")
    sync = device.startswith("cuda")
    out = {"device": device, "dataset": dataset, "host_state": hb,
           "backend_caveat": BACKEND_CAVEAT, "settings": {}}
    for enc, hd in models.values():
        enc.to(device).eval()
        hd.to(device).eval()
    thread_list = threads if device == "cpu" else (None,)
    for bsz in batches:
        batch = synthetic_batch(dataset, bsz, device)
        for th in thread_list:
            if th is not None:
                torch.set_num_threads(th)
            key = f"b{bsz}_t{th}" if th else f"b{bsz}"
            samples = {name: [] for name in models}
            for _ in range(abab_rounds):
                for name, (enc, hd) in models.items():
                    def fn(enc=enc, hd=hd):
                        z = enc(**batch)["global_embedding"]
                        return hd(z, dataset)
                    samples[name].extend(_time_forward(fn, warmup, timed, sync))
            out["settings"][key] = {name: _summ(v) for name, v in samples.items()}
    return out


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------
_MEM_SCRIPT = r"""
import sys, json, torch
sys.path.insert(0, {repo!r})
torch.set_num_threads({threads})
from src.methodology_v2.compression.benchmark import synthetic_batch
from src.methodology_v2.compression.student import build_encoder, build_heads, StudentSpec
import dataclasses
_f = {{x.name for x in dataclasses.fields(StudentSpec)}}
spec = StudentSpec(**{{k: v for k, v in json.loads({spec_json!r}).items() if k in _f}})
enc = build_encoder(spec, seed=0).eval(); hd = build_heads(spec, 0).eval()
b = synthetic_batch({dataset!r}, {batch})
def hwm():
    for line in open('/proc/self/status'):
        if line.startswith('VmHWM'):
            return int(line.split()[1]) * 1024
base = hwm()
with torch.no_grad():
    for _ in range(3):
        hd(enc(**b)['global_embedding'], {dataset!r})
print(json.dumps({{'vmhwm_bytes': hwm(), 'baseline_bytes': base}}))
"""


def memory_axis(spec_dict: dict, dataset: str = "JNU", batch: int = 1,
                threads: int = 1, repo: Path | None = None) -> dict:
    """Peak RSS (VmHWM) of a fresh subprocess: model construction + 3
    forwards. Baseline = RSS after imports/model build, before forward."""
    from ..registry import REPO_ROOT
    repo = str(repo or REPO_ROOT)
    code = _MEM_SCRIPT.format(repo=repo, threads=threads,
                              spec_json=json.dumps(spec_dict),
                              dataset=dataset, batch=batch)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, cwd=repo)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    d = json.loads(r.stdout.strip().split("\n")[-1])
    d.update({"dataset": dataset, "batch": batch, "threads": threads,
              "vmhwm_mb": d["vmhwm_bytes"] / 2**20,
              "forward_increment_mb": (d["vmhwm_bytes"] - d["baseline_bytes"]) / 2**20})
    return d
