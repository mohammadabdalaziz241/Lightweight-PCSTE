#!/usr/bin/env python
"""Q8 deployment measurements for `lightweight_pcste_q8_v1`.

Reuses `benchmarks/efficiency_v1/bench_common.py` so the deterministic
VALIDATION inputs, the model-only / end-to-end scopes and the host are
identical to the frozen FP32 benchmark. TEST is never touched here.

Stages: size | gpu_probe | agreement | latency | throughput | memory
"""
from __future__ import annotations

import argparse
import csv
import contextlib
import io
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
import bench_common as B  # noqa: E402

from src.methodology_v2.compression import quantization as Q  # noqa: E402
from src.methodology_v2.compression.student import build_encoder, half_4x1_spec  # noqa: E402
from src.methodology_v2.experiment.heads import DatasetHeads  # noqa: E402

DATASETS = B.DATASETS
CANON = B.CANON
REP_CELL = {"fold": 1, "seed": 42}          # same deterministic cell as efficiency_v1
CPU_WARMUP, CPU_TIMED, ROUNDS = 100, 1000, 4
THR_BATCH, THR_WARMUP, THR_TIMED = 32, 20, 100


def frozen_k1(fold=1, seed=42):
    with (HERE / "Q8_FROZEN_MODEL_TABLE.csv").open() as fh:
        row = next(r for r in csv.DictReader(fh)
                   if int(r["fold"]) == fold and int(r["seed"]) == seed)
    got = B.sha256_file(CANON / row["k1_checkpoint"])
    if got != row["k1_sha256"]:
        raise RuntimeError(f"K1 checkpoint hash mismatch for GF{fold}/S{seed}")
    ck = torch.load(CANON / row["k1_checkpoint"], map_location="cpu", weights_only=False)
    enc = build_encoder(half_4x1_spec("mean_of_remaining"), seed=0)
    enc.load_state_dict(ck["encoder"], strict=True)
    hd = DatasetHeads()
    hd.load_state_dict(ck["heads"], strict=True)
    return enc.eval(), hd.eval(), row


def save_bytes(obj) -> int:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getbuffer().nbytes


# ---------------------------------------------------------------------------
def stage_size():
    enc, hd, row = frozen_k1()
    q_enc, q_hd = Q.apply_q8_simulated(enc), Q.apply_q8_simulated(hd)
    qd_enc, info_e = Q.cpu_dynamic_quantize(enc)
    qd_hd, info_h = Q.cpu_dynamic_quantize(hd)

    fp32_state = {**{f"encoder.{k}": v.float() for k, v in enc.state_dict().items()},
                  **{f"heads.{k}": v.float() for k, v in hd.state_dict().items()}}
    q8_state = {**{f"encoder.{k}": v for k, v in q_enc.int8_state.items()},
                **{f"heads.{k}": v for k, v in q_hd.int8_state.items()}}
    packed_state = {**{f"encoder.{k}": v for k, v in qd_enc.state_dict().items()},
                    **{f"heads.{k}": v for k, v in qd_hd.state_dict().items()}}

    fp32_b = save_bytes(fp32_state)
    q8_b = save_bytes(q8_state)
    packed_b = save_bytes(packed_state)

    n8 = q_enc.n_int8_params + q_hd.n_int8_params
    n32 = q_enc.n_fp32_params + q_hd.n_fp32_params
    n_scales = sum(v.numel() for k, v in q8_state.items() if k.endswith(".scale"))
    theoretical = n8 * 1 + n32 * 4 + n_scales * 4      # int8 weights + fp32 rest + fp32 scales

    # Full-S1 reference from the frozen efficiency benchmark (not recomputed here)
    eff = json.loads((PUB / "benchmarks/efficiency_v1/parameter_size_results.json").read_text())
    full_s1_b = eff["per_model"]["Full-S1"]["complete_inference_model"]["serialized_state_dict_bytes"]
    k1_eff_b = eff["per_model"]["K1"]["complete_inference_model"]["serialized_state_dict_bytes"]

    out = {
        "representative_cell": REP_CELL,
        "selection_rule": "same deterministic cell as lightweight_pcste_efficiency_v1 (lowest fold, then lowest seed)",
        "k1_checkpoint_sha256": row["k1_sha256"],
        "method": "actual torch.save bytes of a state-dict-only payload; never an estimate",
        "parameter_count_note": "Q8 does NOT change the parameter count (1,379,813 for encoder + 4 heads). "
                                "Only the storage precision of allowlisted Linear weights changes.",
        "n_params_total": n8 + n32,
        "n_params_int8": n8, "n_params_fp32": n32,
        "fraction_params_int8": n8 / (n8 + n32),
        "n_scale_values": int(n_scales),
        "k1_fp32_state_dict_bytes": fp32_b,
        "k1_fp32_state_dict_bytes_efficiency_v1": k1_eff_b,
        "q8_compact_int8_state_bytes": q8_b,
        "q8_theoretical_weight_bytes": int(theoretical),
        "cpu_dynamic_packed_state_bytes": packed_b,
        "cpu_dynamic_engine": info_e["engine"],
        "cpu_dynamic_n_linears": info_e["n_dynamic_linears"] + info_h["n_dynamic_linears"],
        "full_s1_fp32_state_dict_bytes": full_s1_b,
        "reduction_q8_vs_k1_fp32": 1 - q8_b / fp32_b,
        "reduction_q8_vs_full_s1_fp32": 1 - q8_b / full_s1_b,
        "ratio_k1_fp32_over_q8": fp32_b / q8_b,
        "ratio_full_s1_over_q8": full_s1_b / q8_b,
        "separation_rule": "the packed torch.ao artifact carries runtime packing/observer overhead and is reported "
                           "SEPARATELY from the compact int8 state and from the theoretical weight bytes; "
                           "the three are never mixed into one number",
        "max_weight_abs_err": max(q_enc.max_weight_abs_err, q_hd.max_weight_abs_err),
    }
    (HERE / "results/q8_size_results.json").write_text(json.dumps(out, indent=1) + "\n")
    for k in ("k1_fp32_state_dict_bytes", "q8_compact_int8_state_bytes",
              "q8_theoretical_weight_bytes", "cpu_dynamic_packed_state_bytes",
              "full_s1_fp32_state_dict_bytes"):
        print(f"  {k:38s} {out[k]:>12,} B  ({out[k]/2**20:7.3f} MiB)")
    print(f"  reduction Q8 vs K1 FP32   {out['reduction_q8_vs_k1_fp32']*100:.2f}%  ({out['ratio_k1_fp32_over_q8']:.3f}x)")
    print(f"  reduction Q8 vs Full-S1   {out['reduction_q8_vs_full_s1_fp32']*100:.2f}%  ({out['ratio_full_s1_over_q8']:.3f}x)")


# ---------------------------------------------------------------------------
def stage_gpu_probe():
    """Establish empirically whether a true INT8 CUDA path exists."""
    enc, hd, _ = frozen_k1()
    out = {"question": "does the preserved Q8 implementation provide true hardware INT8 CUDA execution?"}
    qd_enc, info = Q.cpu_dynamic_quantize(enc)
    out["cpu_dynamic_engine"] = info["engine"]
    out["cpu_dynamic_backend_is_cpu_only"] = True
    try:
        qd_enc.to("cuda")
        prep = B.prepare_inputs("cuda")
        with torch.inference_mode():
            qd_enc(**prep["CWRU"]["batch"])
        out["cpu_dynamic_on_cuda"] = "unexpectedly succeeded"
        out["true_int8_cuda"] = "UNCERTAIN - investigate"
    except Exception as e:
        out["cpu_dynamic_on_cuda"] = f"{type(e).__name__}: {str(e)[:300]}"
        out["true_int8_cuda"] = False
    out["sim_representation_compute_dtype"] = "float32 (weights are dequantised before compute)"
    out["conclusion"] = (
        "NO true hardware INT8 CUDA execution. The 'sim' representation dequantises int8 weights to fp32 and "
        "computes in fp32; the 'cpu_dynamic' representation uses a CPU-only torch.ao backend. Any GPU timing of "
        "Q8 is therefore SIMULATED Q8 (fp32 arithmetic on quantised-then-dequantised weights) and must not be "
        "presented as INT8 GPU acceleration.")
    out["forbidden_alternatives_not_used"] = ["TensorRT", "torch.compile", "custom kernels", "new quantization library"]
    (HERE / "results/q8_gpu_probe.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


# ---------------------------------------------------------------------------
def stage_agreement():
    """sim vs cpu_dynamic prediction agreement on VALIDATION only (never TEST)."""
    import pandas as pd
    from src.pcste_v2.representation import ItemBuilder
    from src.pcste_v2.protocol_cwru_native12 import load_allowlist
    from src.pcste_v2 import representation as REPR
    from src.methodology_v2.encoder import collate_representations
    from src.methodology_v2.experiment.heads import CLASS_ORDERS
    REPR.NATIVE12_ALLOWLIST = load_allowlist(PUB / "protocols/datasets/cwru_native12_load_v1")

    enc, hd, _ = frozen_k1()
    sim_e, sim_h = Q.apply_q8_simulated(enc).model, Q.apply_q8_simulated(hd).model
    dyn_e, _ = Q.cpu_dynamic_quantize(enc)
    dyn_h, _ = Q.cpu_dynamic_quantize(hd)
    for m in (sim_e, sim_h, dyn_e, dyn_h):
        m.eval()

    pdir = CANON / "pcste_v2/protocol/global_v2_final_s0s1_v1"
    man = pd.read_csv(pdir / "global_v2_fold_1.csv", dtype={"fault_severity": str})
    val = man.loc[man["split"] == "validation"]
    assert set(val["split"]) == {"validation"}, "TEST GUARD"
    builder = ItemBuilder(pdir, 1, ("v1",), {}, False)
    idx = man.set_index("window_id", drop=False)

    rows, per_ds = [], {}
    with torch.inference_mode():
        for ds in DATASETS:
            ids = list(val.loc[val["dataset"] == ds, "window_id"])[:200]   # frozen manifest order
            agree, n, maxdiff = 0, 0, 0.0
            for lo in range(0, len(ids), 64):
                chunk = ids[lo:lo + 64]
                items = [builder.build(idx.loc[w]) for w in chunk]
                batch = collate_representations([it["streams"]["g0_c0"] for it in items])
                ls = sim_h(sim_e(**batch)["global_embedding"], ds)
                ld = dyn_h(dyn_e(**batch)["global_embedding"], ds)
                ps, pd_ = torch.softmax(ls, -1), torch.softmax(ld, -1)
                agree += int((ls.argmax(-1) == ld.argmax(-1)).sum())
                maxdiff = max(maxdiff, float((ps - pd_).abs().max()))
                n += len(chunk)
            per_ds[ds] = {"n_validation_windows": n, "top1_agreement": agree / n,
                          "max_abs_prob_diff": maxdiff}
            rows.append({"dataset": ds, **per_ds[ds]})
            print(f"  {ds:9s} n={n:4d} top1 agreement {agree/n:.4f}  max |dprob| {maxdiff:.5f}")
    out = {"scope": "VALIDATION only - TEST is never used for this check",
           "fold": 1, "seed": 42, "selection": "first up to 200 validation windows per dataset in frozen manifest order",
           "purpose": "bound the numerics gap between the evaluated accuracy representation (sim) and the "
                      "CPU deployment representation (cpu_dynamic), without a second TEST touch",
           "per_dataset": per_ds,
           "overall_top1_agreement": float(np.mean([v["top1_agreement"] for v in per_ds.values()]))}
    (HERE / "results/q8_sim_vs_cpu_dynamic_agreement.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"  overall equal-domain top-1 agreement {out['overall_top1_agreement']:.4f}")


# ---------------------------------------------------------------------------
def _summarise(ts):
    ms = [t * 1e3 for t in ts]
    return {"n": len(ms), "mean_ms": statistics.fmean(ms), "median_ms": statistics.median(ms),
            "sd_ms": statistics.stdev(ms), "p95_ms": float(np.percentile(ms, 95)),
            "min_ms": min(ms), "max_ms": max(ms)}


def stage_latency(inference_mode: bool = True):
    """Primary run uses torch.inference_mode() — the deployment-correct setting and
    the mode BOTH plans state. A secondary run without it is retained for
    comparability with the frozen efficiency_v1 latency numbers, whose
    implementation omitted inference_mode (see Q8_FINAL_REPORT.md)."""
    torch.set_num_threads(1)
    enc, hd, _ = frozen_k1()
    dyn_e, info = Q.cpu_dynamic_quantize(enc)
    dyn_h, _ = Q.cpu_dynamic_quantize(hd)
    for m in (enc, hd, dyn_e, dyn_h):
        m.eval()
    models = {"K1-FP32": (enc, hd), "Q8-cpu_dynamic": (dyn_e, dyn_h)}
    prep = B.prepare_inputs("cpu")
    fams = tuple(models)
    per_round = CPU_TIMED // ROUNDS

    raw, summ = [], []
    for ds in DATASETS:
        for scope in ("model_only", "end_to_end"):
            fns = {}
            for fam in fams:
                e, h = models[fam]
                fns[fam] = (B.model_only_fn(ds, prep, e, h) if scope == "model_only"
                            else B.end_to_end_fn(ds, prep, e, h, "cpu"))
            ctx = torch.inference_mode if inference_mode else contextlib.nullcontext
            with ctx():
                for fam in fams:
                    for _ in range(CPU_WARMUP):
                        fns[fam]()
                samples = {f: [] for f in fams}
                for rnd in range(ROUNDS):
                    order = fams if rnd % 2 == 0 else tuple(reversed(fams))   # ABAB
                    for fam in order:
                        for i in range(per_round):
                            t0 = time.perf_counter()
                            fns[fam]()
                            dt = time.perf_counter() - t0
                            samples[fam].append(dt)
                            raw.append({"device": "cpu", "dataset": ds, "scope": scope,
                                        "family": fam, "inference_mode": inference_mode,
                                        "round": rnd, "iteration": i, "latency_ms": dt * 1e3})
            for fam in fams:
                s = _summarise(samples[fam])
                s.update({"device": "cpu", "dataset": ds, "scope": scope, "family": fam,
                          "batch": 1, "threads": 1, "warmup": CPU_WARMUP, "rounds": ROUNDS,
                          "inference_mode": inference_mode})
                summ.append(s)
            a = next(r for r in summ if r["dataset"] == ds and r["scope"] == scope and r["family"] == "K1-FP32")
            b = next(r for r in summ if r["dataset"] == ds and r["scope"] == scope and r["family"] == "Q8-cpu_dynamic")
            print(f"  cpu {ds:9s} {scope:11s} K1 {a['median_ms']:8.3f} | Q8 {b['median_ms']:8.3f} | "
                  f"speedup {a['median_ms']/b['median_ms']:.3f}x", flush=True)

    suffix = "" if inference_mode else "_no_inference_mode"
    with (HERE / f"results/q8_latency_raw{suffix}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(raw[0].keys())); w.writeheader(); w.writerows(raw)
    cols = ["device", "dataset", "scope", "family", "inference_mode", "batch", "threads",
            "warmup", "rounds", "n", "mean_ms", "median_ms", "sd_ms", "p95_ms", "min_ms", "max_ms"]
    with (HERE / f"results/q8_latency_summary{suffix}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        w.writerows([{c: r[c] for c in cols} for r in summ])
    print(f"  engine={info['engine']}  wrote q8_latency_raw.csv / q8_latency_summary.csv")


def stage_throughput():
    torch.set_num_threads(1)
    enc, hd, _ = frozen_k1()
    dyn_e, _ = Q.cpu_dynamic_quantize(enc)
    dyn_h, _ = Q.cpu_dynamic_quantize(hd)
    for m in (enc, hd, dyn_e, dyn_h):
        m.eval()
    models = {"K1-FP32": (enc, hd), "Q8-cpu_dynamic": (dyn_e, dyn_h)}
    prep = B.prepare_inputs("cpu")
    rows = []
    for ds in DATASETS:
        b1 = prep[ds]["batch"]
        big = {k: v.repeat(THR_BATCH, *([1] * (v.dim() - 1))) for k, v in b1.items()}
        for fam, (e, h) in models.items():
            def fn(e=e, h=h):
                with torch.inference_mode():
                    return h(e(**big)["global_embedding"], ds)
            for _ in range(THR_WARMUP):
                fn()
            ts = []
            for _ in range(THR_TIMED):
                t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
            med = statistics.median(ts)
            rows.append({"device": "cpu", "dataset": ds, "family": fam, "batch": THR_BATCH,
                         "status": "ok", "median_batch_ms": med * 1e3,
                         "windows_per_second": THR_BATCH / med,
                         "per_window_ms": med * 1e3 / THR_BATCH, "n": len(ts)})
            print(f"  cpu b{THR_BATCH} {ds:9s} {fam:16s} {THR_BATCH/med:8.2f} win/s", flush=True)
    with (HERE / "results/q8_throughput_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def stage_memory():
    """CPU resident model bytes: a clear, interpretable quantity."""
    enc, hd, _ = frozen_k1()
    dyn_e, _ = Q.cpu_dynamic_quantize(enc)
    dyn_h, _ = Q.cpu_dynamic_quantize(hd)
    q_enc, q_hd = Q.apply_q8_simulated(enc), Q.apply_q8_simulated(hd)

    def tensor_bytes(m):
        return sum(p.numel() * p.element_size() for p in m.parameters()) + \
               sum(b.numel() * b.element_size() for b in m.buffers())
    out = {
        "scope": "CPU model resident parameter+buffer bytes (a directly interpretable quantity)",
        "k1_fp32_bytes": tensor_bytes(enc) + tensor_bytes(hd),
        "q8_sim_bytes_note": "the sim representation holds DEQUANTISED fp32 weights in memory, so its runtime "
                             "footprint equals FP32 by construction; it is a numerical stand-in, not a compact runtime",
        "q8_sim_runtime_bytes": tensor_bytes(q_enc.model) + tensor_bytes(q_hd.model),
        "q8_cpu_dynamic_packed_state_bytes": save_bytes(
            {**{f"e.{k}": v for k, v in dyn_e.state_dict().items()},
             **{f"h.{k}": v for k, v in dyn_h.state_dict().items()}}),
        "gpu_note": "No GPU memory figure is reported for Q8: there is no true INT8 CUDA path, so a GPU "
                    "measurement would reflect fp32/dequantised runtime memory and would NOT reflect the stored "
                    "INT8 artifact size. Reporting it as a Q8 memory saving would be misleading.",
        "headline_eligible": False,
        "headline_rationale": "CPU packed-state bytes mix weights with torch.ao packing metadata; parameter-tensor "
                              "bytes for the sim representation are fp32 by construction. Neither is a clean "
                              "'Q8 runtime memory' number, so memory is reported as descriptive only.",
    }
    (HERE / "results/q8_memory_results.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("size", "gpu_probe", "agreement", "latency", "throughput", "memory"))
    ap.add_argument("--no-inference-mode", action="store_true",
                    help="secondary run matching the frozen efficiency_v1 implementation")
    a = ap.parse_args()
    (HERE / "results").mkdir(exist_ok=True)
    if a.stage == "latency":
        stage_latency(inference_mode=not a.no_inference_mode)
        return
    {"size": stage_size, "gpu_probe": stage_gpu_probe, "agreement": stage_agreement,
     "throughput": stage_throughput, "memory": stage_memory}[a.stage]()


if __name__ == "__main__":
    main()
