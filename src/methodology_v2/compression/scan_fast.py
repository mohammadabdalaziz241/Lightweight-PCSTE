"""Optional faster selective-scan backend for Part-6 processes ONLY.

The primary chain keeps `src.methodology_v2.encoder.ssm.selective_scan`
(the parity-verified reference loop). This module provides:

* `selective_scan_chunked` — the SAME recurrence
      h_t = exp(delta_t A) h_{t-1} + (delta_t B_t) x_t,  y_t = C_t h_t + D x_t
  evaluated chunk-wise: an intra-chunk loop of `chunk` steps batched over
  ALL chunks in parallel, an inter-chunk carry of T/chunk steps, and the
  exact algebraic recombination h[i,k] = Q[i,k] * h_in[i] + h_local[i,k]
  (Q = running product of the chunk's decays). Sequential python-level
  steps fall from T to chunk + T/chunk (24 -> ~11 at chunk 8).
  Numerical stabilisation: decays are always formed as products of the
  per-step factors exp(delta A) in (0, 1] — never as exp(L_t - L_s) of
  cumulative log-sums, and never via divisions of cumulative products —
  so unbounded softplus(delta) can under-flow a decay to 0 exactly as the
  reference does, but can never overflow or cancel. (The naive log-space
  form was considered and rejected on precision grounds; disclosed.)
* optional torch.compile wrappers (opt-in; failure is disclosed, never
  silently ignored).
* the parity harness: max |dy| vs the vendored official reference on
  synthetic + worst-case-delta cases (unit level now; the 1,000-window
  real-data gate and 100 % validation-prediction agreement are executed
  later by the Part-6 CLI `scan-parity`, and the backend is usable in a
  registered run only if the sealed approval file says all_pass=true).
* `use_scan_backend()` — an explicit context manager that swaps the
  module-level function the frozen MambaRefBlock calls; default OFF.
"""
from __future__ import annotations

import contextlib
import json
import math
import time
from pathlib import Path

import torch

from ..encoder import ssm as _ssm
from ..encoder.third_party.official_selective_scan_ref import selective_scan_ref
from .guards import Part6GuardError
from .protocol import PART6_DIR

PARITY_MAX_ABS = 1e-4
DEFAULT_CHUNK = 8
APPROVAL_FILE = PART6_DIR / "scan_backend_parity.json"


def selective_scan_reference(x, delta, a_mat, b, c, d_skip):
    return _ssm.selective_scan(x, delta, a_mat, b, c, d_skip)


def selective_scan_chunked(x: torch.Tensor, delta: torch.Tensor,
                           a_mat: torch.Tensor, b: torch.Tensor,
                           c: torch.Tensor, d_skip: torch.Tensor,
                           chunk: int = DEFAULT_CHUNK) -> torch.Tensor:
    """Same signature/semantics as ssm.selective_scan.
    x, delta: (N, T, d); a_mat: (d, s); b, c: (N, T, s); d_skip: (d,)."""
    n, t, d = x.shape
    s = a_mat.shape[1]
    chunk = max(1, min(int(chunk), t))
    nc = math.ceil(t / chunk)
    tp = nc * chunk
    if tp != t:                                    # zero-pad: identity steps
        pad = tp - t
        x_p = torch.nn.functional.pad(x, (0, 0, 0, pad))
        delta_p = torch.nn.functional.pad(delta, (0, 0, 0, pad))
        b_p = torch.nn.functional.pad(b, (0, 0, 0, pad))
    else:
        x_p, delta_p, b_p = x, delta, b
    da = torch.exp(delta_p.unsqueeze(-1) * a_mat)              # (N,tp,d,s)
    dbx = (delta_p.unsqueeze(-1) * b_p.unsqueeze(2)) * x_p.unsqueeze(-1)
    da = da.reshape(n, nc, chunk, d, s)
    dbx = dbx.reshape(n, nc, chunk, d, s)

    # intra-chunk local scan (zero initial state) + running decay products
    h_loc = torch.zeros(n, nc, d, s, dtype=x.dtype, device=x.device)
    q = torch.ones(n, nc, d, s, dtype=x.dtype, device=x.device)
    locs, qs = [], []
    for k in range(chunk):
        h_loc = da[:, :, k] * h_loc + dbx[:, :, k]
        q = da[:, :, k] * q
        locs.append(h_loc)
        qs.append(q)
    h_local = torch.stack(locs, dim=2)              # (N,nc,chunk,d,s)
    q_all = torch.stack(qs, dim=2)                  # prefix products

    # inter-chunk carry (sequential over chunks)
    h_in = [torch.zeros(n, d, s, dtype=x.dtype, device=x.device)]
    for i in range(nc - 1):
        h_in.append(q_all[:, i, -1] * h_in[-1] + h_local[:, i, -1])
    h_in = torch.stack(h_in, dim=1)                 # (N,nc,d,s)

    h = h_local + q_all * h_in.unsqueeze(2)         # exact recombination
    h = h.reshape(n, tp, d, s)[:, :t]
    y = torch.einsum("ntds,nts->ntd", h, c)
    return y + d_skip * x


def maybe_compile(fn, mode: str | None = None, probe_device: str = "cpu"):
    """torch.compile if it actually works on this host (a small probe
    call is executed because inductor errors surface lazily); returns
    (callable, note). Failure falls back to eager WITH a disclosed note —
    never silently."""
    try:
        compiled = torch.compile(fn, mode=mode) if mode else torch.compile(fn)
        case = synthetic_case(2, 5, 4, 3, seed=0, device=probe_device)
        args = [case[k] for k in ("x", "delta", "a_mat", "b", "c", "d_skip")]
        y = compiled(*args)
        ref = selective_scan_reference(*args)
        if not torch.isfinite(y).all() or float((y - ref).abs().max()) > 1e-4:
            return fn, "torch.compile probe mismatch -> eager"
        return compiled, "compiled"
    except Exception as e:
        return fn, f"torch.compile unavailable -> eager ({type(e).__name__})"


@contextlib.contextmanager
def use_scan_backend(name: str = "reference", chunk: int = DEFAULT_CHUNK,
                     compile_mode: str | None = None,
                     require_approval: bool = True):
    """Temporarily route MambaRefBlock through a Part-6 scan backend.
    'reference' is a no-op. Non-reference backends require the sealed
    approval file (all_pass=true) unless require_approval=False (tests,
    parity harness itself)."""
    if name == "reference":
        yield "reference"
        return
    if name not in ("chunked", "compiled_reference"):
        raise Part6GuardError(f"unknown scan backend {name}")
    if require_approval and not backend_approved(name):
        raise Part6GuardError(
            f"scan backend {name!r} is NOT approved: run the parity gate and "
            "seal scan_backend_parity.json (all_pass + human_approved) first")
    if name == "chunked":
        fn = lambda x, delta, a, b, c, d: selective_scan_chunked(  # noqa: E731
            x, delta, a, b, c, d, chunk=chunk)
        note = "chunked"
        if compile_mode is not None:
            fn, note = maybe_compile(fn, compile_mode)
            note = f"chunked+{note}"
    else:
        original_ref = _ssm.selective_scan
        fn, note = maybe_compile(original_ref, compile_mode)
        note = f"reference+{note}"
    original = _ssm.selective_scan
    _ssm.selective_scan = fn
    try:
        yield note
    finally:
        _ssm.selective_scan = original


def backend_approved(name: str = "chunked") -> bool:
    """A backend is usable in a registered run only if the sealed parity
    record says all_pass for THAT backend and a human approved it."""
    if not APPROVAL_FILE.exists():
        return False
    rec = json.loads(APPROVAL_FILE.read_text())
    entry = rec.get("backends", {}).get(name, {})
    return bool(entry.get("all_pass")) and bool(rec.get("human_approved"))


# ---------------------------------------------------------------------------
# parity harness
# ---------------------------------------------------------------------------
def _ref_official(x, delta, a_mat, b, c, d_skip):
    """Vendored official selective_scan_ref expects (B, D, L) layouts."""
    u = x.transpose(1, 2).contiguous()                       # (N,d,T)
    dl = delta.transpose(1, 2).contiguous()
    bb = b.transpose(1, 2).contiguous()                      # (N,s,T)
    cc = c.transpose(1, 2).contiguous()
    y = selective_scan_ref(u, dl, a_mat, bb, cc, D=d_skip, z=None,
                           delta_bias=None, delta_softplus=False,
                           return_last_state=False)
    return y.transpose(1, 2)


def synthetic_case(n: int, t: int, d: int, s: int, seed: int,
                   delta_scale: float = 1.0, delta_mode: str = "softplus",
                   dtype=torch.float32, device="cpu") -> dict:
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, t, d, generator=g)
    raw = torch.randn(n, t, d, generator=g) * delta_scale
    if delta_mode == "softplus":
        delta = torch.nn.functional.softplus(raw)
    elif delta_mode == "huge":                # worst case: unbounded softplus
        delta = torch.nn.functional.softplus(raw.abs() + 50.0) * delta_scale
    elif delta_mode == "tiny":
        delta = torch.full((n, t, d), 1e-6)
    else:
        raise ValueError(delta_mode)
    a_mat = -torch.exp(torch.log(torch.arange(1, s + 1, dtype=torch.float32))
                       .repeat(d, 1) + 0.1 * torch.randn(d, s, generator=g))
    b = torch.randn(n, t, s, generator=g)
    c = torch.randn(n, t, s, generator=g)
    d_skip = torch.randn(d, generator=g)
    tens = {"x": x, "delta": delta, "a_mat": a_mat, "b": b, "c": c,
            "d_skip": d_skip}
    return {k: v.to(dtype=dtype, device=device) for k, v in tens.items()}


def parity_case(case: dict, chunk: int = DEFAULT_CHUNK,
                with_grad: bool = True) -> dict:
    """Forward (and gradient) parity of chunked vs reference vs official."""
    args = [case[k] for k in ("x", "delta", "a_mat", "b", "c", "d_skip")]
    if with_grad:
        args_ref = [a.clone().requires_grad_(True) for a in args]
        args_ch = [a.clone().requires_grad_(True) for a in args]
    else:
        args_ref, args_ch = args, args
    y_ref = selective_scan_reference(*args_ref)
    y_ch = selective_scan_chunked(*args_ch, chunk=chunk)
    with torch.no_grad():
        y_off = _ref_official(*[a.detach() for a in args_ref])
    out = {"max_abs_chunked_vs_reference": float((y_ch - y_ref).abs().max()),
           "max_abs_chunked_vs_official": float((y_ch - y_off).abs().max()),
           "max_abs_reference_vs_official": float((y_ref - y_off).abs().max()),
           "finite": bool(torch.isfinite(y_ch).all())}
    if with_grad:
        w = torch.randn_like(y_ref)
        (y_ref * w).sum().backward()
        (y_ch * w).sum().backward()
        gm = 0.0
        for a, bb in zip(args_ref, args_ch):
            if a.grad is not None:
                diff = (a.grad - bb.grad).abs().max()
                gm = max(gm, float(diff))
        out["max_abs_grad_diff"] = gm
    out["pass"] = (out["finite"]
                   and out["max_abs_chunked_vs_reference"] < PARITY_MAX_ABS
                   and out["max_abs_chunked_vs_official"] < PARITY_MAX_ABS)
    return out


def synthetic_parity_suite(chunk: int = DEFAULT_CHUNK, device="cpu") -> dict:
    """Unit-level suite incl. worst-case delta; real-window gate is a CLI
    step (needs finished checkpoints and the sealed reader)."""
    cases = [
        ("typical_T24", dict(n=6, t=24, d=32, s=16, seed=1)),
        ("typical_T23", dict(n=6, t=23, d=32, s=16, seed=2)),
        ("T_not_multiple_of_chunk", dict(n=4, t=13, d=8, s=4, seed=3)),
        ("T_lt_chunk", dict(n=4, t=5, d=8, s=4, seed=4)),
        ("huge_delta_worst_case", dict(n=4, t=24, d=16, s=16, seed=5,
                                       delta_mode="huge", delta_scale=10.0)),
        ("tiny_delta", dict(n=4, t=24, d=16, s=16, seed=6, delta_mode="tiny")),
        ("large_scale_inputs", dict(n=4, t=24, d=16, s=16, seed=7,
                                    delta_scale=5.0)),
    ]
    results = {}
    for name, kw in cases:
        case = synthetic_case(device=device, **kw)
        results[name] = parity_case(case, chunk=chunk)
    return {"chunk": chunk, "threshold_max_abs": PARITY_MAX_ABS,
            "cases": results,
            "all_pass": all(r["pass"] for r in results.values())}


def time_backends(n: int, t: int, d: int, s: int, reps: int = 5,
                  chunk: int = DEFAULT_CHUNK, device="cpu") -> dict:
    case = synthetic_case(n, t, d, s, seed=0, device=device)
    args = [case[k] for k in ("x", "delta", "a_mat", "b", "c", "d_skip")]

    def timeit(fn):
        fn(*args)
        if device != "cpu":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn(*args)
        if device != "cpu":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps

    return {"reference_s": timeit(selective_scan_reference),
            "chunked_s": timeit(lambda *a: selective_scan_chunked(*a, chunk=chunk)),
            "shape": {"n": n, "t": t, "d": d, "s": s}, "device": device}
