"""Q8 post-training quantization (registered recipe) + exploratory scaffolds.

Registered recipe Q8 (Stage 1; TEST-eligible after the sealed session):
  * per-output-channel SYMMETRIC int8 weights on every nn.Linear in the
    ALLOWLIST: stem.proj, coords.proj, temporal.layers.*.{fwd,bwd}.
    {in_proj, x_proj, out_proj}, mixer.score.0, mixer.score.2,
    mixer.value, mixer.gate, mixer.context, heads.heads.*
  * DENYLIST (stays fp32, asserted): *.dt_proj (an nn.Linear!), A_log, D,
    conv1d, every LayerNorm, the recurrent state / exp(delta*A) path
    (which is not a parameterised module and is therefore untouched by
    weight quantization by construction).
  * Two deployment representations:
      - "sim"  : int8 weight-only, dequantised to fp32 for compute
                 (identical numerics on CPU and GPU — the accuracy
                 representation);
      - "cpu_dynamic": torch.ao dynamic quantization (int8 weights per
                 channel + dynamic int8 activations, fbgemm/x86) applied
                 ONLY to allowlisted modules via a qualified-name qconfig
                 dict (the CPU deployment representation).
  * measured serialized bytes of (a) fp32 state_dict, (b) the compact
    int8 state_dict (int8 tensors + fp32 per-channel scales + untouched
    fp32 tensors), (c) the torch.ao dynamic-quantized state_dict.
  Latency gains are NOT claimed for Q8 (plan §7).

Exploratory (VAL-only, never TEST, never registered): fp16-all-but-
sensitive, W4 per-group weight-only, static W8A8 with TRAIN-window
calibration, leave-one-tensor sensitivity map — scaffolds only.
"""
from __future__ import annotations

import copy
import io
import re
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .guards import Part6GuardError

ALLOW_PATTERNS = (
    r"^stem\.proj$",
    r"^stem\.proj_a$", r"^stem\.proj_b$",                  # DW low-rank stem
    r"^coords\.proj$",
    r"^temporal\.layers\.\d+\.(fwd|bwd)\.(in_proj|x_proj|out_proj)$",
    r"^mixer\.score\.[02]$",
    r"^mixer\.(value|gate|context)$",
    r"^(heads\.)?heads\.[A-Z]+$",           # DatasetHeads alone or wrapped
)
DENY_PATTERNS = (
    r"\.dt_proj$", r"\.A_log$", r"\.D$", r"\.conv1d$", r"norm$",
    r"^temporal\.norm$", r"^mixer\.norm$",
)
FP32_TENSOR_PATTERNS = (r"\.dt_proj\.", r"\.A_log$", r"\.D$", r"\.conv1d\.",
                        r"norm\.(weight|bias)$")


def is_allowlisted(qualified_name: str) -> bool:
    return any(re.search(p, qualified_name) for p in ALLOW_PATTERNS)


def is_denylisted(qualified_name: str) -> bool:
    return any(re.search(p, qualified_name) for p in DENY_PATTERNS)


def q8_module_plan(model: nn.Module) -> dict:
    """{qualified_name: 'int8'|'fp32'} for every leaf module + assertions
    that no denylisted module is quantized and every allowlisted Linear is."""
    plan = {}
    for name, mod in model.named_modules():
        if len(list(mod.children())) > 0:
            continue
        if isinstance(mod, nn.Linear) and is_allowlisted(name):
            if is_denylisted(name):
                raise Part6GuardError(f"{name} both allow- and denylisted")
            plan[name] = "int8"
        else:
            plan[name] = "fp32"
    for name, kind in plan.items():
        if kind == "int8" and is_denylisted(name):     # pragma: no cover
            raise Part6GuardError(f"denylisted module {name} would be quantized")
    return plan


# ---------------------------------------------------------------------------
# per-output-channel symmetric int8
# ---------------------------------------------------------------------------
def quantize_weight_per_channel(w: torch.Tensor, bits: int = 8
                                ) -> tuple[torch.Tensor, torch.Tensor]:
    """w (out, in) -> (q int8 (out,in), scale fp32 (out,)). Symmetric,
    zero-point 0, scale = max|w_row| / qmax."""
    qmax = 2 ** (bits - 1) - 1
    amax = w.detach().abs().amax(dim=1).clamp_min(1e-12)
    scale = (amax / qmax).to(torch.float32)
    q = torch.round(w.detach() / scale[:, None]).clamp(-qmax, qmax)
    return q.to(torch.int8), scale


def dequantize_weight(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * scale[:, None]


@dataclass
class Q8Result:
    model: nn.Module                 # simulated (dequantised weights)
    plan: dict
    int8_state: dict                 # compact serialisable state
    n_int8_params: int
    n_fp32_params: int
    max_weight_abs_err: float


def apply_q8_simulated(model: nn.Module) -> Q8Result:
    """Deep-copies `model`; replaces every allowlisted Linear weight by its
    dequantised int8 version (biases stay fp32); returns the compact int8
    state (for byte measurement) and the fp32-compute simulated model."""
    m = copy.deepcopy(model).cpu().eval()
    plan = q8_module_plan(m)
    int8_state, n8, n32, max_err = {}, 0, 0, 0.0
    quantized = set()
    for name, mod in m.named_modules():
        if plan.get(name) == "int8":
            q, s = quantize_weight_per_channel(mod.weight.data)
            deq = dequantize_weight(q, s)
            max_err = max(max_err, float((deq - mod.weight.data).abs().max()))
            mod.weight.data.copy_(deq)
            int8_state[f"{name}.weight.int8"] = q
            int8_state[f"{name}.weight.scale"] = s
            if mod.bias is not None:
                int8_state[f"{name}.bias"] = mod.bias.data.clone()
            n8 += mod.weight.numel()
            quantized.add(name)
    for k, v in m.state_dict().items():
        mod_name = k.rsplit(".", 1)[0]
        if mod_name in quantized:
            continue
        int8_state[k] = v.clone()
        n32 += v.numel()
        for pat in FP32_TENSOR_PATTERNS:
            if re.search(pat, k):
                if v.dtype != torch.float32:      # pragma: no cover
                    raise Part6GuardError(f"{k} must stay fp32")
    assert_fp32_denylist(int8_state)
    return Q8Result(m, plan, int8_state, n8, n32, max_err)


def assert_fp32_denylist(state: dict) -> None:
    """Every denylisted tensor present must be a float32 tensor and no
    '.int8' companion may exist for it."""
    for k, v in state.items():
        for pat in FP32_TENSOR_PATTERNS:
            if re.search(pat, k):
                if k.endswith(".int8") or k.endswith(".scale"):
                    raise Part6GuardError(f"denylisted tensor quantized: {k}")
                if isinstance(v, torch.Tensor) and v.dtype != torch.float32:
                    raise Part6GuardError(f"denylisted tensor not fp32: {k}")


def serialized_bytes(state: dict) -> int:
    """Actual torch.save bytes of a state dict (measured, not estimated)."""
    buf = io.BytesIO()
    torch.save(state, buf)
    return buf.getbuffer().nbytes


def cpu_dynamic_quantize(model: nn.Module) -> tuple[nn.Module, dict]:
    """torch.ao dynamic quantization restricted to allowlisted Linears
    (per-channel symmetric int8 weights, dynamic int8 activations)."""
    import torch.ao.quantization as tq
    m = copy.deepcopy(model).cpu().eval()
    plan = q8_module_plan(m)
    qspec = {name: tq.per_channel_dynamic_qconfig
             for name, kind in plan.items() if kind == "int8"}
    engines = torch.backends.quantized.supported_engines
    engine = "fbgemm" if "fbgemm" in engines else ("x86" if "x86" in engines
                                                    else engines[-1])
    torch.backends.quantized.engine = engine
    qm = tq.quantize_dynamic(m, qconfig_spec=qspec, dtype=torch.qint8,
                             inplace=False)
    n_dyn = sum(1 for _, mod in qm.named_modules()
                if mod.__class__.__name__ == "Linear"
                and mod.__class__.__module__.startswith("torch.ao.nn.quantized"))
    return qm, {"engine": engine, "n_dynamic_linears": n_dyn,
                "n_planned": sum(1 for v in plan.values() if v == "int8")}


def q8_report(model: nn.Module, include_dynamic: bool = True) -> dict:
    fp32_bytes = serialized_bytes({k: v.clone() for k, v in
                                   model.state_dict().items()})
    res = apply_q8_simulated(model)
    out = {"plan_int8_modules": sorted(k for k, v in res.plan.items()
                                       if v == "int8"),
           "plan_fp32_modules": sorted(k for k, v in res.plan.items()
                                       if v == "fp32"),
           "n_int8_params": res.n_int8_params,
           "n_fp32_params": res.n_fp32_params,
           "fraction_int8": res.n_int8_params / max(
               res.n_int8_params + res.n_fp32_params, 1),
           "fp32_state_bytes": fp32_bytes,
           "int8_compact_state_bytes": serialized_bytes(res.int8_state),
           "max_weight_abs_err": res.max_weight_abs_err}
    out["compression_ratio_bytes"] = fp32_bytes / out["int8_compact_state_bytes"]
    if include_dynamic:
        try:
            qm, info = cpu_dynamic_quantize(model)
            out["cpu_dynamic"] = info
            out["cpu_dynamic_state_bytes"] = serialized_bytes(qm.state_dict())
        except Exception as e:                       # pragma: no cover
            out["cpu_dynamic"] = {"error": f"{type(e).__name__}: {e}"}
    return out


# ---------------------------------------------------------------------------
# exploratory scaffolds — VAL-only, never registered
# ---------------------------------------------------------------------------
EXPLORATORY_TAG = "EXPLORATORY_VAL_ONLY_NOT_TEST_REGISTERED"


def apply_fp16_all_but_sensitive(model: nn.Module) -> nn.Module:
    """Cast every parameter/buffer to fp16 except A_log, dt_proj.*, D,
    conv1d.*, LayerNorms; the model is returned in mixed precision for a
    VALIDATION-only accuracy probe. Tagged exploratory."""
    m = copy.deepcopy(model).cpu().eval()
    for name, p in list(m.named_parameters()) + list(m.named_buffers()):
        keep = any(re.search(pat, name) for pat in FP32_TENSOR_PATTERNS)
        if not keep and p.dtype == torch.float32:
            p.data = p.data.half().float()      # fp16 round-trip simulation
    m.exploratory_tag = EXPLORATORY_TAG
    return m


def quantize_weight_per_group(w: torch.Tensor, bits: int = 4,
                              group: int = 64) -> torch.Tensor:
    """Asymmetric-free symmetric per-group weight-only fake quant along
    the input dim; returns the dequantised fp32 weight (VAL-only)."""
    out_f, in_f = w.shape
    qmax = 2 ** (bits - 1) - 1
    pad = (-in_f) % group
    wp = torch.nn.functional.pad(w.detach(), (0, pad))
    g = wp.reshape(out_f, -1, group)
    amax = g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = amax / qmax
    q = torch.round(g / scale).clamp(-qmax, qmax)
    return (q * scale).reshape(out_f, -1)[:, :in_f]


def apply_w4_inout_proj(model: nn.Module, group: int = 64) -> nn.Module:
    """W4 per-group weight-only on in_proj/out_proj only (Quamba2-style
    probe); everything else as Q8. VAL-only scaffold."""
    res = apply_q8_simulated(model)
    m = res.model
    for name, mod in m.named_modules():
        if re.search(r"\.(in_proj|out_proj)$", name) and isinstance(mod, nn.Linear):
            mod.weight.data.copy_(quantize_weight_per_group(
                mod.weight.data, bits=4, group=group))
    m.exploratory_tag = EXPLORATORY_TAG
    return m


class StaticActObserver:
    """Per-module symmetric per-tensor activation range observer for a
    static W8A8 probe; calibration inputs must be TRAIN windows (the
    caller passes only TRAIN ids — asserted by the CLI). VAL-only."""

    def __init__(self):
        self.amax: dict[str, float] = {}
        self._hooks = []

    def attach(self, model: nn.Module, plan: dict) -> None:
        for name, mod in model.named_modules():
            if plan.get(name) == "int8":
                self._hooks.append(mod.register_forward_pre_hook(
                    lambda m_, args, n=name: self._obs(n, args[0])))

    def _obs(self, name, x):
        v = float(x.detach().abs().max())
        self.amax[name] = max(self.amax.get(name, 0.0), v)

    def detach(self):
        for h in self._hooks:
            h.remove()


def apply_static_w8a8(model: nn.Module, act_amax: dict) -> nn.Module:
    """Simulated static W8A8: Q8 weights + fake-quant of each allowlisted
    Linear's INPUT with a fixed per-tensor scale from calibration."""
    res = apply_q8_simulated(model)
    m = res.model
    for name, mod in m.named_modules():
        if res.plan.get(name) == "int8" and name in act_amax:
            s = max(act_amax[name], 1e-12) / 127.0

            def pre(m_, args, s=s):
                x = args[0]
                return (torch.round(x / s).clamp(-127, 127) * s,) + tuple(args[1:])
            mod.register_forward_pre_hook(pre)
    m.exploratory_tag = EXPLORATORY_TAG
    return m


def leave_one_tensor_variants(model: nn.Module):
    """Yield (module_name, model_with_only_that_module_quantized) for the
    sensitivity map (VAL-only)."""
    plan = q8_module_plan(model)
    for name in sorted(k for k, v in plan.items() if v == "int8"):
        m = copy.deepcopy(model).cpu().eval()
        mod = dict(m.named_modules())[name]
        q, s = quantize_weight_per_channel(mod.weight.data)
        mod.weight.data.copy_(dequantize_weight(q, s))
        m.exploratory_tag = EXPLORATORY_TAG
        yield name, m
