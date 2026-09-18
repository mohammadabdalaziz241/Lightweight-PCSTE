"""Same-fold teacher ensembles, deterministic inference, TRAIN+VAL caching
and teacher-saturation diagnostics.

Teacher set for cell (fold f, seed s):
  K1/B1/dw_k1 : S1(f,42) + S1(f,1337) + S1(f,2026)   (frozen primary best.pt)
  K0          : S0(f,42) + S0(f,1337) + S0(f,2026)
  F1          : registered 10 %-label S1(f, s) cell (single teacher)
Cross-fold ensembles are forbidden (guards.assert_same_fold); a missing
seed makes the cache build FAIL — nothing is substituted.

Ensemble rule (PENDING pre-registration decision `ensemble_rule`):
  mean_prob_at_T : p_T = mean_k softmax(z_k / T)
  mean_logits    : p_T = softmax(mean_k z_k / T)
Both are deterministic; the cache stores the RAW per-seed logits so the
rule can be applied (and audited) at training time.

Cache = one .npz per (fold, teacher set) over TRAIN+VAL windows only,
keyed by window id, dataset, split; each teacher's checkpoint hash, the
encoder-config hash, the ensemble-rule text and the Part-6 version are
stored in the sidecar JSON and re-verified on load. Values are exact
(eval mode, dropout 0 by architecture, no augmentation, fp32).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..encoder import PCSTE, collate_representations
from ..encoder.mixer import HzGatedCrossBandMixer
from ..experiment.heads import CLASS_ORDERS, DatasetHeads
from .guards import (CheckpointRef, Part6GuardError, assert_no_test_windows,
                     assert_read_only_primary, assert_same_fold,
                     load_checkpoint_payload, primary_run_id,
                     resolve_checkpoint)
from .protocol import (KD_TEMPERATURE, PART6_RESULTS, PART6_VERSION, SEEDS,
                       config_hash)

DATASETS = ("CWRU", "JNU", "HIT", "MAFAULDA")
MAX_CLASSES = max(len(c) for c in CLASS_ORDERS.values())     # 10
ENSEMBLE_RULES = ("mean_prob_at_T", "mean_logits")


# ---------------------------------------------------------------------------
# alpha recorder (forward hook; the frozen encoder does not return alpha)
# ---------------------------------------------------------------------------
class AlphaRecorder:
    """Captures the Hz-mixer band-attention alpha (B, F) of every forward
    without modifying the frozen mixer: a hook on the mixer records its
    inputs (h, phi_f, band_mask), a hook on mixer.score records the raw
    scores; alpha is recomputed with the mixer's own masking + softmax
    (identical arithmetic: masked_fill(-inf) -> softmax -> nan_to_num ->
    * mask). Works under autograd (student side) and no_grad (teacher)."""

    def __init__(self, mixer: HzGatedCrossBandMixer):
        if not isinstance(mixer, HzGatedCrossBandMixer):
            raise Part6GuardError("AlphaRecorder needs the frozen mixer class")
        self._band_mask = None
        self._scores = None
        self.alpha = None
        self._h1 = mixer.register_forward_pre_hook(self._pre)
        self._h2 = mixer.score.register_forward_hook(self._score_hook)

    def _pre(self, module, args):
        self._band_mask = args[2]

    def _score_hook(self, module, args, output):
        a = output.squeeze(-1)
        a = a.masked_fill(~self._band_mask, float("-inf"))
        alpha = torch.softmax(a, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        self.alpha = alpha * self._band_mask.to(alpha.dtype)

    def remove(self) -> None:
        self._h1.remove()
        self._h2.remove()


# ---------------------------------------------------------------------------
# teacher sets
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TeacherSet:
    name: str                       # "s1" | "s0" | "s1_l010"
    fold: int
    refs: tuple                     # CheckpointRef, ordered by SEEDS
    ensemble_rule: str

    @property
    def hashes(self) -> dict:
        return {r.run_id: r.sha256 for r in self.refs}

    def to_dict(self) -> dict:
        return {"name": self.name, "fold": self.fold,
                "ensemble_rule": self.ensemble_rule,
                "members": [{"run_id": r.run_id, "sha256": r.sha256,
                             "best_epoch": r.best_epoch,
                             "seed": r.seed, "fold": r.fold}
                            for r in self.refs]}


def teacher_arm_and_pct(name: str) -> tuple[str, int]:
    if name == "s1":
        return "s1", 100
    if name == "s0":
        return "s0", 100
    if name == "s1_l010":
        return "s1", 10
    raise Part6GuardError(f"unknown teacher set {name}")


def discover_teacher_set(name: str, fold: int, ensemble_rule: str,
                         root: Path | None = None,
                         seeds: tuple = SEEDS) -> TeacherSet:
    """Resolve + hash-verify every same-fold member; FAIL if any is
    missing/incomplete; refuse cross-fold members structurally."""
    if ensemble_rule not in ENSEMBLE_RULES:
        raise Part6GuardError(f"ensemble_rule must be one of {ENSEMBLE_RULES}")
    arm, pct = teacher_arm_and_pct(name)
    refs, missing = [], []
    for s in seeds:
        rid = primary_run_id(arm, fold, s, pct)
        try:
            refs.append(resolve_checkpoint(rid, root=root))
        except Part6GuardError as e:
            missing.append(f"{rid}: {e}")
    if missing:
        raise Part6GuardError(
            "teacher set incomplete — no substitution allowed:\n  "
            + "\n  ".join(missing))
    assert_same_fold(fold, refs, context=f"teacher set {name} fold {fold}")
    return TeacherSet(name=name, fold=fold, refs=tuple(refs),
                      ensemble_rule=ensemble_rule)


def load_teacher_model(ref: CheckpointRef, device: str = "cpu"
                       ) -> tuple[PCSTE, DatasetHeads]:
    """Deterministic inference mode: eval(), dropout is 0.0 by frozen
    architecture, no augmentation exists in the pipeline."""
    ck = load_checkpoint_payload(ref)
    enc = PCSTE()
    enc.load_state_dict(ck["encoder"], strict=True)
    heads = DatasetHeads()
    heads.load_state_dict(ck["heads"], strict=True)
    enc.to(device).eval()
    heads.to(device).eval()
    for p in list(enc.parameters()) + list(heads.parameters()):
        p.requires_grad_(False)
    return enc, heads


# ---------------------------------------------------------------------------
# ensemble rule
# ---------------------------------------------------------------------------
def ensemble_soft_targets(per_seed_logits: torch.Tensor, rule: str,
                          temperature: float = KD_TEMPERATURE) -> torch.Tensor:
    """per_seed_logits (K, B, C) -> soft targets p_T (B, C)."""
    if rule == "mean_prob_at_T":
        return torch.softmax(per_seed_logits / temperature, dim=-1).mean(0)
    if rule == "mean_logits":
        return torch.softmax(per_seed_logits.mean(0) / temperature, dim=-1)
    raise Part6GuardError(f"unknown ensemble rule {rule}")


# ---------------------------------------------------------------------------
# cache building
# ---------------------------------------------------------------------------
def cache_dir(root: Path | None = None) -> Path:
    return (root or PART6_RESULTS) / "teacher_cache"


def cache_paths(name: str, fold: int, root: Path | None = None
                ) -> tuple[Path, Path]:
    d = cache_dir(root)
    return d / f"teacher_{name}_f{fold}.npz", d / f"teacher_{name}_f{fold}.json"


@torch.no_grad()
def teacher_forward(enc: PCSTE, heads: DatasetHeads, reps: list,
                    datasets: list[str], device: str = "cpu",
                    with_band_summaries: bool = False) -> dict:
    """One exact teacher pass over a list of representations. Returns
    per-head logits (all four heads), embedding, alpha, band summaries."""
    rec = AlphaRecorder(enc.mixer)
    try:
        batch = collate_representations(reps)
        batch = {k: v.to(device) for k, v in batch.items()}
        out = enc(**batch)
        z = out["global_embedding"]
        logits = {ds: heads(z, ds).cpu().numpy() for ds in DATASETS}
        alpha = rec.alpha.cpu().numpy()
        res = {"logits": logits, "embedding": z.cpu().numpy(),
               "alpha": alpha,
               "band_mask": out["band_mask"].cpu().numpy(),
               "n_bands": alpha.shape[1]}
        if with_band_summaries:
            res["band_summaries"] = out["band_summaries"].cpu().numpy()
        return res
    finally:
        rec.remove()


def _pad_bands(a: np.ndarray, n_bands: int) -> np.ndarray:
    if a.shape[1] == n_bands:
        return a
    out = np.zeros((a.shape[0], n_bands) + a.shape[2:], dtype=a.dtype)
    out[:, :a.shape[1]] = a
    return out


def build_teacher_cache(tset: TeacherSet, manifest: pd.DataFrame,
                        rep_fn, out_root: Path | None = None,
                        device: str = "cpu", chunk: int = 64,
                        with_band_summaries: bool = True,
                        window_ids: list[str] | None = None,
                        encoder_state_override: dict | None = None
                        ) -> tuple[Path, Path]:
    """Cache exact teacher outputs on TRAIN+VAL windows of `tset.fold`.

    manifest: fold manifest indexed by window_id (any split filtering
              is re-done here; TEST ids are refused structurally).
    rep_fn:   window_id -> (tensor, frequency_hz, time_seconds).
    encoder_state_override: tests only (tiny fake teachers)."""
    out_root = out_root or PART6_RESULTS
    assert_read_only_primary(out_root)
    if window_ids is None:
        window_ids = list(manifest.index[manifest["split"].isin(
            ["train", "validation"])])
    assert_no_test_windows(window_ids, "teacher cache")
    for w in window_ids:
        if manifest.loc[w, "split"] not in ("train", "validation"):
            raise Part6GuardError(f"{w}: split {manifest.loc[w, 'split']}")
    n_bands = 33
    n = len(window_ids)
    ds_of = [str(manifest.loc[w, "dataset"]) for w in window_ids]
    split_of = [str(manifest.loc[w, "split"]) for w in window_ids]
    arrays: dict[str, np.ndarray] = {
        "window_id": np.array(window_ids, dtype="U"),
        "dataset": np.array(ds_of, dtype="U"),
        "split": np.array(split_of, dtype="U"),
        "band_mask": np.zeros((n, n_bands), dtype=bool),
    }
    k_seeds = len(tset.refs)
    for ds in DATASETS:
        arrays[f"logits_{ds}"] = np.full(
            (k_seeds, n, len(CLASS_ORDERS[ds])), np.nan, dtype=np.float32)
    arrays["embedding"] = np.zeros((k_seeds, n, 192), dtype=np.float32)
    arrays["alpha"] = np.zeros((k_seeds, n, n_bands), dtype=np.float32)
    if with_band_summaries:
        arrays["band_summaries"] = np.zeros((k_seeds, n, n_bands, 192),
                                            dtype=np.float16)
    order = np.argsort(np.array(ds_of, dtype="U"), kind="stable")
    for ki, ref in enumerate(tset.refs):
        if encoder_state_override is None:
            enc, heads = load_teacher_model(ref, device)
        else:                                       # tests: tiny stand-ins
            enc, heads = encoder_state_override[ref.run_id]
            enc.eval()
            heads.eval()
        for ds in DATASETS:
            idx = [int(i) for i in order if ds_of[i] == ds]
            for lo in range(0, len(idx), chunk):
                sub = idx[lo:lo + chunk]
                reps = [rep_fn(window_ids[i]) for i in sub]
                out = teacher_forward(enc, heads, reps, [ds] * len(sub),
                                      device, with_band_summaries)
                for d2 in DATASETS:
                    arrays[f"logits_{d2}"][ki, sub] = out["logits"][d2]
                arrays["embedding"][ki, sub] = out["embedding"]
                arrays["alpha"][ki, sub] = _pad_bands(out["alpha"], n_bands)
                bm = _pad_bands(out["band_mask"], n_bands)
                if ki == 0:
                    arrays["band_mask"][sub] = bm
                else:
                    if not np.array_equal(arrays["band_mask"][sub], bm):
                        raise Part6GuardError("band mask differs across "
                                              "teachers (impossible)")
                if with_band_summaries:
                    arrays["band_summaries"][ki, sub] = _pad_bands(
                        out["band_summaries"], n_bands).astype(np.float16)
        del enc, heads
    for ds in DATASETS:
        own = np.array([d == ds for d in ds_of])
        if np.isnan(arrays[f"logits_{ds}"][:, own]).any():
            raise Part6GuardError(f"NaN logits left for {ds}")
    npz_p, json_p = cache_paths(tset.name, tset.fold, out_root)
    npz_p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_p, **arrays)
    content_hash = hashlib.sha256()
    for k in sorted(arrays):
        content_hash.update(k.encode())
        content_hash.update(np.ascontiguousarray(arrays[k]).tobytes())
    meta = {"part6_version": PART6_VERSION,
            "teacher_set": tset.to_dict(),
            "fold": tset.fold, "n_windows": n,
            "splits": sorted(set(split_of)),
            "datasets": {ds: int(sum(1 for d in ds_of if d == ds))
                         for ds in DATASETS},
            "n_bands_padded": n_bands,
            "with_band_summaries": with_band_summaries,
            "band_summaries_dtype": "float16" if with_band_summaries else None,
            "encoder_config_hash": config_hash(PCSTE().cfg.to_dict()),
            "inference": "eval(); dropout 0.0 (frozen arch); no "
                         "augmentation; fp32; per-dataset chunks of "
                         f"{chunk}",
            "content_sha256": content_hash.hexdigest(),
            "keys": sorted(arrays)}
    json_p.write_text(json.dumps(meta, indent=1, sort_keys=True))
    return npz_p, json_p


class TeacherCache:
    """RAM view of a cache with provenance re-verification."""

    def __init__(self, name: str, fold: int, root: Path | None = None,
                 expected_hashes: dict | None = None):
        npz_p, json_p = cache_paths(name, fold, root)
        if not npz_p.exists() or not json_p.exists():
            raise Part6GuardError(f"teacher cache missing: {npz_p}")
        self.meta = json.loads(json_p.read_text())
        if self.meta["fold"] != fold or self.meta["teacher_set"]["name"] != name:
            raise Part6GuardError("cache identity mismatch")
        z = np.load(npz_p, allow_pickle=False)
        self.arrays = {k: z[k] for k in z.files}
        h = hashlib.sha256()
        for k in sorted(self.arrays):
            h.update(k.encode())
            h.update(np.ascontiguousarray(self.arrays[k]).tobytes())
        if h.hexdigest() != self.meta["content_sha256"]:
            raise Part6GuardError("teacher cache content hash mismatch")
        if expected_hashes is not None:
            got = {m["run_id"]: m["sha256"]
                   for m in self.meta["teacher_set"]["members"]}
            if got != expected_hashes:
                raise Part6GuardError(
                    f"cache teachers {got} != registry teachers {expected_hashes}")
        assert_no_test_windows(list(self.arrays["window_id"]), "cache load")
        self.index = {w: i for i, w in enumerate(self.arrays["window_id"])}
        self.members = [m["run_id"] for m in
                        self.meta["teacher_set"]["members"]]
        self.seed_index = {m["seed"]: i for i, m in
                          enumerate(self.meta["teacher_set"]["members"])}

    def rows(self, window_ids: list[str]) -> np.ndarray:
        try:
            return np.array([self.index[w] for w in window_ids])
        except KeyError as e:
            raise Part6GuardError(f"window not in teacher cache: {e}")

    def per_seed_logits(self, ds: str, window_ids: list[str]) -> torch.Tensor:
        """(K, B, C_ds) raw logits of the window's OWN dataset head."""
        r = self.rows(window_ids)
        return torch.from_numpy(self.arrays[f"logits_{ds}"][:, r])

    def single_alpha(self, seed: int, window_ids: list[str]) -> torch.Tensor:
        r = self.rows(window_ids)
        return torch.from_numpy(self.arrays["alpha"][self.seed_index[seed], r])

    def band_mask(self, window_ids: list[str]) -> torch.Tensor:
        return torch.from_numpy(self.arrays["band_mask"][self.rows(window_ids)])


# ---------------------------------------------------------------------------
# teacher-confidence (saturation) diagnostic — explanatory only
# ---------------------------------------------------------------------------
def saturation_diagnostic(cache: TeacherCache, rule: str,
                          temperature: float = KD_TEMPERATURE,
                          split: str = "train") -> dict:
    """Per dataset: mean softmax entropy (T=1 and at T), top-1 margin,
    confidence distribution of the ensemble soft targets and of each
    single teacher. Never used to change any hyper-parameter."""
    out = {"rule": rule, "temperature": temperature, "split": split,
           "per_dataset": {}}
    ds_arr = cache.arrays["dataset"]
    sp_arr = cache.arrays["split"]
    for ds in DATASETS:
        sel = np.where((ds_arr == ds) & (sp_arr == split))[0]
        if sel.size == 0:
            continue
        z = torch.from_numpy(cache.arrays[f"logits_{ds}"][:, sel])   # K,B,C
        p1 = ensemble_soft_targets(z, rule, temperature=1.0)
        pT = ensemble_soft_targets(z, rule, temperature=temperature)
        ent1 = -(p1 * torch.log(p1.clamp_min(1e-12))).sum(-1)
        entT = -(pT * torch.log(pT.clamp_min(1e-12))).sum(-1)
        top2 = torch.topk(p1, 2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]
        conf = top2[:, 0]
        singles = []
        for k in range(z.shape[0]):
            pk = torch.softmax(z[k], -1)
            ek = -(pk * torch.log(pk.clamp_min(1e-12))).sum(-1)
            singles.append({"mean_entropy_T1": float(ek.mean()),
                            "mean_top1": float(pk.max(-1).values.mean())})
        qs = [0.05, 0.25, 0.5, 0.75, 0.95]
        out["per_dataset"][ds] = {
            "n": int(sel.size), "n_classes": int(z.shape[-1]),
            "max_entropy": float(np.log(z.shape[-1])),
            "ensemble_mean_entropy_T1": float(ent1.mean()),
            "ensemble_mean_entropy_at_T": float(entT.mean()),
            "ensemble_mean_top1_margin": float(margin.mean()),
            "ensemble_top1_confidence_quantiles": {
                str(q): float(torch.quantile(conf, q)) for q in qs},
            "fraction_top1_above_0.99": float((conf > 0.99).float().mean()),
            "single_teachers": singles}
    return out
