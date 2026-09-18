"""Final matched S0 vs S1 stage (final_s0_s1.v1): global-fold composition over the fixed CWRU load partition, the native12
patch grids, and the masked-reconstruction SSL machinery adapted to the v2 / N2b representation.

Global-fold composition: CWRU keeps ONE fixed native12 load partition (TRAIN 0+1 hp, VAL 2 hp, TEST 3 hp) in every global fold;
JNU, HIT and MaFaulDa contribute their own retained fold k. Nothing is regenerated: the CWRU windows come from
pcste_v2_cwru_native12_load_v1 and the other three from their accepted registries, so canonical window identities are preserved.

SSL: the accepted Part-5C masked-reconstruction objective (ReconstructionProbe = learned mask token + X1 post-mixer context +
per-token MLP decoder, masked valid-cell MSE) run on the v2 N2b representation through the SAME PCSTEv2 core that S0 and S1 use.
The decoder and mask token are pretraining-only and are discarded at S1. Checkpoint selection uses the accepted
`mean_nmse.v2` selector (mean over datasets of masked-cell MSE divided by that dataset's masked-target energy on FIXED
validation masks), implemented in src/pcste_v2/selector.py.
"""
from __future__ import annotations

import hashlib, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.methodology_v2.registry import REPO_ROOT
from src.methodology_v2.encoder.ssl_design import ReconstructionProbe, generate_mask, window_rng
from src.methodology_v2.experiment.trainers import MASK_RATIO, MASK_GEOMETRY, SSL_EPOCHS, OPTIMIZER_SPEC, validation_mask_seed
from .model import PCSTEv2, PCSTEv2Config, collate_v2
from .selector import DEFAULT_SELECTOR, selector_value
from .protocol import sealed_windows, _harmonise, MAF_V2_VERSION
from .protocol_cwru_native12_load import PROTOCOL_ID as CWRU_LOAD_ID, PROTOCOL_VERSION as CWRU_LOAD_VERSION, verify_protocol as verify_cwru_load

FINAL_SPEC_VERSION = "final_s0_s1.v1"
GLOBAL_FOLDS = (1, 2, 3)
# native12 v2 patch grids (16x8 patches). CWRU differs from the sealed 48 kHz grid because native12 CWRU has 129 bins, not 513.
NATIVE12_GRIDS = {"CWRU": (9, 23), "JNU": (33, 24), "HIT": (17, 24), "MAFAULDA": (33, 24)}
SSL_SELECTOR = DEFAULT_SELECTOR                     # "mean_nmse.v2"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ global-fold composition
def build_global_fold(fold: int, cwru_dir: Path, maf_dir: Path) -> pd.DataFrame:
    """CWRU = the ONE fixed load partition (fold_id relabelled to k); JNU/HIT = sealed fold k; MaFaulDa = stratified fold k."""
    verify_cwru_load(cwru_dir)
    cw = pd.read_csv(cwru_dir / "cwru_v2_windows_fold_1.csv", dtype={"or_clock_position": str})
    others = sealed_windows(fold, ("JNU", "HIT"))
    mf = pd.read_csv(maf_dir / f"mafaulda_v2_windows_fold_{fold}.csv")
    parts = [_harmonise(cw, f"cwru:{CWRU_LOAD_VERSION}"), _harmonise(others, "sealed:methodology_v2.part3b.v1"), _harmonise(mf, f"mafaulda:{MAF_V2_VERSION}")]
    df = pd.concat(parts, ignore_index=True); df["fold_id"] = fold
    assert df.window_id.is_unique, f"global fold {fold}: duplicate window ids"
    return df


def write_global_folds(out_dir: Path, cwru_dir: Path, maf_dir: Path, steps_per_epoch: dict[str, int]) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True); info = {}
    cw_ids = None
    for f in GLOBAL_FOLDS:
        df = build_global_fold(f, cwru_dir, maf_dir); p = out_dir / f"global_v2_fold_{f}.csv"; df.to_csv(p, index=False)
        c = df[df.dataset == "CWRU"]; key = (tuple(sorted(c.window_id)), tuple(sorted(c.split)))
        if cw_ids is None: cw_ids = key
        assert key == cw_ids, f"global fold {f}: CWRU membership differs from fold 1 (must be identical)"
        assert (c.native_sampling_rate_hz == 12000).all()
        info[f"fold_{f}"] = {"sha256": sha256_file(p), "n_windows": len(df), "counts": {f"{d}/{s}": int(n) for (d, s), n in df.groupby(["dataset", "split"]).size().items()}}
    info |= {"cwru_protocol_dir": str(Path(cwru_dir).resolve().relative_to(REPO_ROOT.resolve())), "cwru_protocol_id": CWRU_LOAD_ID, "cwru_native12": True,
             "cwru_protocol_master": json.loads((cwru_dir / "cwru_v2_hashes.json").read_text())["master"], "mafaulda_protocol": "v2_stratified",
             "final_spec_version": FINAL_SPEC_VERSION, "steps_per_epoch": steps_per_epoch,
             "composition": "CWRU: one fixed native12 load partition in every global fold (TRAIN 0+1hp, VAL 2hp, TEST 3hp). JNU/HIT: sealed methodology_v2 fold k. MaFaulDa: mafaulda_v2.stratified.v1 fold k, radial-only. No split was regenerated.",
             "cwru_membership_identical_across_folds": True}
    (out_dir / "global_v2_hashes.json").write_text(json.dumps(info, indent=1, sort_keys=True)); return info


# ------------------------------------------------------------------ SSL on the v2 representation
def collated_grid(batch: dict) -> tuple[int, int]:
    """Token grid (F, T) that patchify will produce for this collated batch (it pads each axis up to the patch multiple)."""
    from src.methodology_v2.encoder.patchify import PATCH_F, PATCH_T
    _, bins, frames = batch["spec_g0_c0"].shape
    return (-(-int(bins) // PATCH_F), -(-int(frames) // PATCH_T))


def build_patch_mask_v2(metas: list[tuple[str, str]], seed: int, epoch: int, fixed_validation: bool,
                        grid_shape: tuple[int, int] | None = None) -> torch.Tensor:
    """(B, F, T) boolean SSL mask over the native12 patch grids. TRAIN masks vary with (seed, epoch, window); validation masks
    use (validation_mask_seed(seed), epoch 0) so they are identical across every checkpoint comparison.

    grid_shape MUST be the token grid of the collated batch: a chunk containing only narrow-band windows (e.g. an all-CWRU
    validation chunk: 129 bins x 184 frames -> 9 x 23) collates smaller than the four-dataset maximum (33 x 24). Only the
    container size depends on it; the per-window mask CONTENT is generated from that window's own native12 grid and is identical
    either way."""
    if grid_shape is None:
        grid_shape = (max(g[0] for g in NATIVE12_GRIDS.values()), max(g[1] for g in NATIVE12_GRIDS.values()))
    fmax, tmax = grid_shape
    pm = torch.zeros(len(metas), fmax, tmax, dtype=torch.bool)
    for i, (ds, wid) in enumerate(metas):
        fb, tp = NATIVE12_GRIDS[ds]
        assert fb <= fmax and tp <= tmax, f"{ds} grid {(fb, tp)} exceeds the collated grid {(fmax, tmax)}"
        rng = window_rng(validation_mask_seed(seed), 0, wid) if fixed_validation else window_rng(seed, epoch, wid)
        pm[i, :fb, :tp] = torch.from_numpy(generate_mask(np.ones((fb, tp), dtype=bool), MASK_RATIO, MASK_GEOMETRY, rng))
    return pm


def mask_plan_hash(metas: list[tuple[str, str]], seed: int, epoch: int, fixed_validation: bool) -> str:
    pm = build_patch_mask_v2(metas, seed, epoch, fixed_validation)
    return hashlib.sha256(pm.numpy().tobytes()).hexdigest()


class SSLModuleV2(torch.nn.Module):
    """ReconstructionProbe built on the SAME PCSTEv2 core that S0/S1 use, so the pretrained encoder loads into PCSTEv2 directly."""

    def __init__(self):
        super().__init__()
        self.encoder = PCSTEv2(PCSTEv2Config())            # plain N2b encoder; state_dict keys are core.*
        self.probe = ReconstructionProbe(self.encoder.core)

    def forward(self, batch: dict, patch_mask: torch.Tensor) -> dict:
        return self.probe(spec=batch["spec_g0_c0"], frequency_hz=batch["frequency_hz_g0_c0"],
                          time_seconds=batch["time_seconds_g0_c0"], cell_mask=batch["cell_mask_g0_c0"], patch_mask=patch_mask)

    def encoder_state(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.encoder.state_dict().items()}

    def pretraining_only_parameters(self) -> list[str]:
        return [f"probe.{n}" for n, _ in self.probe.named_parameters() if not n.startswith("encoder.")]


@torch.no_grad()
def ssl_validation_metrics(model: SSLModuleV2, store: dict, val_ids: list[str], ds_of: dict, seed: int, dev: str, chunk: int = 32) -> dict:
    """Per-dataset masked-cell MSE and masked-target ENERGY on the FIXED validation masks (forward only, no gradient).
    The energy is the denominator of the accepted mean_nmse.v2 selector: mean of target^2 over exactly the same masked cells."""
    model.eval(); acc = {}
    for lo in range(0, len(val_ids), chunk):
        ids = val_ids[lo:lo + chunk]; metas = [(ds_of[w], w) for w in ids]
        b = {k: v.to(dev) for k, v in collate_v2([store[w] for w in ids], 1, 1, False).items()}
        pm = build_patch_mask_v2(metas, seed, 0, True, collated_grid(b)).to(dev)
        out = model(b, pm)
        from src.methodology_v2.encoder.patchify import patchify
        patches, pcell, tm, _ = patchify(b["spec_g0_c0"], b["cell_mask_g0_c0"])
        tgt = patches.squeeze(3).reshape(*out["pred"].shape); lm = out["loss_mask"]
        e = ((tgt ** 2) * lm).sum(dim=(1, 2, 3)) / lm.sum(dim=(1, 2, 3)).clamp(min=1)
        for i, (ds, _) in enumerate(metas):
            a = acc.setdefault(ds, {"mse": [], "energy": []}); a["mse"].append(float(out["per_window_mse"][i])); a["energy"].append(float(e[i]))
    model.train()
    return {ds: {"mse": float(np.mean(v["mse"])), "energy": float(np.mean(v["energy"])), "n": len(v["mse"])} for ds, v in acc.items()}


def ssl_selector_value(metrics: dict, name: str = SSL_SELECTOR) -> float:
    return selector_value(name, {d: m["mse"] for d, m in metrics.items()}, {d: m["energy"] for d, m in metrics.items()})


def canonical_state_hash(state: dict) -> str:
    """Hash over sorted parameter names, shapes, dtypes and tensor bytes (not the serialised file, whose metadata may differ)."""
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k].detach().cpu().contiguous()
        h.update(k.encode()); h.update(str(tuple(t.shape)).encode()); h.update(str(t.dtype).encode()); h.update(t.numpy().tobytes())
    return h.hexdigest()
