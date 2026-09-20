"""PC-STE v2 CWRU intervention (cwru_xspec.v1): class/physical-specimen-balanced CWRU sampling (C1) and the cross-specimen
supervised-contrastive auxiliary loss (C2). Reference components copied verbatim from cwru_intervention_kit/
(specimen_sampling.py sha256 e83c2c5dd008b4bb, cross_specimen_loss.py sha256 f7dd49a01fc6fc8a); integration helpers below map them onto the
executor's batch stream (tuples (dataset, class, window_id)) and the sealed CWRU v2 physical-specimen registry. Default-off:
no existing variant's configuration or batch stream changes (test-pinned)."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from hashlib import sha256
from math import ceil
from numbers import Integral
import random
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.methodology_v2.experiment.heads import CLASS_ORDERS

CWRU_XSPEC_VERSION = "cwru_xspec.v1"
CWRU_PLAN_VERSION = "cwru-specimen-plan-v1"          # the seed/epoch digest label used inside make_epoch_plan
CWRU_ITEMS_PER_STEP = 16
XSPEC_TAU = 0.10
XSPEC_LAMBDA = 0.10
KIT_SHA256 = {"specimen_sampling.py": "e83c2c5dd008b4bb1ddcb2bc48c7c939ac809e05d96170e48630a60a8faeaf41", "cross_specimen_loss.py": "f7dd49a01fc6fc8aa649222b5c9aae3c55eb8a935959f33a1de914bc0f0be178"}


# =========================================================================== kit: specimen_sampling.py (verbatim)
from collections import defaultdict
from hashlib import sha256
from math import ceil
from numbers import Integral
import random
from typing import Sequence


def make_epoch_plan(
    indices: Sequence[int],
    labels: Sequence[int],
    specimen_ids: Sequence[str],
    *,
    batch_size: int,
    steps: int,
    seed: int,
    epoch: int,
) -> list[list[int]]:
    """Return exactly ``steps`` CWRU batches with fixed ``batch_size``.

    Class slots rotate as evenly as possible; within each class, physical
    specimens rotate evenly. Each class has >=2 specimens in every batch.
    Windows are shuffled within specimen, with no duplicate index in a batch.
    Sampling repeats across batches as needed; the epoch length is unchanged.
    Recording/load frequencies within a specimen are not rebalanced here.
    """
    if not (len(indices) == len(labels) == len(specimen_ids)) or len(indices) == 0:
        raise ValueError("Nonempty index, label and specimen arrays must align")
    if any(not isinstance(x, Integral) or isinstance(x, bool) for x in indices):
        raise ValueError("Indices must be integers")
    if any(not isinstance(x, Integral) or isinstance(x, bool) for x in labels):
        raise ValueError("Labels must be integer class IDs")
    indices = [int(x) for x in indices]
    labels = [int(x) for x in labels]
    if len(set(indices)) != len(indices):
        raise ValueError("TRAIN indices must be unique")
    if any(not isinstance(s, str) or not s.strip() for s in specimen_ids):
        raise ValueError("Physical specimen IDs must be nonempty strings")
    for name, value in (("batch_size", batch_size), ("steps", steps),
                        ("seed", seed), ("epoch", epoch)):
        if not isinstance(value, Integral) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
    if batch_size < 1 or steps < 1 or epoch < 0:
        raise ValueError("Batch size/steps must be positive and epoch nonnegative")

    pools = defaultdict(list)
    specimen_class = {}
    for idx, label, sid in zip(indices, labels, specimen_ids):
        if sid in specimen_class and specimen_class[sid] != label:
            raise ValueError("One specimen ID has conflicting class labels")
        specimen_class[sid] = label
        pools[label, sid].append(idx)
    classes = sorted(set(labels))
    if len(classes) < 2 or batch_size < 2 * len(classes):
        raise ValueError("Need >=2 classes and >=2 batch slots per class")
    groups = {c: sorted(s for cc, s in pools if cc == c) for c in classes}
    for c in classes:
        if len(groups[c]) < 2:
            raise ValueError(f"Class {c} has fewer than two TRAIN specimens")
        max_quota = ceil(batch_size / len(classes))
        required = ceil(max_quota / len(groups[c]))
        if any(len(pools[c, s]) < required for s in groups[c]):
            raise ValueError(f"Class {c} lacks enough distinct windows per specimen")

    digest = sha256(f"cwru-specimen-plan-v1:{int(seed)}:{int(epoch)}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:16], "big"))
    rng.shuffle(classes)
    for c in classes:
        rng.shuffle(groups[c])
    decks = {key: sorted(value) for key, value in pools.items()}
    for deck in decks.values():
        rng.shuffle(deck)
    cursor = {key: 0 for key in pools}
    group_cursor = {c: 0 for c in classes}

    def draw(key, used):
        # A full traversal includes every window, even across a reshuffle.
        for _ in range(2 * len(decks[key])):
            if cursor[key] == len(decks[key]):
                rng.shuffle(decks[key])
                cursor[key] = 0
            idx = decks[key][cursor[key]]
            cursor[key] += 1
            if idx not in used:
                return idx
        raise RuntimeError("Cannot draw a distinct TRAIN window for this batch")

    base, remainder = divmod(batch_size, len(classes))
    plan = []
    for step in range(steps):
        extra = {classes[(step * remainder + j) % len(classes)]
                 for j in range(remainder)}
        batch, used = [], set()
        for c in classes:
            quota = base + int(c in extra)
            for _ in range(quota):
                sid = groups[c][group_cursor[c] % len(groups[c])]
                group_cursor[c] += 1
                idx = draw((c, sid), used)
                batch.append(idx)
                used.add(idx)
        rng.shuffle(batch)
        plan.append(batch)
    return plan


def replace_positions(base_batch: Sequence, positions: Sequence[int], replacements: Sequence):
    """Replace only the designated CWRU slots in an already generated batch.

    Consume the original mixed sampler first, including its old CWRU draws,
    to preserve its RNG progression and all non-CWRU items.
    """
    if len(positions) != len(replacements) or len(set(positions)) != len(positions):
        raise ValueError("Replacement positions must be unique and align with values")
    if any(not isinstance(p, Integral) or p < 0 or p >= len(base_batch) for p in positions):
        raise ValueError("Replacement position outside base batch")
    result = list(base_batch)
    for position, replacement in zip(positions, replacements):
        result[position] = replacement
    return result


# =========================================================================== kit: cross_specimen_loss.py (verbatim)
import math
import torch
import torch.nn.functional as F


def pair_masks(labels, specimen_ids):
    """Same-class/same-specimen pairs are excluded, never false negatives."""
    if labels.ndim != 1 or specimen_ids.shape != labels.shape:
        raise ValueError("Class and specimen IDs must be aligned 1D tensors")
    if labels.dtype != torch.long or specimen_ids.dtype != torch.long:
        raise ValueError("Class and specimen IDs must have torch.long dtype")
    if labels.device != specimen_ids.device:
        raise ValueError("IDs must be on the same device")
    same_class = labels[:, None].eq(labels[None, :])
    same_specimen = specimen_ids[:, None].eq(specimen_ids[None, :])
    if bool((same_specimen & ~same_class).any()):
        raise ValueError("A physical specimen ID has conflicting class labels")
    positive = same_class & ~same_specimen
    negative = ~same_class
    eligible = positive | negative
    return positive, eligible


def cross_specimen_loss(features, labels, specimen_ids, temperature=0.1):
    """Equal class -> equal specimen -> equal anchor reduction.

    Positive: same CWRU fault class, different physical specimen.
    Negative: different CWRU fault class.
    Denominator: positives plus negatives, excluding self/same-specimen peers.
    Temperature and proposed coefficient are prospective, not tuned estimates.
    """
    if features.ndim != 2 or features.shape[0] != labels.numel():
        raise ValueError("Expected features [N,D] aligned with the IDs")
    if features.shape[0] < 4 or features.shape[1] < 1:
        raise ValueError("Need multiple classes and specimens")
    if not features.is_floating_point():
        raise ValueError("Features must be floating point")
    if features.device != labels.device:
        raise ValueError("Features and IDs must be on the same device")
    if features.device.type not in ("cpu", "cuda"):
        raise ValueError("Reference implementation supports CPU/CUDA")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("Nonfinite features")
    positive, eligible = pair_masks(labels, specimen_ids)
    counts = positive.sum(dim=1)
    if bool((counts == 0).any()) or labels.unique().numel() < 2:
        raise ValueError("Every anchor needs a positive from another specimen and negatives")

    # Explicitly disable outer AMP for the similarity and logsumexp operations.
    with torch.autocast(device_type=features.device.type, enabled=False):
        z = F.normalize(features.float(), p=2, dim=1, eps=1e-12)
        similarities = (z @ z.T) / temperature
        log_denominator = torch.logsumexp(
            similarities.masked_fill(~eligible, -torch.inf), dim=1
        )
        log_ratio = log_denominator[:, None] - similarities
        anchor_loss = torch.where(positive, log_ratio, 0.0).sum(1) / counts
        class_losses = []
        for class_id in labels.unique():
            class_mask = labels.eq(class_id)
            specimen_losses = []
            for specimen_id in specimen_ids[class_mask].unique():
                group_mask = class_mask & specimen_ids.eq(specimen_id)
                specimen_losses.append(anchor_loss[group_mask].mean())
            class_losses.append(torch.stack(specimen_losses).mean())
        return torch.stack(class_losses).mean()


# =========================================================================== integration helpers
def cwru_train_universe(man: pd.DataFrame) -> dict:
    """Sealed CWRU TRAIN universe from the global manifest (window_id, class, physical_specimen from the registry column).
    Verifies the protocol premise: the three existing classes, >= 2 TRAIN specimens per class, one class per specimen."""
    cw = man[(man.dataset == "CWRU") & (man.split == "train")].sort_values("window_id").reset_index(drop=True)
    assert len(cw) > 0 and (cw.split == "train").all()
    classes = list(CLASS_ORDERS["CWRU"]); assert set(cw["class"]) == set(classes), f"CWRU TRAIN classes {sorted(set(cw['class']))} != {classes}"
    per_spec = cw.groupby("physical_specimen")["class"].nunique(); assert (per_spec == 1).all(), "a physical specimen carries more than one class"
    for c, g in cw.groupby("class"):
        assert g.physical_specimen.nunique() >= 2, f"class {c} has fewer than two TRAIN specimens"
    specs = sorted(cw.physical_specimen.unique()); spec_index = {s: i for i, s in enumerate(specs)}
    return {"window_ids": list(cw.window_id), "labels": [classes.index(c) for c in cw["class"]], "specimens": list(cw.physical_specimen), "spec_index": spec_index,
            "class_names": classes, "n_windows": len(cw), "windows_per_class_specimen": {f"{c}|{s}": int(n) for (c, s), n in cw.groupby(["class", "physical_specimen"]).size().items()}}


def balanced_stream(stream: list[list[tuple]], universe: dict, steps_per_epoch: int, epochs: int, seed: int) -> tuple[list[list[tuple]], dict]:
    """C1/C2 stream: the ORIGINAL mixed stream is consumed first (RNG progression and all JNU/HIT/MaFaulDa items preserved);
    only the CWRU positions of each batch are replaced by the deterministic class/specimen-balanced plan of that epoch."""
    assert len(stream) == epochs * steps_per_epoch
    idx = list(range(universe["n_windows"])); wid = universe["window_ids"]; cls = universe["class_names"]; labels = universe["labels"]
    out, plan_hash = [], hashlib.sha256(); n_replaced = 0
    for ep in range(epochs):
        plan = make_epoch_plan(idx, labels, universe["specimens"], batch_size=CWRU_ITEMS_PER_STEP, steps=steps_per_epoch, seed=seed, epoch=ep)
        for k, batch_plan in enumerate(plan):
            base = stream[ep * steps_per_epoch + k]; positions = [i for i, it in enumerate(base) if it[0] == "CWRU"]
            assert len(positions) == CWRU_ITEMS_PER_STEP, f"expected {CWRU_ITEMS_PER_STEP} CWRU items per step, got {len(positions)}"
            repl = [("CWRU", cls[labels[j]], wid[j]) for j in batch_plan]
            out.append(replace_positions(base, positions, repl)); n_replaced += len(positions)
            plan_hash.update(("|".join(map(str, batch_plan)) + "\n").encode())
    return out, {"plan_version": CWRU_PLAN_VERSION, "plan_sha256": plan_hash.hexdigest(), "n_cwru_items_replaced": n_replaced, "steps_per_epoch": steps_per_epoch, "epochs": epochs, "seed": seed}


def non_cwru_preserved(original: list[list[tuple]], replaced: list[list[tuple]]) -> bool:
    return len(original) == len(replaced) and all(len(a) == len(b) and all(x == y for x, y in zip(a, b) if x[0] != "CWRU") and [x[0] for x in a] == [y[0] for y in b] for a, b in zip(original, replaced))


def cwru_group_tensors(batch_items: list[tuple], universe: dict, device) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    """Positions of the CWRU items in a micro-batch chunk plus their class and specimen id tensors (torch.long)."""
    wid_to_spec = dict(zip(universe["window_ids"], universe["specimens"])); cls = universe["class_names"]
    pos = [i for i, it in enumerate(batch_items) if it[0] == "CWRU"]
    labels = torch.tensor([cls.index(batch_items[i][1]) for i in pos], dtype=torch.long, device=device)
    specs = torch.tensor([universe["spec_index"][wid_to_spec[batch_items[i][2]]] for i in pos], dtype=torch.long, device=device)
    return pos, labels, specs


def pairing_manifest(universe: dict, plan_info: dict, fold: int) -> pd.DataFrame:
    """TRAIN-only pairing manifest: every CWRU TRAIN window with its class and physical specimen (the positive/negative structure)."""
    cls_of = [universe["class_names"][l] for l in universe["labels"]]; spec_per_class = defaultdict(set)
    for c, s in zip(cls_of, universe["specimens"]):
        spec_per_class[c].add(s)
    return pd.DataFrame({"fold": fold, "window_id": universe["window_ids"], "class": cls_of, "physical_specimen": universe["specimens"],
                         "n_train_specimens_in_class": [len(spec_per_class[c]) for c in cls_of], "n_positive_specimens": [len(spec_per_class[c]) - 1 for c in cls_of],
                         "plan_sha256": plan_info["plan_sha256"], "plan_version": plan_info["plan_version"]})
