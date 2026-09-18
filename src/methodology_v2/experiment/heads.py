"""Dataset-specific downstream heads — single linear layers ONLY.

Class orders are FROZEN from the Part-1/2 taxonomy (verified against the
sealed manifests by tests, never assumed): CWRU uses the fault_type
field of its 3-class benchmark; JNU/HIT/MaFaulDa use original_label.
The identical head configuration serves S0 and S1; paired runs
initialize heads from the same deterministic head seed.
"""
from __future__ import annotations

import hashlib

import torch
import torch.nn as nn

D_MODEL = 192

CLASS_ORDERS: dict[str, tuple[str, ...]] = {
    "CWRU": ("inner_race", "outer_race", "ball"),
    "JNU": ("n", "ib", "ob", "tb"),          # Healthy/Inner/Outer/Roller
    "HIT": ("0", "1", "2"),                  # Healthy/Inner/Outer
    "MAFAULDA": ("normal", "imbalance", "horizontal-misalignment",
                 "vertical-misalignment",
                 "underhang/ball_fault", "underhang/cage_fault",
                 "underhang/outer_race",
                 "overhang/ball_fault", "overhang/cage_fault",
                 "overhang/outer_race"),
}

LABEL_FIELD = {"CWRU": "fault_type", "JNU": "original_label",
               "HIT": "original_label", "MAFAULDA": "original_label"}


def window_class(dataset: str, row) -> str:
    """Frozen mapping from a sealed window-manifest row to its class."""
    return str(row[LABEL_FIELD[dataset]])


def head_seed(fold: int, seed: int) -> int:
    """Deterministic shared head-init seed for a paired S0/S1 run."""
    d = hashlib.sha256(f"heads|{fold}|{seed}".encode()).digest()
    return int.from_bytes(d[:4], "little")


class DatasetHeads(nn.Module):
    """Four single linear heads over the shared 192-d global embedding.
    No hidden layers, no dataset-specific encoder branches."""

    def __init__(self, init_seed: int | None = None):
        super().__init__()
        if init_seed is not None:
            torch.manual_seed(init_seed)
        self.heads = nn.ModuleDict({
            ds: nn.Linear(D_MODEL, len(classes))
            for ds, classes in CLASS_ORDERS.items()})

    def forward(self, z: torch.Tensor, dataset: str) -> torch.Tensor:
        return self.heads[dataset](z)

    def config(self) -> dict:
        return {ds: {"in": D_MODEL, "out": len(c), "type": "Linear",
                     "classes": list(c)}
                for ds, c in CLASS_ORDERS.items()}
