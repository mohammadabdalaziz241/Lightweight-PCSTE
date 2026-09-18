"""Mixed-dataset batch collation for frozen Part-4C representations.

Minimal completion padding + explicit masks only. No interpolation, no
spectral resizing, no fabricated frequency content: a HIT sample in a
mixed batch is zero-padded to the batch tensor shape for mechanical
stacking, but every padded cell is masked and (test-proven) cannot
influence any output.
"""
from __future__ import annotations

import numpy as np
import torch


def collate_representations(items: list[tuple[np.ndarray, np.ndarray,
                                              np.ndarray]]) -> dict:
    """items: [(tensor (bins, frames) float32, frequency_hz (bins,),
    time_seconds (frames,)), ...] — e.g. straight from
    part4c_reader.get_representation output + meta."""
    max_b = max(x.shape[0] for x, _, _ in items)
    max_t = max(x.shape[1] for x, _, _ in items)
    n = len(items)
    spec = torch.zeros(n, max_b, max_t, dtype=torch.float32)
    mask = torch.zeros(n, max_b, max_t, dtype=torch.bool)
    freq = torch.zeros(n, max_b, dtype=torch.float32)
    time = torch.zeros(n, max_t, dtype=torch.float32)
    for i, (x, f, t) in enumerate(items):
        b, tt = x.shape
        spec[i, :b, :tt] = torch.as_tensor(np.array(x, dtype=np.float32))
        mask[i, :b, :tt] = True
        freq[i, :b] = torch.as_tensor(np.array(f, dtype=np.float32))
        time[i, :tt] = torch.as_tensor(np.array(t, dtype=np.float32))
    return {"spec": spec, "cell_mask": mask,
            "frequency_hz": freq, "time_seconds": time}
