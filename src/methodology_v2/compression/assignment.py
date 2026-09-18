"""Static per-host execution assignment for the sealed Stage-3 registry.

EXECUTION METADATA ONLY: this layer schedules WHERE each sealed run
executes; it never alters the sealed registry, architectures, seeds,
losses, checkpoints or statistics (the sealed part6_hashes.csv covers
none of these files and stays untouched).

Deterministic balanced rule (a Latin-square over the 3x3x3 core):

    host = HOSTS[(ARM_INDEX[arm] + (fold - 1) + SEED_INDEX[seed]) mod 3]

with HOSTS = (otter135, otter134, otter133), ARM_INDEX k1=0 / c_small=1 /
k0=2, SEED_INDEX 42=0 / 1337=1 / 2026=2. Properties (test-enforced):
every host receives exactly 9 runs = 3 K1 + 3 C_small + 3 K0, and within
EACH arm one run per fold and one per seed — so each host also carries
each fold exactly 3x and each seed exactly 3x; hostname is uncorrelated
with arm, fold and seed. k1_f1_s42 lands on otter135 (already running
there when the rule was fixed).

The provenance file (execution_assignment.json) records the rule, the
per-host lists and the registry/seal hashes it was derived from; the
loader re-derives the rule and fails closed on ANY mismatch, so the file
cannot drift from the rule or the sealed registry.

Redistribution policy: workers only ever claim runs assigned to their
own host (stale-lock reclaim included). Moving runs between hosts
requires an explicit researcher authorization recorded in a NEW
assignment file — no worker does it automatically.
"""
from __future__ import annotations

import json
import socket
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .guards import Part6GuardError
from .protocol import PART6_DIR

HOSTS = ("otter135", "otter134", "otter133")
ARM_INDEX = {"k1": 0, "c_small": 1, "k0": 2}
SEED_INDEX = {42: 0, 1337: 1, 2026: 2}
ASSIGNMENT_FILE = "execution_assignment.json"


def local_host() -> str:
    return socket.gethostname().split(".")[0]


def host_for(arm: str, fold: int, seed: int) -> str:
    if arm not in ARM_INDEX:
        raise Part6GuardError(f"no assignment rule for arm {arm!r}")
    if int(seed) not in SEED_INDEX:
        raise Part6GuardError(f"unknown seed {seed}")
    return HOSTS[(ARM_INDEX[arm] + (int(fold) - 1) + SEED_INDEX[int(seed)]) % 3]


def build_assignment(enabled_rows: pd.DataFrame) -> dict[str, list[str]]:
    """{host: [run_id, ...]} preserving registry row order."""
    out: dict[str, list[str]] = {h: [] for h in HOSTS}
    for _, r in enabled_rows.iterrows():
        out[host_for(r["arm"], int(r["fold"]), int(r["seed"]))].append(
            str(r["run_id"]))
    return out


def assignment_path(base: Path | None = None) -> Path:
    return (base or PART6_DIR) / ASSIGNMENT_FILE


def write_assignment(enabled_rows: pd.DataFrame, registry_sha256: str,
                     part6_master_hash: str, base: Path | None = None) -> Path:
    hosts = build_assignment(enabled_rows)
    doc = {
        "purpose": "EXECUTION METADATA ONLY — schedules where each sealed "
                   "run executes; the sealed scientific registry is not "
                   "modified by this file",
        "rule": "host = HOSTS[(ARM_INDEX[arm] + fold - 1 + SEED_INDEX[seed]) "
                "mod 3]",
        "host_order": list(HOSTS),
        "arm_index": ARM_INDEX,
        "seed_index": {str(k): v for k, v in SEED_INDEX.items()},
        "hosts": hosts,
        "runs_per_host": {h: len(v) for h, v in hosts.items()},
        "registry_sha256": registry_sha256,
        "part6_master_hash": part6_master_hash,
        "redistribution_policy": "workers claim (and stale-reclaim) ONLY "
                                 "their own host's runs; moving runs between "
                                 "hosts requires an explicit researcher "
                                 "authorization recorded in a new assignment "
                                 "file",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    p = assignment_path(base)
    p.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return p


def load_assignment(enabled_rows: pd.DataFrame, registry_sha256: str,
                    part6_master_hash: str, base: Path | None = None) -> dict:
    """Load + verify fail-closed: hashes must match the sealed registry
    and the per-host lists must equal the deterministic rule exactly."""
    p = assignment_path(base)
    if not p.exists():
        raise Part6GuardError(
            f"execution assignment missing ({p}) — write it with the "
            "'write-assignment' command before starting a worker")
    doc = json.loads(p.read_text())
    if doc.get("registry_sha256") != registry_sha256:
        raise Part6GuardError("assignment file was derived from a DIFFERENT "
                              "registry (sha mismatch) — fail closed")
    if doc.get("part6_master_hash") != part6_master_hash:
        raise Part6GuardError("assignment file part6 master hash mismatch")
    expected = build_assignment(enabled_rows)
    if doc.get("hosts") != expected:
        raise Part6GuardError(
            "assignment file lists do not match the deterministic rule over "
            "the sealed registry (hand-edited?) — fail closed")
    n = sum(len(v) for v in expected.values())
    if n != len(enabled_rows) or sorted(
            r for v in expected.values() for r in v) != sorted(
            enabled_rows["run_id"].astype(str)):
        raise Part6GuardError("assignment does not cover the enabled rows "
                              "exactly once")
    return doc
