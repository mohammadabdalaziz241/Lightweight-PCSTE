"""Fail-closed guards for Part 6.

* TEST guards: every Stage 0-4 entry point that touches window ids or a
  manifest passes through `assert_no_test_windows` / `train_val_only`;
  Part-3B window ids embed the split name (f{fold}:{ds}:...:{split}:...),
  so the check is structural, not a convention. TEST access is possible
  ONLY through a `TestSessionToken` minted by test_policy.open_test_session
  (Stage 5).
* Checkpoint provenance: primary checkpoints are opened read-only, their
  sha256 verified against the primary run's own test_seal.json (which
  holds the checkpoint hash but NO test metric — reading it never
  exposes TEST results), fold/seed/arm parsed from the run id and
  cross-checked with the checkpoint payload.
* Same-fold rule: any teacher/initialisation checkpoint used for cell
  (fold, seed) must carry the same fold; cross-fold use raises.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from ..integrity import sha256_file
from ..part3b_windows import PART3B_DIR
from .protocol import (FOLDS, PRIMARY_DOWNSTREAM, PRIMARY_SSL, SEEDS)


class Part6GuardError(AssertionError):
    """Raised loudly on any Part-6 protocol violation (fail closed)."""


# ---------------------------------------------------------------------------
# TEST guards
# ---------------------------------------------------------------------------
_WID_RE = re.compile(r"^f(?P<fold>\d):(?P<ds>[A-Z]+):.*?:(?P<split>train|validation|test):\d+-\d+$")


def split_of_window_id(window_id: str) -> str:
    m = _WID_RE.match(str(window_id))
    if not m:
        raise Part6GuardError(f"unparseable Part-3B window id: {window_id!r}")
    return m.group("split")


def fold_of_window_id(window_id: str) -> int:
    m = _WID_RE.match(str(window_id))
    if not m:
        raise Part6GuardError(f"unparseable Part-3B window id: {window_id!r}")
    return int(m.group("fold"))


def assert_no_test_windows(window_ids, context: str = "") -> None:
    """Refuse ANY test-split window id (Stages 0-4)."""
    bad = [w for w in window_ids if split_of_window_id(w) == "test"]
    if bad:
        raise Part6GuardError(
            f"TEST windows are forbidden here ({context or 'stage 0-4'}): "
            f"{len(bad)} offending ids, e.g. {bad[0]}")


def train_val_only(manifest: pd.DataFrame, context: str = "") -> pd.DataFrame:
    """Return the TRAIN+VALIDATION view of a fold manifest; the returned
    frame structurally cannot contain test rows."""
    if "split" not in manifest.columns:
        raise Part6GuardError("manifest lacks a split column")
    view = manifest[manifest["split"].isin(["train", "validation"])].copy()
    if (view["split"] == "test").any():         # pragma: no cover
        raise Part6GuardError("test rows survived the filter")
    ids = view.index if view.index.name == "window_id" else view["window_id"]
    assert_no_test_windows(list(ids), context or "train_val_only")
    return view


def load_fold_manifest(fold: int, allow_test: bool = False,
                       token: "TestSessionToken | None" = None
                       ) -> pd.DataFrame:
    """Sealed Part-3B manifest of a fold, TRAIN+VAL only unless a valid
    Stage-5 token is presented."""
    if fold not in FOLDS:
        raise Part6GuardError(f"fold {fold} not in {FOLDS}")
    man = pd.read_csv(PART3B_DIR / f"window_manifest_fold_{fold}.csv"
                      ).set_index("window_id")
    if allow_test:
        if token is None or not token.valid_for(fold):
            raise Part6GuardError(
                "TEST manifest access requires a valid TestSessionToken "
                "(Stage 5 sealed session only)")
        return man
    return train_val_only(man, f"fold {fold} manifest")


@dataclass(frozen=True)
class TestSessionToken:
    """Minted only by test_policy.open_test_session after the pre-test
    ledger is committed and test_seal.json is written. Never construct
    by hand outside that function (tests use it deliberately)."""
    session_id: str
    seal_sha256: str
    folds: tuple

    def valid_for(self, fold: int) -> bool:
        return bool(self.session_id) and bool(self.seal_sha256) \
            and fold in self.folds


# ---------------------------------------------------------------------------
# primary checkpoint provenance
# ---------------------------------------------------------------------------
_RUN_RE = re.compile(r"^(?P<arm>s0|s1)_f(?P<fold>\d)_s(?P<seed>\d+)_l(?P<pct>\d{3})$")
_SSL_RE = re.compile(r"^ssl_f(?P<fold>\d)_s(?P<seed>\d+)$")


@dataclass(frozen=True)
class CheckpointRef:
    run_id: str
    kind: str            # "downstream" | "ssl"
    arm: str             # "s0" | "s1" | "ssl"
    fold: int
    seed: int
    label_pct: int       # 100 for primary; 10 for F1 teachers; 0 for ssl
    path: Path
    sha256: str
    best_epoch: int | None
    best_val_macro_f1: float | None

    @property
    def cell(self) -> tuple[int, int]:
        return (self.fold, self.seed)


def primary_run_id(arm: str, fold: int, seed: int, pct: int = 100) -> str:
    return f"{arm}_f{fold}_s{seed}_l{pct:03d}"


def parse_run_id(run_id: str) -> dict:
    m = _RUN_RE.match(run_id)
    if m:
        return {"kind": "downstream", "arm": m.group("arm"),
                "fold": int(m.group("fold")), "seed": int(m.group("seed")),
                "pct": int(m.group("pct"))}
    m = _SSL_RE.match(run_id)
    if m:
        return {"kind": "ssl", "arm": "ssl", "fold": int(m.group("fold")),
                "seed": int(m.group("seed")), "pct": 0}
    raise Part6GuardError(f"unknown primary run id {run_id!r}")


def resolve_checkpoint(run_id: str, root: Path | None = None,
                       require_complete: bool = True) -> CheckpointRef:
    """Locate a primary best.pt, verify it hashes to the value recorded
    by the primary run itself (test_seal.json for downstream — which
    contains NO test metric — or completion.json for SSL). Never reads
    test_report.json."""
    info = parse_run_id(run_id)
    if info["fold"] not in FOLDS or info["seed"] not in SEEDS:
        raise Part6GuardError(f"{run_id}: fold/seed outside registry")
    if root is None:
        root = PRIMARY_DOWNSTREAM if info["kind"] == "downstream" else PRIMARY_SSL
    d = Path(root) / run_id
    st_p = d / "state.json"
    if not st_p.exists():
        raise Part6GuardError(f"{run_id}: no state.json (run not started)")
    st = json.loads(st_p.read_text())
    if require_complete and st.get("status") != "COMPLETE":
        raise Part6GuardError(f"{run_id}: status={st.get('status')} != COMPLETE")
    bp = d / "best.pt"
    if not bp.exists():
        raise Part6GuardError(f"{run_id}: best.pt missing")
    if info["kind"] == "downstream":
        seal_p = d / "test_seal.json"
        if not seal_p.exists():
            raise Part6GuardError(f"{run_id}: test_seal.json missing — "
                                  "checkpoint not frozen yet")
        seal = json.loads(seal_p.read_text())
        if "macro_domain_f1_test" in seal:      # pragma: no cover
            raise Part6GuardError("test_seal.json unexpectedly carries a "
                                  "TEST metric — refusing to read further")
        expected = seal["best_checkpoint_sha256"]
        best_epoch = seal.get("best_epoch")
        best_val = seal.get("best_val_macro_f1")
    else:
        comp = json.loads((d / "completion.json").read_text())
        expected = comp["best_checkpoint_sha256"]
        best_epoch = comp.get("best_epoch")
        best_val = None
    got = sha256_file(bp)
    if got != expected:
        raise Part6GuardError(
            f"{run_id}: best.pt sha256 {got[:12]} != recorded {expected[:12]} "
            "(fail closed)")
    return CheckpointRef(run_id=run_id, kind=info["kind"], arm=info["arm"],
                         fold=info["fold"], seed=info["seed"],
                         label_pct=info["pct"], path=bp, sha256=got,
                         best_epoch=best_epoch, best_val_macro_f1=best_val)


def load_checkpoint_payload(ref: CheckpointRef) -> dict:
    """Read-only torch.load with payload/identity cross-checks."""
    ck = torch.load(ref.path, map_location="cpu", weights_only=False)
    if ck.get("run_id") != ref.run_id:
        raise Part6GuardError(f"{ref.run_id}: payload run_id {ck.get('run_id')}")
    if ref.kind == "downstream":
        for k in ("encoder", "heads"):
            if k not in ck:
                raise Part6GuardError(f"{ref.run_id}: payload lacks {k!r}")
        if ref.best_epoch is not None and int(ck["epoch"]) != int(ref.best_epoch):
            raise Part6GuardError(f"{ref.run_id}: payload epoch != sealed epoch")
    else:
        if "encoder" not in ck:
            raise Part6GuardError(f"{ref.run_id}: SSL payload lacks encoder")
    return ck


def assert_same_fold(cell_fold: int, refs: list[CheckpointRef],
                     context: str = "") -> None:
    bad = [r.run_id for r in refs if r.fold != cell_fold]
    if bad:
        raise Part6GuardError(
            f"cross-fold checkpoint use is forbidden ({context}): cell fold "
            f"{cell_fold} but {bad}")


def assert_same_cell(fold: int, seed: int, ref: CheckpointRef,
                     context: str = "") -> None:
    if ref.fold != fold or ref.seed != seed:
        raise Part6GuardError(
            f"{context}: checkpoint {ref.run_id} is not the same-cell "
            f"checkpoint of (fold {fold}, seed {seed})")


def assert_read_only_primary(path: Path) -> None:
    """Part 6 must never write under the primary result trees."""
    p = Path(path).resolve()
    for forbidden in (PRIMARY_DOWNSTREAM.resolve(), PRIMARY_SSL.resolve()):
        if forbidden == p or forbidden in p.parents:
            raise Part6GuardError(f"refusing to write inside primary tree: {p}")
