"""Shared atomic work queue for multi-machine Stage-3 execution.

All workers (otter135/133/134) consume the SAME sealed registry over the
shared filesystem. Claiming is hardlink-based — `os.link(tmp, lock)` is
atomic on local filesystems AND on NFS (the classic NFS-safe protocol),
so two workers can never own the same run:

  claim   : write a unique temp file containing a random token, then
            os.link(tmp, queue/<run_id>.lock). Exactly one link() ever
            succeeds; the winner re-reads the lock and verifies its own
            token before proceeding (paranoia against exotic FS races).
  heartbeat: the owner (worker process every ~60 s AND the training
            process at every epoch boundary) rewrites the lock via
            atomic os.replace, preserving the token. A mismatching token
            on heartbeat means the lock was broken -> the trainer aborts
            loudly (this kills any orphaned run whose lock was reclaimed).
  stale   : heartbeat older than STALE_AFTER_S (default 1800 s; epochs
            are ~300-500 s, so a live run can never look stale).
  reclaim : os.rename(lock, queue/<run_id>.stale.<ts>.<host>.<pid>) —
            rename succeeds for exactly ONE contender; the loser gets
            FileNotFoundError and walks away. The winner then claims
            normally (racing any third worker through link()).
  release : owner-verified unlink + append-only history record
            (queue/<run_id>.history.jsonl) with the outcome.
  failed  : a run whose claim ended 'failed' (2 in-claim attempts) is
            skipped by workers until a human clears the record.

Nothing here trains, loads TEST, or reads checkpoints.
"""
from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path

from .guards import Part6GuardError
from .protocol import PART6_RESULTS

HEARTBEAT_INTERVAL_S = 60
STALE_AFTER_S = 1800
CLAIM_TOKEN_ENV = "PART6_CLAIM_TOKEN"


def queue_dir(root: Path | None = None) -> Path:
    return (root or PART6_RESULTS) / "queue"


def lock_path(run_id: str, root: Path | None = None) -> Path:
    return queue_dir(root) / f"{run_id}.lock"


def history_path(run_id: str, root: Path | None = None) -> Path:
    return queue_dir(root) / f"{run_id}.history.jsonl"


def _now() -> float:
    return time.time()


def read_lock(run_id: str, root: Path | None = None) -> dict | None:
    p = lock_path(run_id, root)
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


@dataclass(frozen=True)
class ClaimHandle:
    run_id: str
    token: str
    root: Path | None = None


def claim_run(run_id: str, meta: dict, root: Path | None = None
              ) -> ClaimHandle | None:
    """Atomically claim `run_id`. Returns a handle or None if already
    claimed. `meta` (host, gpu, pid, git commit, registry hash, ...) is
    stored verbatim in the lock."""
    qd = queue_dir(root)
    qd.mkdir(parents=True, exist_ok=True)
    token = os.urandom(16).hex()
    rec = {"run_id": run_id, "token": token,
           "host": socket.gethostname(), "pid": os.getpid(),
           "claimed_at": _now(), "heartbeat_at": _now(), **meta}
    tmp = qd / f".claim.{run_id}.{token}"
    tmp.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str))
    try:
        os.link(tmp, lock_path(run_id, root))
    except FileExistsError:
        tmp.unlink(missing_ok=True)
        return None
    finally:
        tmp.unlink(missing_ok=True)
    back = read_lock(run_id, root)
    if back is None or back.get("token") != token:      # pragma: no cover
        return None
    return ClaimHandle(run_id=run_id, token=token, root=root)


def _verify_owner(run_id: str, token: str, root: Path | None) -> dict:
    d = read_lock(run_id, root)
    if d is None:
        raise Part6GuardError(f"{run_id}: lock vanished — ownership lost")
    if d.get("token") != token:
        raise Part6GuardError(
            f"{run_id}: lock token mismatch — this claim was broken/"
            "reclaimed; abort immediately (never train without the lock)")
    return d


def heartbeat(handle_or_run_id, token: str | None = None,
              root: Path | None = None, extra: dict | None = None) -> None:
    """Owner-verified heartbeat (atomic replace). Accepts a ClaimHandle
    or (run_id, token) — the latter is what the training process uses via
    the CLAIM_TOKEN_ENV environment variable."""
    if isinstance(handle_or_run_id, ClaimHandle):
        run_id, token, root = (handle_or_run_id.run_id,
                               handle_or_run_id.token,
                               handle_or_run_id.root)
    else:
        run_id = handle_or_run_id
    d = _verify_owner(run_id, token, root)
    d["heartbeat_at"] = _now()
    if extra:
        d.update(extra)
    tmp = queue_dir(root) / f".hb.{run_id}.{token}"
    tmp.write_text(json.dumps(d, indent=1, sort_keys=True, default=str))
    os.replace(tmp, lock_path(run_id, root))


def is_stale(lock_data: dict, now: float | None = None,
             stale_after_s: float = STALE_AFTER_S) -> bool:
    return (now or _now()) - float(lock_data.get("heartbeat_at", 0)) \
        > stale_after_s


def reclaim_stale(run_id: str, root: Path | None = None,
                  stale_after_s: float = STALE_AFTER_S) -> bool:
    """Break a STALE lock. Exactly one contender wins the rename; the
    caller must then claim_run() normally. Returns True iff this process
    broke the lock."""
    d = read_lock(run_id, root)
    if d is None or not is_stale(d, stale_after_s=stale_after_s):
        return False
    dest = queue_dir(root) / (f"{run_id}.stale.{int(_now())}."
                              f"{socket.gethostname()}.{os.getpid()}")
    try:
        os.rename(lock_path(run_id, root), dest)
    except FileNotFoundError:
        return False
    rec = dict(d)
    rec.update({"broken_at": _now(), "broken_by_host": socket.gethostname(),
                "broken_by_pid": os.getpid(), "event": "stale_lock_broken"})
    with open(history_path(run_id, root), "a") as f:
        f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
    return True


def release(handle: ClaimHandle, outcome: str, extra: dict | None = None
            ) -> None:
    """Owner-verified release. outcome: 'complete' | 'failed'."""
    if outcome not in ("complete", "failed"):
        raise Part6GuardError(f"unknown outcome {outcome}")
    d = _verify_owner(handle.run_id, handle.token, handle.root)
    d.update({"released_at": _now(), "event": outcome, **(extra or {})})
    with open(history_path(handle.run_id, handle.root), "a") as f:
        f.write(json.dumps(d, sort_keys=True, default=str) + "\n")
    os.unlink(lock_path(handle.run_id, handle.root))


def failure_recorded(run_id: str, root: Path | None = None) -> bool:
    """True if any claim on this run ended 'failed' (workers then skip it
    until a human clears/renames the history record)."""
    p = history_path(run_id, root)
    if not p.exists():
        return False
    return any(json.loads(ln).get("event") == "failed"
               for ln in p.read_text().splitlines() if ln.strip())


def janitor_archive_stale_complete(run_id: str, root: Path | None = None
                                   ) -> bool:
    """Archive a STALE lock left behind on a COMPLETE run (e.g. the
    owning worker died after the run finished). The caller must have
    verified state == COMPLETE. Never touches a fresh lock."""
    d = read_lock(run_id, root)
    if d is None or not is_stale(d):
        return False
    dest = queue_dir(root) / (f"{run_id}.released-janitor.{int(_now())}."
                              f"{socket.gethostname()}.{os.getpid()}")
    try:
        os.rename(lock_path(run_id, root), dest)
    except FileNotFoundError:
        return False
    rec = dict(d)
    rec.update({"event": "janitor_archived_after_complete",
                "archived_at": _now(),
                "archived_by_host": socket.gethostname()})
    with open(history_path(run_id, root), "a") as f:
        f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
    return True


def epoch_heartbeat_from_env(run_id: str, root: Path | None = None,
                             extra: dict | None = None) -> bool:
    """Called by the training process at each epoch boundary when it was
    launched by a worker (token in the environment). Verifies ownership
    and refreshes the heartbeat; raises if the lock was broken. Returns
    False (no-op) when not running under a worker claim."""
    token = os.environ.get(CLAIM_TOKEN_ENV)
    if not token:
        return False
    heartbeat(run_id, token=token, root=root, extra=extra)
    return True
