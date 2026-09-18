"""Part-6 TEST policy: ONE sealed session at the very end.

Sequence (enforced, fail closed):
  1. all registered Part-6 runs COMPLETE (best.pt frozen by the
     validation rule) and PTQ variants prepared;
  2. `write_pre_test_ledger` -> pre_test_ledger.csv (run_id / model_id,
     best epoch, val MacroDomainF1, checkpoint sha256) — to be COMMITTED
     (git) before the session; `open_test_session` verifies the ledger's
     git blob is committed (HEAD contains it byte-identically) unless
     `require_committed=False` (tests only);
  3. `open_test_session` writes test_seal.json (ledger hash, model list,
     timestamp, git head) BEFORE any TEST window is loaded and returns a
     TestSessionToken — the ONLY object that unlocks TEST manifests
     (guards.load_fold_manifest(allow_test=True, token=...));
  4. every evaluation appends one row to test_touch_ledger.csv (model_id,
     checkpoint sha, fold, n_test_windows, timestamp, session id);
     `assert_touch_allowed` refuses a second evaluation of the same
     model_id unless an explicit documented integrity failure record
     exists (integrity_failures.json) for it;
  5. the session is closed with a summary; nothing may be added after.

Stages 0-4 never import a token: nothing there can evaluate TEST.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..integrity import sha256_file
from ..registry import REPO_ROOT
from .guards import Part6GuardError, TestSessionToken
from .protocol import FOLDS, PART6_DIR, PART6_RESULTS, PART6_VERSION

PRE_TEST_LEDGER = "pre_test_ledger.csv"
TEST_SEAL = "test_seal.json"
TOUCH_LEDGER = "test_touch_ledger.csv"
INTEGRITY_FAILURES = "integrity_failures.json"
TOUCH_COLUMNS = ["session_id", "model_id", "checkpoint_sha256", "fold",
                 "n_test_windows", "macro_domain_f1_test", "evaluated_at"]


def session_dir(root: Path | None = None) -> Path:
    return (root or PART6_RESULTS) / "test_session"


def write_pre_test_ledger(models: list[dict], out_dir: Path | None = None
                          ) -> Path:
    """models: [{model_id, run_id, variant ('fp32'|'q8'), best_epoch,
    val_macro_domain_f1, checkpoint_sha256, fold, seed}]. Written to the
    PROTOCOL directory (to be committed), not to results."""
    out_dir = out_dir or PART6_DIR
    if not models:
        raise Part6GuardError("empty pre-test ledger")
    df = pd.DataFrame(models)
    need = {"model_id", "run_id", "variant", "best_epoch",
            "val_macro_domain_f1", "checkpoint_sha256", "fold", "seed"}
    if not need <= set(df.columns):
        raise Part6GuardError(f"ledger columns missing: {need - set(df.columns)}")
    if df["model_id"].duplicated().any():
        raise Part6GuardError("duplicate model_id in pre-test ledger")
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / PRE_TEST_LEDGER
    df.sort_values("model_id").to_csv(p, index=False)
    return p


def _git_head(repo: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def _committed_identically(path: Path, repo: Path) -> bool:
    """True iff HEAD contains `path` with byte-identical content."""
    try:
        rel = str(Path(path).resolve().relative_to(repo.resolve()))
        blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=repo,
                              capture_output=True, check=True).stdout
        return hashlib.sha256(blob).hexdigest() == sha256_file(path)
    except Exception:
        return False


def open_test_session(ledger_path: Path | None = None,
                      out_root: Path | None = None,
                      require_committed: bool = True,
                      repo: Path | None = None) -> TestSessionToken:
    """Verify the pre-test ledger is committed, write test_seal.json
    BEFORE any TEST data is touched, mint the session token."""
    ledger_path = ledger_path or (PART6_DIR / PRE_TEST_LEDGER)
    repo = repo or REPO_ROOT
    if not ledger_path.exists():
        raise Part6GuardError("pre_test_ledger.csv missing — write and commit "
                              "it before opening the TEST session")
    if require_committed and not _committed_identically(ledger_path, repo):
        raise Part6GuardError("pre_test_ledger.csv is not committed "
                              "byte-identically at HEAD (fail closed)")
    sd = session_dir(out_root)
    if (sd / TEST_SEAL).exists():
        raise Part6GuardError("a Part-6 TEST session already exists — one "
                              "sealed session only (append-only ledger)")
    sd.mkdir(parents=True, exist_ok=True)
    ledger = pd.read_csv(ledger_path)
    ledger_hash = sha256_file(ledger_path)
    session_id = hashlib.sha256(
        f"{ledger_hash}|{datetime.now(timezone.utc).isoformat()}".encode()
    ).hexdigest()[:16]
    seal = {"part6_version": PART6_VERSION, "session_id": session_id,
            "pre_test_ledger_sha256": ledger_hash,
            "n_models": int(len(ledger)),
            "model_ids": sorted(ledger["model_id"].astype(str)),
            "git_head": _git_head(repo),
            "sealed_at": datetime.now(timezone.utc).isoformat(),
            "rule": "TEST windows are loaded only after this seal; every "
                    "model in the ledger is evaluated exactly once; every "
                    "touch is appended to test_touch_ledger.csv; nothing "
                    "is added or dropped afterwards"}
    (sd / TEST_SEAL).write_text(json.dumps(seal, indent=1, sort_keys=True))
    seal_hash = sha256_file(sd / TEST_SEAL)
    if not (sd / TOUCH_LEDGER).exists():
        pd.DataFrame(columns=TOUCH_COLUMNS).to_csv(sd / TOUCH_LEDGER, index=False)
    return TestSessionToken(session_id=session_id, seal_sha256=seal_hash,
                            folds=tuple(FOLDS))


def _touch_ledger(out_root: Path | None) -> pd.DataFrame:
    p = session_dir(out_root) / TOUCH_LEDGER
    if not p.exists():
        raise Part6GuardError("touch ledger missing — session not open")
    return pd.read_csv(p)


def assert_touch_allowed(model_id: str, token: TestSessionToken,
                         out_root: Path | None = None) -> None:
    sd = session_dir(out_root)
    if not (sd / TEST_SEAL).exists():
        raise Part6GuardError("no test seal — session not open")
    seal = json.loads((sd / TEST_SEAL).read_text())
    if seal["session_id"] != token.session_id:
        raise Part6GuardError("token does not belong to the open session")
    if model_id not in seal["model_ids"]:
        raise Part6GuardError(f"{model_id} is not in the sealed pre-test "
                              "ledger — nothing may be added")
    led = _touch_ledger(out_root)
    prior = led[led["model_id"] == model_id]
    if len(prior):
        fails = {}
        fp = sd / INTEGRITY_FAILURES
        if fp.exists():
            fails = json.loads(fp.read_text())
        if model_id not in fails:
            raise Part6GuardError(
                f"{model_id} was already evaluated on TEST in this session; a "
                "second evaluation requires a documented integrity failure "
                f"record in {INTEGRITY_FAILURES}")


def record_touch(model_id: str, checkpoint_sha256: str, fold: int,
                 n_test_windows: int, macro_domain_f1_test: float,
                 token: TestSessionToken, out_root: Path | None = None) -> None:
    sd = session_dir(out_root)
    row = {"session_id": token.session_id, "model_id": model_id,
           "checkpoint_sha256": checkpoint_sha256, "fold": fold,
           "n_test_windows": n_test_windows,
           "macro_domain_f1_test": macro_domain_f1_test,
           "evaluated_at": datetime.now(timezone.utc).isoformat()}
    led = _touch_ledger(out_root)
    led = pd.concat([led, pd.DataFrame([row])], ignore_index=True)
    led.to_csv(sd / TOUCH_LEDGER, index=False)


def document_integrity_failure(model_id: str, reason: str,
                               out_root: Path | None = None) -> None:
    sd = session_dir(out_root)
    fp = sd / INTEGRITY_FAILURES
    rec = json.loads(fp.read_text()) if fp.exists() else {}
    rec[model_id] = {"reason": reason,
                     "documented_at": datetime.now(timezone.utc).isoformat()}
    fp.write_text(json.dumps(rec, indent=1, sort_keys=True))


def close_test_session(token: TestSessionToken, out_root: Path | None = None
                       ) -> dict:
    sd = session_dir(out_root)
    seal = json.loads((sd / TEST_SEAL).read_text())
    led = _touch_ledger(out_root)
    evaluated = set(led["model_id"].astype(str))
    summary = {"session_id": token.session_id,
               "n_models_sealed": len(seal["model_ids"]),
               "n_touches": int(len(led)),
               "models_never_evaluated": sorted(set(seal["model_ids"]) - evaluated),
               "closed_at": datetime.now(timezone.utc).isoformat(),
               "touch_ledger_sha256": sha256_file(sd / TOUCH_LEDGER)}
    (sd / "session_summary.json").write_text(json.dumps(summary, indent=1,
                                                        sort_keys=True))
    return summary
