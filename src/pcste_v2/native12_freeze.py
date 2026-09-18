"""Protocol freeze digest and launch preflight for pcste_v2_cwru_native12_specimen_v1.

The freeze digest is the SHA-256 of a deterministic bundle index (sorted "relative_path sha256" lines) over: the native12 CWRU protocol
directory, the global native12 protocol directory (manifests, normalisers, registries), the frozen prospective criteria and run
assignment, the official source inventory, and the code files that define the representation, sampler, model and executor.
Excluded from the digest input: PROTOCOL_STATUS.json, FREEZE_DIGEST.json (the digest file itself), timestamps and machine-specific
absolute paths (the index stores repository-relative paths only). A git commit, the inventory hash or the package hash is NOT the digest.
"""
from __future__ import annotations

import hashlib, json
from pathlib import Path

from src.methodology_v2.registry import REPO_ROOT

CWRU_DIR = "pcste_v2/protocol/cwru_native12_specimen_v1"
GLOBAL_DIR = "pcste_v2/protocol/global_v2_native12_MAFv2_v1"
MIG_DIR = "analysis/pcste_v2_cwru_native12_migration"
STATUS_FILE = "PROTOCOL_STATUS.json"; DIGEST_FILE = "FREEZE_DIGEST.json"; INDEX_FILE = "FREEZE_BUNDLE_INDEX.txt"
EXCLUDED_NAMES = {STATUS_FILE, DIGEST_FILE, INDEX_FILE}
CODE_FILES = ["scripts/pcste_v2/run.py", "scripts/pcste_v2/fit_normalizers.py"]
CODE_GLOBS = [("src/pcste_v2", "*.py"), ("src/methodology_v2/encoder", "*.py"), ("src/methodology_v2/experiment", "*.py")]
FROZEN_INPUTS = [f"{MIG_DIR}/NATIVE12_PROSPECTIVE_CRITERIA.json", f"{MIG_DIR}/NATIVE12_RUN_ASSIGNMENT.csv", f"{MIG_DIR}/OFFICIAL_NATIVE12_SOURCE_INVENTORY.json"]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def bundle_index(repo: Path = REPO_ROOT, cwru_dir: str = CWRU_DIR, global_dir: str = GLOBAL_DIR, frozen_inputs: list[str] | None = None) -> list[tuple[str, str]]:
    files: list[Path] = []
    for d in (cwru_dir, global_dir):
        files += [p for p in sorted((repo / d).rglob("*")) if p.is_file() and p.name not in EXCLUDED_NAMES]
    files += [repo / f for f in CODE_FILES] + [p for d, g in CODE_GLOBS for p in sorted((repo / d).glob(g))] + [repo / f for f in (FROZEN_INPUTS if frozen_inputs is None else frozen_inputs)]
    out = []
    for p in files:
        if not p.exists(): raise FileNotFoundError(f"freeze bundle input missing: {p.relative_to(repo)}")
        out.append((p.relative_to(repo).as_posix(), _sha(p)))
    return sorted(set(out))


def digest_of(index: list[tuple[str, str]]) -> str:
    return hashlib.sha256("".join(f"{path} {sha}\n" for path, sha in index).encode()).hexdigest()


def write_freeze(repo: Path = REPO_ROOT, status: str = "NOT_FROZEN", training_allowed: bool = False, note: str = "", cwru_dir: str = CWRU_DIR, global_dir: str = GLOBAL_DIR, frozen_inputs: list[str] | None = None, protocol_id: str = "pcste_v2_cwru_native12_specimen_v1") -> dict:
    idx = bundle_index(repo, cwru_dir, global_dir, frozen_inputs); dg = digest_of(idx); cd = repo / cwru_dir
    (cd / INDEX_FILE).write_text("".join(f"{path} {sha}\n" for path, sha in idx))
    (cd / DIGEST_FILE).write_text(json.dumps({"protocol_id": protocol_id, "freeze_digest_sha256": dg, "n_bundle_files": len(idx), "index_file": INDEX_FILE, "digest_input": "sorted 'relative_path sha256' lines of the bundle index (this file, PROTOCOL_STATUS.json and the index file excluded)"}, indent=1))
    (cd / STATUS_FILE).write_text(json.dumps({"protocol_id": protocol_id, "status": status, "training_allowed": training_allowed, "freeze_digest_sha256": dg, "note": note}, indent=1))
    return {"digest": dg, "n_files": len(idx)}


def read_status(repo: Path = REPO_ROOT, cwru_dir: str = CWRU_DIR) -> dict:
    return json.loads((repo / cwru_dir / STATUS_FILE).read_text())


def preflight(repo: Path, protocol_dir: Path, fold: int, authorize_digest: str | None, check_bytes: bool = True) -> dict:
    """Launch gate for native12 protocols. Raises PermissionError with the precise reason; returns the verified allowlist."""
    from src.pcste_v2.protocol_cwru_native12 import verify_any_native12, load_allowlist, assert_native12_source, NATIVE_RATE_HZ
    import pandas as pd
    info = json.loads((protocol_dir / "global_v2_hashes.json").read_text()); cwru_dir = repo / info["cwru_protocol_dir"]
    fi = info.get("freeze_inputs"); st = read_status(repo, info["cwru_protocol_dir"])
    if st.get("status") != "FROZEN": raise PermissionError(f"native12 preflight: protocol status is {st.get('status')} (must be FROZEN)")
    if not st.get("training_allowed", False): raise PermissionError("native12 preflight: training_allowed=false in PROTOCOL_STATUS.json")
    dg = digest_of(bundle_index(repo, info["cwru_protocol_dir"], str(protocol_dir.relative_to(repo)), fi))
    if dg != st.get("freeze_digest_sha256"): raise PermissionError(f"native12 preflight: freeze digest mismatch (bundle {dg[:12]} vs status {str(st.get('freeze_digest_sha256'))[:12]}): a frozen input changed")
    if authorize_digest != dg: raise PermissionError("native12 preflight: launch authorization absent or does not match the freeze digest (--authorize-launch <digest>)")
    verify_any_native12(cwru_dir); allow = load_allowlist(cwru_dir)
    man = pd.read_csv(protocol_dir / f"global_v2_fold_{fold}.csv", dtype={"fault_severity": str}); cw = man[man.dataset == "CWRU"]
    if not (cw.native_sampling_rate_hz == NATIVE_RATE_HZ).all(): raise PermissionError("native12 preflight: a CWRU manifest row is not 12000 Hz")
    for sf, var in cw[["source_file", "mat_variable"]].drop_duplicates().itertuples(index=False):
        assert_native12_source(sf, var, allow, check_bytes=check_bytes)
    check_registry(pd.read_csv(protocol_dir / "normalizers" / f"registry_fold_{fold}.csv"), info[f"fold_{fold}"]["sha256"], str(protocol_dir.relative_to(repo)))
    return allow


def check_registry(r, manifest_sha: str, manifest_dir: str = GLOBAL_DIR) -> None:
    """Normaliser registry gate: fitted on THIS manifest, files inside the native12 protocol directory, no 48 kHz key."""
    if not (r.manifest_sha256 == manifest_sha).all(): raise PermissionError("native12 preflight: normaliser registry was not fitted on this protocol's manifest (stale normaliser)")
    if not r.file.str.startswith(str(manifest_dir)).all(): raise PermissionError("native12 preflight: a normaliser file lies outside the native12 protocol directory")
    if "CWRU48" in set(r.key): raise PermissionError("native12 preflight: a 48 kHz CWRU normaliser is registered")
