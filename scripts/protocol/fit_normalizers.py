"""Fit versioned N2b normalisers (+ envelope cell statistics) for a global v2 protocol fold from TRAIN rows only."""
import argparse, json, sys, hashlib
from pathlib import Path
import pandas as pd
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from src.pcste_v2.representation import fit_normalizers, N2B_VERSION, N2B_FLOOR
from src.pcste_v2.protocol import verify_global
ap = argparse.ArgumentParser(); ap.add_argument("--protocol", required=True); ap.add_argument("--fold", type=int, required=True)
ap.add_argument("--grids", nargs="+", default=["v1", "mr_short", "mr_long"]); ap.add_argument("--maf-channels", type=int, default=1)
ap.add_argument("--no-envelope", action="store_true"); ap.add_argument("--workers", type=int, default=8)
a = ap.parse_args()
pdir = REPO / "pcste_v2/protocol" / a.protocol; sha = verify_global(pdir, a.fold)
man = pd.read_csv(pdir / f"global_v2_fold_{a.fold}.csv")
rows = fit_normalizers(pdir, a.fold, man, tuple(a.grids), {"MAFAULDA": a.maf_channels}, not a.no_envelope, a.workers)
reg = pdir / "normalizers" / f"registry_fold_{a.fold}.csv"
old = pd.read_csv(reg) if reg.exists() else pd.DataFrame()
new = pd.concat([old, pd.DataFrame(rows)]).drop_duplicates(["grid", "key", "channel"], keep="last").sort_values(["grid", "key", "channel"])
new["manifest_sha256"] = sha; new.to_csv(reg, index=False)
print(new[["grid", "key", "channel", "n_windows", "n_floored_bins", "min_std_raw", "sha256"]].to_string(index=False))
