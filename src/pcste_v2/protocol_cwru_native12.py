"""CWRU strictly native 12 kHz specimen-disjoint protocol: pcste_v2_cwru_native12_specimen_v1 (cwru_native12.specimen.v1).

Source eligibility is an explicit, byte-verified ALLOWLIST: the 105 acquisitions of the official "12k Drive End Bearing Fault Data"
and "12k Fan End Bearing Fault Data" catalogues (analysis/pcste_v2_cwru_native12_migration/OFFICIAL_NATIVE12_SOURCE_INVENTORY.json,
sha256 bd97cebc…), each matched to a local file by SHA-256. Nothing is discovered by directory scan. The official 48k DE fault
category (52 acquisitions), the 48k normal baselines 97-100.mat and every local alias / derivative are DENYLISTED and can never enter
a native12 manifest, normaliser, cache or initialisation. Fold roles are the current v2 P_B specimen roles (cwru_v2_PB_v1) restricted
to the native12 universe; specimen identities are the conservative end/class/diameter identities of the current registry (loads,
orientations, channels and repetitions do not create specimens). Sensor policy (unchanged, class-independent): the accelerometer on
the FAULTED bearing housing (DE_time for DE-fault acquisitions, FE_time for FE-fault acquisitions); fault_bearing_end and
sensor_location are recorded as separate facts. Windows: 1.0 s = 12000 samples, stride 0.5 s TRAIN / 1.0 s VAL and TEST, tail dropped,
identical to the v2 executor's CWRU12 setting (STFT grid v1: n_fft 256, hop 64). TEST access in the builder is limited to metadata,
bytes hashes, structural sample counts and deterministic window indices.
"""
from __future__ import annotations

import hashlib, json, re
from pathlib import Path

import numpy as np
import pandas as pd

from src.methodology_v2.registry import REPO_ROOT
from .protocol_cwru import FOLDS as PB_FOLDS, CLASS_OF, NOMINAL_RPM_BY_LOAD, WINDOW_S, STRIDE_S

PROTOCOL_ID = "pcste_v2_cwru_native12_specimen_v1"
PROTOCOL_VERSION = "cwru_native12.specimen.v1"
NATIVE_RATE_HZ = 12000
INVENTORY = REPO_ROOT / "analysis/pcste_v2_cwru_native12_migration/OFFICIAL_NATIVE12_SOURCE_INVENTORY.json"
INVENTORY_SHA256 = "bd97cebced8be3a1425ba5edbff116386edf7b63756282376d1c715a0a22f611"
LOCAL_ROOTS = ("data/raw", "data/raw_cwru_12k_fe", "data/raw_native12_official")          # where allowlisted bytes may live
DENY_ROOTS = ("data/raw_cwru_48k", "data/raw/normal")                                      # historical 48k namespace (never scanned for sources)
NORMAL_BASELINES = {"97.mat", "98.mat", "99.mat", "100.mat"}
CLASS_NAME = {"Ball": "ball", "InnerRace": "inner_race", "OuterRace": "outer_race"}
FOLDS = PB_FOLDS                                                                           # roles preserved from cwru_v2.P_B.v1
STFT_CWRU12 = {"grid": "v1", "n_fft": 256, "hop": 64, "window_samples": int(WINDOW_S * NATIVE_RATE_HZ), "transform": "log1p|STFT| (sealed rep_of)", "padding": "none (frames = 1 + (n - n_fft)//hop)"}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_inventory() -> dict:
    raw = INVENTORY.read_bytes(); assert hashlib.sha256(raw).hexdigest() == INVENTORY_SHA256, "official inventory hash mismatch (fail closed)"
    return json.loads(raw)


def local_hash_map(roots=LOCAL_ROOTS + DENY_ROOTS) -> dict[str, list[str]]:
    """sha256 -> list of local relative paths (all aliases), over the allow roots AND the historical 48k namespace (for denylist accounting)."""
    out: dict[str, list[str]] = {}
    for root in roots:
        for p in sorted((REPO_ROOT / root).rglob("*.mat")):
            out.setdefault(sha256_file(p), []).append(str(p.relative_to(REPO_ROOT)))
    return out


def _mat_header(path: Path) -> dict:
    import scipy.io as sio
    m = sio.loadmat(path); ks = [k for k in m if not k.startswith("__")]
    return {"vars": ks, "n": {k: int(m[k].shape[0]) for k in ks if k.endswith("_time")}, "rpm": (float(np.ravel(m[[k for k in ks if k.endswith("RPM")][0]])[0]) if any(k.endswith("RPM") for k in ks) else np.nan)}


def build_source_registry() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(allowlist registry of the 105 native12 acquisitions with local paths/aliases; denylist accounting of 48k sources and their local aliases)."""
    inv = load_inventory(); loc = local_hash_map(); rows, deny = [], []
    for f in inv["native12_files"]:
        paths = [p for p in loc.get(f["file_sha256"], []) if not p.startswith("data/raw_cwru_48k")]
        assert paths, f"official {f['official_filename']} ({f['official_label']}) not present locally: download from {f['source_url']} (fail closed)"
        primary = sorted(paths, key=lambda p: (not p.startswith("data/raw_native12_official"), p))[0]; hdr = _mat_header(REPO_ROOT / primary)
        end = f["fault_bearing_end"]; cls = CLASS_NAME[f["fault_class"]]; dia = int(f["fault_diameter_mil"]); loc_code = {"ball": "B", "inner_race": "IR", "outer_race": "OR"}[cls]
        spec = f"{end}_{loc_code}{dia:03d}"; assert spec == f["conservative_identity_candidate"]
        sensor = end                                                        # policy: accelerometer on the faulted bearing housing (class-independent)
        cands = [v for v in hdr["vars"] if v.endswith(f"_{sensor}_time")]; assert len(cands) == 1, (f["official_filename"], sensor, hdr["vars"])   # the 28-mil files carry internal ids (e.g. X056) that differ from the file number
        var = cands[0]; internal_id = var.split("_")[0]
        m = re.match(r"(IR|B|OR)(\d{3})(@\d+)?_(\d)", f["official_label"]); orient = f.get("outer_race_orientation_clock"); load = int(f["load_hp"])
        alias_mismatch = [p for p in paths if "@" in Path(p).name and orient is not None and f"@{int(orient)}_" not in Path(p).name]
        rpm = hdr["rpm"]; rpm_src = "file_RPM_variable" if np.isfinite(rpm) else "nominal_by_load(site_table)"
        rows.append(dict(official_file_number=int(f["official_file_number"]), official_filename=f["official_filename"], official_label=f["official_label"], source_category=f["source_category"], source_url=f["source_url"], source_page=f["source_page"],
                         sha256=f["file_sha256"], file_size_bytes=int(f["file_size_bytes"]), local_path=primary, local_aliases="|".join(paths), local_alias_label_mismatch="|".join(alias_mismatch),
                         recording_id=f"cwru_n12_{sensor}_{int(f['official_file_number'])}", internal_id=internal_id, physical_specimen=spec, group_id=f"cwrun12_{spec}",
                         fault_bearing_end=end, sensor_location=sensor, bearing_family="SKF_6205_DE" if end == "DE" else "SKF_6203_FE", bearing_factor_set="CWRU_DE_6205" if end == "DE" else "CWRU_FE_6203",
                         fault_type=cls, fault_severity=f"{dia} mil", or_clock_position=("" if orient is None or (isinstance(orient, float) and np.isnan(orient)) else str(int(orient))), load=f"{load}hp",
                         rpm=(rpm if np.isfinite(rpm) else NOMINAL_RPM_BY_LOAD[str(load)]), rpm_source=rpm_src, native_sampling_rate_hz=NATIVE_RATE_HZ, channel_rate_hz=NATIVE_RATE_HZ, mat_variable=var,
                         has_DE_signal="DE" in {c["sensor_location"] for c in f["channel_headers"]}, has_FE_signal="FE" in {c["sensor_location"] for c in f["channel_headers"]}, has_BA_signal="BA" in {c["sensor_location"] for c in f["channel_headers"]},
                         n_samples=hdr["n"][var], duration_s=round(hdr["n"][var] / NATIVE_RATE_HZ, 3), eligibility="NATIVE12_ALLOWLISTED", identity_status="conservative end/class/diameter identity (CWRU documents one test bearing per fault size and location; not serial-number certified)"))
    # denylist: the 52 official 48k DE fault acquisitions (not downloaded by the audit; matched locally by the MAT internal id X<file number>
    # in the historical 48k namespace) and the 48k normal baselines 97-100 with their Normal_* aliases (matched by hash duplicates).
    local48 = {}
    for h, paths in loc.items():
        for pth in paths:
            if pth.startswith("data/raw_cwru_48k") and "normal" not in pth.lower():
                hdr = _mat_header(REPO_ROOT / pth); ids = sorted({v.split("_")[0] for v in hdr["vars"] if v.endswith("_time")}); local48[pth] = (ids, h)
    for f in inv["excluded_official_48k_fault_files"]:
        num = int(f["official_file_number"]); paths = [pth for pth, (ids, h) in local48.items() if f"X{num:03d}" in ids or f"X{num}" in ids]
        if not paths:                                                   # CWRU quirk: a few files carry a foreign internal id; match the official label to the local stem
            paths = [pth for pth in local48 if Path(pth).stem == f["official_label"] + "_DE48k"]
        deny.append(dict(official_filename=f["official_filename"], official_label=f.get("official_label", ""), category="official_48k_DE_fault", official_file_number=num, local_aliases="|".join(sorted(paths)), local_sha256="|".join(sorted({local48[pth][1] for pth in paths})),
                         reason="48k DE source category: excluded as an entire acquisition (no channel salvage, no resampling, no relabelling)"))
    for h, paths in loc.items():
        if any("normal" in pth.lower() for pth in paths):
            deny.append(dict(official_filename="97-100.mat (48k normal baseline)", official_label="Normal_*", category="official_48k_normal_baseline", official_file_number=-1, local_aliases="|".join(sorted(paths)), local_sha256=h, reason="48k normal baseline (sealed registry documents 48k even in the 12k-named directory); no Healthy class"))
    unmatched48 = [pth for pth in local48 if not any(pth in (d["local_aliases"] or "") for d in deny)]
    for pth in unmatched48:
        deny.append(dict(official_filename="(local 48k file without official id match)", official_label="", category="historical_48k_namespace", official_file_number=-1, local_aliases=pth, local_sha256=local48[pth][1], reason="in the historical 48k namespace; never a native12 source"))
    D = pd.DataFrame(deny); R = pd.DataFrame(rows).sort_values(["physical_specimen", "or_clock_position", "load"]).reset_index(drop=True)
    deny_sha = {x for v in D.local_sha256.fillna("") for x in v.split("|") if x}
    assert len(R) == 105 and R.sha256.is_unique and R.recording_id.is_unique and not (set(R.sha256) & deny_sha) and not (set(R.local_path) & {a for v in D.local_aliases.fillna("") for a in v.split("|")}), "allowlist/denylist overlap or count mismatch"
    return R, D


def assign_split(spec: str, fold: int) -> str:
    if spec in FOLDS[fold]["test"]: return "test"
    if spec in FOLDS[fold]["validation"]: return "validation"
    return "train"


def build_windows(rec: pd.DataFrame, fold: int) -> tuple[pd.DataFrame, list[dict]]:
    """Deterministic window indices (structural only). Exclusion rules: policy channel missing -> acquisition excluded; n_samples < window -> excluded."""
    rows, excl = [], []; w = int(round(WINDOW_S * NATIVE_RATE_HZ))
    for r in rec.itertuples():
        split = assign_split(r.physical_specimen, fold); stride = int(round(STRIDE_S[split] * NATIVE_RATE_HZ))
        if r.n_samples < w:
            excl.append(dict(fold=fold, recording_id=r.recording_id, reason=f"n_samples {r.n_samples} < window {w}")); continue
        for s in range(0, r.n_samples - w + 1, stride):
            rows.append(dict(protocol_version=PROTOCOL_VERSION, fold_id=fold, dataset="CWRU", split=split, window_id=f"n12f{fold}:CWRU:{r.recording_id}:{r.sensor_location}:{split}:{s}-{s + w}", group_id=r.group_id,
                             physical_specimen=r.physical_specimen, recording_id=r.recording_id, source_file=r.local_path, mat_variable=r.mat_variable, source_sha256=r.sha256, official_file_number=r.official_file_number,
                             original_label=r.official_label.split("_")[0], fault_type=r.fault_type, fault_severity=r.fault_severity, bearing=r.fault_bearing_end, fault_bearing_end=r.fault_bearing_end, sensor_location=r.sensor_location,
                             bearing_factor_set=r.bearing_factor_set, channel=r.sensor_location, native_sampling_rate_hz=NATIVE_RATE_HZ, window_duration_seconds=WINDOW_S, stride_seconds=STRIDE_S[split], start_sample=s, end_sample=s + w,
                             rpm=r.rpm, load=r.load, or_clock_position=r.or_clock_position))
    return pd.DataFrame(rows), excl


def validate(rec: pd.DataFrame, win: dict[int, pd.DataFrame], deny: pd.DataFrame) -> list[dict]:
    out = []; deny_sha = {x for v in deny.local_sha256.fillna("") for x in v.split("|") if x}
    assert (rec.native_sampling_rate_hz == NATIVE_RATE_HZ).all() and (rec.channel_rate_hz == NATIVE_RATE_HZ).all() and not (set(rec.sha256) & deny_sha)
    for fold, w in win.items():
        assert (w.native_sampling_rate_hz == NATIVE_RATE_HZ).all() and (w.end_sample - w.start_sample == int(WINDOW_S * NATIVE_RATE_HZ)).all() and not (set(w.source_sha256) & deny_sha)
        for split in ("train", "validation", "test"):
            sp = w[w.split == split]
            for cls in ("inner_race", "ball", "outer_race"):
                c = sp[sp.fault_type == cls]; specs = sorted(c.physical_specimen.unique())
                out.append(dict(fold=fold, split=split, fault_type=cls, n_specimens=len(specs), specimens="|".join(specs), n_recordings=c.recording_id.nunique(), n_windows=len(c), sizes="|".join(sorted(c.fault_severity.unique())),
                                fault_ends="|".join(sorted(c.fault_bearing_end.unique())), sensors="|".join(sorted(c.sensor_location.unique())), loads="|".join(sorted(c.load.unique()))))
                assert len(specs) >= 1, f"fold {fold} {split} {cls}: class missing"
                if split == "train": assert len(specs) >= 2, f"fold {fold} TRAIN {cls}: fewer than two specimens"
        s_tr, s_va, s_te = (set(w[w.split == k].physical_specimen) for k in ("train", "validation", "test"))
        assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te) and (s_tr | s_va | s_te) == set(rec.physical_specimen)
        for k in ("recording_id", "source_sha256"):
            g = w.groupby(k).split.nunique(); assert (g == 1).all(), f"fold {fold}: a {k} appears in more than one split"
        assert w.window_id.is_unique
    tests = [set(FOLDS[f]["test"]) for f in FOLDS]; assert not (tests[0] & tests[1]) and not (tests[0] & tests[2]) and not (tests[1] & tests[2])
    return out


def de_fe_tables(rec: pd.DataFrame, win: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for fold, w in win.items():
        for meaning, col in (("fault_bearing_end", "fault_bearing_end"), ("sensor_location", "sensor_location")):
            for (split, cls, v), g in w.groupby(["split", "fault_type", col]):
                rows.append(dict(fold=fold, meaning=meaning, split=split, fault_type=cls, value=v, n_specimens=g.physical_specimen.nunique(), n_recordings=g.recording_id.nunique(), n_windows=len(g)))
    return pd.DataFrame(rows)


def specimen_table(rec: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec, g in rec.groupby("physical_specimen"):
        rows.append(dict(physical_specimen=spec, fault_type=g.fault_type.iloc[0], fault_severity=g.fault_severity.iloc[0], fault_bearing_end=g.fault_bearing_end.iloc[0], bearing_family=g.bearing_family.iloc[0], n_recordings=len(g), orientations="|".join(sorted({o for o in g.or_clock_position if o})), loads="|".join(sorted(g.load.unique())),
                         official_files="|".join(map(str, sorted(g.official_file_number))), has_FE_signal_all=bool(g.has_FE_signal.all()), identity_provenance="official catalogue label (end, class, diameter); one physical test bearing per label per CWRU documentation", identity_confidence="conservative (orientations merged; not serial-number certified)",
                         **{f"role_fold{f}": assign_split(spec, f) for f in FOLDS}))
    return pd.DataFrame(rows)


def write_protocol(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True); rec, deny = build_source_registry()
    rec.to_csv(out_dir / "cwru_v2_recordings.csv", index=False); deny.to_csv(out_dir / "cwru_n12_denylist.csv", index=False)
    allow = {r.local_path: {"sha256": r.sha256, "official_filename": r.official_filename, "source_url": r.source_url, "mat_variable": r.mat_variable, "native_sampling_rate_hz": NATIVE_RATE_HZ} for r in rec.itertuples()}
    (out_dir / "cwru_n12_allowlist.json").write_text(json.dumps({"protocol_id": PROTOCOL_ID, "inventory_sha256": INVENTORY_SHA256, "n": len(allow), "files": allow}, indent=1, sort_keys=True))
    win, excl = {}, []
    for f in FOLDS:
        w, e = build_windows(rec, f); win[f] = w; excl += e; w.to_csv(out_dir / f"cwru_v2_windows_fold_{f}.csv", index=False)
    summary = validate(rec, win, deny); pd.DataFrame(summary).to_csv(out_dir / "cwru_v2_fold_summary.csv", index=False)
    de_fe_tables(rec, win).to_csv(out_dir / "cwru_n12_de_fe_tables.csv", index=False); specimen_table(rec).to_csv(out_dir / "cwru_n12_specimens.csv", index=False)
    pd.DataFrame(excl, columns=["fold", "recording_id", "reason"]).to_csv(out_dir / "cwru_n12_structural_exclusions.csv", index=False)
    (out_dir / "cwru_v2_folds.json").write_text(json.dumps({"protocol_id": PROTOCOL_ID, "protocol_version": PROTOCOL_VERSION, "roles_inherited_from": "cwru_v2.P_B.v1 (pcste_v2/protocol/cwru_v2_PB_v1/cwru_v2_folds.json)", "folds": FOLDS, "window_s": WINDOW_S, "stride_s": STRIDE_S, "native_rate_hz": NATIVE_RATE_HZ, "stft_cwru12": STFT_CWRU12,
                                                            "sensor_policy": "accelerometer on the faulted bearing housing (DE_time for DE-fault, FE_time for FE-fault); class-independent; fault_bearing_end and sensor_location recorded separately", "structural_rules": "policy channel missing -> acquisition excluded; n_samples < 12000 -> excluded; tail dropped; no resampling"}, indent=1))
    hashes = {p.name: sha256_file(p) for p in sorted(out_dir.iterdir()) if p.is_file() and p.name not in ("cwru_v2_hashes.json", "PROTOCOL_STATUS.json")}
    master = hashlib.sha256("".join(f"{k}:{v}\n" for k, v in sorted(hashes.items())).encode()).hexdigest()
    (out_dir / "cwru_v2_hashes.json").write_text(json.dumps({"protocol_id": PROTOCOL_ID, "files": hashes, "master": master, "native12": True}, indent=1))
    return {"recordings": len(rec), "denylisted": len(deny), "windows": {f: len(w) for f, w in win.items()}, "structural_exclusions": excl, "master_hash": master}


def verify_protocol(out_dir: Path) -> str:
    h = json.loads((out_dir / "cwru_v2_hashes.json").read_text()); assert h.get("native12") is True and h.get("protocol_id") == PROTOCOL_ID
    for name, sha in h["files"].items():
        assert sha256_file(out_dir / name) == sha, f"native12 protocol file {name} changed (fail closed)"
    return h["master"]


def verify_any_native12(out_dir: Path) -> str:
    """Verify any native12 CWRU protocol directory by dispatching on the protocol id recorded in its hash file."""
    pid = json.loads((out_dir / "cwru_v2_hashes.json").read_text()).get("protocol_id")
    if pid == PROTOCOL_ID:
        return verify_protocol(out_dir)
    if pid == "pcste_v2_cwru_native12_load_v1":
        from .protocol_cwru_native12_load import verify_protocol as vp_load
        return vp_load(out_dir)
    raise AssertionError(f"unknown native12 CWRU protocol id {pid}")


def load_allowlist(out_dir: Path) -> dict:
    return json.loads((out_dir / "cwru_n12_allowlist.json").read_text())["files"]


def assert_native12_source(source_file: str, mat_variable: str, allow: dict, check_bytes: bool = True) -> None:
    """Loader/preflight gate: the file must be allowlisted, its bytes unchanged, the variable the policy channel, the rate 12000."""
    a = allow.get(source_file)
    if a is None: raise PermissionError(f"native12 gate: {source_file} is not an allowlisted native 12 kHz CWRU source")
    if a["mat_variable"] != mat_variable: raise PermissionError(f"native12 gate: {source_file}: variable {mat_variable} is not the allowlisted policy channel {a['mat_variable']}")
    if int(a["native_sampling_rate_hz"]) != NATIVE_RATE_HZ: raise PermissionError(f"native12 gate: {source_file}: rate {a['native_sampling_rate_hz']} != 12000")
    if check_bytes and sha256_file(REPO_ROOT / source_file) != a["sha256"]: raise PermissionError(f"native12 gate: {source_file}: bytes differ from the allowlisted official acquisition")
