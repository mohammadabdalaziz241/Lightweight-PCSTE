"""pcste_v2_cwru_native12_specimen_v1: source gate, disjointness, reproducibility, sensor policy, streams, other-dataset preservation,
historical default-off behaviour and launch-preflight refusals. Metadata/bytes/structure only: no optimiser step, no TEST statistics."""
import hashlib, json, sys, tempfile, shutil
from pathlib import Path
import numpy as np, pandas as pd, pytest, torch
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts/pcste_v2"))
from src.pcste_v2 import protocol_cwru_native12 as P, native12_freeze as F, representation as REPR, cwru_intervention as CI
from src.pcste_v2.protocol import label_subset_frame
from src.methodology_v2.experiment.samplers import SupervisedSampler
CD = REPO / F.CWRU_DIR; GD = REPO / F.GLOBAL_DIR; BD = REPO / "pcste_v2/protocol/global_v2_PB_MAFv2_v1"
REC = pd.read_csv(CD / "cwru_v2_recordings.csv", dtype={"or_clock_position": str}); ALLOW = P.load_allowlist(CD); DENY = pd.read_csv(CD / "cwru_n12_denylist.csv")
INFO = json.loads((GD / "global_v2_hashes.json").read_text())


def test_allowlist_matches_official_inventory_and_local_bytes():
    inv = P.load_inventory(); off = {f["file_sha256"] for f in inv["native12_files"]}
    assert len(ALLOW) == 105 == len(REC) and set(REC.sha256) == off and set(REC.source_category) == {"official_12k_DE_fault", "official_12k_FE_fault"}
    for sf, a in ALLOW.items():
        assert hashlib.sha256((REPO / sf).read_bytes()).hexdigest() == a["sha256"] and a["native_sampling_rate_hz"] == 12000
        P.assert_native12_source(sf, a["mat_variable"], ALLOW, check_bytes=False)


def test_rejections_48k_relabelled_resampled_alias_and_stale_normaliser(tmp_path):
    with pytest.raises(PermissionError, match="not an allowlisted"):
        P.assert_native12_source("data/raw_cwru_48k/ball/B007_0_DE48k.mat", "X122_DE_time", ALLOW)                 # 48k acquisition
    with pytest.raises(PermissionError, match="not an allowlisted"):
        P.assert_native12_source("data/raw/normal/Normal_0HP.mat", "X097_DE_time", ALLOW)                            # 48k normal baseline in the 12k directory
    sf = REC.local_path.iloc[0]; a = ALLOW[sf]
    with pytest.raises(PermissionError, match="policy channel"):
        P.assert_native12_source(sf, "X109_DE_time", ALLOW)                                                          # 48k variable relabelled onto an allowlisted path
    fake = ALLOW | {"data/raw_cwru_48k/ball/B007_0_DE48k.mat": a | {"native_sampling_rate_hz": 12000, "mat_variable": "X122_DE_time"}}
    with pytest.raises(PermissionError, match="bytes differ"):
        P.assert_native12_source("data/raw_cwru_48k/ball/B007_0_DE48k.mat", "X122_DE_time", fake)                    # relabelled 48k bytes never match an official native12 hash
    derived = tmp_path / "resampled.mat"; derived.write_bytes((REPO / sf).read_bytes()[:-16] + b"\x00" * 16)          # a derivative/resampled copy: bytes never match
    with pytest.raises(PermissionError, match="bytes differ"):
        P.assert_native12_source(str(derived), a["mat_variable"], {str(derived): a}, check_bytes=True)
    # misleading local alias: files named OR021@12_* are official 246-249 (orientation 3); the registry records the official orientation
    r = REC[REC.local_path.str.contains("OR021@12_")]; assert len(r) == 4 and set(r.official_filename) == {"246.mat", "247.mat", "248.mat", "249.mat"} and set(r.or_clock_position) == {"3"} and r.local_alias_label_mismatch.notna().all()
    # denylist covers all 52 official 48k files + 4 normal baselines with local aliases; no overlap with the allowlist
    assert (DENY.category.value_counts().to_dict() == {"official_48k_DE_fault": 52, "official_48k_normal_baseline": 4}) and DENY.local_aliases.notna().all()
    assert not (set(a for v in DENY.local_aliases for a in v.split("|")) & set(ALLOW)) and not ({x for v in DENY.local_sha256 for x in v.split("|")} & {a["sha256"] for a in ALLOW.values()})
    # stale / foreign normaliser registry
    reg = pd.read_csv(GD / "normalizers/registry_fold_1.csv"); F.check_registry(reg, INFO["fold_1"]["sha256"])
    with pytest.raises(PermissionError, match="stale"):
        F.check_registry(reg, "0" * 64)
    with pytest.raises(PermissionError, match="outside"):
        F.check_registry(reg.assign(file=reg.file.str.replace("global_v2_native12_MAFv2_v1", "global_v2_PB_MAFv2_v1")), INFO["fold_1"]["sha256"])
    with pytest.raises(PermissionError, match="48 kHz"):
        F.check_registry(pd.concat([reg, reg.iloc[[0]].assign(key="CWRU48")]), INFO["fold_1"]["sha256"])


def test_loader_gate():
    REPR.NATIVE12_ALLOWLIST = ALLOW; REPR._cwru_channel.cache_clear()
    try:
        r = REC.iloc[0]; x = REPR._cwru_channel(r.local_path, r.mat_variable); assert x.shape[0] == r.n_samples
        with pytest.raises(PermissionError):
            REPR._cwru_channel("data/raw_cwru_48k/ball/B007_0_DE48k.mat", "X122_DE_time")
    finally:
        REPR.NATIVE12_ALLOWLIST = None; REPR._cwru_channel.cache_clear()


@pytest.mark.parametrize("fold", [1, 2, 3])
def test_fold_disjointness_support_and_pairing(fold):
    w = pd.read_csv(CD / f"cwru_v2_windows_fold_{fold}.csv", dtype={"or_clock_position": str}); g = pd.read_csv(GD / f"global_v2_fold_{fold}.csv", dtype={"fault_severity": str})
    for k in ("physical_specimen", "recording_id", "source_sha256", "source_file"):
        assert (w.groupby(k).split.nunique() == 1).all()
    assert w.window_id.is_unique and (w.native_sampling_rate_hz == 12000).all() and (w.end_sample - w.start_sample == 12000).all() and set(w.split) == {"train", "validation", "test"}
    for split in ("train", "validation", "test"):
        assert set(w[w.split == split].fault_type) == {"inner_race", "outer_race", "ball"}
    tr = w[w.split == "train"]; assert (tr.groupby("fault_type").physical_specimen.nunique() >= 2).all()
    roles = json.loads((CD / "cwru_v2_folds.json").read_text())["folds"][str(fold)]; pb = json.loads((REPO / "pcste_v2/protocol/cwru_v2_PB_v1/cwru_v2_folds.json").read_text())["folds"][str(fold)]
    assert roles == pb and set(w[w.split == "test"].physical_specimen) == set(pb["test"]) and set(w[w.split == "validation"].physical_specimen) == set(pb["validation"])
    # TRAIN pairing plan and normaliser fit set: TRAIN only
    pm = pd.read_csv(REPO / "analysis/pcste_v2_cwru_native12_migration" / f"NATIVE12_PAIRING_MANIFEST_TRAIN_fold{fold}_seed42.csv"); assert set(pm.window_id) == set(g.window_id[(g.dataset == "CWRU") & (g.split == "train")])
    reg = pd.read_csv(GD / f"normalizers/registry_fold_{fold}.csv"); assert int(reg[reg.key == "CWRU12"].n_windows.iloc[0]) == int(((g.dataset == "CWRU") & (g.split == "train")).sum()) and "CWRU48" not in set(reg.key)
    # sensor policy: class-independent, sensor == faulted bearing end; both sensors in every TRAIN class
    assert (w.sensor_location == w.fault_bearing_end).all() and (tr.groupby("fault_type").sensor_location.nunique() == 2).all()


def test_reproducible_generation(tmp_path):
    out = tmp_path / "proto"; P.write_protocol(out)
    for name, sha in json.loads((CD / "cwru_v2_hashes.json").read_text())["files"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == sha, name
    assert P.verify_protocol(CD) == json.loads((out / "cwru_v2_hashes.json").read_text())["master"]


def test_other_datasets_unchanged_and_default_off_config():
    for f in (1, 2, 3):
        a = pd.read_csv(BD / f"global_v2_fold_{f}.csv", dtype={"fault_severity": str}); b = pd.read_csv(GD / f"global_v2_fold_{f}.csv", dtype={"fault_severity": str})
        assert a[a.dataset != "CWRU"].sort_values("window_id").reset_index(drop=True).equals(b[b.dataset != "CWRU"].sort_values("window_id").reset_index(drop=True))
        for key in ("JNU", "HIT", "MAFAULDA"):
            assert (BD / f"normalizers/fold_{f}/v1__{key}__c0.npz").read_bytes() == (GD / f"normalizers/fold_{f}/v1__{key}__c0.npz").read_bytes()
    import run as R; assert R.CWRU_INTERVENTION.get("n2b") is None and R.protocol_info("global_v2_PB_MAFv2_v1").get("cwru_native12") in (None, False) and R.protocol_info("global_v2_native12_MAFv2_v1")["cwru_native12"] is True
    assert INFO["steps_per_epoch"] == {"1": 214, "2": 212, "3": 223}


def test_streams_c1_equals_c2_non_cwru_preserved_positives():
    f = 1; man = pd.read_csv(GD / f"global_v2_fold_{f}.csv", dtype={"fault_severity": str}); base = pd.read_csv(BD / f"global_v2_fold_{f}.csv", dtype={"fault_severity": str}); spe = int(INFO["steps_per_epoch"][str(f)])
    def stream(m):
        torch.manual_seed(42); np.random.seed(42); s = SupervisedSampler(label_subset_frame(m), 1.0, 42); return [s.next_batch() for _ in range(2 * spe)]
    c0, sb = stream(man), stream(base); u = CI.cwru_train_universe(man); c1, i1 = CI.balanced_stream(c0, u, spe, 2, 42); c2, i2 = CI.balanced_stream(c0, u, spe, 2, 42)
    assert c1 == c2 and i1["plan_sha256"] == i2["plan_sha256"] and CI.non_cwru_preserved(c0, c1)
    assert all([x for x in a if x[0] != "CWRU"] == [x for x in b if x[0] != "CWRU"] for a, b in zip(sb, c0))     # non-CWRU draws identical to the current baseline
    tr = set(man.window_id[(man.dataset == "CWRU") & (man.split == "train")]); wid_spec = dict(zip(u["window_ids"], u["specimens"]))
    for b in c1:
        cw = [it for it in b if it[0] == "CWRU"]; assert len(cw) == 16 and len({it[2] for it in cw}) == 16 and {it[2] for it in cw} <= tr
        for i, it in enumerate(cw):
            assert sum(1 for j, jt in enumerate(cw) if j != i and jt[1] == it[1] and wid_spec[jt[2]] != wid_spec[it[2]]) >= 1


def test_preflight_refusals(monkeypatch):
    pdir = GD; dg = F.digest_of(F.bundle_index(REPO)); real = F.read_status(REPO)
    monkeypatch.setattr(F, "read_status", lambda repo=REPO, cwru_dir=None: {"status": "NOT_FROZEN", "training_allowed": False, "freeze_digest_sha256": dg})
    with pytest.raises(PermissionError, match="must be FROZEN"): F.preflight(REPO, pdir, 1, dg, check_bytes=False)
    monkeypatch.setattr(F, "read_status", lambda repo=REPO, cwru_dir=None: {"status": "FROZEN", "training_allowed": False, "freeze_digest_sha256": dg})
    with pytest.raises(PermissionError, match="training_allowed=false"): F.preflight(REPO, pdir, 1, dg, check_bytes=False)
    monkeypatch.setattr(F, "read_status", lambda repo=REPO, cwru_dir=None: {"status": "FROZEN", "training_allowed": True, "freeze_digest_sha256": "0" * 64})
    with pytest.raises(PermissionError, match="digest mismatch"): F.preflight(REPO, pdir, 1, dg, check_bytes=False)
    monkeypatch.setattr(F, "read_status", lambda repo=REPO, cwru_dir=None: {"status": "FROZEN", "training_allowed": True, "freeze_digest_sha256": dg})
    with pytest.raises(PermissionError, match="authorization"): F.preflight(REPO, pdir, 1, None, check_bytes=False)
    with pytest.raises(PermissionError, match="authorization"): F.preflight(REPO, pdir, 1, "deadbeef", check_bytes=False)
    allow = F.preflight(REPO, pdir, 1, dg, check_bytes=False); assert len(allow) == 105                                   # would pass only under an authorised, frozen, training-allowed state
    assert real["status"] == "FROZEN" and isinstance(real["training_allowed"], bool)                                        # launch permission is the researcher-controlled status flag, not a test constant


# ------------------------------------------------------------------- final matched S0/S1 stage (final_s0_s1.v1)
def test_final_stage_composition_and_ssl_contract():
    """Global-fold composition (fixed CWRU, per-fold others), native12 grids, label-free SSL sampler, mask determinism,
    encoder/decoder separation and the accepted mean_nmse.v2 selector."""
    import numpy as _np
    from src.pcste_v2 import final_s0_s1 as FS
    from src.pcste_v2.selector import DEFAULT_SELECTOR, selector_value
    from src.methodology_v2.experiment.samplers import SSLSampler
    G = REPO / "pcste_v2/protocol/global_v2_final_s0s1_v1"; info = json.loads((G / "global_v2_hashes.json").read_text())
    assert FS.SSL_SELECTOR == DEFAULT_SELECTOR == "mean_nmse.v2" and info["steps_per_epoch"] == {"1": 214, "2": 212, "3": 223}
    cw = {}
    for f in (1, 2, 3):
        m = pd.read_csv(G / f"global_v2_fold_{f}.csv", dtype={"fault_severity": str})
        c = m[m.dataset == "CWRU"]; cw[f] = (tuple(sorted(c.window_id)), tuple(sorted(zip(c.window_id, c.split))))
        assert (c.native_sampling_rate_hz == 12000).all() and set(c.split) == {"train", "validation", "test"}
        assert (pd.read_csv(G / f"normalizers/registry_fold_{f}.csv").key != "CWRU48").all()
    assert cw[1] == cw[2] == cw[3], "CWRU membership and split assignment must be identical in every global fold"
    # CWRU normaliser statistics identical across folds (only the recorded fold label differs)
    zs = [_np.load(G / f"normalizers/fold_{f}/v1__CWRU12__c0.npz") for f in (1, 2, 3)]
    for k in ("mean", "std_denominator", "std_raw", "frequency_hz", "floor"):
        assert all(_np.array_equal(zs[0][k], z[k]) for z in zs[1:]), k
    # label-free sampler contract: label-bearing columns are structurally rejected
    m1 = pd.read_csv(G / "global_v2_fold_1.csv", dtype={"fault_severity": str}); tr = m1[m1.split == "train"]
    SSLSampler(tr[["dataset", "group_id", "window_id"]], 42)
    with pytest.raises(AssertionError):
        SSLSampler(tr[["dataset", "group_id", "window_id", "class"]], 42)
    # mask determinism: TRAIN masks vary by epoch, VAL masks are fixed; exact 60% of the native12 grid
    metas = [("CWRU", "w1"), ("JNU", "w2"), ("HIT", "w3"), ("MAFAULDA", "w4")]
    assert FS.mask_plan_hash(metas, 42, 0, True) == FS.mask_plan_hash(metas, 42, 7, True)      # VAL masks ignore the epoch
    assert FS.mask_plan_hash(metas, 42, 0, False) != FS.mask_plan_hash(metas, 42, 1, False)    # TRAIN masks vary by epoch
    assert FS.mask_plan_hash(metas, 42, 0, False) != FS.mask_plan_hash(metas, 1337, 0, False)  # and by seed
    pm = FS.build_patch_mask_v2(metas, 42, 0, False)
    for i, (ds, _) in enumerate(metas):
        fb, tp = FS.NATIVE12_GRIDS[ds]; assert int(pm[i, :fb, :tp].sum()) == round(0.60 * fb * tp) and int(pm[i].sum()) == round(0.60 * fb * tp)
    # encoder/decoder separation and the S1 loading contract
    torch.manual_seed(42); mod = FS.SSLModuleV2()
    enc = mod.encoder_state(); assert all(k.startswith("core.") for k in enc) and len(enc) == 101
    assert sum(p.numel() for p in mod.parameters()) - sum(p.numel() for p in mod.encoder.parameters()) == 119552
    from src.pcste_v2.model import PCSTEv2, PCSTEv2Config
    torch.manual_seed(0); fresh = PCSTEv2(PCSTEv2Config()); fresh.load_state_dict(enc, strict=True)     # strict coverage into the downstream model
    assert FS.canonical_state_hash(fresh.state_dict()) == FS.canonical_state_hash(enc)
    assert not any("mask_token" in k or "decoder" in k or "ctx_proj" in k for k in enc)
    # S0 and the SSL run start from the SAME initial encoder under the common factory
    torch.manual_seed(42); s0 = PCSTEv2(PCSTEv2Config()); torch.manual_seed(42); ssl = FS.SSLModuleV2()
    assert FS.canonical_state_hash(s0.state_dict()) == FS.canonical_state_hash(ssl.encoder.state_dict())
    # selector arithmetic
    assert abs(selector_value("mean_nmse.v2", {"A": 1.0, "B": 4.0}, {"A": 2.0, "B": 8.0}) - 0.5) < 1e-12


def test_ssl_mask_matches_collated_grid_including_single_dataset_chunks():
    """The mask container must match the collated token grid. An all-CWRU chunk collates to 9x23, not the 33x24 four-dataset
    maximum; the per-window mask CONTENT must be identical in both containers."""
    import pandas as _pd
    from src.pcste_v2 import final_s0_s1 as FS, representation as REPR, protocol_cwru_native12 as P
    from src.pcste_v2.representation import ItemBuilder
    from src.pcste_v2.model import collate_v2
    from src.methodology_v2.encoder.patchify import patchify
    G = REPO / "pcste_v2/protocol/global_v2_final_s0s1_v1"; REPR.NATIVE12_ALLOWLIST = P.load_allowlist(REPO / "pcste_v2/protocol/cwru_native12_load_v1")
    man = _pd.read_csv(G / "global_v2_fold_1.csv", dtype={"fault_severity": str}); mi = man.set_index("window_id")
    ib = ItemBuilder(G, 1, ("v1",), {"MAFAULDA": 1}, False)
    try:
        for ds, expect in (("CWRU", (9, 23)), ("JNU", (33, 24))):
            ids = list(man[(man.dataset == ds) & (man.split == "validation")].window_id[:4])
            b = collate_v2([ib.build(_pd.Series(mi.loc[w].to_dict() | {"window_id": w})) for w in ids], 1, 1, False)
            grid = FS.collated_grid(b); assert grid == expect, (ds, grid, expect)
            _, _, tm, _ = patchify(b["spec_g0_c0"], b["cell_mask_g0_c0"]); assert tuple(tm.shape[1:]) == grid
            pm = FS.build_patch_mask_v2([(ds, w) for w in ids], 42, 0, True, grid)
            assert pm.shape[1:] == tm.shape[1:], "mask container must match the collated token grid"
            wide = FS.build_patch_mask_v2([(ds, w) for w in ids], 42, 0, True)           # four-dataset maximum container
            fb, tp = FS.NATIVE12_GRIDS[ds]
            assert torch.equal(pm[:, :fb, :tp], wide[:, :fb, :tp]), "mask content must not depend on the container size"
            assert int(pm.sum()) == len(ids) * round(0.60 * fb * tp)
        # a mixed chunk still uses the wide grid
        mixed = list(man[(man.dataset == "CWRU") & (man.split == "validation")].window_id[:2]) + list(man[(man.dataset == "JNU") & (man.split == "validation")].window_id[:2])
        b = collate_v2([ib.build(_pd.Series(mi.loc[w].to_dict() | {"window_id": w})) for w in mixed], 1, 1, False)
        assert FS.collated_grid(b) == (33, 24)
    finally:
        REPR.NATIVE12_ALLOWLIST = None
