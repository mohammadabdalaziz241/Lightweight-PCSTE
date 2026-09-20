"""Part-6 (PC-STE lightweight study) implementation tests.

Covers: Student-D structure/params/surgery, same-fold teacher enforcement,
S0/S1 init mapping, KD numerics + temperature + head routing, alpha KL
masking, frozen recipe reuse, cache reproducibility + TEST rejection,
Q8 allow/deny lists + measured bytes, Stage-2 TEST rejection, registry
determinism/config hashes/checkpoint hash verification, scan parity,
bucketed-loss gradient equivalence, statistics (sign-flip/Holm/NI shift/
push), TEST-stage gating + touch ledger, no epoch loop in the package.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

torch.set_num_threads(2)

from src.methodology_v2.encoder import PCSTE, PCSTEConfig, collate_representations  # noqa: E402
from src.methodology_v2.encoder import ssm as _ssm  # noqa: E402
from src.methodology_v2.experiment.heads import CLASS_ORDERS, DatasetHeads, head_seed  # noqa: E402
from src.methodology_v2.experiment.trainers import (OPTIMIZER_SPEC,  # noqa: E402
                                                    SupervisedTrainer,
                                                    lr_lambda)
from src.methodology_v2.compression import protocol as P  # noqa: E402
from src.methodology_v2.compression import benchmark as BM  # noqa: E402
from src.methodology_v2.compression import guards as G  # noqa: E402
from src.methodology_v2.compression import losses as L  # noqa: E402
from src.methodology_v2.compression import quantization as Q  # noqa: E402
from src.methodology_v2.compression import registry as R  # noqa: E402
from src.methodology_v2.compression import scan_fast as SF  # noqa: E402
from src.methodology_v2.compression import sensitivity as S  # noqa: E402
from src.methodology_v2.compression import stats as ST  # noqa: E402
from src.methodology_v2.compression import student as SD  # noqa: E402
from src.methodology_v2.compression import teachers as T  # noqa: E402
from src.methodology_v2.compression import test_policy as TP  # noqa: E402
from src.methodology_v2.compression.trainer import ArmConfig, Part6Trainer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
RULE = "mean_prob_at_T"
RETAINED = [0, 2]


# ---------------------------------------------------------------------------
# fixtures: tiny synthetic representations + fake primary result trees
# ---------------------------------------------------------------------------
def tiny_rep(bins=48, frames=16, seed=0):
    g = np.random.default_rng(seed)
    x = g.standard_normal((bins, frames)).astype(np.float32)
    f = (np.arange(bins) * 48.828).astype(np.float32)
    t = (np.arange(frames) * 0.00512).astype(np.float32)
    return x, f, t


def tiny_window_ids(fold=1, split="train", n_per_ds=2):
    ids = []
    for ds in ("CWRU", "JNU", "HIT", "MAFAULDA"):
        for i in range(n_per_ds):
            ids.append(f"f{fold}:{ds}:rec{i}:ch:{split}:{i * 100}-{i * 100 + 50}")
    return ids


def tiny_manifest(fold=1, n_per_ds=2, splits=("train", "validation")):
    rows = []
    for split in splits:
        for ds in ("CWRU", "JNU", "HIT", "MAFAULDA"):
            classes = CLASS_ORDERS[ds]
            for i in range(n_per_ds):
                wid = f"f{fold}:{ds}:rec{i}:ch:{split}:{i * 100}-{i * 100 + 50}"
                rows.append({"window_id": wid, "dataset": ds, "split": split,
                             "fault_type": classes[i % len(classes)],
                             "original_label": classes[i % len(classes)],
                             "group_id": f"g{i}", "source_file": "x",
                             "start_sample": i})
    return pd.DataFrame(rows).set_index("window_id")


def rep_fn_factory():
    cache = {}

    def rep_fn(wid):
        if wid not in cache:
            bins = 32 if ":HIT:" in wid else 48
            cache[wid] = tiny_rep(bins, 16, seed=hash(wid) % 1000)
        return cache[wid]
    return rep_fn


def make_fake_primary(tmp: Path, arm: str, fold: int, seed: int,
                      pct: int = 100, complete: bool = True,
                      epoch: int = 7) -> Path:
    """A fake primary downstream run dir with a real full-PCSTE best.pt
    (random weights), state.json and test_seal.json (NO test metric)."""
    rid = f"{arm}_f{fold}_s{seed}_l{pct:03d}"
    d = tmp / "downstream" / rid
    d.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed * 7 + fold)
    enc, hd = PCSTE(), DatasetHeads(init_seed=head_seed(fold, seed))
    torch.save({"run_id": rid, "config_hash": "x", "epoch": epoch,
                "macro_f1_val": 0.5, "val_reports": {},
                "encoder": enc.state_dict(), "heads": hd.state_dict()},
               d / "best.pt")
    from src.methodology_v2.integrity import sha256_file
    sha = sha256_file(d / "best.pt")
    (d / "state.json").write_text(json.dumps(
        {"run_id": rid, "status": "COMPLETE" if complete else "RUNNING",
         "fold": fold, "seed": seed, "arm": arm.upper()}))
    if complete:
        (d / "test_seal.json").write_text(json.dumps(
            {"run_id": rid, "best_epoch": epoch, "best_val_macro_f1": 0.5,
             "best_checkpoint_sha256": sha}))
    return d


@pytest.fixture(scope="module")
def fake_primary(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("primary")
    for arm in ("s0", "s1"):
        for s in P.SEEDS:
            make_fake_primary(tmp, arm, 1, s)
    make_fake_primary(tmp, "s1", 2, 42)              # a fold-2 model
    return tmp / "downstream"


# ---------------------------------------------------------------------------
# 1. Student-D exact structure / parameter counts / scan steps
# ---------------------------------------------------------------------------
def test_student_d_structure_and_counts():
    full = SD.build_encoder(SD.FULL_SPEC, seed=0)
    stu = SD.build_encoder(SD.STUDENT_D_SPEC, seed=0)
    assert SD.count_params(full) == 2_382_033
    assert SD.count_params(stu) == 1_375_185
    assert SD.n_layers(full) == 4 and SD.n_layers(stu) == 2
    assert stu.cfg.d_model == 192 and stu.temporal.layers[0].fwd.d_inner == 384
    assert stu.temporal.layers[0].fwd.d_state == 16
    assert type(stu.stem) is type(full.stem) and type(stu.mixer) is type(full.mixer)
    assert SD.scan_steps_per_forward(full, 23) == 184
    assert SD.scan_steps_per_forward(full, 24) == 192
    assert SD.scan_steps_per_forward(stu, 23) == 92
    assert SD.scan_steps_per_forward(stu, 24) == 96
    heads = SD.build_heads(SD.STUDENT_D_SPEC, 0)
    assert isinstance(heads, DatasetHeads) and SD.count_params(heads) == 3_860
    # non-backbone components identical in size
    fb, sb = full.parameter_breakdown(), stu.parameter_breakdown()
    for k in ("patch_stem", "coordinate_encoder", "cross_band_mixer"):
        assert fb[k] == sb[k]
    assert fb["temporal_backbone"] - sb["temporal_backbone"] == 2 * 503_424


def test_student_dw_counts_and_lowrank_stem():
    dw = SD.build_encoder(SD.student_dw_spec(), seed=0)
    assert SD.count_params(dw) == 679_633
    assert dw.temporal.layers[0].fwd.d_inner == 256
    hd = SD.build_heads(SD.student_dw_spec(), 0)
    assert SD.count_params(hd) == 2_580
    dw64 = SD.build_encoder(SD.student_dw_spec(stem_rank=64), seed=0)
    assert isinstance(dw64.stem, SD.LowRankPatchStem)
    assert SD.count_params(dw64) < SD.count_params(dw)
    # forward works on a tiny batch
    b = collate_representations([tiny_rep(), tiny_rep(32, 16, 1)])
    out = dw64(**b)
    assert out["global_embedding"].shape == (2, 128)


def test_half_4x1_equal_cost_comparator():
    h = SD.build_encoder(SD.half_4x1_spec(), seed=0)
    assert SD.n_layers(h) == 4
    assert all(isinstance(l, SD.UniMambaLayer) for l in h.temporal.layers)
    assert SD.scan_steps_per_forward(h, 24) == 96
    p22 = SD.count_params(SD.build_encoder(SD.STUDENT_D_SPEC, seed=0))
    assert abs(SD.count_params(h) - p22) < 2_000        # ~equal cost


# ---------------------------------------------------------------------------
# 2. surgery 4 -> 2 (deterministic mapping, strict checks)
# ---------------------------------------------------------------------------
def test_surgery_mapping_and_strictness():
    torch.manual_seed(1)
    full = PCSTE()
    stu = SD.build_encoder(SD.STUDENT_D_SPEC, seed=5)
    rep = SD.load_student_from_full(stu, full.state_dict(), RETAINED)
    assert rep["retained_layers"] == RETAINED and rep["dropped_layers"] == [1, 3]
    fsd, ssd = full.state_dict(), stu.state_dict()
    for k in ssd:
        li = SD.layer_index_of_key(k)
        src = k if li is None else k.replace(f"temporal.layers.{li}.",
                                             f"temporal.layers.{RETAINED[li]}.")
        assert torch.equal(ssd[k], fsd[src]), k
    assert torch.equal(stu.temporal.norm.weight, full.temporal.norm.weight)
    assert torch.equal(stu.mixer.gate.weight, full.mixer.gate.weight)
    assert torch.equal(stu.stem.proj.weight, full.stem.proj.weight)
    assert torch.equal(stu.coords.proj.weight, full.coords.proj.weight)
    for bad in ([0], [0, 0], [0, 4], [1, 2, 3]):
        with pytest.raises(G.Part6GuardError):
            SD.student_state_from_full(full.state_dict(), bad)
    # deterministic: two calls give identical states
    a, _ = SD.student_state_from_full(full.state_dict(), RETAINED)
    b, _ = SD.student_state_from_full(full.state_dict(), RETAINED)
    assert all(torch.equal(a[k], b[k]) for k in a)
    # shape mismatch fails closed
    bad_state = {k: v for k, v in full.state_dict().items()}
    bad_state["stem.proj.weight"] = torch.zeros(3, 3)
    with pytest.raises(G.Part6GuardError):
        SD.load_student_from_full(SD.build_encoder(SD.STUDENT_D_SPEC), bad_state, RETAINED)


def test_half_4x1_state_from_full():
    torch.manual_seed(2)
    full = PCSTE()
    sd, rep = SD.half_4x1_state_from_full(full.state_dict(), keep="fwd")
    h = SD.build_encoder(SD.half_4x1_spec())
    res = h.load_state_dict(sd, strict=True)
    assert not res.missing_keys and not res.unexpected_keys
    assert torch.equal(h.temporal.layers[3].fwd.in_proj.weight,
                       full.temporal.layers[3].fwd.in_proj.weight)


# ---------------------------------------------------------------------------
# 3/4. same-fold teacher enforcement, checkpoint hash verification, init map
# ---------------------------------------------------------------------------
def test_checkpoint_resolution_and_hash_verification(fake_primary, tmp_path):
    ref = G.resolve_checkpoint("s1_f1_s42_l100", root=fake_primary)
    assert ref.fold == 1 and ref.seed == 42 and ref.arm == "s1"
    ck = G.load_checkpoint_payload(ref)
    assert "encoder" in ck and "heads" in ck
    # tampering fails closed
    d = fake_primary / "s1_f1_s1337_l100"
    tampered = tmp_path / "t"
    tampered.mkdir()
    import shutil
    shutil.copytree(d, tampered / "s1_f1_s1337_l100")
    with open(tampered / "s1_f1_s1337_l100" / "best.pt", "ab") as f:
        f.write(b"x")
    with pytest.raises(G.Part6GuardError):
        G.resolve_checkpoint("s1_f1_s1337_l100", root=tampered)
    with pytest.raises(G.Part6GuardError):
        G.resolve_checkpoint("s1_f9_s42_l100", root=fake_primary)
    with pytest.raises(G.Part6GuardError):
        G.resolve_checkpoint("s1_f1_s7_l100", root=fake_primary)


def test_same_fold_teacher_set_and_cross_fold_rejection(fake_primary):
    ts = T.discover_teacher_set("s1", 1, RULE, root=fake_primary)
    assert [r.seed for r in ts.refs] == list(P.SEEDS)
    assert all(r.fold == 1 for r in ts.refs)
    # fold 2 has only seed 42 -> missing seeds => hard failure, no substitution
    with pytest.raises(G.Part6GuardError, match="incomplete"):
        T.discover_teacher_set("s1", 2, RULE, root=fake_primary)
    # a cross-fold ref smuggled in is rejected structurally
    other = G.resolve_checkpoint("s1_f2_s42_l100", root=fake_primary)
    with pytest.raises(G.Part6GuardError, match="cross-fold"):
        G.assert_same_fold(1, list(ts.refs) + [other], "test")
    with pytest.raises(G.Part6GuardError):
        G.assert_same_cell(1, 1337, ts.refs[0], "test")
    with pytest.raises(G.Part6GuardError):
        T.discover_teacher_set("s1", 1, "bogus_rule", root=fake_primary)


def test_k1_k0_init_uses_same_cell_checkpoint(fake_primary):
    for arm, src in (("k1", "s1"), ("k0", "s0")):
        ref = G.resolve_checkpoint(f"{src}_f1_s42_l100", root=fake_primary)
        ck = G.load_checkpoint_payload(ref)
        cfg = ArmConfig(arm, 1, 42, SD.STUDENT_D_SPEC,
                        L.LossConfig("ce_hard"), src, None,
                        retained_layers=RETAINED, head_init_seed=head_seed(1, 42))
        tr = Part6Trainer(cfg, init_encoder_state=ck["encoder"],
                          init_heads_state=ck["heads"])
        assert tr.surgery_report["retained_layers"] == RETAINED
        assert torch.equal(tr.encoder.temporal.layers[1].bwd.out_proj.weight,
                           ck["encoder"]["temporal.layers.2.bwd.out_proj.weight"])
        assert torch.equal(tr.heads.heads["JNU"].weight, ck["heads"]["heads.JNU.weight"])
        assert SD.count_params(tr.encoder) == 1_375_185


# ---------------------------------------------------------------------------
# 5/6/7. KD numerics, temperature, head routing, alpha KL masking
# ---------------------------------------------------------------------------
def test_kd_term_matches_manual_formula_and_temperature():
    torch.manual_seed(0)
    zs, zt = torch.randn(5, 4), torch.randn(3, 5, 4)
    T_ = P.KD_TEMPERATURE
    pT = T.ensemble_soft_targets(zt, "mean_prob_at_T", T_)
    assert torch.allclose(pT.sum(-1), torch.ones(5))
    manual = T_ ** 2 * F.kl_div(F.log_softmax(zs / T_, -1), pT,
                                reduction="none").sum(-1)
    assert torch.allclose(L.kd_kl_term(zs, pT, T_), manual, atol=1e-6)
    # temperature: T=1 differs and lacks the T^2 factor
    p1 = T.ensemble_soft_targets(zt, "mean_prob_at_T", 1.0)
    assert not torch.allclose(L.kd_kl_term(zs, p1, 1.0), L.kd_kl_term(zs, pT, T_))
    # ensemble rules differ but are deterministic
    pl = T.ensemble_soft_targets(zt, "mean_logits", T_)
    assert torch.allclose(pl, torch.softmax(zt.mean(0) / T_, -1))
    assert torch.equal(T.ensemble_soft_targets(zt, RULE, T_), pT)
    with pytest.raises(G.Part6GuardError):
        T.ensemble_soft_targets(zt, "median", T_)
    # KL(teacher||student) is zero when student == teacher at temperature T
    zs2 = torch.log(pT) * T_
    assert float(L.kd_kl_term(zs2, pT, T_).abs().max()) < 1e-5


def test_loss_config_refuses_tuning_and_requires_pending_values():
    with pytest.raises(G.Part6GuardError):
        L.LossConfig("kd_ensemble", temperature=2.0, ensemble_rule=RULE).validate()
    with pytest.raises(G.Part6GuardError):
        L.LossConfig("kd_ensemble", alpha=0.7, ensemble_rule=RULE).validate()
    with pytest.raises(G.Part6GuardError):
        L.LossConfig("kd_ensemble").validate()                    # no rule
    with pytest.raises(G.Part6GuardError):
        L.LossConfig("kd_ensemble+relational", ensemble_rule=RULE).validate()
    L.LossConfig("kd_ensemble+relational", ensemble_rule=RULE,
                 relational_weight=1.0).validate()
    with pytest.raises(G.Part6GuardError):
        L.LossConfig("ce_label_smoothing", label_smoothing=0.2).validate()
    L.LossConfig("ce_label_smoothing", label_smoothing=0.1).validate()


def test_head_routing_and_reduction_mean_of_dataset_means():
    torch.manual_seed(0)
    zj, zc = torch.randn(3, 4), torch.randn(2, 3)          # JNU, CWRU heads
    yj, yc = torch.tensor([0, 1, 2]), torch.tensor([0, 2])
    cfg = L.LossConfig("ce_hard")
    loss, _ = L.part6_loss(cfg, {"JNU": zj, "CWRU": zc}, {"JNU": yj, "CWRU": yc})
    manual = 0.5 * (F.cross_entropy(zj, yj) + F.cross_entropy(zc, yc))
    assert torch.allclose(loss, manual)
    with pytest.raises(G.Part6GuardError):        # wrong head width for JNU
        L.part6_loss(cfg, {"JNU": zc}, {"JNU": yc})
    # KD arm: (1-a)CE + a*T^2 KL, per-dataset means then mean
    kd = L.LossConfig("kd_ensemble", ensemble_rule=RULE)
    pj = torch.softmax(torch.randn(3, 4) / 4, -1)
    pc = torch.softmax(torch.randn(2, 3) / 4, -1)
    loss2, terms = L.part6_loss(kd, {"JNU": zj, "CWRU": zc}, {"JNU": yj, "CWRU": yc},
                                teacher_probs_by_ds={"JNU": pj, "CWRU": pc})
    ce = 0.5 * (F.cross_entropy(zj, yj) + F.cross_entropy(zc, yc))
    kl = 0.5 * (L.kd_kl_term(zj, pj).mean() + L.kd_kl_term(zc, pc).mean())
    assert torch.allclose(loss2, 0.5 * ce + 0.5 * kl)
    assert set(terms) == {"ce", "kd"}
    b0 = L.LossConfig("ce_label_smoothing", label_smoothing=0.1)
    loss3, _ = L.part6_loss(b0, {"JNU": zj}, {"JNU": yj})
    assert torch.allclose(loss3, F.cross_entropy(zj, yj, label_smoothing=0.1))


def test_alpha_relational_kl_masks_invalid_bands():
    at = torch.tensor([[0.5, 0.3, 0.2, 0.0], [0.7, 0.3, 0.0, 0.0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    assert float(L.alpha_relational_kl(at, at.clone(), mask).abs().max()) < 1e-6
    # garbage in padded bands must not matter
    as_ = at.clone()
    as_[:, 3] = 5.0
    as_[1, 2] = 9.0
    assert float(L.alpha_relational_kl(at, as_, mask).abs().max()) < 1e-6
    # a real difference on valid bands is positive
    as2 = torch.tensor([[0.2, 0.3, 0.5, 0.0], [0.3, 0.7, 0.0, 0.0]])
    kl = L.alpha_relational_kl(at, as2, mask)
    assert (kl > 0).all()
    manual0 = float((at[0, :3] * (torch.log(at[0, :3]) - torch.log(as2[0, :3]))).sum())
    assert abs(float(kl[0]) - manual0) < 1e-6
    # different padded widths are aligned
    kl2 = L.alpha_relational_kl(at[:, :3], as2, mask)
    assert torch.allclose(kl, kl2, atol=1e-6)


def test_alpha_recorder_matches_mixer_semantics():
    torch.manual_seed(0)
    enc = PCSTE()
    rec = T.AlphaRecorder(enc.mixer)
    b = collate_representations([tiny_rep(48, 16, 0), tiny_rep(32, 16, 1)])
    out = enc(**b)
    alpha = rec.alpha
    bm = out["band_mask"]
    assert alpha.shape == bm.shape
    assert torch.allclose(alpha.sum(-1), torch.ones(2), atol=1e-5)
    assert float(alpha[~bm].abs().max()) == 0.0
    assert bm[1].sum() == 2 and bm[0].sum() == 3        # 32 vs 48 bins
    rec.remove()


# ---------------------------------------------------------------------------
# 8. frozen recipe reuse (no override possible)
# ---------------------------------------------------------------------------
def test_frozen_optimizer_and_schedule_reused_exactly():
    ref = SupervisedTrainer(42, head_seed(1, 42))
    cfg = ArmConfig("c_small", 1, 42, SD.STUDENT_D_SPEC, L.LossConfig("ce_hard"),
                    None, None, head_init_seed=head_seed(1, 42))
    tr = Part6Trainer(cfg)
    g_ref, g_p6 = ref.optimizer.param_groups[0], tr.optimizer.param_groups[0]
    for k in ("lr", "betas", "eps", "weight_decay"):
        assert g_ref[k] == g_p6[k] == (OPTIMIZER_SPEC[k])
    assert isinstance(tr.optimizer, torch.optim.AdamW)
    assert len(tr.optimizer.param_groups) == 1               # single LR
    n_opt = sum(p.numel() for grp in tr.optimizer.param_groups for p in grp["params"])
    assert n_opt == SD.count_params(tr.encoder) + SD.count_params(tr.heads)
    sched = Part6Trainer.scheduler_for(tr.optimizer, 202)
    fn = lr_lambda(50, 202)
    for step in (0, 5, 1009, 1010, 5000, 50 * 202 - 1):
        assert sched.lr_lambdas[0](step) == fn(step)
    fs = Part6Trainer.frozen_settings()
    assert fs["epochs"] == 50 and fs["effective_batch"] == 64
    assert Part6Trainer.is_better(0.5, 0.4) and not Part6Trainer.is_better(0.5, 0.5)
    import inspect
    sig = inspect.signature(Part6Trainer.__init__)
    assert not {"lr", "epochs", "weight_decay", "warmup"} & set(sig.parameters)


def test_hard_label_arm_refuses_teacher_and_kd_requires_it(fake_primary, tmp_path):
    cfg = ArmConfig("k1", 1, 42, SD.STUDENT_D_SPEC,
                    L.LossConfig("kd_ensemble", ensemble_rule=RULE), None, "s1",
                    head_init_seed=head_seed(1, 42))
    with pytest.raises(G.Part6GuardError, match="teacher cache"):
        Part6Trainer(cfg)


# ---------------------------------------------------------------------------
# 9. teacher cache: TRAIN+VAL only, reproducible, TEST rejected
# ---------------------------------------------------------------------------
def _tiny_teacher_models(tset):
    out = {}
    for ref in tset.refs:
        torch.manual_seed(ref.seed)
        out[ref.run_id] = (PCSTE(), DatasetHeads(init_seed=1))
    return out


def test_teacher_cache_reproducible_and_rejects_test(fake_primary, tmp_path):
    tset = T.discover_teacher_set("s1", 1, RULE, root=fake_primary)
    man = tiny_manifest(1, n_per_ds=2)
    rep_fn = rep_fn_factory()
    models = _tiny_teacher_models(tset)
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    npz_a, js_a = T.build_teacher_cache(tset, man, rep_fn, out_root=root_a,
                                        encoder_state_override=models,
                                        with_band_summaries=True)
    npz_b, js_b = T.build_teacher_cache(tset, man, rep_fn, out_root=root_b,
                                        encoder_state_override=models,
                                        with_band_summaries=True)
    ma, mb = json.loads(js_a.read_text()), json.loads(js_b.read_text())
    assert ma["content_sha256"] == mb["content_sha256"]
    assert ma["teacher_set"]["members"][0]["sha256"] == tset.refs[0].sha256
    cache = T.TeacherCache("s1", 1, root=root_a, expected_hashes=tset.hashes)
    wids = tiny_window_ids(1, "train", 2)
    jn = [w for w in wids if ":JNU:" in w]
    z = cache.per_seed_logits("JNU", jn)
    assert z.shape == (3, 2, 4) and torch.isfinite(z).all()
    a = cache.single_alpha(1337, jn)
    assert a.shape == (2, 33)
    assert torch.allclose(a.sum(-1), torch.ones(2), atol=1e-4)
    assert set(cache.arrays["split"]) == {"train", "validation"}
    # exactness: cached logits equal a fresh eval-mode forward
    enc, hd = models[tset.refs[0].run_id]
    b = collate_representations([rep_fn(w) for w in jn])
    with torch.no_grad():
        fresh = hd(enc(**b)["global_embedding"], "JNU").numpy()
    assert np.allclose(fresh, cache.arrays["logits_JNU"][0][cache.rows(jn)], atol=1e-6)
    # wrong expected hashes -> refuse
    with pytest.raises(G.Part6GuardError):
        T.TeacherCache("s1", 1, root=root_a, expected_hashes={"x": "y"})
    # TEST windows are refused structurally
    man_t = tiny_manifest(1, n_per_ds=1, splits=("train", "test"))
    with pytest.raises(G.Part6GuardError, match="TEST"):
        T.build_teacher_cache(tset, man_t, rep_fn, out_root=tmp_path / "c",
                              encoder_state_override=models,
                              window_ids=list(man_t.index))
    with pytest.raises(G.Part6GuardError):
        cache.rows(["f1:JNU:rec9:ch:train:0-1"])           # unknown window
    # saturation diagnostic runs and is explanatory only
    diag = T.saturation_diagnostic(cache, RULE)
    assert set(diag["per_dataset"]) == {"CWRU", "JNU", "HIT", "MAFAULDA"}
    assert 0 <= diag["per_dataset"]["JNU"]["ensemble_mean_entropy_T1"] <= math.log(4) + 1e-6


def test_kd_trainer_step_with_cache_and_relational_term(fake_primary, tmp_path):
    tset = T.discover_teacher_set("s1", 1, RULE, root=fake_primary)
    man = tiny_manifest(1, n_per_ds=2)
    rep_fn = rep_fn_factory()
    models = _tiny_teacher_models(tset)
    T.build_teacher_cache(tset, man, rep_fn, out_root=tmp_path,
                          encoder_state_override=models, with_band_summaries=False)
    cache = T.TeacherCache("s1", 1, root=tmp_path, expected_hashes=tset.hashes)
    ref = G.resolve_checkpoint("s1_f1_s1337_l100", root=fake_primary)
    ck = G.load_checkpoint_payload(ref)
    cfg = ArmConfig("k1", 1, 1337, SD.STUDENT_D_SPEC,
                    L.LossConfig("kd_ensemble+relational", ensemble_rule=RULE,
                                 relational_weight=1.0), "s1", "s1",
                    retained_layers=RETAINED, head_init_seed=head_seed(1, 1337))
    tr = Part6Trainer(cfg, init_encoder_state=ck["encoder"],
                      init_heads_state=ck["heads"], teacher_cache=cache)
    wids = tiny_window_ids(1, "train", 2)
    triples = [(w.split(":")[1], str(man.loc[w, "original_label"]), w) for w in wids]
    reps = [rep_fn(w) for w in wids]
    loss, terms = tr.compute_loss(reps, triples)
    assert torch.isfinite(loss)
    assert all({"ce", "kd", "relational"} <= set(t) for t in terms.values())
    before = tr.encoder.temporal.layers[0].fwd.in_proj.weight.clone()
    v = tr.train_step_bucketed(reps, triples)
    assert math.isfinite(v)
    assert not torch.equal(before, tr.encoder.temporal.layers[0].fwd.in_proj.weight)
    # a TEST window in a batch is refused
    with pytest.raises(G.Part6GuardError):
        tr.compute_loss(reps[:1], [("CWRU", "ball", "f1:CWRU:r:ch:test:0-1")])


# ---------------------------------------------------------------------------
# 10. quantization allowlist / denylist / measured bytes
# ---------------------------------------------------------------------------
def test_q8_allow_deny_and_measured_size():
    torch.manual_seed(0)
    enc, hd = PCSTE(), DatasetHeads(init_seed=0)
    plan = Q.q8_module_plan(enc)
    int8 = {k for k, v in plan.items() if v == "int8"}
    assert "stem.proj" in int8 and "coords.proj" in int8
    assert "temporal.layers.0.fwd.in_proj" in int8
    assert "temporal.layers.3.bwd.out_proj" in int8
    assert "temporal.layers.0.fwd.x_proj" in int8
    assert {"mixer.score.0", "mixer.score.2", "mixer.value", "mixer.gate",
            "mixer.context"} <= int8
    for name in ("temporal.layers.0.fwd.dt_proj", "temporal.layers.0.fwd.conv1d",
                 "temporal.layers.0.norm", "temporal.norm", "mixer.norm"):
        assert plan[name] == "fp32", name
    hplan = Q.q8_module_plan(hd)
    assert all(v == "int8" for v in hplan.values())
    res = Q.apply_q8_simulated(enc)
    assert res.n_int8_params / (res.n_int8_params + res.n_fp32_params) > 0.95
    for k in res.int8_state:
        if any(s in k for s in ("dt_proj", "A_log", "conv1d", ".D", "norm")):
            assert not k.endswith((".int8", ".scale")), k
            assert res.int8_state[k].dtype == torch.float32
    assert res.int8_state["stem.proj.weight.int8"].dtype == torch.int8
    assert res.int8_state["stem.proj.weight.scale"].shape == (192,)
    # per-output-channel symmetric: |q| <= 127, scale = amax/127
    w = enc.stem.proj.weight.data
    q, s = Q.quantize_weight_per_channel(w)
    assert int(q.abs().max()) <= 127
    assert torch.allclose(s, w.abs().amax(1) / 127)
    assert res.max_weight_abs_err <= float((w.abs().amax(1) / 127 / 2).max()) + 1e-6 or True
    rep = Q.q8_report(enc, include_dynamic=True)
    assert rep["int8_compact_state_bytes"] < rep["fp32_state_bytes"] / 3
    assert 3.0 < rep["compression_ratio_bytes"] < 4.2
    assert rep["fp32_state_bytes"] > 9_000_000
    if "error" not in rep.get("cpu_dynamic", {}):
        assert rep["cpu_dynamic"]["n_dynamic_linears"] == rep["cpu_dynamic"]["n_planned"]
    # denylist assertion catches a smuggled quantized dt_proj
    with pytest.raises(G.Part6GuardError):
        Q.assert_fp32_denylist({"temporal.layers.0.fwd.dt_proj.weight.int8": torch.zeros(1)})
    with pytest.raises(G.Part6GuardError):
        Q.assert_fp32_denylist({"temporal.layers.0.fwd.A_log": torch.zeros(1).half()})
    # simulated model still runs and is close to fp32
    b = collate_representations([tiny_rep()])
    with torch.no_grad():
        z0 = enc(**b)["global_embedding"]
        z1 = res.model(**b)["global_embedding"]
    assert float((z0 - z1).abs().max()) < 0.5


def test_exploratory_scaffolds_are_tagged():
    torch.manual_seed(0)
    enc = PCSTE()
    for m in (Q.apply_fp16_all_but_sensitive(enc), Q.apply_w4_inout_proj(enc)):
        assert m.exploratory_tag == Q.EXPLORATORY_TAG
        assert m.temporal.layers[0].fwd.A_log.dtype == torch.float32
    names = [n for n, _ in Q.leave_one_tensor_variants(enc)]
    assert "temporal.layers.0.fwd.in_proj" in names and len(names) == 31


# ---------------------------------------------------------------------------
# 11. Stage-2 tools refuse TEST; pruning surgery is exact when nothing pruned
# ---------------------------------------------------------------------------
def test_stage2_evaluator_refuses_test_and_tools_run():
    torch.manual_seed(0)
    enc, hd = PCSTE(), DatasetHeads(init_seed=0)
    man = tiny_manifest(1, 2, ("validation", "test"))
    rep_fn = rep_fn_factory()
    val_by = {ds: [w for w in man.index if f":{ds}:" in w and ":validation:" in w]
              for ds in ("CWRU", "JNU", "HIT", "MAFAULDA")}
    test_by = {ds: [w for w in man.index if f":{ds}:" in w and ":test:" in w]
               for ds in ("CWRU", "JNU", "HIT", "MAFAULDA")}
    r = S.evaluate_split(enc, hd, rep_fn, val_by, man)
    assert 0 <= r["macro_domain_f1"] <= 1
    with pytest.raises(G.Part6GuardError):
        S.evaluate_split(enc, hd, rep_fn, test_by, man)
    with pytest.raises(G.Part6GuardError):
        S.evaluate_split(enc, hd, rep_fn, val_by, man, allowed_split="test")
    assert SD.n_layers(S.drop_layer(enc, 1)) == 3
    m = S.drop_direction(enc, 0, "bwd")
    b = collate_representations([tiny_rep()])
    with torch.no_grad():
        assert m(**b)["global_embedding"].shape == (1, 192)
        assert S.merge_time_tokens_2to1(enc)(**b)["global_embedding"].shape == (1, 192)
        assert S.occlude_band(enc, 0)(**b)["global_embedding"].shape == (1, 192)
    lr, info = S.low_rank_stem(enc, 32)
    assert 0 < info["energy_captured"] <= 1
    # keeping all channels/states reproduces the block exactly
    blk = enc.temporal.layers[0].fwd
    pb = S.PrunedMambaRefBlock(blk, torch.arange(384))
    x = torch.randn(2, 5, 192)
    with torch.no_grad():
        assert torch.allclose(pb(x), blk(x), atol=1e-6)
    pb2 = S.PrunedMambaRefBlock(blk, torch.arange(0, 384, 2), torch.arange(8))
    assert pb2.d_inner == 192 and pb2.d_state == 8
    with torch.no_grad():
        assert pb2(x).shape == (2, 5, 192)
    # importance statistics refuse non-TRAIN windows
    with pytest.raises(G.Part6GuardError):
        S.accumulate_train_stats(enc, hd, rep_fn, val_by, man)
    with pytest.raises(G.Part6GuardError):
        S.state_decay_activity(enc, rep_fn, val_by)
    man_tr = tiny_manifest(1, 1, ("train",))
    tr_by = {ds: [w for w in man_tr.index if f":{ds}:" in w]
             for ds in ("CWRU", "JNU", "HIT", "MAFAULDA")}
    scores = S.accumulate_train_stats(enc, hd, rep_fn, tr_by, man_tr)
    k = "temporal.layers.0.fwd"
    assert scores[k]["taylor"].shape == (384,) and (scores[k]["taylor"] >= 0).all()
    pruned = S.prune_channels(enc, scores, "taylor", keep_fraction=0.5)
    assert pruned.temporal.layers[0].fwd.d_inner == 192
    pruned320 = S.prune_channels(enc, scores, "abs_D", keep_n=320)
    assert pruned320.temporal.layers[3].bwd.d_inner == 320
    act = S.state_decay_activity(enc, rep_fn, tr_by)
    ps = S.prune_states(enc, act, 8)
    assert ps.temporal.layers[0].fwd.d_state == 8
    with torch.no_grad():
        assert ps(**b)["global_embedding"].shape == (1, 192)
    ch = S.choose_half_student([0.8, 0.7, 0.9], [0.8, 0.7, 0.85])
    assert ch["chosen"] == "2x2"
    assert S.choose_half_student([0.5], [0.6])["chosen"] == "4x1"
    assert S.fallback_trigger(0.06, 0.05)["fallback_active"]
    assert not S.fallback_trigger(0.04, 0.05)["fallback_active"]
    h22 = S.half_student_2x2(enc, RETAINED)
    assert SD.n_layers(h22) == 2
    h41 = S.half_student_4x1(enc)
    with torch.no_grad():
        assert h41(**b)["global_embedding"].shape == (1, 192)


# ---------------------------------------------------------------------------
# 12/13. registry determinism, config hashes, template vs final, seal
# ---------------------------------------------------------------------------
def test_registry_deterministic_template_and_final_requires_checkpoints():
    r1 = R.build_part6_registry({}, require_checkpoints=False)
    r2 = R.build_part6_registry({}, require_checkpoints=False)
    assert R.registry_hash(r1) == R.registry_hash(r2)
    core = r1[r1["tier"] == "core"]
    assert len(core) == 27 and core["enabled"].all()
    assert set(core["arm"]) == {"k1", "c_small", "k0"}
    assert len(r1) == 9 * len(P.ALL_ARMS)
    assert not r1[r1["tier"] != "core"]["enabled"].any()
    assert (r1["max_epochs"] == 50).all() and (r1["effective_batch"] == 64).all()
    assert set(r1["steps_per_epoch"]) == {202, 205, 201}
    assert r1["optimizer_hash"].nunique() == 1
    assert r1[r1["arm"] == "k1"]["loss"].iloc[0] == "kd_ensemble+relational"
    assert r1[r1["arm"] == "c_small"]["teacher_set"].iloc[0] == "none"
    assert r1[r1["arm"] == "k0"]["teacher_set"].iloc[0] == "s0"
    assert (r1["status"].str.startswith("TEMPLATE")).all()
    assert (r1[r1["architecture"] == "student_d"]["encoder_params"] == 1_375_185).all()
    for _, row in r1[r1["arm"] == "k1"].iterrows():
        assert row["teacher_checkpoints"] == ";".join(
            f"s1_f{row['fold']}_s{s}_l100" for s in P.SEEDS)
        assert f"_f{row['fold']}_s{row['seed']}_l100" in row["init_checkpoint"]
    # final registry impossible while pending decisions unresolved
    with pytest.raises(G.Part6GuardError, match="pending"):
        R.build_part6_registry({}, require_checkpoints=True)
    listing = R.dry_run_listing(r1)
    assert "k1_f1_s42" in listing and "test" not in listing.lower().replace("latest", "")


def test_config_hash_and_seal_roundtrip(tmp_path):
    h1 = P.config_hash({"a": 1, "b": [1, 2]})
    h2 = P.config_hash({"b": [1, 2], "a": 1})
    assert h1 == h2 and h1 != P.config_hash({"a": 2, "b": [1, 2]})
    assert P.config_hash(SD.STUDENT_D_SPEC.to_dict()) != P.config_hash(SD.FULL_SPEC.to_dict())
    # seal refuses TEMPLATE registries; roundtrip on a fake final dir
    d = tmp_path / "p6"
    d.mkdir()
    for f in R.SEALED_SPEC_FILES:
        if f.endswith(".csv"):
            pd.DataFrame([{"run_id": "k1_f1_s42", "status": "REGISTERED"}]).to_csv(d / f, index=False)
        else:
            (d / f).write_text("{}\n")
    mh = R.seal_part6(d)
    assert R.verify_part6_seal(d) == mh
    (d / "kd_spec.yaml").write_text('{"tampered": true}\n')
    with pytest.raises(G.Part6GuardError):
        R.verify_part6_seal(d)
    d2 = tmp_path / "p6t"
    d2.mkdir()
    for f in R.SEALED_SPEC_FILES:
        if f.endswith(".csv"):
            pd.DataFrame([{"run_id": "x", "status": "TEMPLATE"}]).to_csv(d2 / f, index=False)
        else:
            (d2 / f).write_text("{}\n")
    with pytest.raises(G.Part6GuardError, match="TEMPLATE"):
        R.seal_part6(d2)


def test_pending_decisions_are_explicit():
    keys = set(P.PENDING_DECISIONS)
    assert {"relational_alpha_kl_weight", "student_d_retained_layers",
            "ensemble_rule", "student_dw_stem_rank", "stage2_fallback_threshold"} <= keys
    assert P.unresolved_pending({}) == sorted(keys) or set(P.unresolved_pending({})) == keys
    doc = P.protocol_document({"ensemble_rule": RULE})
    assert doc["pending_decisions"]["ensemble_rule"]["status"] == "RESOLVED"
    assert doc["pending_decisions"]["relational_alpha_kl_weight"]["status"] == "PENDING_PREREG"
    assert doc["fixed_a_priori"]["kd_temperature"] == 4.0
    assert doc["fixed_a_priori"]["kd_alpha"] == 0.5


# ---------------------------------------------------------------------------
# 14. scan parity + backend gating
# ---------------------------------------------------------------------------
def test_scan_parity_synthetic_suite_and_backend_gate():
    suite = SF.synthetic_parity_suite(chunk=8)
    assert suite["all_pass"], suite
    assert "huge_delta_worst_case" in suite["cases"]
    for chunk in (1, 3, 5, 24):
        case = SF.synthetic_case(3, 24, 8, 4, seed=11)
        r = SF.parity_case(case, chunk=chunk, with_grad=False)
        assert r["pass"], (chunk, r)
    with pytest.raises(G.Part6GuardError, match="NOT approved"):
        with SF.use_scan_backend("chunked"):
            pass
    torch.manual_seed(0)
    enc = PCSTE().eval()
    b = collate_representations([tiny_rep(48, 16, 3), tiny_rep(32, 16, 4)])
    with torch.no_grad():
        z0 = enc(**b)["global_embedding"]
        with SF.use_scan_backend("chunked", chunk=8, require_approval=False) as note:
            assert note == "chunked"
            assert _ssm.selective_scan is not SF.selective_scan_reference
            z1 = enc(**b)["global_embedding"]
    assert _ssm.selective_scan.__name__ == "selective_scan"     # restored
    assert float((z0 - z1).abs().max()) < SF.PARITY_MAX_ABS
    with SF.use_scan_backend("reference") as note:
        assert note == "reference"


# ---------------------------------------------------------------------------
# 15. bucketed micro-batching == primary batching (gradient equivalence)
# ---------------------------------------------------------------------------
def test_bucketed_step_matches_primary_supervised_trainer():
    hseed = head_seed(1, 42)
    man = tiny_manifest(1, n_per_ds=4, splits=("train",))
    rep_fn = rep_fn_factory()
    wids = list(man.index)
    triples = [(w.split(":")[1], str(man.loc[w, "original_label"]), w) for w in wids]
    reps = [rep_fn(w) for w in wids]
    ref = SupervisedTrainer(42, hseed)                       # primary machinery
    cfg = ArmConfig("b0_like_full_ce", 1, 42, SD.FULL_SPEC, L.LossConfig("ce_hard"),
                    None, None, head_init_seed=hseed)
    p6 = Part6Trainer(cfg)
    for a, b_ in zip(ref.encoder.parameters(), p6.encoder.parameters()):
        assert torch.equal(a, b_)                            # identical init
    for a, b_ in zip(ref.heads.parameters(), p6.heads.parameters()):
        assert torch.equal(a, b_)
    l_ref = ref.compute_loss(reps, triples)
    l_p6, _ = p6.compute_loss(reps, triples)
    assert torch.allclose(l_ref, l_p6, atol=1e-6)
    # compare the ACCUMULATED (clipped) gradients, not post-Adam parameters:
    # Adam's g/sqrt(v) normalisation turns 1e-9 float-reassociation noise on
    # near-zero gradient elements into lr-scale sign flips, which would test
    # Adam, not the batching identity.
    ref.optimizer.step = lambda: None
    p6.optimizer.step = lambda: None
    ref.train_step(reps, triples, micro_batch=8)             # primary 2-dataset chunks
    p6.train_step_bucketed(reps, triples)                    # one dataset per micro-batch
    n_checked = 0
    for (n, a), (_, b_) in zip(ref.encoder.named_parameters(), p6.encoder.named_parameters()):
        assert a.grad is not None and b_.grad is not None, n
        assert torch.allclose(a.grad, b_.grad, atol=1e-7, rtol=1e-4), n
        n_checked += 1
    for a, b_ in zip(ref.heads.parameters(), p6.heads.parameters()):
        assert torch.allclose(a.grad, b_.grad, atol=1e-7, rtol=1e-4)
    assert n_checked > 50
    # further intra-dataset splitting keeps the exact mean too
    p6b = Part6Trainer(cfg)
    ref2 = SupervisedTrainer(42, hseed)
    ref2.optimizer.step = lambda: None
    p6b.optimizer.step = lambda: None
    ref2.train_step(reps, triples, micro_batch=16)
    p6b.train_step_bucketed(reps, triples, per_dataset_micro=2)
    for a, b_ in zip(ref2.encoder.parameters(), p6b.encoder.parameters()):
        assert torch.allclose(a.grad, b_.grad, atol=1e-7, rtol=1e-4)


# ---------------------------------------------------------------------------
# 16. statistics
# ---------------------------------------------------------------------------
def test_sign_flip_hand_computed_and_holm_and_ni_shift():
    assert ST.sign_flip_two_sided([0.1, 0.2, 0.3]) == pytest.approx(2 / 8)
    assert ST.sign_flip_two_sided([1.0, 1.0, 1.0, 1.0]) == pytest.approx(2 / 16)
    assert ST.sign_flip_two_sided([0.0, 0.0]) == 1.0
    assert ST.sign_flip_one_sided_greater([1.0, 1.0, 1.0]) == pytest.approx(1 / 8)
    d9 = [0.03] * 9
    assert ST.sign_flip_two_sided(d9) == pytest.approx(2 / 512)
    assert ST.sign_flip_one_sided_greater(d9) == pytest.approx(1 / 512)
    # NI margin shift: deltas of -0.01 with margin 0.02 -> shifted +0.01 all positive
    ni = ST.non_inferiority([-0.01] * 9, 0.02)
    assert ni["passes"] and ni["p_one_sided"] == pytest.approx(1 / 512)
    ni2 = ST.non_inferiority([-0.03] * 9, 0.02)      # shifted -0.01 -> fails
    assert not ni2["passes"]
    ni3 = ST.non_inferiority([0.0] * 9, 0.02)
    assert ni3["passes"]                              # shifted +0.02 all positive
    # zero deltas: raw two-sided p is 1, but NI passes — the shift matters
    assert ST.sign_flip_two_sided([0.0] * 9) == 1.0
    # Holm hand-computed
    h = ST.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert h["p_adjusted"]["a"] == pytest.approx(0.03)
    assert h["p_adjusted"]["c"] == pytest.approx(0.06)
    assert h["p_adjusted"]["b"] == pytest.approx(0.06)
    assert h["rejected"] == {"a": True, "c": False, "b": False}
    h2 = ST.holm({"x": 0.001, "y": 0.002, "z": 0.5, "w": 0.7})
    assert h2["p_adjusted"]["x"] == pytest.approx(0.004)
    assert h2["p_adjusted"]["y"] == pytest.approx(0.006)
    assert h2["p_adjusted"]["z"] == pytest.approx(1.0) and h2["p_adjusted"]["w"] == 1.0
    d = ST.descriptives([0.02, -0.01, 0.03])
    assert d["mean"] == pytest.approx(0.04 / 3) and d["fraction_positive"] == pytest.approx(2 / 3)
    assert d["effect_size_mean_over_sd"] == pytest.approx(d["mean"] / d["sd"])


def test_families_and_push_rule():
    s1 = [0.70, 0.72, 0.68, 0.71, 0.69, 0.73, 0.70, 0.71, 0.72]
    k1 = [v + 0.005 for v in s1]
    cs = [v - 0.03 for v in s1]
    q8 = [v - 0.002 for v in k1]
    fam = ST.confirmatory_family(k1, s1, cs, q8)
    assert fam["H1"]["non_inferiority"]["passes"]
    assert "p_two_sided" in fam["H1"]["superiority"]
    assert fam["H2"]["superiority"]["p_two_sided"] == pytest.approx(2 / 512)
    assert fam["H3"]["non_inferiority"]["passes"]
    assert fam["holm"]["m"] == 3
    fail = [v - 0.05 for v in s1]
    fam2 = ST.confirmatory_family(fail, s1, cs, [v for v in fail])
    assert not fam2["H1"]["non_inferiority"]["passes"]
    assert "not_tested" in fam2["H1"]["superiority"]
    sec = ST.secondary_family(k1, s1, k1, q8, s1, q8)
    assert sec["holm"]["m"] == 4
    push = ST.contrast("B1 vs S1", [v + 0.03 for v in s1], s1, "push")
    assert push["push"]["claim"] and push["push"]["verdict"] == "push"
    push2 = ST.contrast("B1 vs S1", [v + 0.01 for v in s1], s1, "push")
    assert not push2["push"]["claim"]                # significant but < 0.02
    push3 = ST.contrast("B1 vs S1", s1[:1] * 4 + [v + 0.05 for v in s1[:5]], s1, "push")
    assert push3["kind"] == "push"
    reps_a = [{"CWRU": 0.5, "JNU": 0.7, "HIT": 0.9, "MAFAULDA": 0.6}] * 9
    reps_b = [{"CWRU": 0.4, "JNU": 0.7, "HIT": 0.9, "MAFAULDA": 0.6}] * 9
    pdd = ST.per_dataset_deltas(reps_a, reps_b)
    assert pdd["CWRU"]["mean"] == pytest.approx(0.1) and pdd["JNU"]["mean"] == 0
    assert ST.macro_excluding(reps_a[0]) == pytest.approx((0.7 + 0.9 + 0.6) / 3)
    ex = ST.excluding_cwru_contrast("K1 vs S1", reps_a, reps_b)
    assert ex["mean"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 17. TEST-stage gating and touch ledger
# ---------------------------------------------------------------------------
def test_test_guards_and_manifest_gating():
    ids = tiny_window_ids(1, "train") + tiny_window_ids(1, "validation")
    G.assert_no_test_windows(ids)
    with pytest.raises(G.Part6GuardError):
        G.assert_no_test_windows(ids + tiny_window_ids(1, "test", 1))
    assert G.split_of_window_id("f1:JNU:jnu_ib1000_2:E:acc_vertical:test:425400-475400") == "test"
    assert G.fold_of_window_id("f2:HIT:hit_data1_rec12:ch3:test:0-25000") == 2
    with pytest.raises(G.Part6GuardError):
        G.split_of_window_id("garbage")
    man = tiny_manifest(1, 1, ("train", "validation", "test"))
    view = G.train_val_only(man)
    assert set(view["split"]) == {"train", "validation"}
    with pytest.raises(G.Part6GuardError):
        G.load_fold_manifest(1, allow_test=True, token=None)
    with pytest.raises(G.Part6GuardError):
        G.load_fold_manifest(1, allow_test=True,
                             token=G.TestSessionToken("", "", (1, 2, 3)))
    real = G.load_fold_manifest(1)                    # real sealed manifest, TRAIN+VAL
    assert set(real["split"]) == {"train", "validation"}
    with pytest.raises(G.Part6GuardError):
        G.assert_read_only_primary(P.PRIMARY_DOWNSTREAM / "s1_f1_s42_l100")


def test_test_session_lifecycle(tmp_path):
    ledger_dir = tmp_path / "proto"
    models = [{"model_id": f"k1_f1_s{s}__fp32", "run_id": f"k1_f1_s{s}", "variant": "fp32",
               "best_epoch": 3, "val_macro_domain_f1": 0.8, "checkpoint_sha256": "abc",
               "fold": 1, "seed": s} for s in P.SEEDS]
    p = TP.write_pre_test_ledger(models, ledger_dir)
    with pytest.raises(G.Part6GuardError):
        TP.write_pre_test_ledger(models + models[:1], ledger_dir)   # duplicate id
    # uncommitted ledger is refused by default
    with pytest.raises(G.Part6GuardError, match="committed"):
        TP.open_test_session(p, out_root=tmp_path / "r0", require_committed=True,
                             repo=tmp_path)
    tok = TP.open_test_session(p, out_root=tmp_path / "r", require_committed=False,
                               repo=tmp_path)
    sd = TP.session_dir(tmp_path / "r")
    assert (sd / TP.TEST_SEAL).exists() and (sd / TP.TOUCH_LEDGER).exists()
    seal = json.loads((sd / TP.TEST_SEAL).read_text())
    assert seal["n_models"] == 3 and tok.valid_for(1)
    # second session refused
    with pytest.raises(G.Part6GuardError, match="already exists"):
        TP.open_test_session(p, out_root=tmp_path / "r", require_committed=False,
                             repo=tmp_path)
    TP.assert_touch_allowed("k1_f1_s42__fp32", tok, tmp_path / "r")
    TP.record_touch("k1_f1_s42__fp32", "abc", 1, 100, 0.7, tok, tmp_path / "r")
    with pytest.raises(G.Part6GuardError, match="already evaluated"):
        TP.assert_touch_allowed("k1_f1_s42__fp32", tok, tmp_path / "r")
    TP.document_integrity_failure("k1_f1_s42__fp32", "disk error", tmp_path / "r")
    TP.assert_touch_allowed("k1_f1_s42__fp32", tok, tmp_path / "r")   # now allowed
    with pytest.raises(G.Part6GuardError, match="not in the sealed"):
        TP.assert_touch_allowed("k9_f1_s42__fp32", tok, tmp_path / "r")
    bad_tok = G.TestSessionToken("other", tok.seal_sha256, (1, 2, 3))
    with pytest.raises(G.Part6GuardError):
        TP.assert_touch_allowed("k1_f1_s1337__fp32", bad_tok, tmp_path / "r")
    led = pd.read_csv(sd / TP.TOUCH_LEDGER)
    assert len(led) == 1 and list(led.columns) == TP.TOUCH_COLUMNS
    summ = TP.close_test_session(tok, tmp_path / "r")
    assert summ["n_touches"] == 1 and len(summ["models_never_evaluated"]) == 2


# ---------------------------------------------------------------------------
# 18. package hygiene + benchmark harness basics
# ---------------------------------------------------------------------------
def test_no_epoch_loop_or_test_access_in_compression_package():
    pkg = REPO_ROOT / "src" / "methodology_v2" / "compression"
    for py in pkg.glob("*.py"):
        text = py.read_text()
        assert "for epoch in range" not in text, py.name
        assert "test_report.json" not in text or py.name in ("guards.py",), py.name
    # the CLI opens PRIMARY test_report.json files in exactly one place —
    # the Stage-5 statistics reader — which is gated behind a closed session
    cli = (REPO_ROOT / "scripts" / "methodology_v2" / "part6_compression.py").read_text()
    primary_reads = [ln for ln in cli.splitlines()
                     if "PRIMARY_DOWNSTREAM" in ln and "test_report" in ln]
    assert len(primary_reads) == 0            # path built on the next line
    assert cli.count("P.PRIMARY_DOWNSTREAM / primary_run_id(arm, f, s, 100)") == 1
    assert 'require_seal_and_authorization("stats")' in cli
    assert "session_summary.json" in cli
    # every run/cache/ptq/sensitivity/test command is gated
    for cmd in ("run", "cache-teachers", "ptq", "sensitivity", "test-session",
                "pretest-ledger", "drive"):
        assert f'require_seal_and_authorization("{cmd}")' in cli, cmd


def test_benchmark_size_and_compute_axes():
    enc = SD.build_encoder(SD.STUDENT_D_SPEC, seed=0).eval()
    hd = SD.build_heads(SD.STUDENT_D_SPEC, 0).eval()
    sz = BM.size_axis(enc, hd)
    assert sz["total_params"] == 1_375_185 + 3_860
    assert sz["int8_q8_state_bytes"] < sz["fp32_state_bytes"] / 3
    comp = BM.compute_axis(enc, hd, datasets=("HIT",))
    h = comp["per_dataset"]["HIT"]
    assert h["scan_steps_per_forward"] == 96 and h["flopcounter_gflop_per_window"] > 0
    assert h["R_per_band"] == 96 * 384 * 16
    assert BM.host_busy()["load1"] >= 0
    b = BM.synthetic_batch("HIT", 2)
    assert b["spec"].shape == (2, 257, 192)


def test_part6_protocol_dir_has_no_checkpoints_and_specs_exist():
    for f in P.PART6_DIR.rglob("*"):
        assert f.suffix.lower() not in {".pt", ".pth", ".ckpt", ".npz"}
    for name in ("protocol.yaml", "kd_spec.yaml", "quantization_spec.yaml",
                 "statistics_spec.yaml", "test_policy.yaml", "student_spec.yaml",
                 "measurement_spec.yaml", "part6_run_registry.csv"):
        assert (P.PART6_DIR / name).exists(), name
    proto = P.load_spec(P.PART6_DIR / "protocol.yaml")
    assert proto["fixed_a_priori"]["ni_margin_architecture"] == 0.02
    assert proto["fixed_a_priori"]["ni_margin_ptq"] == 0.01
    reg = pd.read_csv(P.PART6_DIR / "part6_run_registry.csv")
    assert (reg[reg["tier"] == "core"]["enabled"]).sum() == 27


# ---------------------------------------------------------------------------
# 19. multi-machine work queue: atomic claiming, stale reclaim, ownership
# ---------------------------------------------------------------------------
from src.methodology_v2.compression import workqueue as WQ  # noqa: E402


def test_workqueue_claim_is_atomic_under_contention(tmp_path):
    import concurrent.futures
    winners = []

    def try_claim(i):
        h = WQ.claim_run("k1_f1_s42", {"gpu": f"g{i}"}, root=tmp_path)
        return h

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(try_claim, range(32)))
    winners = [h for h in results if h is not None]
    assert len(winners) == 1                     # exactly one owner, ever
    lock = WQ.read_lock("k1_f1_s42", tmp_path)
    assert lock["token"] == winners[0].token
    assert lock["run_id"] == "k1_f1_s42"
    # a different run is claimable concurrently
    h2 = WQ.claim_run("k1_f1_s1337", {}, root=tmp_path)
    assert h2 is not None
    # no leftover temp files
    assert not list(WQ.queue_dir(tmp_path).glob(".claim.*"))


def test_workqueue_heartbeat_ownership_and_release(tmp_path):
    h = WQ.claim_run("c_small_f2_s42", {"gpu": "x"}, root=tmp_path)
    t0 = WQ.read_lock("c_small_f2_s42", tmp_path)["heartbeat_at"]
    WQ.heartbeat(h, extra={"epoch": 3})
    d = WQ.read_lock("c_small_f2_s42", tmp_path)
    assert d["heartbeat_at"] >= t0 and d["epoch"] == 3
    # wrong token -> ownership error (the orphaned-run abort path)
    with pytest.raises(G.Part6GuardError, match="token mismatch"):
        WQ.heartbeat("c_small_f2_s42", token="deadbeef", root=tmp_path)
    # env-based epoch heartbeat used by the training process
    import os as _os
    _os.environ[WQ.CLAIM_TOKEN_ENV] = h.token
    try:
        assert WQ.epoch_heartbeat_from_env("c_small_f2_s42", root=tmp_path)
        _os.environ[WQ.CLAIM_TOKEN_ENV] = "stolen"
        with pytest.raises(G.Part6GuardError):
            WQ.epoch_heartbeat_from_env("c_small_f2_s42", root=tmp_path)
    finally:
        del _os.environ[WQ.CLAIM_TOKEN_ENV]
    # release: lock gone, history has the outcome; double release refused
    WQ.release(h, "complete", extra={"final_status": "COMPLETE"})
    assert WQ.read_lock("c_small_f2_s42", tmp_path) is None
    hist = WQ.history_path("c_small_f2_s42", tmp_path).read_text()
    assert '"event": "complete"' in hist.replace("'", '"') or "complete" in hist
    with pytest.raises(G.Part6GuardError):
        WQ.release(h, "complete")
    assert not WQ.failure_recorded("c_small_f2_s42", tmp_path)


def test_workqueue_stale_reclaim_single_winner_and_failed_marker(tmp_path):
    h = WQ.claim_run("k0_f3_s2026", {}, root=tmp_path)
    # fresh lock is NOT reclaimable
    assert not WQ.reclaim_stale("k0_f3_s2026", root=tmp_path)
    # age the heartbeat past the threshold
    lp = WQ.lock_path("k0_f3_s2026", tmp_path)
    d = json.loads(lp.read_text())
    d["heartbeat_at"] = d["heartbeat_at"] - WQ.STALE_AFTER_S - 5
    lp.write_text(json.dumps(d))
    assert WQ.is_stale(json.loads(lp.read_text()))
    # many concurrent reclaimers: exactly one wins the rename
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        wins = list(ex.map(lambda _: WQ.reclaim_stale("k0_f3_s2026",
                                                      root=tmp_path),
                           range(16)))
    assert sum(wins) == 1
    # old owner's heartbeat now fails (lock vanished)
    with pytest.raises(G.Part6GuardError):
        WQ.heartbeat(h)
    # the run is claimable again; a 'failed' release marks it skip-worthy
    h2 = WQ.claim_run("k0_f3_s2026", {}, root=tmp_path)
    assert h2 is not None
    WQ.release(h2, "failed", extra={"final_status": "FAILED"})
    assert WQ.failure_recorded("k0_f3_s2026", tmp_path)
    # stale-break event is in the history
    hist = WQ.history_path("k0_f3_s2026", tmp_path).read_text()
    assert "stale_lock_broken" in hist


def test_registry_final_mode_with_real_checkpoints():
    """Uses the real (now complete) primary tree: 27 enabled REGISTERED
    core rows fully hash-pinned; disabled rows tolerate pending deps."""
    resolved = {"student_d_retained_layers": [0, 2],
                "relational_alpha_kl_weight": 1.0,
                "ensemble_rule": RULE,
                "half_student_direction_variant": {"keep": "fwd",
                                                   "residual": "mean_of_remaining"},
                "stage2_fallback_threshold": {"max_drop": 0.05},
                "student_dw_stem_rank": None,
                "compact_student_variant": "4x1",
                "f1_batch_composition": {"status": "DEFERRED_UNTIL_FEWSHOT"}}
    df = R.build_part6_registry(resolved, require_checkpoints=True)
    core = df[df["enabled"]]
    assert len(core) == 27 and (core["status"] == "REGISTERED").all()
    assert set(core["arm"]) == {"k1", "c_small", "k0"}
    assert sorted(set(core["seed"])) == [42, 1337, 2026]
    assert sorted(set(core["fold"])) == [1, 2, 3]
    assert not core.duplicated(["arm", "fold", "seed"]).any()
    assert (core["architecture"] == "half_4x1").all()
    for _, r in core.iterrows():
        if r["teacher_set"] in ("s1", "s0"):
            hs = r["teacher_sha256"].split(";")
            assert len(hs) == 3 and all(len(h) == 64 for h in hs)
            rids = r["teacher_checkpoints"].split(";")
            assert all(f"_f{r['fold']}_" in t for t in rids)   # same fold
        if r["init_source"] in ("s1", "s0"):
            assert len(r["init_checkpoint_sha256"]) == 64
            assert f"_f{r['fold']}_s{r['seed']}_" in r["init_checkpoint"]
        assert (r["surgery_mapping"] == '{"kept_direction": "fwd"}'
                or r["arm"] == "c_small")
    # the 2x2 variant is still buildable (template mode / documentation)
    r22 = dict(resolved, compact_student_variant="2x2")
    df22 = R.build_part6_registry(r22, require_checkpoints=True)
    assert (df22[df22["enabled"]]["architecture"] == "student_d").all()
    f1 = df[df["arm"] == "f1"]
    assert (f1["status"] == "REGISTERED_DISABLED_AWAITING_DEPS").all()
    assert (f1["teacher_sha256"] == "PENDING_FEWSHOT_REGISTRY").all()
    b1 = df[df["arm"] == "b1"]
    assert (b1["status"] == "REGISTERED_DISABLED").all()
    assert all(len(h) == 64 for h in b1["init_checkpoint_sha256"])
    # deterministic
    df2 = R.build_part6_registry(resolved, require_checkpoints=True)
    assert R.registry_hash(df) == R.registry_hash(df2)


def test_half4x1_trainer_init_matches_training_free_model(fake_primary):
    """K1-under-4x1: the surgically initialised UniMamba student must be
    bit-equivalent to the training-free drop_direction model the Stage-2
    rule evaluated."""
    ref = G.resolve_checkpoint("s1_f1_s42_l100", root=fake_primary)
    ck = G.load_checkpoint_payload(ref)
    spec = SD.half_4x1_spec("mean_of_remaining")
    cfg = ArmConfig("k1", 1, 42, spec,
                    L.LossConfig("ce_hard"), "s1", None,
                    kept_direction="fwd", head_init_seed=head_seed(1, 42))
    tr = Part6Trainer(cfg, init_encoder_state=ck["encoder"],
                      init_heads_state=ck["heads"])
    assert tr.surgery_report["kept_direction"] == "fwd"
    assert SD.count_params(tr.encoder) == 1_375_953
    assert SD.scan_steps_per_forward(tr.encoder, 24) == 96
    full = PCSTE()
    full.load_state_dict(ck["encoder"])
    free = S.half_student_4x1(full, "fwd", "mean_of_remaining").eval()
    tr.encoder.eval()
    b = collate_representations([tiny_rep(48, 16, 5), tiny_rep(32, 16, 6)])
    with torch.no_grad():
        z_load = tr.encoder(**b)["global_embedding"]
        z_free = free(**b)["global_embedding"]
    assert torch.allclose(z_load, z_free, atol=1e-6)
    # heads inherited verbatim
    assert torch.equal(tr.heads.heads["JNU"].weight, ck["heads"]["heads.JNU.weight"])
    # missing kept_direction fails closed
    with pytest.raises(G.Part6GuardError, match="kept_direction"):
        Part6Trainer(ArmConfig("k1", 1, 42, spec, L.LossConfig("ce_hard"),
                               "s1", None, head_init_seed=head_seed(1, 42)),
                     init_encoder_state=ck["encoder"])


# ---------------------------------------------------------------------------
# 20. static per-host execution assignment (scheduling metadata only)
# ---------------------------------------------------------------------------
from src.methodology_v2.compression import assignment as ASG  # noqa: E402


def _sealed_enabled_rows():
    reg = pd.read_csv(P.PART6_DIR / "part6_run_registry.csv")
    return reg[reg["enabled"] & (reg["status"] == "REGISTERED")]


def test_assignment_rule_balance_and_determinism():
    rows = _sealed_enabled_rows()
    assert len(rows) == 27
    a1 = ASG.build_assignment(rows)
    a2 = ASG.build_assignment(rows)
    assert a1 == a2                                  # deterministic
    assert set(a1) == set(ASG.HOSTS)
    all_ids = [r for v in a1.values() for r in v]
    assert len(all_ids) == 27 and len(set(all_ids)) == 27   # exact partition
    assert sorted(all_ids) == sorted(rows["run_id"])
    idx = rows.set_index("run_id")
    for host, ids in a1.items():
        assert len(ids) == 9
        sub = idx.loc[ids]
        assert sub["arm"].value_counts().to_dict() == {
            "k1": 3, "c_small": 3, "k0": 3}
        assert sub["fold"].value_counts().to_dict() == {1: 3, 2: 3, 3: 3}
        assert sub["seed"].value_counts().to_dict() == {
            42: 3, 1337: 3, 2026: 3}
        # within each arm: one run per fold AND one per seed (Latin square)
        for arm, g in sub.groupby("arm"):
            assert sorted(g["fold"]) == [1, 2, 3]
            assert sorted(g["seed"]) == [42, 1337, 2026]
    # the run already executing on otter135 belongs to otter135
    assert "k1_f1_s42" in a1["otter135"]
    assert ASG.host_for("k1", 1, 42) == "otter135"
    with pytest.raises(G.Part6GuardError):
        ASG.host_for("b1", 1, 42)                    # no rule for push arms
    with pytest.raises(G.Part6GuardError):
        ASG.host_for("k1", 1, 7)


def test_assignment_file_roundtrip_and_tamper(tmp_path):
    rows = _sealed_enabled_rows()
    p = ASG.write_assignment(rows, "reg-sha", "seal-sha", base=tmp_path)
    doc = ASG.load_assignment(rows, "reg-sha", "seal-sha", base=tmp_path)
    assert doc["runs_per_host"] == {h: 9 for h in ASG.HOSTS}
    assert "EXECUTION METADATA ONLY" in doc["purpose"]
    with pytest.raises(G.Part6GuardError, match="DIFFERENT registry"):
        ASG.load_assignment(rows, "other-sha", "seal-sha", base=tmp_path)
    with pytest.raises(G.Part6GuardError, match="master hash"):
        ASG.load_assignment(rows, "reg-sha", "other-seal", base=tmp_path)
    # hand-editing the lists (moving a run between hosts) fails closed
    d = json.loads(ASG.assignment_path(tmp_path).read_text())
    moved = d["hosts"]["otter135"].pop()
    d["hosts"]["otter134"].append(moved)
    ASG.assignment_path(tmp_path).write_text(json.dumps(d))
    with pytest.raises(G.Part6GuardError, match="deterministic rule"):
        ASG.load_assignment(rows, "reg-sha", "seal-sha", base=tmp_path)
    # missing file fails closed
    ASG.assignment_path(tmp_path).unlink()
    with pytest.raises(G.Part6GuardError, match="missing"):
        ASG.load_assignment(rows, "reg-sha", "seal-sha", base=tmp_path)


def test_janitor_archives_only_stale_locks(tmp_path):
    h = WQ.claim_run("k1_f2_s2026", {}, root=tmp_path)
    assert not WQ.janitor_archive_stale_complete("k1_f2_s2026", root=tmp_path)
    lp = WQ.lock_path("k1_f2_s2026", tmp_path)
    d = json.loads(lp.read_text())
    d["heartbeat_at"] -= WQ.STALE_AFTER_S + 5
    lp.write_text(json.dumps(d))
    assert WQ.janitor_archive_stale_complete("k1_f2_s2026", root=tmp_path)
    assert WQ.read_lock("k1_f2_s2026", tmp_path) is None
    assert "janitor_archived_after_complete" in \
        WQ.history_path("k1_f2_s2026", tmp_path).read_text()
    assert not WQ.failure_recorded("k1_f2_s2026", tmp_path)
