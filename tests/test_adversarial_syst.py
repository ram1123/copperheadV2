"""Tests for the systematics-adversarial VBF DNN loss and its variation plumbing.

Run with:  pixi run -e default python -m pytest tests/test_adversarial_syst.py -q
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import run_stage2_vbf
from modules import systematics as syst
from MVA_training.VBF_run3 import train_dnn


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "dnn_run2_vbf.yaml"


# --------------------------------------------------------------------------
# Synthetic Stage-1 field list, shaped like the real 2018 parquet schema.
# --------------------------------------------------------------------------
JEC_SOURCES_CORRELATED = ["Absolute", "BBEC1", "EC2", "FlavorQCD", "HF", "RelativeBal", "Total"]
JEC_SOURCES_YEAR = ["Absolute_2018", "BBEC1_2018", "EC2_2018", "HF_2018", "RelativeSample_2018"]
JET_BASES = [
    "jet1_pt", "jet1_eta", "jet1_phi", "jet1_btagUParTAK4QvG",
    "jet2_pt", "jet2_eta", "jet2_phi", "jet2_btagUParTAK4QvG",
    "jj_mass", "jj_mass_log", "jj_dEta", "rpt", "ll_zstar_log",
    "mmj_min_dEta", "pt_centrality", "njets", "nBtagLoose", "nBtagMedium",
    "nsoftjets5", "htsoft2",
]
DIMUON_BASES = [
    "dimuon_mass", "dimuon_pt", "dimuon_pt_log", "dimuon_rapidity",
    "dimuon_ebe_mass_res", "dimuon_ebe_mass_res_rel",
]
NO_VARIATION_BASES = ["dimuon_cos_theta_cs", "dimuon_phi_cs"]


def make_stage1_fields() -> set:
    fields = set(NO_VARIATION_BASES)
    for base in JET_BASES + DIMUON_BASES:
        fields.add(f"{base}_nominal")
    for base in JET_BASES:
        for src in JEC_SOURCES_CORRELATED + JEC_SOURCES_YEAR:
            for d in ("up", "down"):
                fields.add(f"{base}_{src}_{d}")
    for base in DIMUON_BASES:
        for d in ("up", "down"):
            fields.add(f"{base}_mu_roccor_{d}")
    fields |= {"event", "wgt_nominal"}
    return fields


# --------------------------------------------------------------------------
# Discovery parity with Stage-2
# --------------------------------------------------------------------------
def test_run_stage2_reexports_the_shared_helpers():
    """run_stage2_vbf must not carry a second copy of the discovery logic."""
    assert run_stage2_vbf.discover_shape_systs is syst.discover_shape_systs
    assert run_stage2_vbf.feature_name_for_variation is syst.feature_name_for_variation


def test_discovery_matches_stage2_on_fixed_field_list():
    fields = make_stage1_fields()
    discovered = run_stage2_vbf.discover_shape_systs(fields)
    stage2_used = syst.stage2_shape_variations(fields)

    # stage2_shape_variations is discover_shape_systs minus the "log_" suffixes
    assert stage2_used == [s for s in discovered if not s.startswith("log_")]
    # 13 sources x up/down, exactly as measured on the real Run-2 inputs
    assert len(stage2_used) == 26
    assert "Total_up" in stage2_used and "mu_roccor_down" in stage2_used
    assert "RelativeSample_2018_up" in stage2_used
    # the jj_mass_log_* branches must not leak in as "log_..." pseudo-variations
    assert not any(s.startswith("log_") for s in stage2_used)


def test_year_canonicalisation():
    assert syst.canonical_variation_name("Total_up") == "Total_up"
    assert syst.canonical_variation_name("mu_roccor_down") == "mu_roccor_down"
    assert syst.canonical_variation_name("FlavorQCD_up") == "FlavorQCD_up"
    for token in ("2018", "2017", "2016", "2016APV"):
        assert (
            syst.canonical_variation_name(f"Absolute_{token}_up")
            == "Absolute_yearDecor_up"
        )
        assert (
            syst.canonical_variation_name(f"RelativeSample_{token}_down")
            == "RelativeSample_yearDecor_down"
        )
    # every year collapses onto the same 26-slot axis
    fields = make_stage1_fields()
    canon = syst.canonical_variation_list(syst.stage2_shape_variations(fields))
    assert len(canon) == 26


def test_sweep_variation_set_is_four():
    fields = make_stage1_fields()
    allv = syst.stage2_shape_variations(fields)
    sweep = syst.select_variations(allv, "sweep")
    assert sorted(sweep) == ["Total_down", "Total_up", "mu_roccor_down", "mu_roccor_up"]
    assert syst.select_variations(allv, "full") == allv


def test_resolve_variation_columns_agrees_with_stage2_resolution():
    """The per-feature map must reproduce feature_name_for_variation exactly."""
    import yaml

    cfg = yaml.safe_load(CONFIG.read_text())
    features = [f for f in cfg["features"]["training"] if not f.startswith("year_")]
    fields = make_stage1_fields()
    variations = syst.stage2_shape_variations(fields)

    # raises on any disagreement
    syst.assert_matches_stage2(features, variations, fields)

    resolved = syst.resolve_variation_columns(features, variations, fields)
    # a JEC variation shifts the jet-side features only
    jec = resolved["Total_up"]
    assert "jet1_pt_nominal" in jec and jec["jet1_pt_nominal"] == "jet1_pt_Total_up"
    assert "dimuon_mass" not in jec           # falls back to nominal
    assert "dimuon_cos_theta_cs" not in jec   # no varied branch at all
    assert "nsoftjets5_nominal" not in jec    # soft-drop pinned to nominal
    assert "htsoft2_nominal" not in jec       # soft-drop pinned to nominal
    # mu_roccor shifts the dimuon-side features only
    roc = resolved["mu_roccor_up"]
    assert roc["dimuon_mass"] == "dimuon_mass_mu_roccor_up"
    assert "jet1_pt_nominal" not in roc
    # measured counts on the real inputs: 15 JEC-shifted, 6 roccor-shifted
    assert len(jec) == 15
    assert len(roc) == 6


# --------------------------------------------------------------------------
# Loss algebra
# --------------------------------------------------------------------------
def _reference_original_loss(logits, targets, wb, pos_weight, label_smoothing, normalize_in_batch):
    """Literal copy of the pre-change inline block in train_one_fold."""
    loss_raw = train_dnn.bce_with_logits_loss(
        logits=logits,
        targets=targets,
        weights=None,
        pos_weight=pos_weight,
        label_smoothing=label_smoothing,
        normalize_in_batch=False,
    )
    loss_w = train_dnn.make_loss_weights(
        w=wb, normalize_in_batch=normalize_in_batch, clip_abs_max=None
    )
    return torch.sum(loss_raw * loss_w) / (torch.sum(torch.abs(loss_w)) + 1e-12)


def test_training_loop_loss_is_the_original_arithmetic():
    torch.manual_seed(0)
    logits = torch.randn(64)
    y = (torch.rand(64) > 0.5).float()
    w = torch.randn(64)  # HEP weights can be negative
    ref = _reference_original_loss(logits, y, w, None, 0.04, True)
    got = train_dnn.training_loop_loss(
        logits=logits, targets=y, wb=w, pos_weight=None,
        label_smoothing=0.04, normalize_in_batch=True,
    )
    assert torch.equal(ref, got)


def test_adversarial_penalty_matches_the_specified_formula():
    torch.manual_seed(1)
    B, V = 32, 5
    logits_var = torch.randn(B, V)
    p_nom = torch.sigmoid(torch.randn(B))
    y = (torch.rand(B) > 0.5).float()
    w = torch.randn(B)

    cfg = train_dnn.load_config(str(CONFIG))
    adv = train_dnn.AdversarialConfig(
        lam=1.0, detach_nominal=False,
        consistency_label_smoothing="inherit", variation_chunk=0,
    )
    got = train_dnn.adversarial_penalty(logits_var, p_nom, y, w, None, cfg, adv)

    expected = torch.zeros(())
    for v in range(V):
        lv = logits_var[:, v]
        expected = expected + 2.0 * _reference_original_loss(
            lv, p_nom, w, None, cfg.label_smoothing, cfg.normalize_in_batch
        )
        expected = expected + _reference_original_loss(
            lv, y, w, None, cfg.label_smoothing, cfg.normalize_in_batch
        )
    assert torch.allclose(got, expected, rtol=0, atol=0)

    # 'none' consistency smoothing only changes the consistency half
    adv_ns = replace(adv, consistency_label_smoothing="none")
    got_ns = train_dnn.adversarial_penalty(logits_var, p_nom, y, w, None, cfg, adv_ns)
    assert not torch.allclose(got, got_ns)


def test_variation_chunking_partitions_all_variations():
    assert train_dnn._variation_chunks(26, 0) == [(0, 26)]
    assert train_dnn._variation_chunks(26, 30) == [(0, 26)]
    assert train_dnn._variation_chunks(26, 8) == [(0, 8), (8, 16), (16, 24), (24, 26)]
    for chunk in (0, 1, 3, 4, 26, 100):
        spans = train_dnn._variation_chunks(26, chunk)
        assert sum(hi - lo for lo, hi in spans) == 26
        assert spans[0][0] == 0 and spans[-1][1] == 26


# --------------------------------------------------------------------------
# Dataset: variation rows fall back to nominal where a feature does not shift
# --------------------------------------------------------------------------
def test_dataset_variation_rows_fall_back_to_nominal():
    features = ["a", "b", "c"]
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0], "b": [10.0, 20.0], "c": [100.0, 200.0],
            "label": [1.0, 0.0], "wgt_nominal": [0.5, -0.5],
            f"a{syst.VARIATION_COL_SEP}V_up": [1.5, 2.5],
        }
    )
    spec = [("V_up", {"a": f"a{syst.VARIATION_COL_SEP}V_up"})]
    ds = train_dnn.ParquetDataset(df, features, "label", "wgt_nominal", "float32", spec)
    x, y, w, xv = ds[0]
    assert xv.shape == (1, 3)
    assert float(xv[0, 0]) == pytest.approx(1.5)      # shifted
    assert float(xv[0, 1]) == pytest.approx(10.0)     # nominal fallback
    assert float(xv[0, 2]) == pytest.approx(100.0)    # nominal fallback
    assert torch.allclose(x, torch.tensor([1.0, 10.0, 100.0]))

    # without a spec the dataset is the original 3-tuple
    ds0 = train_dnn.ParquetDataset(df, features, "label", "wgt_nominal", "float32")
    assert len(ds0[0]) == 3


def test_dataset_replaces_non_finite_varied_values_with_nominal():
    features = ["a", "b"]
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0], "b": [10.0, 20.0],
            "label": [1.0, 0.0], "wgt_nominal": [1.0, 1.0],
            f"a{syst.VARIATION_COL_SEP}V_up": [np.nan, np.inf],
        }
    )
    spec = [("V_up", {"a": f"a{syst.VARIATION_COL_SEP}V_up"})]
    ds = train_dnn.ParquetDataset(df, features, "label", "wgt_nominal", "float32", spec)
    assert float(ds[0][3][0, 0]) == pytest.approx(1.0)
    assert float(ds[1][3][0, 0]) == pytest.approx(2.0)


# --------------------------------------------------------------------------
# End-to-end: lambda=0 with --use_adversarial reproduces the switch-off run
# --------------------------------------------------------------------------
def _write_synthetic_folds(data_dir: Path, cfg, variations, n=512, seed=7):
    rng = np.random.default_rng(seed)
    feats = list(cfg.training_features)
    physical = [f for f in feats if not f.startswith("year_")]
    col_to_feat = {}
    for split in ("train", "validation", "evaluation"):
        for i in range(cfg.n_folds):
            data = {f: rng.normal(size=n).astype(np.float32) for f in physical}
            for f in feats:
                if f.startswith("year_"):
                    data[f] = np.zeros(n, dtype=np.float32)
            data["year_2018"] = np.ones(n, dtype=np.float32)
            data["label"] = (rng.random(n) > 0.5).astype(np.float32)
            data["wgt_nominal"] = rng.normal(loc=1.0, scale=0.2, size=n).astype(np.float32)
            data["event"] = np.arange(n, dtype=np.int64)
            data["process"] = ["p"] * n
            data["process_group"] = ["g"] * n
            for v in variations:
                for f in physical[:5]:
                    col = syst.variation_column_name(f, v)
                    data[col] = data[f] + rng.normal(scale=0.1, size=n).astype(np.float32)
                    col_to_feat[col] = f
            pd.DataFrame(data).to_parquet(data_dir / f"data_df_{split}_{i}.parquet", index=False)

    manifest = {
        "training_features": feats,
        "systematic_variations": {
            "enabled": True,
            "variation_set": "sweep",
            "canonical_variations": list(variations),
            "n_variations": len(variations),
            "n_augmented_columns": len(col_to_feat),
            "column_to_nominal_feature": col_to_feat,
        },
    }
    (data_dir / "preprocess_manifest.json").write_text(json.dumps(manifest, indent=2))


def _tiny_cfg():
    cfg = train_dnn.load_config(str(CONFIG))
    return replace(
        cfg,
        device="cpu",
        epochs=2,
        batch_size=64,
        n_folds=4,
        es_enable=False,
        plots_enable=False,
        save_torchscript=False,
        save_history=False,
        amp_enable=False,
        num_workers=0,
    )


def _state_dict_from(out_dir: Path):
    ckpt = torch.load(out_dir / "fold0" / "best.pt", map_location="cpu", weights_only=False)
    return ckpt["model_state"]


def _assert_states_identical(a, b):
    assert set(a) == set(b)
    for k in a:
        assert torch.equal(a[k], b[k]), f"tensor '{k}' differs"


@pytest.mark.slow
def test_lambda_zero_reproduces_switch_off_bitwise(tmp_path):
    """AC3 / section 5.2 + 4.2 evidence, on synthetic data.

    Same seed, same fold parquets: `--use_adversarial` with lambda=0 must give
    bit-identical weights to no `--use_adversarial` at all. This is what proves
    the variation forward pass consumes no RNG and does not touch BatchNorm
    running statistics.
    """
    cfg = _tiny_cfg()
    variations = ["Total_down", "Total_up", "mu_roccor_down", "mu_roccor_up"]
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_synthetic_folds(data_dir, cfg, variations)

    spec = train_dnn.load_variation_spec(str(data_dir))
    assert [v for v, _ in spec] == sorted(variations)

    out_off = tmp_path / "off"
    train_dnn.set_seed(cfg.seed)
    train_dnn.train_one_fold(0, cfg, str(data_dir), str(out_off))

    out_zero = tmp_path / "lam0"
    adv0 = train_dnn.AdversarialConfig(
        lam=0.0, detach_nominal=False,
        consistency_label_smoothing="inherit", variation_chunk=0,
    )
    train_dnn.set_seed(cfg.seed)
    train_dnn.train_one_fold(0, cfg, str(data_dir), str(out_zero), adv=adv0, variation_spec=spec)

    _assert_states_identical(_state_dict_from(out_off), _state_dict_from(out_zero))


@pytest.mark.slow
def test_nonzero_lambda_actually_changes_training(tmp_path):
    """Guards against a penalty that is silently multiplied out to nothing."""
    cfg = _tiny_cfg()
    variations = ["Total_down", "Total_up", "mu_roccor_down", "mu_roccor_up"]
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_synthetic_folds(data_dir, cfg, variations)
    spec = train_dnn.load_variation_spec(str(data_dir))

    out_off = tmp_path / "off"
    train_dnn.set_seed(cfg.seed)
    train_dnn.train_one_fold(0, cfg, str(data_dir), str(out_off))

    out_lam = tmp_path / "lam"
    adv = train_dnn.AdversarialConfig(
        lam=1.0, detach_nominal=False,
        consistency_label_smoothing="inherit", variation_chunk=0,
    )
    train_dnn.set_seed(cfg.seed)
    train_dnn.train_one_fold(0, cfg, str(data_dir), str(out_lam), adv=adv, variation_spec=spec)

    off, lam = _state_dict_from(out_off), _state_dict_from(out_lam)
    assert any(not torch.equal(off[k], lam[k]) for k in off)


@pytest.mark.slow
def test_variation_chunking_is_gradient_equivalent(tmp_path):
    """Chunked accumulation must give the same result as one chunk."""
    cfg = _tiny_cfg()
    variations = ["Total_down", "Total_up", "mu_roccor_down", "mu_roccor_up"]
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_synthetic_folds(data_dir, cfg, variations)
    spec = train_dnn.load_variation_spec(str(data_dir))

    base = train_dnn.AdversarialConfig(
        lam=0.5, detach_nominal=True,
        consistency_label_smoothing="inherit", variation_chunk=0,
    )
    out_one = tmp_path / "one"
    train_dnn.set_seed(cfg.seed)
    train_dnn.train_one_fold(0, cfg, str(data_dir), str(out_one), adv=base, variation_spec=spec)

    out_chunk = tmp_path / "chunked"
    train_dnn.set_seed(cfg.seed)
    train_dnn.train_one_fold(
        0, cfg, str(data_dir), str(out_chunk),
        adv=replace(base, variation_chunk=2), variation_spec=spec,
    )

    a, b = _state_dict_from(out_one), _state_dict_from(out_chunk)
    for k in a:
        assert torch.allclose(a[k].float(), b[k].float(), rtol=1e-5, atol=1e-6), k


# --------------------------------------------------------------------------
# Degenerate-variation detection (the constant -1 QvG JEC branches)
# --------------------------------------------------------------------------
def test_degenerate_detection_catches_near_constant_sentinels():
    """Regression: an `n_unique == 1` test misses a 99.4%-sentinel column.

    On the real Run-2 inputs, 11 of the 48 JEC-varied QvG columns reach n_unique
    in the hundreds -- because the missing/non-finite fallback hands a few rows
    their real nominal value back -- while still being >99.4% sentinel. A
    uniqueness test leaves those in the "degenerate dropped" comparison set and
    hollows out the comparison.
    """
    from MVA_training.VBF_run3.preprocess_dnn import (
        DEGENERATE_MODAL_SHARE,
        find_degenerate_variation_columns,
    )

    n = 20000
    rng = np.random.default_rng(3)
    nominal = rng.uniform(-1.0, 1.0, n)

    pure = np.full(n, -1.0)                    # n_unique == 1
    leaky = pure.copy()
    leak = rng.choice(n, size=int(0.004 * n), replace=False)
    leaky[leak] = nominal[leak]                # n_unique in the dozens, 99.6% sentinel
    genuine = nominal + rng.normal(0, 0.02, n)  # a real shift

    df = pd.DataFrame(
        {"f": nominal, "v_pure": pure, "v_leaky": leaky, "v_genuine": genuine}
    )
    nominal_of = {"v_pure": "f", "v_leaky": "f", "v_genuine": "f"}
    flagged = find_degenerate_variation_columns(
        df, ["v_pure", "v_leaky", "v_genuine"], nominal_of
    )

    assert "v_pure" in flagged
    assert "v_leaky" in flagged, "near-constant sentinel must be caught, not just exact constants"
    assert "v_genuine" not in flagged
    assert len(np.unique(leaky)) > 1, "the leaky column must defeat an n_unique==1 test"
    assert flagged["v_leaky"]["n_unique_variation"] > 1
    assert flagged["v_leaky"]["modal_share_variation"] >= DEGENERATE_MODAL_SHARE
    assert flagged["v_leaky"]["modal_value"] == pytest.approx(-1.0)


def test_degenerate_detection_spares_a_feature_with_its_own_sentinel():
    """A nominal feature that is itself mostly one value must not be flagged."""
    from MVA_training.VBF_run3.preprocess_dnn import find_degenerate_variation_columns

    n = 20000
    rng = np.random.default_rng(4)
    nominal = np.zeros(n)
    live = rng.choice(n, size=int(0.005 * n), replace=False)
    nominal[live] = rng.uniform(1.0, 5.0, live.size)   # 99.5% zeros in NOMINAL too
    varied = nominal.copy()
    varied[live] *= 1.01

    df = pd.DataFrame({"f": nominal, "v": varied})
    flagged = find_degenerate_variation_columns(df, ["v"], {"v": "f"})
    assert flagged == {}, "nominal-side guard must spare a genuinely spiky feature"


# --------------------------------------------------------------------------
# Variant B (high-score-restricted consistency term)
# --------------------------------------------------------------------------
def _adv(**kw):
    base = dict(lam=1.0, detach_nominal=False,
                consistency_label_smoothing="inherit", variation_chunk=0)
    base.update(kw)
    return train_dnn.AdversarialConfig(**base)


def test_variant_b_restricts_the_consistency_term_to_high_score_events():
    """VARIANT B: only events with arctanh(p_nominal) > cut enter the consistency term.

    The label term is left unmasked here -- that is the distinction between the
    variant-B reading and the three-step one, which `mask_label_term` selects.
    """
    torch.manual_seed(6)
    B, V, CUT = 256, 3, 2.0
    logits_var = torch.randn(B, V)
    # spread p_nominal so some events sit above tanh(2) = 0.96403 and some below
    p_nom = torch.rand(B)
    y, w = (torch.rand(B) > 0.5).float(), torch.randn(B)
    cfg = train_dnn.load_config(str(CONFIG))

    got = train_dnn.adversarial_penalty(
        logits_var, p_nom, y, w, None, cfg, _adv(high_score_cut=CUT))

    keep = p_nom > np.tanh(CUT)
    assert 0 < int(keep.sum()) < B, "test needs a mix of passing and failing events"
    expected = torch.zeros(())
    for v in range(V):
        lv = logits_var[:, v]
        expected = expected + 2.0 * _reference_original_loss(
            lv[keep], p_nom[keep], w[keep], None,
            cfg.label_smoothing, cfg.normalize_in_batch)
        expected = expected + _reference_original_loss(
            lv, y, w, None, cfg.label_smoothing, cfg.normalize_in_batch)
    assert torch.allclose(got, expected, rtol=0, atol=0)

    # the cut must actually change the answer
    unmasked = train_dnn.adversarial_penalty(logits_var, p_nom, y, w, None, cfg, _adv())
    assert not torch.allclose(got, unmasked)


def test_variant_b_cut_matches_arctanh_definition():
    """`p > tanh(cut)` is used instead of `arctanh(p) > cut` for numerical safety.

    The two are equivalent in exact arithmetic. In float32 they can only disagree
    for p within one ULP of tanh(cut) -- there both candidates round to the *same*
    float32, so the disagreement is not representable in the precision the model
    runs at. The test asserts equivalence everywhere outside that band, and pins
    the band itself rather than pretending it does not exist.
    """
    rng = np.random.default_rng(0)
    p = np.concatenate([
        np.array([0.5, 0.9, 0.96, 0.99, 0.999, 0.999999]),
        rng.uniform(0.0, 1.0, 5000),
    ]).astype(np.float32)

    for cut in (1.0, 2.0, 3.0):
        thr = np.tanh(cut)
        # exclude only the sub-ULP band around the threshold
        outside = np.abs(p.astype(np.float64) - thr) > np.spacing(np.float32(thr))
        by_tanh = torch.from_numpy(p) > thr
        by_arctanh = torch.from_numpy(np.arctanh(p.astype(np.float64))) > cut
        assert torch.equal(
            by_tanh[torch.from_numpy(outside)], by_arctanh[torch.from_numpy(outside)]
        ), f"cut={cut}"

    # arctanh is the thing being avoided: it overflows where the cut selects.
    with np.errstate(divide="ignore"):
        assert np.isinf(np.arctanh(np.float64(1.0)))
    assert np.isfinite(np.tanh(2.0))


def test_variant_b_empty_selection_contributes_zero():
    """A batch with no high-score event must give exactly 0, not NaN.

    Every term has to be under the cut for the total to vanish, so this is the
    three-step configuration (`mask_label_term`) -- the one phase 4 ran.
    """
    torch.manual_seed(7)
    B, V = 32, 3
    logits_var = torch.randn(B, V)
    p_nom = torch.rand(B) * 0.5          # all far below tanh(2)
    y, w = (torch.rand(B) > 0.5).float(), torch.randn(B)
    cfg = train_dnn.load_config(str(CONFIG))
    got = train_dnn.adversarial_penalty(
        logits_var, p_nom, y, w, None, cfg,
        _adv(high_score_cut=2.0, mask_label_term=True))
    assert float(got) == 0.0 and torch.isfinite(got)


def test_variant_b_empty_selection_is_still_differentiable():
    """Regression: the empty-selection zero must keep its graph.

    At initialisation the model predicts ~0.5 everywhere, so no event clears a
    cut of 2.0 (p > 0.9640) and EVERY run hits this path on its first batch. A
    bare `new_zeros(())` has no grad_fn and the training loop's
    `scaler.scale(term).backward()` then raises "element 0 of tensors does not
    require grad". Checking the value alone does not catch it -- this asserts the
    backward actually runs and yields a zero gradient.
    """
    torch.manual_seed(11)
    B, V = 32, 3
    lin = torch.nn.Linear(4, 1)
    x = torch.randn(B * V, 4)
    logits_var = lin(x).reshape(B, V)
    p_nom = torch.rand(B) * 0.5          # all far below tanh(2)
    y, w = (torch.rand(B) > 0.5).float(), torch.randn(B)
    cfg = train_dnn.load_config(str(CONFIG))

    pen = train_dnn.adversarial_penalty(
        logits_var, p_nom, y, w, None, cfg,
        _adv(high_score_cut=2.0, mask_label_term=True))

    assert float(pen) == 0.0
    assert pen.requires_grad, "the zero must stay attached to the graph"
    pen.backward()                        # must not raise
    assert lin.weight.grad is not None
    assert torch.all(lin.weight.grad == 0), "an empty selection must give zero gradient"


def test_default_formula_is_the_specified_penalty():
    """The default penalty is `sum_i [2*loss(pred_i, pred_nom) + loss(pred_i, label)]`."""
    torch.manual_seed(8)
    B, V = 40, 5
    logits_var, p_nom = torch.randn(B, V), torch.sigmoid(torch.randn(B))
    y, w = (torch.rand(B) > 0.5).float(), torch.randn(B)
    cfg = train_dnn.load_config(str(CONFIG))
    default = train_dnn.adversarial_penalty(logits_var, p_nom, y, w, None, cfg, _adv())
    expected = torch.zeros(())
    for v in range(V):
        lv = logits_var[:, v]
        expected = expected + 2.0 * _reference_original_loss(
            lv, p_nom, w, None, cfg.label_smoothing, cfg.normalize_in_batch)
        expected = expected + _reference_original_loss(
            lv, y, w, None, cfg.label_smoothing, cfg.normalize_in_batch)
    assert torch.allclose(default, expected, rtol=0, atol=0)


# --------------------------------------------------------------------------
# The three-step schedule: step 2 (label-only) and the [score_cut] on every term
# --------------------------------------------------------------------------
def test_step2_label_only_drops_the_consistency_term():
    """STEP 2: lambda * sum_i original_loss(pred_i, label), nothing else."""
    torch.manual_seed(21)
    B, V = 48, 4
    logits_var, p_nom = torch.randn(B, V), torch.sigmoid(torch.randn(B))
    y, w = (torch.rand(B) > 0.5).float(), torch.randn(B)
    cfg = train_dnn.load_config(str(CONFIG))

    got = train_dnn.adversarial_penalty(
        logits_var, p_nom, y, w, None, cfg, _adv(label_only=True))

    expected = torch.zeros(())
    for v in range(V):
        expected = expected + _reference_original_loss(
            logits_var[:, v], y, w, None, cfg.label_smoothing, cfg.normalize_in_batch)
    assert torch.allclose(got, expected, rtol=0, atol=0)

    # p_nominal must not enter at all: perturbing it cannot move the answer.
    other = train_dnn.adversarial_penalty(
        logits_var, torch.sigmoid(torch.randn(B)), y, w, None, cfg, _adv(label_only=True))
    assert torch.allclose(got, other, rtol=0, atol=0)


def test_step2_label_only_respects_the_score_cut_when_masked():
    """STEP 2 with a cut: the label term is restricted to arctanh(p_nom) > cut."""
    torch.manual_seed(22)
    B, V, CUT = 256, 3, 1.0
    logits_var = torch.randn(B, V)
    p_nom = torch.rand(B)
    y, w = (torch.rand(B) > 0.5).float(), torch.randn(B)
    cfg = train_dnn.load_config(str(CONFIG))

    got = train_dnn.adversarial_penalty(
        logits_var, p_nom, y, w, None, cfg,
        _adv(label_only=True, high_score_cut=CUT, mask_label_term=True))

    keep = p_nom > np.tanh(CUT)
    assert 0 < int(keep.sum()) < B, "test needs a mix of passing and failing events"
    expected = torch.zeros(())
    for v in range(V):
        expected = expected + _reference_original_loss(
            logits_var[keep, v], y[keep], w[keep], None,
            cfg.label_smoothing, cfg.normalize_in_batch)
    assert torch.allclose(got, expected, rtol=0, atol=0)


def test_step3_masks_both_terms_when_mask_label_term_is_set():
    """STEP 3: sum_i [ 2*consistency + label ], every term under [score_cut]."""
    torch.manual_seed(23)
    B, V, CUT = 256, 3, 0.6
    logits_var = torch.randn(B, V)
    p_nom = torch.rand(B)
    y, w = (torch.rand(B) > 0.5).float(), torch.randn(B)
    cfg = train_dnn.load_config(str(CONFIG))

    got = train_dnn.adversarial_penalty(
        logits_var, p_nom, y, w, None, cfg,
        _adv(high_score_cut=CUT, mask_label_term=True))

    keep = p_nom > np.tanh(CUT)
    assert 0 < int(keep.sum()) < B
    expected = torch.zeros(())
    for v in range(V):
        lv = logits_var[keep, v]
        expected = expected + 2.0 * _reference_original_loss(
            lv, p_nom[keep], w[keep], None, cfg.label_smoothing, cfg.normalize_in_batch)
        expected = expected + _reference_original_loss(
            lv, y[keep], w[keep], None, cfg.label_smoothing, cfg.normalize_in_batch)
    assert torch.allclose(got, expected, rtol=0, atol=0)

    # and it differs from the variant-B reading, which leaves the label term whole
    variant_b = train_dnn.adversarial_penalty(
        logits_var, p_nom, y, w, None, cfg, _adv(high_score_cut=CUT))
    assert not torch.allclose(got, variant_b)


def test_score_cut_all_true_is_the_unmasked_formula():
    """`score_cut = all True` is expressed as no cut at all, for both steps."""
    torch.manual_seed(24)
    B, V = 64, 4
    logits_var, p_nom = torch.randn(B, V), torch.sigmoid(torch.randn(B))
    y, w = (torch.rand(B) > 0.5).float(), torch.randn(B)
    cfg = train_dnn.load_config(str(CONFIG))
    for kw in ({"label_only": True}, {}):
        no_cut = train_dnn.adversarial_penalty(
            logits_var, p_nom, y, w, None, cfg, _adv(**kw))
        # a cut of -inf keeps every event; mask_label_term must then be a no-op
        masked = train_dnn.adversarial_penalty(
            logits_var, p_nom, y, w, None, cfg,
            _adv(high_score_cut=-30.0, mask_label_term=True, **kw))
        assert torch.allclose(no_cut, masked, rtol=0, atol=1e-6)


def test_step2_empty_selection_is_zero_and_differentiable():
    """Warm-started or not, a batch with no event above the cut must not NaN."""
    torch.manual_seed(25)
    B, V = 32, 3
    lin = torch.nn.Linear(4, 1)
    logits_var = lin(torch.randn(B * V, 4)).reshape(B, V)
    p_nom = torch.rand(B) * 0.5          # all far below tanh(2)
    y, w = (torch.rand(B) > 0.5).float(), torch.randn(B)
    cfg = train_dnn.load_config(str(CONFIG))

    pen = train_dnn.adversarial_penalty(
        logits_var, p_nom, y, w, None, cfg,
        _adv(label_only=True, high_score_cut=2.0, mask_label_term=True))
    assert float(pen) == 0.0
    assert pen.requires_grad
    pen.backward()
    assert torch.all(lin.weight.grad == 0)


def test_unmasked_label_term_survives_an_empty_selection():
    """With mask_label_term off, an empty cut must not silently kill the label term.

    The consistency term is dropped for that batch, but the label term does not
    depend on the cut and must still contribute -- otherwise variant B would
    quietly become a no-op on exactly the batches where the model is worst.
    """
    torch.manual_seed(26)
    B, V = 32, 3
    logits_var, p_nom = torch.randn(B, V), torch.rand(B) * 0.5
    y, w = (torch.rand(B) > 0.5).float(), torch.randn(B)
    cfg = train_dnn.load_config(str(CONFIG))

    got = train_dnn.adversarial_penalty(
        logits_var, p_nom, y, w, None, cfg, _adv(high_score_cut=2.0))
    expected = torch.zeros(())
    for v in range(V):
        expected = expected + _reference_original_loss(
            logits_var[:, v], y, w, None, cfg.label_smoothing, cfg.normalize_in_batch)
    assert float(got) != 0.0
    assert torch.allclose(got, expected, rtol=0, atol=0)


def test_new_flags_default_off_leave_the_specified_formula_untouched():
    """label_only / mask_label_term must not perturb any pre-existing behaviour."""
    torch.manual_seed(27)
    B, V = 40, 5
    logits_var, p_nom = torch.randn(B, V), torch.sigmoid(torch.randn(B))
    y, w = (torch.rand(B) > 0.5).float(), torch.randn(B)
    cfg = train_dnn.load_config(str(CONFIG))
    for kw in ({}, {"high_score_cut": 1.0}):
        a = train_dnn.adversarial_penalty(logits_var, p_nom, y, w, None, cfg, _adv(**kw))
        b = train_dnn.adversarial_penalty(
            logits_var, p_nom, y, w, None, cfg,
            _adv(label_only=False, mask_label_term=False, **kw))
        assert torch.equal(a, b)
