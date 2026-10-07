"""C8 drift detectors on synthetic data with known answers (CPU or GPU, seconds)."""

import numpy as np
import polars as pl
import pytest

from xnids.drift import explain, ks, trigger
from xnids.drift.adwin import ConfidenceADWIN
from xnids.drift.mmd import MMDTest
from xnids.drift.monitor import DriftMonitor
from xnids.models import zoo
from xnids.models.bundle import Bundle
from xnids.models.preprocess import Preprocessor

RNG = np.random.default_rng(0)
NAMES = ["a", "b", "c"]


def test_ks_flags_only_the_shifted_feature():
    ref = RNG.normal(size=(20000, 3))
    win = RNG.normal(size=(5000, 3))
    win[:, 1] += 0.5
    k = ks.ks_features(ref, win, NAMES)
    assert ks.significant(k, 0.05) == ["b"] and ks.ranked(k)[0] == "b"
    assert ks.significant(k, 0.05, min_stat=0.5) == []          # effect-size floor removes a moderate shift


def test_ks_quiet_on_same_distribution():
    k = ks.ks_features(RNG.normal(size=(20000, 3)), RNG.normal(size=(5000, 3)), NAMES)
    assert ks.significant(k, 0.05) == []


def test_mmd_detects_joint_shift_ks_misses():
    # same marginals, different correlation: per-feature KS cannot see it, MMD can
    ref = RNG.normal(size=(4000, 2))
    z = RNG.normal(size=4000)
    win = np.c_[z, z + 0.1 * RNG.normal(size=4000)] / np.sqrt([1, 1.01])
    m = MMDTest(n_ref=1000, n_win=1000, n_perm=100, pca_dims=None, device="cpu").fit(ref)
    assert m.test(win)[1] < 0.05
    assert m.test(RNG.normal(size=(4000, 2)))[1] > 0.05
    assert ks.significant(ks.ks_features(ref, win, ["x", "y"]), 0.05) == []


def test_adwin_cuts_after_mean_change():
    a = ConfidenceADWIN(0.002)
    assert a.update(np.clip(RNG.normal(0.95, 0.02, 3000), 0, 1)) == []
    cuts = a.update(np.clip(RNG.normal(0.70, 0.02, 3000), 0, 1))
    assert cuts and 3000 <= cuts[0] < 3300


def test_atc_recovers_error_rate():
    conf_val = RNG.uniform(0.5, 1, 10000)
    correct = conf_val > 0.6                     # errors live at low confidence
    atc = trigger.ATC().fit(conf_val, correct)
    assert atc.source_err == pytest.approx(0.2, abs=0.02)
    assert atc.estimate_error(RNG.uniform(0.5, 0.75, 5000)) == pytest.approx(0.4, abs=0.03)
    worth, per_hour = trigger.worth_acting(0.1, trigger.CostConfig(flows_per_hour=1000, adaptation_cost=1e9))
    assert per_hour == pytest.approx(100) and not worth


def test_explain_sentences():
    ref = RNG.uniform(1, 2, 5000)
    assert "3.0x higher" in explain.describe(ref, ref * 3, "flow_duration", 0.9)
    z = np.r_[np.zeros(4000), ref[:1000]]
    assert "zero in 80%" in explain.describe(ref, z, "fwd_pkt_cnt", 0.8)
    assert explain.pretty("fwd_pkt_len_tot") == "forward packet length total"


def test_monitor_end_to_end_flags_switch_only():
    def frame(n, shift, return_y=False):
        y = (RNG.random(n) < 0.3).astype(int)                 # same 30% attack mix in every window
        X = RNG.normal(size=(n, 3)).astype(np.float32)
        X[:, 1] += 2.0 * y                                     # attacks differ on feature b
        X[:, 0] += shift                                       # the drift: feature a moves
        df = pl.DataFrame(X, schema=NAMES)
        return (df, y) if return_y else df

    Xs, y = frame(20000, 0.0, return_y=True)
    pre = Preprocessor().fit(Xs)
    m = zoo.build("lda", {}, device="cpu").fit(pre.transform(Xs), y, pre.transform(Xs), y)
    s = m.score(pre.transform(Xs))
    b = Bundle(m, pre, float(np.quantile(s[y == 1], 0.05)), NAMES, {})
    mon = DriftMonitor(b, Xs, y, {"reference": {"n": 5000}, "mmd": {"n_perm": 50, "n_ref": 500, "n_win": 500,
                                                                     "pca_dims": None}})
    reps = mon.run([frame(3000, 0.0), frame(3000, 0.0), frame(3000, 1.5), frame(3000, 1.5)])
    assert [r.detectors["ks_effect"] for r in reps] == [False, False, True, True]
    assert reps[2].top_features[0] == "a" and reps[2].detectors["combined"]
    assert set(reps[0].features()) >= {"ks_max", "mmd_p", "adwin_flag", "est_fpr_rise", "attack_share_pred"}
    assert reps[2].to_json().startswith("{")
