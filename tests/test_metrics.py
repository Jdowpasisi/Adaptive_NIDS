"""Hand-computed toy cases for the C5 metrics."""

import numpy as np
import pytest

from xnids.eval import metrics


def test_threshold_catches_at_least_dr():
    # 20 attacks with scores 0.05 .. 1.00; 95% DR -> k = floor(0.05 * 20) = 1 -> t = second-lowest = 0.10
    s = np.arange(1, 21) / 20
    y = np.ones(20)
    assert metrics.threshold_at_dr(y, s, 0.95) == pytest.approx(0.10)
    assert metrics.rates(y, s, 0.10)[1] == pytest.approx(19 / 20)


def test_threshold_ignores_benign_scores():
    y = np.array([0, 0, 0, 1, 1, 1, 1])
    s = np.array([0.99, 0.98, 0.1, 0.2, 0.3, 0.4, 0.5])
    # 4 attacks, k = floor(0.05 * 4) = 0 -> t = lowest attack score 0.2, so all 4 attacks are caught
    assert metrics.threshold_at_dr(y, s, 0.95) == pytest.approx(0.2)


def test_rates_and_confusion_by_hand():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    s = np.array([.9, .8, .7, .1, .95, .6, .3, .2, .1, .05])
    tp, fp, tn, fn = metrics.confusion(y, s, 0.5)       # flagged: .9 .8 .7 (attacks), .95 .6 (benign)
    assert (tp, fp, tn, fn) == (3, 2, 4, 1)
    fpr, dr = metrics.rates(y, s, 0.5)
    assert fpr == pytest.approx(2 / 6) and dr == pytest.approx(3 / 4)


def test_mcc_by_hand():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    s = np.array([.9, .8, .7, .1, .95, .6, .3, .2, .1, .05])
    # tp=3 fp=2 tn=4 fn=1 -> (12 - 2) / sqrt(5 * 4 * 6 * 5) = 10 / sqrt(600)
    assert metrics.mcc(y, s, 0.5) == pytest.approx(10 / np.sqrt(600))
    assert metrics.mcc(y, y.astype(float), 0.5) == pytest.approx(1.0)
    assert metrics.mcc(y, np.zeros(10), 0.5) == 0.0      # nothing flagged: degenerate, defined as 0


def test_score_at_threshold_is_flagged():
    y = np.array([1, 0])
    s = np.array([0.5, 0.5])
    assert metrics.rates(y, s, 0.5) == (1.0, 1.0)        # ties at t are flagged (score >= t)


def test_oracle_vs_frozen():
    # target attacks score lower than source: the frozen threshold misses them, the oracle re-picks
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(1000), np.ones(1000)]
    s = np.r_[rng.uniform(0, 0.3, 1000), rng.uniform(0.2, 0.5, 1000)]
    ev = metrics.evaluate(y, s, t=0.6)
    assert ev["dr_at_thr"] == 0 and ev["fpr_at_thr"] == 0
    assert ev["oracle_fpr_at_dr"] < 0.5 and ev["roc_auc"] > 0.9


def test_evaluate_keys_and_no_accuracy():
    y = np.array([0, 1, 0, 1])
    ev = metrics.evaluate(y, np.array([.1, .9, .2, .8]), 0.5)
    assert ev["fpr_at_thr"] == 0 and ev["dr_at_thr"] == 1 and ev["pr_auc"] == 1 and ev["mcc_at_thr"] == 1
    assert "accuracy" not in ev and ev["n"] == 4 and ev["n_attack"] == 2


def test_no_attacks_raises():
    with pytest.raises(ValueError):
        metrics.threshold_at_dr(np.zeros(5), np.ones(5))
