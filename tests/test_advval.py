import numpy as np
import pandas as pd

from xnids.analysis import advval

PARAMS = {"n_estimators": 60, "learning_rate": 0.1, "num_leaves": 15, "n_jobs": 2}


def _domains(shift_cols: dict[str, float], n: int = 3000, seed: int = 0):
    rng = np.random.default_rng(seed)
    cols = ["a", "b", "c", "d"]
    Xs = pd.DataFrame(rng.normal(size=(n, 4)), columns=cols)
    Xt = pd.DataFrame(rng.normal(size=(n, 4)), columns=cols)
    for c, shift in shift_cols.items():
        Xt[c] += shift
    return Xs, Xt


def test_identical_domains_auc_near_half():
    Xs, Xt = _domains({})
    assert abs(advval.domain_auc(Xs, Xt, 0, PARAMS, folds=3).auc - 0.5) < 0.05


def test_shifted_feature_is_detected_and_ranked_first():
    Xs, Xt = _domains({"c": 3.0})
    res = advval.domain_auc(Xs, Xt, 0, PARAMS, folds=3)
    assert res.auc > 0.95 and res.importance.index[0] == "c"


def test_iterative_removal_strips_lab_features_in_order_and_stops():
    Xs, Xt = _domains({"c": 4.0, "a": 1.5})
    curve = advval.iterative_removal(Xs, Xt, 0, PARAMS, folds=3, auc_stop=0.7, max_removed=3)
    assert curve["removed"].dropna().tolist() == ["c", "a"]    # strongest shift first, then stop
    assert curve["auc"].iloc[-1] < 0.7 and curve["auc"].iloc[0] > 0.95


def test_consensus_ranking():
    curves = pd.DataFrame({"seed": [0, 0, 0, 1, 1, 1], "step": [0, 1, 2, 0, 1, 2],
                           "removed": [None, "x", "y", None, "y", "z"]})
    assert advval.consensus_ranking(curves, max_removed=10) == ["y", "x", "z"]
