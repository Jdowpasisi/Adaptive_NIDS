import numpy as np
import pandas as pd

from xnids.analysis.ablation import shares


def _cell(model, src, tgt, kind, k, pt, mcc, fpr=0.01, ofpr=0.5, prauc=0.5):
    return {"model": model, "source": src, "target": tgt, "kind": kind, "k": k, "pair_target": pt, "n_seeds": 3,
            "mcc_at_thr_mean": mcc, "fpr_at_thr_mean": fpr, "oracle_fpr_at_dr_mean": ofpr, "pr_auc_mean": prauc,
            "dr_at_thr_mean": 0.5}


def test_shares_pair_within_and_formula():
    ref = pd.DataFrame([_cell("rf", "a", "a", "within", 0, None, 0.9, ofpr=0.01, prauc=0.99),
                        _cell("rf", "a", "b", "cross", 0, None, 0.1, ofpr=0.8, prauc=0.3),
                        _cell("rf", "a", "c", "cross", 0, None, 0.2, ofpr=0.8, prauc=0.3)])
    cells = pd.DataFrame([
        _cell("rf", "a", "a", "within", 3, "b", 0.85), _cell("rf", "a", "b", "cross", 3, "b", 0.5, ofpr=0.4),
        _cell("rf", "a", "a", "within", 3, "c", 0.70), _cell("rf", "a", "c", "cross", 3, "c", 0.2)])
    sh = shares(cells, ref).set_index("target")
    assert np.isclose(sh.loc["b", "share_mcc_at_thr"], (0.5 - 0.1) / (0.9 - 0.1))
    assert np.isclose(sh.loc["b", "share_oracle_fpr_at_dr"], (0.8 - 0.4) / (0.8 - 0.01))
    assert sh.loc["b", "mcc_at_thr_within_k"] == 0.85 and sh.loc["c", "mcc_at_thr_within_k"] == 0.70   # own pair
    assert sh.loc["c", "share_mcc_at_thr"] == 0.0


def test_no_collapse_gives_nan_share():
    ref = pd.DataFrame([_cell("rf", "a", "a", "within", 0, None, 0.5), _cell("rf", "a", "b", "cross", 0, None, 0.5)])
    cells = pd.DataFrame([_cell("rf", "a", "b", "cross", 1, "b", 0.6)])
    assert np.isnan(shares(cells, ref).share_mcc_at_thr.iloc[0])
