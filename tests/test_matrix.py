import numpy as np
import pandas as pd

from xnids.eval import matrix


def _long():
    rows = []
    for seed in (0, 1, 2):
        for src in ("a", "b"):
            for tgt in ("a", "b"):
                within = src == tgt
                rows.append({"track": "t", "model": "m", "source": src, "target": tgt,
                             "kind": "within" if within else "cross", "seed": seed, "run_id": f"r{seed}{src}",
                             "fpr_at_thr": 0.01 if within else 0.002, "dr_at_thr": 0.95 if within else 0.01,
                             "pr_auc": 0.99 if within else 0.3, "mcc_at_thr": (0.9 if within else 0.05) + seed * 0.01,
                             "oracle_fpr_at_dr": 0.01 if within else 0.8, "roc_auc": 0.99})
    return pd.DataFrame(rows)


def test_aggregate_mean_std_and_traceability():
    agg = matrix.aggregate(_long())
    cell = agg[(agg.source == "a") & (agg.target == "b")].iloc[0]
    assert cell.n_seeds == 3 and np.isclose(cell.mcc_at_thr_mean, 0.06)
    assert np.isclose(cell.mcc_at_thr_std, 0.01) and cell.run_ids == "r0a;r1a;r2a"


def test_h1_checks_flag_collapse_but_not_frozen_fpr():
    h1 = matrix.h1_checks(matrix.aggregate(_long()))
    assert h1.h1_mcc_worse.all() and h1.h1_oracle_fpr_higher.all()
    assert not h1.h1_frozen_fpr_higher.any()      # silent models: frozen-threshold FPR can drop cross-dataset


def test_wide_and_fmt():
    w = matrix.wide(matrix.aggregate(_long()), "t", "fpr_at_thr", pct=True)
    assert w.loc[("m", "a"), "a"] == "1.00 ± 0.00"
    assert matrix.fmt(float("nan"), 0.0) == "–"
