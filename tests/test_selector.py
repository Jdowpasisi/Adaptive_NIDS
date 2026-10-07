"""C11 selector on a synthetic log where the right action depends on a drift feature."""

import numpy as np
import pandas as pd

from xnids.select.selector import DRIFT_COLS, Selector, decision_table


def _log(n_windows=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(n_windows):
        f = {c: rng.random() for c in DRIFT_COLS}
        before = 0.3
        # rule: adabn helps when ks_max is high, fewshot always helps a bit but costs labels, scaling never helps
        out = {"wait": before, "adabn": before + (0.4 if f["ks_max"] > 0.5 else -0.2),
               "fewshot(budget=200,rule=random)": before + 0.15, "scaling": before - 0.05}
        for a, after in out.items():
            rows.append({"scenario_id": f"p{w % 4}:{w}", "pair": f"p{w % 4}", "model": "mlp", "adapter": a, **f,
                         "labels_used": 200 if a.startswith("fewshot") else 0, "mcc_before": before,
                         "mcc_after": after + rng.normal(0, 0.01), "fpr_before": 0.01, "fpr_after": 0.01})
        rows.append({"scenario_id": f"p{w % 4}:{w}", "pair": f"p{w % 4}", "model": "xgb", "adapter": "wait", **f,
                     "labels_used": 0, "mcc_before": 0.5, "mcc_after": 0.5, "fpr_before": 0.01, "fpr_after": 0.01})
    return pd.DataFrame(rows)


def test_decision_table_gain_and_actions():
    t = decision_table(_log(10), label_cost=1e-4)
    assert set(t.action) == {"wait", "adabn", "fewshot(budget=200,rule=random)", "scaling"}   # xgb:wait excluded
    fs = t[t.action.str.startswith("fewshot")].gain
    assert np.allclose(fs, 0.15 - 0.02, atol=0.05)                  # 200 labels x 1e-4 = 0.02 charged
    assert (t[t.action == "wait"].gain == 0).all()


def test_selector_learns_the_rule_and_waits_when_nothing_helps():
    train, test = decision_table(_log(400, 0)), decision_table(_log(100, 1))
    sel = Selector(n_models=3).fit(train)
    chosen = sel.choose_table(test)
    feats = test.groupby("scenario_id").ks_max.first()
    hi = chosen.loc[feats[feats > 0.6].index]
    lo = chosen.loc[feats[feats < 0.4].index]
    assert (hi.action == "adabn").mean() > 0.9
    assert (lo.action == "fewshot(budget=200,rule=random)").mean() > 0.9
    a, g, sd = sel.choose({c: 0.9 for c in DRIFT_COLS})
    assert a == "adabn" and g > 0.2 and sd >= 0
    # with a huge label cost and low ks_max nothing beats waiting
    sel2 = Selector(n_models=1).fit(decision_table(_log(400, 0), label_cost=1.0))
    assert sel2.choose({c: 0.1 for c in DRIFT_COLS})[0] == "wait"
