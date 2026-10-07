"""C11 adapter selector: predict each action's net utility from the label-free drift features, pick the best, or wait.

Decision problem. The deployed model is the source MLP. In a drift window the actions are: wait, the MLP adapters,
and "retrain an XGBoost with the bought labels" (xgb few-shot). The logs (C10) are full-information: every action's
outcome on every window is known, so this is supervised regression, not a bandit or RL problem (Build Guide C10/C11).

Utility of an action in a window:
    gain = metric_after - metric_before(deployed MLP) - label_cost * labels_used
with metric = MCC by default. The Build Guide uses delta FPR; C6 showed FPR cannot see the silent failures (a model
that stops alerting has a LOWER FPR), so FPR is kept as the comparison utility (utility="fpr").

    sel = Selector(drift_cols, label_cost=2e-4).fit(train_log)
    action, gain, std = sel.choose(report.features(), actions)
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

DRIFT_COLS = ["ks_max", "ks_mean", "n_ks_sig", "mmd_stat", "mmd_p", "adwin_flag", "mean_conf", "entropy",
              "est_fpr_rise", "attack_share_pred"]


def decision_table(log: pd.DataFrame, deployed: str = "mlp", utility: str = "mcc",
                   label_cost: float = 2e-4) -> pd.DataFrame:
    """One row per (window, action) with `gain`. Actions = deployed-model adapters (incl. wait) + other models'
    non-wait actions (switching to them), all measured against the deployed model's before-state."""
    before = log[(log.model == deployed) & (log.adapter == "wait")].set_index("scenario_id")
    rows = log[(log.model == deployed) | (log.adapter != "wait")].copy()
    rows["action"] = np.where(rows.model == deployed, rows.adapter, rows.model + ":" + rows.adapter)
    b = before.reindex(rows.scenario_id)
    if utility == "mcc":
        raw = rows.mcc_after.to_numpy() - b.mcc_before.to_numpy()
    elif utility == "fpr":
        raw = b.fpr_before.to_numpy() - rows.fpr_after.to_numpy()
    else:
        raise ValueError(utility)
    rows["gain"] = raw - label_cost * rows.labels_used.to_numpy()
    rows.loc[rows.action == "wait", "gain"] = 0.0
    return rows.reset_index(drop=True)


class Selector:
    def __init__(self, drift_cols: list[str] = DRIFT_COLS, n_models: int = 5, margin: float = 0.005,
                 seed: int = 0, **lgbm) -> None:
        self.drift_cols, self.n_models, self.margin, self.seed = drift_cols, n_models, margin, seed
        self.lgbm = {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20,
                     "verbose": -1} | lgbm

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.drift_cols].astype(float).copy()
        X["action"] = pd.Categorical(df["action"], categories=self.actions_)
        return X

    def fit(self, table: pd.DataFrame) -> "Selector":
        """table: decision_table() rows (needs drift columns, action, gain, scenario_id)."""
        self.actions_ = sorted(table.action.unique())
        X, y = self._X(table), table.gain.to_numpy()
        rng = np.random.default_rng(self.seed)
        scen = table.scenario_id.unique()
        self.models_ = []
        for k in range(self.n_models):                      # bootstrap over WINDOWS (all actions of a window together)
            pick = set(rng.choice(scen, len(scen), replace=True)) if self.n_models > 1 else set(scen)
            m = table.scenario_id.isin(pick).to_numpy()
            self.models_.append(LGBMRegressor(random_state=self.seed + k, **self.lgbm).fit(X[m], y[m]))
        return self

    def predict(self, table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        P = np.stack([m.predict(self._X(table)) for m in self.models_])
        return P.mean(0), P.std(0)

    def choose(self, features: dict, actions: list[str] | None = None) -> tuple[str, float, float]:
        """(action, predicted gain, ensemble std); 'wait' when no action is predicted to beat it by `margin`."""
        acts = [a for a in (actions or self.actions_) if a != "wait"]
        rows = pd.DataFrame([{**features, "action": a} for a in acts])
        mu, sd = self.predict(rows)
        i = int(np.argmax(mu))
        return ("wait", 0.0, 0.0) if mu[i] < self.margin else (acts[i], float(mu[i]), float(sd[i]))

    def choose_table(self, table: pd.DataFrame) -> pd.DataFrame:
        """Vectorised choose() for every window in a decision table: one row per scenario."""
        t = table[table.action != "wait"].copy()
        t["pred"], t["pred_std"] = self.predict(t)
        best = t.loc[t.groupby("scenario_id").pred.idxmax()].set_index("scenario_id")
        best.loc[best.pred < self.margin, ["action", "pred", "pred_std"]] = ["wait", 0.0, 0.0]
        return best[["action", "pred", "pred_std"]]
