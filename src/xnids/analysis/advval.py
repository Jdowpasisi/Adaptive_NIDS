"""C4 adversarial validation: which features simply tell one lab's traffic from another's?

A LightGBM classifier is trained to predict the domain (source = 0, target = 1) from a track's features. The
out-of-fold ROC AUC measures how separable the two domains are (0.5 = indistinguishable). Iterative removal then
drops the most important feature and re-measures, until AUC < `auc_stop` or `max_removed` features are gone.
The ordered removal list is the "lab-telling" ranking passed to C7 (ablation) and C8 (drift explanations).

Data: only the TRAIN split of each dataset is used (a target's train split is its unlabelled pool). The `benign`
subset uses labels only to restrict both domains to benign traffic, so the classifier learns lab differences
rather than attack-mix differences.

CV: Build Guide's StratifiedKFold(5) with the given LightGBM settings, plus early stopping on a 10% slice of each
training fold (never the held-out fold), because lab pairs are often near-separable and converge in a few dozen
trees. Importance is LightGBM 'gain', averaged over the fold models.
"""

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from xnids.data import schema
from xnids.features import tracks
from xnids.utils import paths


def load_domain(dataset: str, track: str, subset: str, n: int, seed: int) -> pd.DataFrame:
    """Up to n rows of the dataset's train split (benign only if subset == 'benign'), sampled with `seed`."""
    feats = schema.features(track)
    split = pl.scan_parquet(paths.SPLITS / f"{dataset}.parquet").filter(pl.col("split") == "train").select("row_id")
    lf = pl.scan_parquet(tracks.processed_path(dataset, track)).join(split, on="row_id")
    if subset == "benign":
        lf = lf.filter(pl.col("y") == 0)
    elif subset != "all":
        raise ValueError(f"subset must be 'benign' or 'all', not {subset!r}")
    df = lf.select("row_id", *feats).sort("row_id").collect()
    if df.height > n:
        idx = np.sort(np.random.default_rng(seed).choice(df.height, n, replace=False))
        df = df[idx]
    return df.select(feats).to_pandas()


@dataclass
class AucResult:
    auc: float
    importance: pd.Series            # gain, averaged over folds, sorted descending
    n_trees: list[int] = field(default_factory=list)


def domain_auc(Xs: pd.DataFrame, Xt: pd.DataFrame, seed: int, params: dict, folds: int = 5,
               early_stopping: int = 30) -> AucResult:
    X = pd.concat([Xs, Xt], ignore_index=True)
    d = np.r_[np.zeros(len(Xs)), np.ones(len(Xt))]
    oof = np.zeros(len(X))
    imp = np.zeros(X.shape[1])
    trees = []
    for k, (tr, te) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, d)):
        Xtr, Xes, dtr, des = train_test_split(X.iloc[tr], d[tr], test_size=0.1, stratify=d[tr],
                                              random_state=seed + k)
        clf = lgb.LGBMClassifier(**params, random_state=seed + k, importance_type="gain", verbose=-1)
        clf.fit(Xtr, dtr, eval_X=Xes, eval_y=des, eval_metric="auc",
                callbacks=[lgb.early_stopping(early_stopping, verbose=False)])
        oof[te] = clf.predict_proba(X.iloc[te], num_iteration=clf.best_iteration_)[:, 1]
        imp += clf.feature_importances_ / folds
        trees.append(int(clf.best_iteration_ or params.get("n_estimators", 0)))
    importance = pd.Series(imp, index=X.columns).sort_values(ascending=False)
    return AucResult(float(roc_auc_score(d, oof)), importance, trees)


def iterative_removal(Xs: pd.DataFrame, Xt: pd.DataFrame, seed: int, params: dict, folds: int,
                      auc_stop: float, max_removed: int) -> pd.DataFrame:
    """One row per step: step 0 uses all features; step k has the top feature of step k-1 removed."""
    cols = list(Xs.columns)
    rows = []
    for step in range(max_removed + 1):
        res = domain_auc(Xs[cols], Xt[cols], seed, params, folds)
        top = res.importance.index[0]
        rows.append({"step": step, "n_features": len(cols), "auc": res.auc, "top_feature": top,
                     "top_gain_share": float(res.importance.iloc[0] / max(res.importance.sum(), 1e-12)),
                     "top5": ";".join(res.importance.index[:5]), "mean_trees": float(np.mean(res.n_trees))})
        if res.auc < auc_stop or step == max_removed or len(cols) == 1:
            break
        cols.remove(top)
    out = pd.DataFrame(rows)
    out["removed"] = [None] + out["top_feature"].iloc[:-1].tolist()   # feature dropped before this step
    return out


def consensus_ranking(curves: pd.DataFrame, max_removed: int) -> list[str]:
    """Order features by mean removal position across seeds (not removed = max_removed + 1)."""
    pos: dict[str, list[int]] = {}
    seeds = curves["seed"].unique()
    for s in seeds:
        removed = curves[(curves["seed"] == s) & curves["removed"].notna()].sort_values("step")["removed"]
        for i, f in enumerate(removed, start=1):
            pos.setdefault(f, []).append(i)
    score = {f: (sum(p) + (max_removed + 1) * (len(seeds) - len(p))) / len(seeds) for f, p in pos.items()}
    return sorted(score, key=lambda f: (score[f], f))
