"""C10: drift windows for the adaptation logs.

A window is a sample of traffic arriving after deployment. It is split in two halves:
  adapt half: what an adapter may see (unlabelled; few-shot may buy labels from it)
  eval half:  held out, labelled, used only to measure the outcome of each action.
Windows are drawn from the target's TRAIN split (its unlabelled pool), never from any test split.

Kinds:
  random  plain sample of the target
  mix     attack mix re-weighted: benign share ~ U[benign_share], attack families ~ Dirichlet(1)
  blend   source validation flows and target flows, target share ~ U[blend_share] (mild drift, where waiting may win)
  time    a contiguous time slice of the target (only when it has timestamps): adapt = earlier half, eval = later half
"""

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class Pool:
    X: pl.DataFrame
    y: np.ndarray
    family: np.ndarray
    ts: np.ndarray | None = None


@dataclass
class Window:
    kind: str
    size: int
    target_share: float
    X_adapt: pl.DataFrame
    y_adapt: np.ndarray
    X_eval: pl.DataFrame
    y_eval: np.ndarray


def _take(p: Pool, idx: np.ndarray) -> tuple[pl.DataFrame, np.ndarray]:
    return p.X[idx], p.y[idx]


def _mix_indices(p: Pool, n: int, rng: np.random.Generator, benign_share: tuple[float, float]) -> np.ndarray:
    fams = [f for f in np.unique(p.family) if f != "Benign"]
    b = rng.uniform(*benign_share)
    w = rng.dirichlet(np.ones(len(fams))) if fams else np.array([])
    counts = {"Benign": int(round(b * n)), **{f: int(round((1 - b) * n * wi)) for f, wi in zip(fams, w, strict=True)}}
    idx = []
    for f, c in counts.items():
        if c <= 0:
            continue
        rows = np.flatnonzero(p.family == f)
        idx.append(rng.choice(rows, c, replace=c > len(rows)))     # rare families are resampled with replacement
    return np.concatenate(idx)


def make_windows(tgt: Pool, src: Pool, cfg: dict, seed: int) -> list[Window]:
    """cfg: the `windows` block of configs/adapt/c10.yaml."""
    rng = np.random.default_rng(seed)
    kinds = dict(cfg["kinds"])
    if tgt.ts is not None:                                     # a third of random + mix become time slices
        moved = (kinds.get("random", 0) + kinds.get("mix", 0)) / 3
        kinds["random"], kinds["mix"] = kinds["random"] * 2 / 3, kinds["mix"] * 2 / 3
        kinds["time"] = moved
    names, probs = list(kinds), np.array(list(kinds.values()), float)
    order = np.argsort(tgt.ts) if tgt.ts is not None else None
    out = []
    for _ in range(cfg["per_pair"]):
        kind = names[rng.choice(len(names), p=probs / probs.sum())]
        size = int(rng.choice(cfg["sizes"]))
        share = 1.0
        if kind == "random":
            idx_t, idx_s = rng.choice(tgt.X.height, size, replace=False), None
        elif kind == "mix":
            idx_t, idx_s = _mix_indices(tgt, size, rng, tuple(cfg["benign_share"])), None
        elif kind == "blend":
            share = float(rng.uniform(*cfg["blend_share"]))
            nt = int(round(share * size))
            idx_t = rng.choice(tgt.X.height, nt, replace=False)
            idx_s = rng.choice(src.X.height, size - nt, replace=False)
        elif kind == "time":
            start = int(rng.integers(0, max(1, tgt.X.height - size)))
            idx_t, idx_s = order[start:start + size], None
        else:
            raise ValueError(kind)
        Xt, yt = _take(tgt, idx_t)
        if idx_s is not None:
            Xs, ys = _take(src, idx_s)
            X, y = pl.concat([Xt, Xs]), np.concatenate([yt, ys])
        else:
            X, y = Xt, yt
        if kind != "time":                                     # time windows keep their order: adapt on the past
            perm = rng.permutation(X.height)
            X, y = X[perm], y[perm]
        h = X.height // 2
        out.append(Window(kind, X.height, share, X[:h], y[:h], X[h:], y[h:]))
    return out
