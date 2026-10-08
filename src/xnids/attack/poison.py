"""C17 attacks on a self-updating unsupervised adapter (Build Guide C17). The attacker injects a fraction rho of the
traffic each adaptation round learns from; they cannot touch labels or the model.

  frog-boiling   flows of the TARGET family, moved a little further toward benign every round (interpolation in
                 signed-log feature space toward randomly paired benign flows): alpha_r = alpha_max * r / rounds
  stat-skew      benign-looking flows whose top-k most target-discriminative features carry extreme values on the
                 target family's side, to drag the BatchNorm means and inflate the variances

  frog_adaptive  (beyond the Build Guide; grey-box) the attacker can QUERY the deployed model's score. Each round
                 it injects target-family flows, interpolated toward benign by a random amount, that the CURRENT model
                 scores just below its alert threshold. Tent's entropy minimisation then pushes exactly the flows at
                 the boundary toward "benign": a frog-boil that follows the model.

All are built from the attacker's OWN material (the source TRAIN split here); evaluation uses held-out test flows.
"""

import numpy as np
import polars as pl
from scipy.stats import ks_2samp


def slog(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def unslog(z: np.ndarray) -> np.ndarray:
    return np.sign(z) * np.expm1(np.abs(z))


class FrogBoiling:
    name = "frog"

    def __init__(self, target: pl.DataFrame, benign: pl.DataFrame, rounds: int, alpha_max: float = 0.8) -> None:
        self.T, self.B = target.to_numpy().astype(np.float64), benign.to_numpy().astype(np.float64)
        self.cols, self.rounds, self.alpha_max = target.columns, rounds, alpha_max

    def alpha(self, r: int) -> float:
        return self.alpha_max * r / self.rounds

    def flows(self, n: int, r: int, rng: np.random.Generator, model=None) -> pl.DataFrame:
        t = self.T[rng.integers(len(self.T), size=n)]
        b = self.B[rng.integers(len(self.B), size=n)]
        a = self.alpha(r)
        z = (1 - a) * slog(t) + a * slog(b)
        return pl.DataFrame(unslog(z).astype(np.float32), schema=self.cols, orient="row")


class StatSkew:
    name = "skew"

    def __init__(self, target: pl.DataFrame, benign: pl.DataFrame, k: int = 5, scale: float = 5.0) -> None:
        self.cols = benign.columns
        self.B = benign.to_numpy().astype(np.float64)
        T = target.to_numpy().astype(np.float64)
        d = [ks_2samp(self.B[:, j], T[:, j]).statistic if np.ptp(self.B[:, j]) + np.ptp(T[:, j]) > 0 else 0.0
             for j in range(self.B.shape[1])]
        self.features = [int(j) for j in np.argsort(d)[::-1][:k]]          # most target-vs-benign discriminative
        self.values = {}
        for j in self.features:
            side = np.sign(np.median(T[:, j]) - np.median(self.B[:, j])) or 1.0
            ext = np.max(np.abs(np.concatenate([self.B[:, j], T[:, j]])))
            self.values[j] = side * max(ext, 1.0) * scale                 # far beyond anything seen, target side
        self.feature_names = [self.cols[j] for j in self.features]

    def flows(self, n: int, r: int, rng: np.random.Generator, model=None) -> pl.DataFrame:
        X = self.B[rng.integers(len(self.B), size=n)].copy()
        for j, v in self.values.items():
            X[:, j] = v
        return pl.DataFrame(X.astype(np.float32), schema=self.cols, orient="row")


class AdaptiveFrog:
    name = "frog_adaptive"

    def __init__(self, target: pl.DataFrame, benign: pl.DataFrame, candidates: int = 20000) -> None:
        self.T, self.B = target.to_numpy().astype(np.float64), benign.to_numpy().astype(np.float64)
        self.cols, self.candidates = target.columns, candidates

    def flows(self, n: int, r: int, rng: np.random.Generator, model=None) -> pl.DataFrame:
        if model is None:
            raise ValueError("the adaptive attacker needs query access to the deployed model")
        m = self.candidates
        t = self.T[rng.integers(len(self.T), size=m)]
        b = self.B[rng.integers(len(self.B), size=m)]
        a = rng.random(m)[:, None]
        X = pl.DataFrame(unslog((1 - a) * slog(t) + a * slog(b)).astype(np.float32), schema=self.cols, orient="row")
        s = model.score(X)
        below = np.flatnonzero(s < model.threshold)
        pick = below[np.argsort(-s[below])][:n]                    # just under the threshold
        if len(pick) < n:                                          # not enough: the lowest-scoring rest
            rest = np.setdiff1d(np.arange(m), pick)
            pick = np.concatenate([pick, rest[np.argsort(s[rest])][: n - len(pick)]])
        return X[np.sort(pick)]


ATTACKS = {"frog": FrogBoiling, "skew": StatSkew, "frog_adaptive": AdaptiveFrog}
