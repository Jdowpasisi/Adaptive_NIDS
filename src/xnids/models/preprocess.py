"""Run-level preprocessing, fit on SOURCE TRAIN only and stored in the model bundle.

    Preprocessor: inf -> NaN -> source-train median (clean.MedianImputer), then drop columns that are constant in
    source train. Model-specific scaling (StandardScaler, signed log) lives inside each model wrapper.

`slog` is a signed log1p. Plain log1p (the Build Guide's MLP recipe) maps the -1 "not applicable" sentinel to -inf
and is undefined for the negative extractor bugs kept in C3, so the neural models use sign(x) * log1p(|x|), which
equals log1p(x) for x >= 0.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl

from xnids.data.clean import MedianImputer, constant_cols


def slog(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


class Preprocessor:
    def __init__(self) -> None:
        self.imputer: MedianImputer | None = None
        self.features_in: list[str] = []
        self.features_out: list[str] = []
        self.dropped_constant: list[str] = []

    def fit(self, X: pl.DataFrame) -> "Preprocessor":
        self.features_in = list(X.columns)
        self.imputer = MedianImputer().fit(X)
        self.dropped_constant = constant_cols(self.imputer.transform(X), self.features_in)
        self.features_out = [c for c in self.features_in if c not in self.dropped_constant]
        return self

    def transform(self, X: pl.DataFrame) -> np.ndarray:
        missing = set(self.features_in) - set(X.columns)
        if missing:
            raise ValueError(f"Preprocessor: input is missing features {sorted(missing)}")
        return self.imputer.transform(X.select(self.features_in)).select(self.features_out).to_numpy().astype(
            np.float32)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({
            "features_in": self.features_in, "features_out": self.features_out,
            "dropped_constant": self.dropped_constant, "imputer": self.imputer.to_dict()}, indent=1))

    @classmethod
    def load(cls, path: Path) -> "Preprocessor":
        d = json.loads(Path(path).read_text())
        p = cls()
        p.features_in, p.features_out, p.dropped_constant = d["features_in"], d["features_out"], d["dropped_constant"]
        p.imputer = MedianImputer.from_dict(d["imputer"])
        return p
