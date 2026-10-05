"""C3 cleaning.

Two kinds of step:
  * dataset-level, decided once before splitting and frozen in the split file:
      dedup_status(): exact duplicates of (features + label) keep one copy; conflicting duplicates
      (same features, different label) are all dropped.
  * run-level, fit on SOURCE TRAIN ONLY inside every training run (C5) and stored in the model bundle:
      MedianImputer (inf -> NaN -> source-train median) and constant_cols().

Negative sentinels are kept as values: LycoSTand and CICFlowMeter write -1 for "not applicable" (e.g. the TCP
initial window of a UDP flow). See DATASET_CARD.md for the counts.
"""

import numpy as np
import polars as pl

DEDUP_STATUS = ("keep", "duplicate", "conflict")


def dedup_status(lf: pl.LazyFrame, features: list[str], label: str = "label",
                 keep_hash: bool = False) -> pl.LazyFrame:
    """row_id -> status in {keep, duplicate, conflict} (plus the vector hash `_h` if keep_hash).

    Rows are compared on a 64-bit hash of their feature vector (NaN == NaN). A collision between two
    different vectors among ~2e7 rows has probability ~1e-5, accepted and noted in the dataset card.
    Of each group of identical (features, label) rows the lowest row_id is kept.
    """
    h = lf.select("row_id", label, pl.struct(features).hash(seed=0).alias("_h"))
    groups = h.group_by("_h").agg(pl.col(label).n_unique().alias("_n_labels"))
    return (
        h.join(groups, on="_h")
        .with_columns(_first=pl.col("row_id").min().over("_h", label))
        .select(
            "row_id",
            pl.when(pl.col("_n_labels") > 1).then(pl.lit("conflict"))
            .when(pl.col("row_id") != pl.col("_first")).then(pl.lit("duplicate"))
            .otherwise(pl.lit("keep")).alias("status"),
            *(["_h"] if keep_hash else []),
        )
    )


def constant_cols(lf: pl.LazyFrame | pl.DataFrame, features: list[str]) -> list[str]:
    """Features with a single distinct value (NaN counts as a value) in the given frame, i.e. source train."""
    counts = lf.lazy().select(pl.col(features).n_unique()).collect().row(0, named=True)
    return [c for c in features if counts[c] <= 1]


class MedianImputer:
    """inf -> NaN, then NaN -> per-column median learnt on the data passed to fit() (source train)."""

    def __init__(self) -> None:
        self.medians_: dict[str, float] | None = None

    def fit(self, X: pl.DataFrame) -> "MedianImputer":
        finite = X.select(pl.when(pl.col(c).is_finite()).then(pl.col(c)).alias(c) for c in X.columns)
        med = finite.median().row(0, named=True)
        self.medians_ = {c: (0.0 if v is None or np.isnan(v) else float(v)) for c, v in med.items()}
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        if self.medians_ is None:
            raise RuntimeError("MedianImputer.transform before fit")
        return X.with_columns(
            pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(pl.lit(m)).cast(X.schema[c]).alias(c)
            for c, m in self.medians_.items() if c in X.columns
        )

    def fit_transform(self, X: pl.DataFrame) -> pl.DataFrame:
        return self.fit(X).transform(X)

    def to_dict(self) -> dict:
        return {"medians": self.medians_}

    @classmethod
    def from_dict(cls, d: dict) -> "MedianImputer":
        imp = cls()
        imp.medians_ = dict(d["medians"])
        return imp
