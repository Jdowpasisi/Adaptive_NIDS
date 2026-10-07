"""C9 adapter interface. Every adapter takes a source-trained bundle and returns a NEW bundle (the original is never
modified), so C10 can apply every adapter to the same starting point and C14 can register the result as a candidate.

    ctx = AdaptContext(src_train=(X, y), src_val=(X, y), tgt_pool=X_unlabelled, tgt_labelled=None, seed=0)
    new_bundle = ADAPTERS["adabn"]().adapt(bundle, ctx)

Threshold rule (Build Guide rule 4): the threshold never comes from target TEST data.
  - input / normalisation adapters (scaling, AdaBN, Tent) keep the frozen source threshold;
  - adapters that retrain the network (CORAL, DANN) re-pick it on SOURCE validation with the adapted model;
  - few-shot may re-pick it on its purchased label budget, and records that it did (meta["threshold_source"]).
"""

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from xnids.eval import metrics
from xnids.models.bundle import Bundle


@dataclass
class AdaptContext:
    src_train: tuple[pl.DataFrame, np.ndarray]
    src_val: tuple[pl.DataFrame, np.ndarray]
    tgt_pool: pl.DataFrame                                  # target TRAIN split, labels never read
    tgt_labelled: tuple[pl.DataFrame, np.ndarray] | None = None
    seed: int = 0
    target_dr: float = 0.95
    params: dict = field(default_factory=dict)


class Adapter(ABC):
    name: str = "base"
    needs_labels: bool = False
    needs_gradients: bool = False                          # only for torch models with BatchNorm (MLP)

    def __init__(self, **params) -> None:
        self.params = params

    @abstractmethod
    def adapt(self, bundle: Bundle, ctx: AdaptContext) -> Bundle: ...

    @staticmethod
    def clone(bundle: Bundle, note: str, **meta) -> Bundle:
        b = copy.deepcopy(bundle)
        b.path = None
        b.meta = {**bundle.meta, "parent_version": bundle.meta.get("version"), "adapter": note, **meta}
        b.meta["version"] = f"{bundle.meta.get('version', 'model')}+{note}"
        return b

    @staticmethod
    def rethreshold_on_source_val(bundle: Bundle, ctx: AdaptContext) -> Bundle:
        X, y = ctx.src_val
        bundle.threshold = metrics.threshold_at_dr(y, bundle.score(X), ctx.target_dr)
        bundle.meta["threshold_source"] = "source validation (re-picked after adaptation)"
        return bundle
