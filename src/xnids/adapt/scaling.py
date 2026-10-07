"""Per-domain normalisation (C7 normalisation variant; reused by the C9 per-domain scaling adapter).

Each domain is mapped through its OWN empirical CDF (a rank / quantile transform to [0, 1]): the source transform is
fit on source train, a target's transform on that target's unlabelled pool (its train split). The model therefore
sees "where a value sits in its own network's distribution" instead of the raw value, which removes per-lab shifts in
location and scale at the cost of any information carried by absolute values. No target labels are used.
"""

import numpy as np
from sklearn.preprocessing import QuantileTransformer

from xnids.adapt.base import AdaptContext, Adapter
from xnids.models.bundle import Bundle, InputMap


class DomainRankNorm:
    def __init__(self, n_quantiles: int = 1000, subsample: int = 200_000, seed: int = 0) -> None:
        self.qt = QuantileTransformer(n_quantiles=n_quantiles, output_distribution="uniform", subsample=subsample,
                                      random_state=seed)

    def fit(self, X: np.ndarray) -> "DomainRankNorm":
        self.qt.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.qt.transform(X).astype(np.float32)


# ---------------------------------------------------------------- C9 per-domain scaling adapter


def _slog_moments(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.sign(Z) * np.log1p(np.abs(Z))
    mu, sd = z.mean(0), z.std(0)
    sd[sd == 0] = 1.0
    return mu, sd


class ScalingAdapter(Adapter):
    """Per-domain scaling: re-estimate each feature's location and scale on the target's unlabelled pool and map
    target traffic onto the source's scale (in signed-log space) before the unchanged model. Equivalent to refitting
    the model's standardiser on the target for models that standardise, and works for every model (trees included).
    The median imputer is likewise refit on the target pool. No labels; the frozen threshold is kept."""
    name = "scaling"

    def adapt(self, bundle: Bundle, ctx: AdaptContext) -> Bundle:
        from xnids.data.clean import MedianImputer

        b = self.clone(bundle, "scaling")
        feats = b.preprocess.features_in
        b.preprocess.imputer = MedianImputer().fit(ctx.tgt_pool.select(feats))      # target medians for NaN / inf
        mu_s, sd_s = _slog_moments(bundle.preprocess.transform(ctx.src_train[0]))
        mu_t, sd_t = _slog_moments(b.preprocess.transform(ctx.tgt_pool))
        b.input_map = InputMap(mu_t, sd_t, mu_s, sd_s)
        b.meta["threshold_source"] = "frozen source threshold"
        return b
