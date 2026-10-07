"""Per-domain normalisation (C7 normalisation variant; reused by the C9 per-domain scaling adapter).

Each domain is mapped through its OWN empirical CDF (a rank / quantile transform to [0, 1]): the source transform is
fit on source train, a target's transform on that target's unlabelled pool (its train split). The model therefore
sees "where a value sits in its own network's distribution" instead of the raw value, which removes per-lab shifts in
location and scale at the cost of any information carried by absolute values. No target labels are used.
"""

import numpy as np
from sklearn.preprocessing import QuantileTransformer


class DomainRankNorm:
    def __init__(self, n_quantiles: int = 1000, subsample: int = 200_000, seed: int = 0) -> None:
        self.qt = QuantileTransformer(n_quantiles=n_quantiles, output_distribution="uniform", subsample=subsample,
                                      random_state=seed)

    def fit(self, X: np.ndarray) -> "DomainRankNorm":
        self.qt.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.qt.transform(X).astype(np.float32)
