"""Maximum Mean Discrepancy two-sample test (RBF kernel, permutation p-value), in torch.

Replaces alibi_detect.cd.MMDDrift (dropped in C0 because current alibi-detect releases conflict with the stack).
The reference sample, its standardisation, optional PCA and the kernel bandwidth (median heuristic) are all fixed at
fit() time from source data only. MMD^2 is the unbiased estimator; the p-value is the share of label permutations
of the pooled sample whose MMD^2 is at least the observed one (with the +1 correction).
"""

import numpy as np
import torch

from xnids.models.preprocess import slog


class MMDTest:
    def __init__(self, n_ref: int = 2000, n_win: int = 2000, n_perm: int = 200, pca_dims: int | None = 20,
                 seed: int = 0, device: str = "auto") -> None:
        self.n_ref, self.n_win, self.n_perm, self.pca_dims, self.seed = n_ref, n_win, n_perm, pca_dims, seed
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device

    def _project(self, X: np.ndarray) -> torch.Tensor:
        Z = (slog(X.astype(np.float64)) - self.mean) / self.std
        if self.components is not None:
            Z = Z @ self.components
        return torch.as_tensor(Z, dtype=torch.float32, device=self.device)

    def fit(self, ref: np.ndarray) -> "MMDTest":
        rng = np.random.default_rng(self.seed)
        R = ref[rng.choice(len(ref), min(self.n_ref, len(ref)), replace=False)]
        Z = slog(R.astype(np.float64))
        self.mean, self.std = Z.mean(0), Z.std(0)
        self.std[self.std == 0] = 1.0
        Z = (Z - self.mean) / self.std
        self.components = None
        if self.pca_dims and Z.shape[1] > self.pca_dims:
            _, _, vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
            self.components = vt[: self.pca_dims].T
        self.ref = self._project(R)
        d2 = torch.cdist(self.ref, self.ref).pow(2)
        self.gamma = 1.0 / float(d2[d2 > 0].median()) if (d2 > 0).any() else 1.0
        return self

    @staticmethod
    def _mmd2(K: torch.Tensor, n: int) -> torch.Tensor:
        m = K.shape[0] - n
        kxx = (K[:n, :n].sum() - K[:n, :n].diagonal().sum()) / (n * (n - 1))
        kyy = (K[n:, n:].sum() - K[n:, n:].diagonal().sum()) / (m * (m - 1))
        return kxx + kyy - 2 * K[:n, n:].mean()

    def test(self, win: np.ndarray) -> tuple[float, float]:
        """(MMD^2, permutation p-value) of the window against the fixed reference."""
        rng = np.random.default_rng(self.seed + len(win))
        W = win[rng.choice(len(win), min(self.n_win, len(win)), replace=False)]
        Z = torch.cat([self.ref, self._project(W)])
        K = torch.exp(-self.gamma * torch.cdist(Z, Z).pow(2))
        n = len(self.ref)
        obs = self._mmd2(K, n)
        gen = torch.Generator(device=self.device).manual_seed(self.seed)
        ge = 0
        for _ in range(self.n_perm):
            p = torch.randperm(len(Z), generator=gen, device=self.device)
            ge += int(self._mmd2(K[p][:, p], n) >= obs)
        return float(obs), (ge + 1) / (self.n_perm + 1)
