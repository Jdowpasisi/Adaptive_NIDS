"""Helpers for adapters that work inside the MLP (BatchNorm statistics, gradients on the penultimate layer)."""

import numpy as np
import polars as pl
import torch
from torch import nn

from xnids.models.bundle import Bundle
from xnids.models.mlp import MLP


def require_mlp(bundle: Bundle, adapter: str) -> MLP:
    if not isinstance(bundle.model, MLP):
        raise TypeError(f"{adapter} needs the MLP (BatchNorm + gradients); got {bundle.model.name}")
    return bundle.model


def tensor(bundle: Bundle, X: pl.DataFrame, n: int | None = None, seed: int = 0) -> torch.Tensor:
    """Network-input tensor for X (preprocess -> optional input map -> the MLP's own scaler), optionally sampled."""
    if n is not None and X.height > n:
        X = X[np.sort(np.random.default_rng(seed).choice(X.height, n, replace=False))]
    m = bundle.model
    return torch.from_numpy(m.scaler.transform(bundle.model_input(X))).to(m.device)


def batches(T: torch.Tensor, bs: int, seed: int, shuffle: bool = True):
    idx = torch.randperm(len(T), generator=torch.Generator().manual_seed(seed)) if shuffle else torch.arange(len(T))
    for i in range(0, len(T), bs):
        b = idx[i:i + bs]
        if len(b) > 1:
            yield T[b.to(T.device)]


def bn_layers(net: nn.Module) -> list[nn.BatchNorm1d]:
    return [m for m in net.modules() if isinstance(m, nn.BatchNorm1d)]


def predicted_attack_share(bundle: Bundle, T: torch.Tensor) -> float:
    net = bundle.model.net
    net.eval()
    with torch.no_grad():
        p = torch.sigmoid(torch.cat([net(b) for b in batches(T, 65536, 0, shuffle=False)]))
    return float((p >= bundle.threshold).float().mean())
