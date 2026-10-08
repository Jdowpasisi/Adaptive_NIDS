"""C15: per-flow novelty against the reference, the window statistic of the calibrated live monitor.

A flow is NOVEL when its distance to the nearest reference flow (standardised MLP embedding, i.e. the network's
penultimate layer) exceeds `tau`, the `flow_quantile` of that distance for held-out source flows. The window
statistic is the novel share. Unlike KS / MMD this ignores changes of MIX (an attack burst or one busy host of the
same network) and reacts to traffic unlike anything in the reference (C15: the C8 statistics could not tell a
known 2017 attack burst from the 2018 network).
"""

import numpy as np
import torch

from xnids.models.bundle import Bundle


class Novelty:
    def __init__(self, bundle: Bundle, batch: int = 2048) -> None:
        if not hasattr(getattr(bundle.model, "net", None), "features"):
            raise ValueError("novelty needs an MLP bundle (its penultimate-layer embedding)")
        self.bundle, self.batch = bundle, batch
        self.device = bundle.model.device
        self.tau: float | None = None

    def embed(self, Z: np.ndarray) -> torch.Tensor:
        """Z = the preprocessed window (bundle.preprocess.transform)."""
        m = self.bundle.model
        T = torch.from_numpy(m.scaler.transform(Z)).to(self.device)
        with torch.no_grad():
            return torch.cat([m.net.features(T[i:i + 8192]) for i in range(0, len(T), 8192)]).float()

    def fit(self, Z_ref: np.ndarray) -> "Novelty":
        net = self.bundle.model.net
        was = net.training
        net.eval()
        E = self.embed(Z_ref)
        net.train(was)
        self.mu, self.sd = E.mean(0), E.std(0) + 1e-6
        self.ref = (E - self.mu) / self.sd
        return self

    def distances(self, Z: np.ndarray) -> np.ndarray:
        self.bundle.model.net.eval()
        Q = (self.embed(Z) - self.mu) / self.sd
        return torch.cat([torch.cdist(Q[i:i + self.batch], self.ref).min(1).values
                          for i in range(0, len(Q), self.batch)]).cpu().numpy()

    def calibrate_tau(self, Z_holdout: np.ndarray, flow_quantile: float) -> float:
        self.tau = float(np.quantile(self.distances(Z_holdout), flow_quantile))
        return self.tau

    def share(self, Z: np.ndarray) -> float:
        if self.tau is None:
            raise RuntimeError("Novelty.tau is not set (calibrate_tau or load it from the calibration file)")
        return float(np.mean(self.distances(Z) > self.tau))
