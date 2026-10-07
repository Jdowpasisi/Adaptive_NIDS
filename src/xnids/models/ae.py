"""Autoencoder anomaly detector, trained on BENIGN source traffic only.

Architecture: signed-log + StandardScaler -> 64-32-16 encoder, 16-32-64 decoder (Linear, BatchNorm1d, ReLU), MSE.
Raw score = per-flow reconstruction error. It is mapped to [0, 1) with err / (err + q95), where q95 is the 95th
percentile of the error on BENIGN SOURCE-VALIDATION flows. The map is monotone, so ranking metrics and the
frozen-threshold rule are unchanged, and it does not saturate: attacks stay distinguishable from each other, which
the confidence-based drift signals in C8 rely on. Labels are used only to select benign rows for training.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from xnids.models.mlp import InputScaler, predict, train_loop
from xnids.models.zoo import BaseModel


class AENet(nn.Module):
    def __init__(self, d_in: int, hidden: tuple[int, ...] = (64, 32, 16)) -> None:
        super().__init__()
        enc, d = [], d_in
        for h in hidden:
            enc += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU()]
            d = h
        dec = []
        for h in list(hidden[-2::-1]) + [d_in]:
            dec += [nn.Linear(d, h)] + ([nn.BatchNorm1d(h), nn.ReLU()] if h != d_in else [])
            d = h
        self.encoder, self.decoder = nn.Sequential(*enc), nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def error(self, x: torch.Tensor) -> torch.Tensor:
        return ((self(x) - x) ** 2).mean(-1)


class Autoencoder(BaseModel):
    name = "ae"
    differentiable = True
    DEFAULTS = {"hidden": [64, 32, 16], "lr": 1e-3, "batch_size": 4096, "max_epochs": 50, "patience": 5}

    def fit(self, Xtr, ytr, Xval, yval):
        p = self.DEFAULTS | self.params
        torch.manual_seed(self.seed)
        Btr, Bval = Xtr[ytr == 0], Xval[yval == 0]
        self.scaler = InputScaler().fit(Btr)
        self.net = AENet(Xtr.shape[1], tuple(p["hidden"])).to(self.device)
        T = lambda a: torch.from_numpy(a).to(self.device)  # noqa: E731
        self.info = train_loop(self.net, nn.MSELoss(), T(self.scaler.transform(Btr)), None,
                               T(self.scaler.transform(Bval)), None, self.seed, p["lr"], p["batch_size"],
                               p["max_epochs"], p["patience"])
        self.q95 = float(np.quantile(self.raw_error(Bval), 0.95))
        self.info["benign_val_err_q95"] = self.q95
        return self

    def raw_error(self, X: np.ndarray) -> np.ndarray:
        return predict(self.net, self.scaler.transform(X), self.device, fn=self.net.error)

    def score(self, X):
        e = self.raw_error(X)
        return e / (e + self.q95)

    def _save(self, d: Path) -> None:
        torch.save(self.net.state_dict(), d / "model.pt")
        (d / "arch.json").write_text(json.dumps({"d_in": self.net.encoder[0].in_features,
                                                 "scaler": self.scaler.to_dict(), "q95": self.q95}))

    def _load(self, d: Path) -> None:
        a = json.loads((d / "arch.json").read_text())
        p = self.DEFAULTS | self.params
        self.scaler, self.q95 = InputScaler.from_dict(a["scaler"]), a["q95"]
        self.net = AENet(a["d_in"], tuple(p["hidden"])).to(self.device)
        self.net.load_state_dict(torch.load(d / "model.pt", map_location=self.device))
        self.net.eval()
