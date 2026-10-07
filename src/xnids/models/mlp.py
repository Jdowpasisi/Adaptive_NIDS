"""MLP (PyTorch): signed-log + StandardScaler -> 256-128-64 (Linear, BatchNorm1d, ReLU, Dropout 0.2) -> 1 logit.

BatchNorm is required: AdaBN re-estimates its running statistics and Tent trains its affine parameters (C9).
`MLPNet.features(x)` returns the 64-d penultimate representation used by Deep CORAL and DANN.
Training: class-weighted BCE (pos_weight = n_benign / n_attack), Adam 1e-3, early stopping on source-val loss.
"""

import copy
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from torch import nn

from xnids.models.preprocess import slog
from xnids.models.zoo import BaseModel, _pos_weight


class InputScaler:
    """slog then standardise with train mean/std (std 0 -> 1). Plain numpy so it serialises to JSON."""

    def fit(self, X: np.ndarray) -> "InputScaler":
        Z = slog(X.astype(np.float64))
        self.mean, self.std = Z.mean(0), Z.std(0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((slog(X.astype(np.float64)) - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "InputScaler":
        s = cls()
        s.mean, s.std = np.asarray(d["mean"]), np.asarray(d["std"])
        return s


def block(d_in: int, d_out: int, dropout: float) -> list[nn.Module]:
    return [nn.Linear(d_in, d_out), nn.BatchNorm1d(d_out), nn.ReLU(), nn.Dropout(dropout)]


class MLPNet(nn.Module):
    def __init__(self, d_in: int, hidden: tuple[int, ...] = (256, 128, 64), dropout: float = 0.2) -> None:
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += block(d, h, dropout)
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x)).squeeze(-1)


def batches(n: int, bs: int, gen: torch.Generator | None, device: str):
    idx = torch.randperm(n, generator=gen, device="cpu").to(device) if gen is not None else torch.arange(n, device=device)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def train_loop(net: nn.Module, loss_fn, Xtr: torch.Tensor, Ytr: torch.Tensor | None, Xval: torch.Tensor,
               Yval: torch.Tensor | None, seed: int, lr: float, batch_size: int, max_epochs: int, patience: int,
               weight_decay: float = 0.0) -> dict:
    """Adam + early stopping on validation loss; restores the best weights. Y=None -> loss_fn(out, X) (AE)."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, bad, history = float("inf"), None, 0, []
    for epoch in range(max_epochs):
        net.train()
        for idx in batches(len(Xtr), batch_size, gen, Xtr.device):
            if len(idx) < 2:          # BatchNorm needs >1 sample
                continue
            xb = Xtr[idx]
            loss = loss_fn(net(xb), Ytr[idx] if Ytr is not None else xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            tot, n = 0.0, 0
            for idx in batches(len(Xval), 65536, None, Xval.device):
                xb = Xval[idx]
                tot += float(loss_fn(net(xb), Yval[idx] if Yval is not None else xb)) * len(idx)
                n += len(idx)
            val = tot / max(n, 1)
        history.append(val)
        if val < best - 1e-6:
            best, best_state, bad = val, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return {"epochs": len(history), "best_epoch": int(np.argmin(history)) + 1, "best_val_loss": best}


def predict(net: nn.Module, X: np.ndarray, device: str, fn=None, batch_size: int = 65536) -> np.ndarray:
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).to(device)
            out.append((fn(xb) if fn else net(xb)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


class MLP(BaseModel):
    name = "mlp"
    differentiable = True
    DEFAULTS = {"hidden": [256, 128, 64], "dropout": 0.2, "lr": 1e-3, "batch_size": 4096, "max_epochs": 50,
                "patience": 5, "weight_decay": 0.0}

    def fit(self, Xtr, ytr, Xval, yval):
        p = self.DEFAULTS | self.params
        torch.manual_seed(self.seed)
        self.scaler = InputScaler().fit(Xtr)
        self.net = MLPNet(Xtr.shape[1], tuple(p["hidden"]), p["dropout"]).to(self.device)
        pw = torch.tensor(_pos_weight(ytr), device=self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        T = lambda a: torch.from_numpy(a).to(self.device)  # noqa: E731
        self.info = train_loop(self.net, loss_fn, T(self.scaler.transform(Xtr)), T(ytr.astype(np.float32)),
                               T(self.scaler.transform(Xval)), T(yval.astype(np.float32)), self.seed, p["lr"],
                               p["batch_size"], p["max_epochs"], p["patience"], p["weight_decay"])
        return self

    def score(self, X):
        return expit(predict(self.net, self.scaler.transform(X), self.device))   # stable sigmoid, no overflow

    def _save(self, d: Path) -> None:
        torch.save(self.net.state_dict(), d / "model.pt")
        (d / "arch.json").write_text(json.dumps({"d_in": self.net.body[0].in_features,
                                                 "scaler": self.scaler.to_dict()}))

    def _load(self, d: Path) -> None:
        a = json.loads((d / "arch.json").read_text())
        p = self.DEFAULTS | self.params
        self.scaler = InputScaler.from_dict(a["scaler"])
        self.net = MLPNet(a["d_in"], tuple(p["hidden"]), p["dropout"]).to(self.device)
        self.net.load_state_dict(torch.load(d / "model.pt", map_location=self.device))
        self.net.eval()
