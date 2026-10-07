"""TabNet (pytorch-tabnet): n_d = n_a = 16, early stopping on source val. Inputs are signed-log transformed
(TabNet's own BatchNorm handles the scale); classes are balanced with sampling weights (weights=1)."""

import json
from pathlib import Path

import numpy as np

from xnids.models.preprocess import slog
from xnids.models.zoo import BaseModel


class TabNet(BaseModel):
    name = "tabnet"
    differentiable = True
    DEFAULTS = {"n_d": 16, "n_a": 16, "n_steps": 3, "max_epochs": 30, "patience": 5, "batch_size": 16384,
                "virtual_batch_size": 512, "lr": 2e-2}

    def _x(self, X: np.ndarray) -> np.ndarray:
        return slog(np.asarray(X, dtype=np.float32))     # float32 throughout: no 1.2 GB float64 copy on 2M rows

    def fit(self, Xtr, ytr, Xval, yval):
        import torch
        from pytorch_tabnet.tab_model import TabNetClassifier

        p = self.DEFAULTS | self.params
        self.est = TabNetClassifier(n_d=p["n_d"], n_a=p["n_a"], n_steps=p["n_steps"], seed=self.seed,
                                    device_name=self.device, optimizer_fn=torch.optim.Adam,
                                    optimizer_params={"lr": p["lr"]}, verbose=0)
        self.est.fit(self._x(Xtr), ytr.astype(int), eval_set=[(self._x(Xval), yval.astype(int))],
                     eval_name=["val"], eval_metric=["logloss"], max_epochs=p["max_epochs"],
                     patience=p["patience"], batch_size=p["batch_size"], virtual_batch_size=p["virtual_batch_size"],
                     weights=1, drop_last=False,
                     # the default explains every training row to compute feature importances: >11 GB of RAM on
                     # 2M rows (OOM-killed the C6 sweep). We do not use TabNet's importances.
                     compute_importance=False)
        self.info = {"best_epoch": int(self.est.best_epoch), "epochs": len(self.est.history["loss"])}
        return self

    def score(self, X):
        return self.est.predict_proba(self._x(X))[:, 1]

    def _save(self, d: Path) -> None:
        self.est.save_model(str(d / "tabnet"))
        (d / "arch.json").write_text(json.dumps({"saved": "tabnet.zip"}))

    def _load(self, d: Path) -> None:
        from pytorch_tabnet.tab_model import TabNetClassifier

        self.est = TabNetClassifier(device_name=self.device)
        self.est.load_model(str(d / "tabnet.zip"))
