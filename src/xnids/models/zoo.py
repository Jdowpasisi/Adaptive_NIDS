"""The seven C5 models behind one interface.

    model = build(name, params, seed, device)
    model.fit(Xtr, ytr, Xval, yval)      # numpy float32 features (already preprocessed), y in {0, 1}
    s = model.score(X)                   # P(attack) in [0, 1], higher = more suspicious
    model.save(dir); Model.load(dir)

Class imbalance is handled inside each model (class weights / scale_pos_weight / weighted BCE), never by resampling,
so the thresholding on source validation sees the real class mix.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import numpy as np


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class BaseModel(ABC):
    name: str = "base"
    differentiable: bool = False          # True for torch models (CORAL / DANN / AdaBN / Tent need gradients)

    def __init__(self, params: dict | None = None, seed: int = 0, device: str = "auto") -> None:
        self.params = dict(params or {})
        self.seed = seed
        self.device = resolve_device(device)
        self.info: dict = {}              # training details logged to MLflow (epochs, trees, ...)

    @abstractmethod
    def fit(self, Xtr: np.ndarray, ytr: np.ndarray, Xval: np.ndarray, yval: np.ndarray) -> "BaseModel": ...

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray: ...

    def save(self, d: Path) -> None:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        (d / "model_meta.json").write_text(json.dumps(
            {"name": self.name, "params": self.params, "seed": self.seed, "info": self.info}, indent=1, default=str))
        self._save(d)

    @classmethod
    def load(cls, d: Path, device: str = "auto") -> "BaseModel":
        meta = json.loads((Path(d) / "model_meta.json").read_text())
        m = MODELS[meta["name"]](meta["params"], meta["seed"], device)
        m.info = meta["info"]
        m._load(Path(d))
        return m

    def _save(self, d: Path) -> None:
        joblib.dump(self.est, d / "model.joblib", compress=3)

    def _load(self, d: Path) -> None:
        self.est = joblib.load(d / "model.joblib")


def _pos_weight(y: np.ndarray) -> float:
    pos = float(np.sum(y == 1))
    return float(np.sum(y == 0)) / pos if pos else 1.0


class LDA(BaseModel):
    """StandardScaler + LinearDiscriminantAnalysis: Cantone et al.'s best cross-dataset generaliser."""
    name = "lda"

    def fit(self, Xtr, ytr, Xval, yval):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.est = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(**self.params)).fit(Xtr, ytr)
        return self

    def score(self, X):
        return self.est.predict_proba(X)[:, 1]


class DecisionTree(BaseModel):
    name = "dt"

    def fit(self, Xtr, ytr, Xval, yval):
        from sklearn.tree import DecisionTreeClassifier

        p = {"class_weight": "balanced"} | self.params
        self.est = DecisionTreeClassifier(random_state=self.seed, **p).fit(Xtr, ytr)
        self.info = {"depth": int(self.est.get_depth()), "leaves": int(self.est.get_n_leaves())}
        return self

    def score(self, X):
        return self.est.predict_proba(X)[:, 1]


class RandomForest(BaseModel):
    name = "rf"

    def fit(self, Xtr, ytr, Xval, yval):
        from sklearn.ensemble import RandomForestClassifier

        p = {"n_estimators": 200, "class_weight": "balanced_subsample", "n_jobs": -1} | self.params
        self.est = RandomForestClassifier(random_state=self.seed, **p).fit(Xtr, ytr)
        return self

    def score(self, X):
        return self.est.predict_proba(X)[:, 1]


class XGBoost(BaseModel):
    """hist trees, up to 1000 rounds, early stopping on source val (aucpr), scale_pos_weight."""
    name = "xgb"

    def fit(self, Xtr, ytr, Xval, yval):
        import xgboost as xgb

        p = {"n_estimators": 1000, "learning_rate": 0.1, "max_depth": 8, "early_stopping_rounds": 30} | self.params
        self.est = xgb.XGBClassifier(tree_method="hist", device=self.device, eval_metric="aucpr",
                                     scale_pos_weight=_pos_weight(ytr), random_state=self.seed, n_jobs=-1, **p)
        self.est.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
        self.est.set_params(device="cpu")      # inputs are numpy on CPU: predict there, no device-mismatch copy
        self.info = {"best_iteration": int(self.est.best_iteration)}
        return self

    def score(self, X):
        return self.est.predict_proba(X)[:, 1]

    def _save(self, d):
        self.est.save_model(d / "model.ubj")

    def _load(self, d):
        import xgboost as xgb

        self.est = xgb.XGBClassifier(device="cpu")
        self.est.load_model(d / "model.ubj")


def _registry() -> dict[str, type[BaseModel]]:
    from xnids.models.ae import Autoencoder
    from xnids.models.mlp import MLP
    from xnids.models.tabnet_wrap import TabNet

    return {m.name: m for m in (LDA, DecisionTree, RandomForest, XGBoost, MLP, Autoencoder, TabNet)}


class _Lazy(dict):
    """Registry filled on first use, so importing zoo does not import torch."""

    def _fill(self):
        if not super().__len__():
            self.update(_registry())

    def __getitem__(self, k):
        self._fill()
        return super().__getitem__(k)

    def __iter__(self):
        self._fill()
        return super().__iter__()

    def __contains__(self, k):
        self._fill()
        return super().__contains__(k)

    def keys(self):
        self._fill()
        return super().keys()


MODELS: dict[str, type[BaseModel]] = _Lazy()


def build(name: str, params: dict | None = None, seed: int = 0, device: str = "auto") -> BaseModel:
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; known: {sorted(MODELS.keys())}")
    return MODELS[name](params, seed, device)
