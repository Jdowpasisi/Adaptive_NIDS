"""Model bundle (Build Guide §4): models/{version}/

    model/            model-specific files (+ model_meta.json); torch models keep their input scaler in arch.json
    preprocess.json   source-train median imputer + dropped constant columns (the role of scaler.pkl in the guide)
    threshold.json    the frozen threshold, the DR it targets, and where it came from
    features.json     track + input feature list (the API rejects any other schema)
    meta.json         config hash, MLflow run id, parent version, training data (dataset, split hash, rows), git commit
    input_map.json    optional (C9 scaling adapter): a per-feature map applied after preprocessing, before the model
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from xnids.models.preprocess import Preprocessor
from xnids.models.zoo import MODELS, BaseModel


class InputMap:
    """Per-feature affine map in signed-log space: z -> (z - mu_from) / sd_from * sd_to + mu_to, back to raw scale.
    With from = target pool and to = source train it moves target traffic onto the source's scale (moment matching)."""

    def __init__(self, mu_from, sd_from, mu_to, sd_to) -> None:
        self.mu_from, self.sd_from = np.asarray(mu_from, float), np.asarray(sd_from, float)
        self.mu_to, self.sd_to = np.asarray(mu_to, float), np.asarray(sd_to, float)

    def transform(self, X: np.ndarray) -> np.ndarray:
        z = np.sign(X) * np.log1p(np.abs(X))
        z = (z - self.mu_from) / self.sd_from * self.sd_to + self.mu_to
        return (np.sign(z) * np.expm1(np.abs(z))).astype(np.float32)

    def to_dict(self) -> dict:
        return {k: getattr(self, k).tolist() for k in ("mu_from", "sd_from", "mu_to", "sd_to")}

    @classmethod
    def from_dict(cls, d: dict) -> "InputMap":
        return cls(d["mu_from"], d["sd_from"], d["mu_to"], d["sd_to"])


@dataclass
class Bundle:
    model: BaseModel
    preprocess: Preprocessor
    threshold: float
    features: list[str]
    meta: dict
    path: Path | None = None
    input_map: "InputMap | None" = None

    def model_input(self, X: pl.DataFrame) -> np.ndarray:
        Z = self.preprocess.transform(X)
        return self.input_map.transform(Z) if self.input_map is not None else Z

    def score(self, X: pl.DataFrame) -> np.ndarray:
        return self.model.score(self.model_input(X))

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        return self.score(X) >= self.threshold

    def save(self, d: Path) -> Path:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        self.model.save(d / "model")
        self.preprocess.save(d / "preprocess.json")
        (d / "threshold.json").write_text(json.dumps({
            "threshold": self.threshold, "target_dr": self.meta.get("target_dr"),
            "chosen_on": "source validation split (frozen; never re-tuned on target)"}, indent=1))
        (d / "features.json").write_text(json.dumps({"track": self.meta.get("track"), "features": self.features}, indent=1))
        (d / "meta.json").write_text(json.dumps(self.meta, indent=1, default=str))
        if self.input_map is not None:
            (d / "input_map.json").write_text(json.dumps(self.input_map.to_dict()))
        self.path = d
        return d

    @classmethod
    def load(cls, d: Path, device: str = "auto") -> "Bundle":
        d = Path(d)
        name = json.loads((d / "model" / "model_meta.json").read_text())["name"]
        return cls(
            model=MODELS[name].load(d / "model", device=device),
            preprocess=Preprocessor.load(d / "preprocess.json"),
            threshold=float(json.loads((d / "threshold.json").read_text())["threshold"]),
            features=json.loads((d / "features.json").read_text())["features"],
            meta=json.loads((d / "meta.json").read_text()),
            path=d,
            input_map=InputMap.from_dict(json.loads((d / "input_map.json").read_text()))
            if (d / "input_map.json").exists() else None,
        )


def bundle_for(model: str, track: str, source: str, seed: int, device: str = "auto") -> Bundle:
    """The bundle trained by the CURRENT configs/train/<track>/<model>.yaml for this source and seed."""
    from xnids.eval import harness
    from xnids.utils import config, paths

    cfg = config.load(paths.CONFIGS / "train" / track / f"{model}.yaml")
    c = next(c for c in harness.expand(cfg) if c["data"]["source"] == source)
    d = paths.MODELS / f"{model}-{track}-{source}-s{seed}-{config.cfg_hash(c)}"
    if not d.exists():
        raise FileNotFoundError(f"no bundle {d.name}; train it with scripts/train.py --config "
                                f"configs/train/{track}/{model}.yaml --source {source} --seeds {seed}")
    return Bundle.load(d, device=device)
