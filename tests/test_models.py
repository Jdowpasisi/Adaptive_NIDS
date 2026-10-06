"""Smoke tests for every C5 model on a small synthetic problem (CPU, seconds each)."""

import numpy as np
import polars as pl
import pytest

from xnids.models import zoo
from xnids.models.preprocess import Preprocessor, slog

FAST = {
    "lda": {}, "dt": {"max_depth": 6}, "rf": {"n_estimators": 20},
    "xgb": {"n_estimators": 50}, "mlp": {"max_epochs": 8, "batch_size": 256},
    "ae": {"max_epochs": 8, "batch_size": 256}, "tabnet": {"max_epochs": 5, "batch_size": 512, "virtual_batch_size": 64},
}


def _data(n=3000, d=8, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    X = rng.normal(size=(n, d)).astype(np.float32)
    X[y == 1, :3] += 2.5                       # attacks differ on 3 features
    X[:, 5] = -1                                # a constant -1 sentinel column (dropped by Preprocessor)
    return X, y


@pytest.mark.parametrize("name", list(FAST))
def test_model_learns_scores_and_roundtrips(name, tmp_path):
    from sklearn.metrics import roc_auc_score

    X, y = _data()
    Xv, yv = _data(seed=1)
    m = zoo.build(name, FAST[name], seed=0, device="cpu").fit(X, y, Xv, yv)
    s = m.score(Xv)
    assert s.shape == (len(Xv),) and np.all((s >= 0) & (s <= 1)) and np.isfinite(s).all()
    # the AE is unsupervised (benign-only training), so it gets a lower bar than the supervised models
    assert roc_auc_score(yv, s) > (0.8 if name == "ae" else 0.9), name
    m.save(tmp_path / name)
    s2 = zoo.MODELS[name].load(tmp_path / name, device="cpu").score(Xv)
    np.testing.assert_allclose(s, s2, rtol=1e-5, atol=1e-6)


def test_mlp_exposes_penultimate_features_and_batchnorm():
    import torch

    X, y = _data()
    m = zoo.build("mlp", FAST["mlp"], device="cpu").fit(X, y, X, y)
    assert m.net.features(torch.from_numpy(m.scaler.transform(X[:5]))).shape == (5, 64)
    assert any(isinstance(mod, torch.nn.BatchNorm1d) for mod in m.net.modules())


def test_slog_handles_sentinel():
    assert np.isfinite(slog(np.array([-1.0, -14.0, 0.0, 1e9]))).all()
    assert slog(np.array([3.0]))[0] == pytest.approx(np.log1p(3.0))


def test_preprocessor_imputes_drops_constants_and_roundtrips(tmp_path):
    df = pl.DataFrame({"a": [1.0, float("inf"), 3.0, 5.0], "b": [7.0, 7.0, 7.0, 7.0], "c": [float("nan"), 1.0, 2.0, 3.0]})
    p = Preprocessor().fit(df)
    assert p.features_out == ["a", "c"] and p.dropped_constant == ["b"]
    out = p.transform(df)
    np.testing.assert_allclose(out, [[1, 2], [3, 1], [3, 2], [5, 3]])
    p.save(tmp_path / "p.json")
    np.testing.assert_allclose(Preprocessor.load(tmp_path / "p.json").transform(df), out)
    with pytest.raises(ValueError, match="missing"):
        p.transform(df.drop("c"))


def test_unknown_model():
    with pytest.raises(KeyError):
        zoo.build("nope")
