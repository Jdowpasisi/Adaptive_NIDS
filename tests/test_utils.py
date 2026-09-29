import numpy as np
import pytest

from xnids.utils import config, log, seed


def test_hash_ignores_key_order():
    a = {"run": {"name": "x", "seeds": [0, 1]}, "model": {"name": "rf"}}
    b = {"model": {"name": "rf"}, "run": {"seeds": [0, 1], "name": "x"}}
    assert config.cfg_hash(a) == config.cfg_hash(b)
    assert len(config.cfg_hash(a)) == 10


def test_hash_changes_with_value():
    assert config.cfg_hash({"a": 1}) != config.cfg_hash({"a": 2})


def test_base_inheritance(tmp_path):
    (tmp_path / "base.yaml").write_text("model: {name: rf, params: {n: 1, depth: 3}}\n")
    (tmp_path / "child.yaml").write_text("_base: base.yaml\nmodel: {params: {n: 5}}\n")
    cfg = config.load(tmp_path / "child.yaml")
    assert cfg == {"model": {"name": "rf", "params": {"n": 5, "depth": 3}}}


def test_flatten():
    assert config.flatten({"a": {"b": 1, "c": [1, 2]}}) == {"a.b": 1, "a.c": "[1, 2]"}


def test_seed_reproducible():
    seed.set_seed(0)
    x = np.random.rand(3)
    seed.set_seed(0)
    assert np.array_equal(x, np.random.rand(3))


def test_mlflow_run_tagged(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setattr(log.paths, "TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(log.paths, "ARTIFACT_ROOT", tmp_path / "mlruns")
    cfg = {"run": {"name": "t"}, "x": 1}
    with log.start_run(cfg, seed=0, experiment="test") as run:
        mlflow.log_metric("m", 1.0)
    runs = log.find_runs(experiment="test", cfg_hash=config.cfg_hash(cfg))
    assert run.info.run_id in set(runs.run_id)
    assert runs.iloc[0]["tags.seed"] == "0"
