"""End-to-end C5 harness on a tiny synthetic two-dataset track (temp data/config dirs, CPU, ~seconds)."""

import json

import numpy as np
import pandas as pd
import polars as pl
import pytest

from xnids.eval import harness
from xnids.models.bundle import Bundle
from xnids.utils import paths


@pytest.fixture
def toy(tmp_path, monkeypatch):
    for attr in ("CONFIGS", "PROCESSED", "SPLITS", "MODELS", "ARTIFACT_ROOT"):
        monkeypatch.setattr(paths, attr, tmp_path / attr.lower())
    monkeypatch.setattr(paths, "TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    (tmp_path / "configs/tracks").mkdir(parents=True)
    (tmp_path / "configs/tracks/toy.yaml").write_text("datasets: [a, b]\nfeatures: [f0, f1, f2, f3]\n")
    rng = np.random.default_rng(0)
    for ds, shift in (("a", 0.0), ("b", 1.5)):          # b: same attack signal, benign shifted (a "new lab")
        n = 4000
        y = (rng.random(n) < 0.25).astype(np.int8)
        X = rng.normal(size=(n, 4)).astype(np.float32)
        X[y == 1, 0] += 3
        X[:, 1] += shift
        df = pl.DataFrame({"row_id": pl.Series(np.arange(n), dtype=pl.UInt32), "y": y,
                           "family": np.where(y == 1, "DoS", "Benign"), "label": np.where(y == 1, "dos", "benign"),
                           **{f"f{i}": X[:, i] for i in range(4)}})
        (paths.PROCESSED / ds).mkdir(parents=True)
        df.write_parquet(paths.PROCESSED / ds / "toy.parquet")
        paths.SPLITS.mkdir(exist_ok=True)
        pl.DataFrame({"row_id": df["row_id"], "split": np.array(["train", "val", "test"])[np.arange(n) % 3],
                      "split_hash": "h"}).write_parquet(paths.SPLITS / f"{ds}.parquet")
    return tmp_path


CFG = {"experiment": "t", "run": {"name": "lda_toy", "seeds": [0, 1]},
       "data": {"track": "toy", "source": "a", "targets": "all", "max_rows_per_split": None},
       "eval": {"target_dr": 0.95}, "model": {"name": "lda", "device": "cpu",
                                              "grid": [{"solver": "svd"}, {"solver": "lsqr", "shrinkage": "auto"}]}}


def test_end_to_end(toy):
    from xnids.utils import log

    table = harness.run_config(CFG)
    assert len(table) == 4                                     # 2 seeds x 2 targets
    assert set(table["kind"]) == {"within", "cross"}
    within = table[table.kind == "within"]
    assert (within["dr_at_thr"] > 0.85).all() and (within["roc_auc"] > 0.95).all()
    # threshold frozen on source val: identical for both targets of a run
    assert table.groupby("run_id")["threshold"].nunique().eq(1).all()

    runs = log.find_runs(experiment="t")
    assert len(runs) == 2 and set(runs["tags.source"]) == {"a"}
    assert {"metrics.a/fpr_at_thr", "metrics.b/dr_at_thr", "metrics.threshold"} <= set(runs.columns)

    bundle_dir = sorted(paths.MODELS.glob("*"))[0]
    assert json.loads((bundle_dir / "threshold.json").read_text())["threshold"] == pytest.approx(
        table[table.run_id == json.loads((bundle_dir / "meta.json").read_text())["run_id"]]["threshold"].iloc[0])
    b = Bundle.load(bundle_dir, device="cpu")
    assert b.meta["training_data"]["dataset"] == "a" and b.meta["parent_version"] is None
    test_a = harness.load_split("a", "toy", "test")
    scores = b.score(test_a.X)
    assert scores.shape == (len(test_a.y),)
    # the bundle reproduces the logged scores
    import mlflow
    rid = runs.sort_values("start_time")["run_id"].iloc[0]
    logged = pd.read_parquet(mlflow.artifacts.download_artifacts(run_id=rid, artifact_path="scores_a.parquet"))
    assert len(logged) == len(test_a.y) and set(logged.columns) == {"row_id", "score", "y"}


def test_expand_all_sources_and_within_cell(toy):
    cfgs = harness.expand({**CFG, "data": {**CFG["data"], "source": "all", "targets": ["b"]}})
    assert [c["data"]["source"] for c in cfgs] == ["a", "b"]
    assert cfgs[0]["data"]["targets"] == ["a", "b"] and cfgs[1]["data"]["targets"] == ["b"]


def test_dev_subsample_applies(toy):
    s = harness.load_split("a", "toy", "train", max_rows=500, min_per_family=50)
    assert 300 < len(s.y) < 800 and s.y.sum() >= 50


def test_resume_skips_finished_runs(toy):
    from xnids.utils import log

    first = harness.run_config(CFG)
    again = harness.run_config(CFG, resume=True)
    assert len(log.find_runs(experiment="t")) == 2                 # nothing re-trained
    pd.testing.assert_frame_equal(first.sort_values(["seed", "target"]).reset_index(drop=True),
                                  again.sort_values(["seed", "target"]).reset_index(drop=True))


def test_drop_features_norm_and_no_bundle(toy):
    from xnids.utils import log

    cfg = {**CFG, "run": {"name": "abl", "seeds": [0], "save_bundle": False},
           "data": {**CFG["data"], "drop_features": ["f1"], "domain_norm": "rank"}}
    t = harness.run_config(cfg)
    assert len(t) == 2 and not paths.MODELS.exists()            # no bundle written
    runs = log.find_runs(experiment="t")
    assert runs["params.n_features"].iloc[0] == "3"             # f1 dropped before preprocessing
    # f1 carried the benign shift between a and b; without it and with per-domain ranks, b looks like a
    cross = t[t.kind == "cross"].iloc[0]
    assert cross.roc_auc > 0.95


def test_domain_rank_norm_is_per_domain():
    from xnids.adapt.scaling import DomainRankNorm

    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(5000, 2)), rng.normal(size=(5000, 2)) * 3 + 10   # shifted and rescaled domain
    za, zb = DomainRankNorm().fit(a).transform(a), DomainRankNorm().fit(b).transform(b)
    assert abs(za.mean() - zb.mean()) < 0.01 and abs(za.std() - zb.std()) < 0.01
