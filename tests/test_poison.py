"""C17 attack generators, continual adapters and guards on a small synthetic MLP (CPU)."""

import numpy as np
import polars as pl
import pytest
import torch

from xnids.adapt.torchutil import bn_layers, tensor
from xnids.attack import guards as G
from xnids.attack.poison import AdaptiveFrog, FrogBoiling, StatSkew, slog
from xnids.eval import metrics
from xnids.models import zoo
from xnids.models.bundle import Bundle
from xnids.models.preprocess import Preprocessor

NAMES = ["a", "b", "c", "d"]


def _data(n, seed, attack_share=0.3):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < attack_share).astype(int)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    X[y == 1, 0] += 3.0
    return pl.DataFrame(np.abs(X) * 10, schema=NAMES), y


@pytest.fixture(scope="module")
def bundle():
    Xs, ys = _data(6000, 0)
    Xv, yv = _data(3000, 1)
    pre = Preprocessor().fit(Xs)
    m = zoo.build("mlp", {"max_epochs": 5, "batch_size": 256, "hidden": [32, 16]}, seed=0, device="cpu").fit(
        pre.transform(Xs), ys, pre.transform(Xv), yv)
    thr = metrics.threshold_at_dr(yv, m.score(pre.transform(Xv)), 0.95)
    return Bundle(m, pre, thr, NAMES, {"version": "toy"}), (Xs, ys)


def test_frog_boiling_schedule_and_endpoints():
    t = pl.DataFrame({"a": [100.0] * 50, "b": [1.0] * 50})
    b = pl.DataFrame({"a": [1.0] * 50, "b": [1.0] * 50})
    f = FrogBoiling(t, b, rounds=10, alpha_max=1.0)
    rng = np.random.default_rng(0)
    assert f.alpha(0) == 0 and f.alpha(10) == 1.0
    assert np.allclose(f.flows(5, 0, rng)["a"], 100.0) and np.allclose(f.flows(5, 10, rng)["a"], 1.0, atol=1e-5)
    mid = f.flows(5, 5, rng)["a"].to_numpy()                      # halfway in signed-log space
    assert np.allclose(slog(mid), (slog(np.array(100.0)) + slog(np.array(1.0))) / 2, atol=1e-4)


def test_stat_skew_targets_the_discriminative_feature_on_the_target_side(bundle):
    _, (X, y) = bundle
    s = StatSkew(X.filter(pl.Series(y == 1)), X.filter(pl.Series(y == 0)), k=1, scale=5.0)
    assert s.feature_names == ["a"]                               # attacks differ on feature a
    f = s.flows(20, 1, np.random.default_rng(0))
    assert (f["a"] > X["a"].max()).all()                          # extreme, on the attack side (higher)
    assert f["b"].std() > 0                                        # the other features stay benign-looking


def test_adaptive_frog_injects_flows_just_below_the_threshold(bundle):
    b, (X, y) = bundle
    att = AdaptiveFrog(X.filter(pl.Series(y == 1)), X.filter(pl.Series(y == 0)), candidates=4000)
    f = att.flows(100, 1, np.random.default_rng(0), model=b)
    s = b.score(f)
    assert (s < b.threshold).all() and np.median(s) > 0.5 * b.threshold
    with pytest.raises(ValueError):
        att.flows(10, 1, np.random.default_rng(0))


def test_trimmed_bn_stats_resist_outliers(bundle):
    b, (X, _) = bundle
    T = tensor(b, X.head(2000))
    net = b.model.net
    plain = G.pool_bn_stats(net, T)
    Tp = torch.cat([T, torch.full((10, T.shape[1]), 1e3)])         # 0.5% extreme rows
    poisoned, trimmed = G.pool_bn_stats(net, Tp), G.pool_bn_stats(net, Tp, trim=0.01)
    d = lambda u, v: float((u[0][0] - v[0][0]).norm())            # noqa: E731  first BN layer's mean
    assert d(trimmed, plain) < 0.2 * d(poisoned, plain)
    before = [m.running_mean.clone() for m in bn_layers(net)]
    assert all(torch.equal(a, m.running_mean) for a, m in zip(before, bn_layers(net), strict=True))   # restored


def test_continual_adabn_blends_with_momentum(bundle):
    b, (X, _) = bundle
    nb = G.continual_adabn(b, X.head(2000) * 3, momentum=0.25)
    pool = G.pool_bn_stats(b.model.net, tensor(b, X.head(2000) * 3))
    m0, m1 = bn_layers(b.model.net)[0], bn_layers(nb.model.net)[0]
    assert torch.allclose(m1.running_mean, 0.75 * m0.running_mean + 0.25 * pool[0][0], atol=1e-4)
    assert not torch.equal(m0.running_mean, m1.running_mean)      # the input bundle is untouched (deep copy)


def test_clip_projects_the_update_onto_the_radius(bundle):
    b, (X, _) = bundle
    nb = G.continual_adabn(b, X.head(2000) * 5, momentum=1.0)
    full = float((G.update_vector("adabn", nb.model.net) - G.update_vector("adabn", b.model.net)).norm())
    cb, norm = G.clip_update("adabn", b, nb, radius=full / 4)
    assert norm == pytest.approx(full)
    after = float((G.update_vector("adabn", cb.model.net) - G.update_vector("adabn", b.model.net)).norm())
    assert after == pytest.approx(full / 4, rel=1e-4)
