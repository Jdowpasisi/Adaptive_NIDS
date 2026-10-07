"""C9 adapters on a small synthetic shift (CPU, seconds). Includes the Build Guide's required unit tests."""

import numpy as np
import polars as pl
import pytest
import torch

from xnids import adapt
from xnids.adapt.base import AdaptContext
from xnids.adapt.coral import coral
from xnids.adapt.torchutil import bn_layers
from xnids.models import zoo
from xnids.models.bundle import Bundle
from xnids.models.preprocess import Preprocessor

NAMES = ["a", "b", "c", "d"]


def _domain(n, shift, seed):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    X[y == 1, 0] += 3.0
    X[:, 1:] = X[:, 1:] * (1 + shift) + 2 * shift          # target: other features rescaled and moved
    return pl.DataFrame(np.abs(X) * 10, schema=NAMES), y


@pytest.fixture(scope="module")
def setup():
    Xs, ys = _domain(6000, 0.0, 0)
    Xv, yv = _domain(3000, 0.0, 1)
    Xt, yt = _domain(6000, 1.0, 2)
    pre = Preprocessor().fit(Xs)
    m = zoo.build("mlp", {"max_epochs": 5, "batch_size": 256}, seed=0, device="cpu").fit(
        pre.transform(Xs), ys, pre.transform(Xv), yv)
    from xnids.eval import metrics
    thr = metrics.threshold_at_dr(yv, m.score(pre.transform(Xv)), 0.95)
    b = Bundle(m, pre, thr, NAMES, {"version": "toy"})
    ctx = AdaptContext((Xs, ys), (Xv, yv), Xt, seed=0, params={"pool_labels": yt})
    return b, ctx


def test_coral_zero_for_identical_inputs():
    h = torch.randn(256, 8)
    assert coral(h, h).item() == pytest.approx(0.0, abs=1e-7)
    assert coral(h, h * 3).item() > 0


def test_adabn_changes_bn_stats_only(setup):
    b, ctx = setup
    nb = adapt.get("adabn").adapt(b, ctx)
    old, new = bn_layers(b.model.net)[0], bn_layers(nb.model.net)[0]
    assert not torch.allclose(old.running_mean, new.running_mean)
    for (n1, p1), (_, p2) in zip(b.model.net.named_parameters(), nb.model.net.named_parameters(), strict=True):
        assert torch.equal(p1, p2), n1                       # no weight changed


def test_tent_changes_only_bn_affine(setup):
    b, ctx = setup
    nb = adapt.get("tent", guard_factor=1e9).adapt(b, ctx)   # guard off so it takes steps
    assert nb.meta["tent_steps"] > 0
    bn_params = {id(t) for m in bn_layers(nb.model.net) for t in (m.weight, m.bias)}
    changed = [n for (n, p1), (_, p2) in zip(b.model.net.named_parameters(), nb.model.net.named_parameters(),
                                              strict=True) if not torch.equal(p1, p2)]
    assert changed and all(".weight" in n or ".bias" in n for n in changed)
    assert all(id(p) in bn_params for n, p in nb.model.net.named_parameters() if n in changed)


@pytest.mark.parametrize("rate,prev,expect", [
    (0.90, 0.30, True),     # runs away above 2x the source rate
    (0.10, 0.20, True),     # collapsing toward "all benign" and still falling
    (0.10, 0.05, False),    # a silent model climbing back toward the source rate is allowed
    (0.30, 0.25, False),    # inside the band
])
def test_tent_guard_rule(rate, prev, expect):
    from xnids.adapt.tent import guard_violation

    assert guard_violation(rate, prev, r_src=0.3, factor=2.0) is expect


def test_tent_records_guard_reference(setup):
    b, ctx = setup
    nb = adapt.get("tent").adapt(b, ctx)
    assert 0 < nb.meta["tent_source_attack_rate"] <= 1 and isinstance(nb.meta["tent_guard_stop"], bool)


@pytest.mark.parametrize("name,params", [("scaling", {}), ("adabn", {}), ("tent", {}), ("coral", {"epochs": 1}),
                                         ("dann", {"epochs": 1}), ("fewshot", {"budget": 50, "rule": "random"}),
                                         ("fewshot", {"budget": 50, "rule": "uncertainty"}),
                                         ("fewshot", {"budget": 50, "rule": "drift"})])
def test_adapter_returns_new_working_bundle(setup, name, params):
    b, ctx = setup
    before = b.score(ctx.tgt_pool[:100]).copy()
    nb = adapt.get(name, **params).adapt(b, ctx)
    s = nb.score(ctx.tgt_pool[:500])
    assert np.isfinite(s).all() and ((s >= 0) & (s <= 1)).all()
    assert nb is not b and nb.meta["parent_version"] == "toy" and "threshold_source" in nb.meta
    np.testing.assert_array_equal(before, b.score(ctx.tgt_pool[:100]))     # the source bundle is untouched


def test_fewshot_reads_only_bought_labels(setup):
    b, ctx = setup
    labels, asked = ctx.params["pool_labels"], []

    def analyst(idx):                                        # the only way few-shot can see a target label
        asked.extend(idx.tolist())
        return labels[idx]

    spy_ctx = AdaptContext(ctx.src_train, ctx.src_val, ctx.tgt_pool, seed=0, params={"pool_labels": analyst})
    nb = adapt.get("fewshot", budget=50).adapt(b, spy_ctx)
    assert nb.meta["labels_used"] == 50 and len(asked) == 50 == len(set(asked))


def test_fewshot_trees_upweight(setup):
    b, ctx = setup
    Xs, ys = ctx.src_train
    m = zoo.build("xgb", {"n_estimators": 30}, device="cpu").fit(b.preprocess.transform(Xs), ys,
                                                                  b.preprocess.transform(ctx.src_val[0]), ctx.src_val[1])
    tb = Bundle(m, b.preprocess, 0.5, NAMES, {"version": "toyx"})
    nb = adapt.get("fewshot", budget=50, target_weight=0.2).adapt(tb, ctx)
    assert nb.meta["fewshot_repeat"] == round(0.25 * 6000 / 50)


def test_scaling_maps_target_moments_onto_source(setup):
    b, ctx = setup
    nb = adapt.get("scaling").adapt(b, ctx)
    zs, zt = b.model_input(ctx.src_train[0]), nb.model_input(ctx.tgt_pool)
    ls, lt = (np.sign(z) * np.log1p(np.abs(z)) for z in (zs, zt))
    np.testing.assert_allclose(ls.mean(0), lt.mean(0), atol=1e-3)


def test_mlp_only_adapters_reject_trees(setup):
    b, ctx = setup
    tb = Bundle(zoo.build("lda", device="cpu").fit(b.preprocess.transform(ctx.src_train[0]), ctx.src_train[1],
                                                   None, None), b.preprocess, 0.5, NAMES, {})
    with pytest.raises(TypeError):
        adapt.get("adabn").adapt(tb, ctx)
