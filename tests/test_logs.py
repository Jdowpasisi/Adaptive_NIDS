import numpy as np
import polars as pl

from xnids.select.logs import Pool, make_windows

CFG = {"per_pair": 40, "sizes": [200, 400], "kinds": {"random": 0.3, "mix": 0.3, "blend": 0.4},
       "blend_share": [0.1, 0.9], "benign_share": [0.5, 0.97], "seed": 0}


def _pool(n, tag, ts=False, seed=0):
    rng = np.random.default_rng(seed)
    fam = rng.choice(["Benign", "DoS", "Recon"], n, p=[0.7, 0.2, 0.1])
    X = pl.DataFrame({"a": rng.normal(size=n), "src": np.full(n, tag)})
    return Pool(X, (fam != "Benign").astype(int), fam, np.arange(n) * 1000 if ts else None)


def test_windows_halves_kinds_and_blend():
    tgt, src = _pool(5000, 1.0), _pool(3000, 0.0, seed=1)
    ws = make_windows(tgt, src, CFG, seed=0)
    assert len(ws) == 40 and {w.kind for w in ws} == {"random", "mix", "blend"}
    for w in ws:
        assert w.X_adapt.height + w.X_eval.height == w.size and w.X_adapt.height == w.size // 2
        tgt_frac = float(np.concatenate([w.X_adapt["src"], w.X_eval["src"]]).mean())
        if w.kind == "blend":
            assert 0.08 < w.target_share < 0.92 and abs(tgt_frac - w.target_share) < 0.01
        else:
            assert tgt_frac == 1.0


def test_mix_window_benign_share_and_determinism():
    tgt, src = _pool(5000, 1.0), _pool(3000, 0.0, seed=1)
    ws = [w for w in make_windows(tgt, src, {**CFG, "kinds": {"mix": 1.0}}, seed=3)]
    shares = [1 - np.concatenate([w.y_adapt, w.y_eval]).mean() for w in ws]
    assert min(shares) >= 0.49 and max(shares) <= 0.98
    again = make_windows(tgt, src, {**CFG, "kinds": {"mix": 1.0}}, seed=3)
    assert all(a.X_eval.equals(b.X_eval) for a, b in zip(ws, again, strict=True))


def test_time_windows_are_contiguous_and_ordered():
    tgt, src = _pool(5000, 1.0, ts=True), _pool(3000, 0.0, seed=1)
    tgt.X = tgt.X.with_columns(t=pl.Series(tgt.ts))
    src.X = src.X.with_columns(t=pl.lit(-1, dtype=pl.Int64))     # same columns, so blend windows can concat
    ws = [w for w in make_windows(tgt, src, CFG, seed=0) if w.kind == "time"]
    assert ws
    for w in ws:
        t = np.concatenate([w.X_adapt["t"], w.X_eval["t"]])
        assert (np.diff(t) == 1000).all()                      # one contiguous, time-ordered slice
