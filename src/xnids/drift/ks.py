"""Per-feature two-sample Kolmogorov-Smirnov tests against the reference sample, Bonferroni-corrected."""

import numpy as np
from scipy.stats import ks_2samp


def ks_features(ref: np.ndarray, win: np.ndarray, names: list[str]) -> dict[str, tuple[float, float]]:
    """{feature: (KS statistic D, p-value)} for every column."""
    out = {}
    for j, name in enumerate(names):
        r = ks_2samp(ref[:, j], win[:, j], method="asymp")
        out[name] = (float(r.statistic), float(r.pvalue))
    return out


def significant(ks: dict[str, tuple[float, float]], alpha: float, min_stat: float = 0.0) -> list[str]:
    """Features with p < alpha / n_features (Bonferroni) and, optionally, an effect size D >= min_stat.

    With 5,000-flow windows against a 20,000-flow reference the test detects tiny, practically irrelevant
    shifts, so the effect-size floor is what separates "different" from "meaningfully different".
    """
    a = alpha / max(len(ks), 1)
    return [f for f, (d, p) in ks.items() if p < a and d >= min_stat]


def ranked(ks: dict[str, tuple[float, float]]) -> list[str]:
    return sorted(ks, key=lambda f: (-ks[f][0], f))


def summary(ks: dict[str, tuple[float, float]], alpha: float, min_stat: float) -> dict[str, float]:
    d = np.array([v[0] for v in ks.values()])
    return {"ks_max": float(d.max()) if len(d) else 0.0, "ks_mean": float(d.mean()) if len(d) else 0.0,
            "n_ks_sig": len(significant(ks, alpha, min_stat))}
