"""C7 / H3: share of the cross-dataset collapse explained by removing (or normalising) lab-telling features.

For a metric where higher is better (MCC, PR-AUC):   collapse = within_0 - cross_0,  recovered = cross_k - cross_0
For a rate where lower is better (oracle FPR@95%DR): collapse = cross_0 - within_0,  recovered = cross_0 - cross_k
share = recovered / collapse, reported raw: < 0 means the intervention made transfer worse, > 1 means the ablated
model transfers better than the original model did within its own dataset.

The Build Guide's formula uses FPR at the frozen threshold: (FPR_cross - FPR_cross_ablated) / (FPR_cross - FPR_within).
It is reported as share_fpr_thr, but C6 showed that cross-dataset failure mostly shows up as lost DETECTION at an
unchanged or lower FPR, so that denominator is often near zero or negative. MCC and oracle FPR capture the collapse;
they are the headline shares.
"""

import numpy as np
import pandas as pd

HIGHER = ("mcc_at_thr", "pr_auc")
LOWER = ("oracle_fpr_at_dr",)


def _share(rec: float, col: float, min_collapse: float = 0.02) -> float:
    return float(rec / col) if col > min_collapse else np.nan


def shares(cells: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """cells: aggregated (model, source, target, k, kind, <metric>_mean); ref: the k = 0 cells (from C6).
    Returns one row per (model, source, target, k) with the within/cross values and the shares."""
    r = ref.set_index(["model", "source", "target"])
    out = []
    for (m, s, t, k), g in cells.groupby(["model", "source", "target", "k"]):
        cross = g[g.kind == "cross"].iloc[0] if (g.kind == "cross").any() else None
        within = g[g.kind == "within"].iloc[0] if (g.kind == "within").any() else None
        if cross is None or (m, s, t) not in r.index or (m, s, s) not in r.index:
            continue
        c0, w0 = r.loc[(m, s, t)], r.loc[(m, s, s)]
        row = {"model": m, "source": s, "target": t, "k": k, "n_seeds": int(cross.get("n_seeds", np.nan))}
        for met in HIGHER + LOWER:
            sign = 1 if met in HIGHER else -1
            row[f"{met}_within0"], row[f"{met}_cross0"] = w0[f"{met}_mean"], c0[f"{met}_mean"]
            row[f"{met}_cross_k"] = cross[f"{met}_mean"]
            row[f"{met}_within_k"] = within[f"{met}_mean"] if within is not None else np.nan
            row[f"share_{met}"] = _share(sign * (cross[f"{met}_mean"] - c0[f"{met}_mean"]),
                                         sign * (w0[f"{met}_mean"] - c0[f"{met}_mean"]))
        f_c0, f_w0, f_ck = c0["fpr_at_thr_mean"], w0["fpr_at_thr_mean"], cross["fpr_at_thr_mean"]
        den = f_c0 - f_w0
        row["share_fpr_thr"] = float((f_c0 - f_ck) / den) if abs(den) > 1e-9 else np.nan
        row["fpr_thr_collapse_denominator"] = den
        row["dr_at_thr_cross0"], row["dr_at_thr_cross_k"] = c0["dr_at_thr_mean"], cross["dr_at_thr_mean"]
        out.append(row)
    return pd.DataFrame(out)
