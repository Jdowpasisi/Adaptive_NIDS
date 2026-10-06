"""Evaluation metrics (Build Guide C5, rule 6).

Headline: the FPR at a threshold frozen on SOURCE validation data, always reported together with the DR that
threshold actually achieves on the target. Also PR-AUC (threshold-free) and MCC at the frozen threshold (for
comparison with Cantone et al.). `oracle_fpr_at_dr` re-picks the threshold on the evaluated data itself, so it is a
DIAGNOSTIC only: it separates "the ranking broke" from "the calibration shifted" and must never be reported as
operational performance. Plain accuracy is deliberately absent.
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def threshold_at_dr(y_val: np.ndarray, s_val: np.ndarray, dr: float = 0.95) -> float:
    """Highest threshold t such that flagging `score >= t` catches at least `dr` of the attacks in (y_val, s_val)."""
    s_att = np.sort(np.asarray(s_val)[np.asarray(y_val) == 1])
    if len(s_att) == 0:
        raise ValueError("threshold_at_dr: no attacks in the validation data")
    k = int(np.floor((1 - dr) * len(s_att)))
    return float(s_att[k])


def confusion(y: np.ndarray, s: np.ndarray, t: float) -> tuple[int, int, int, int]:
    """(tp, fp, tn, fn) when flagging score >= t."""
    y = np.asarray(y).astype(bool)
    pred = np.asarray(s) >= t
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    return tp, fp, int((~pred & ~y).sum()), int((~pred & y).sum())


def rates(y: np.ndarray, s: np.ndarray, t: float) -> tuple[float, float]:
    """(FPR, DR) at threshold t. A rate whose denominator is empty is NaN."""
    tp, fp, tn, fn = confusion(y, s, t)
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    dr = tp / (tp + fn) if tp + fn else float("nan")
    return fpr, dr


def mcc(y: np.ndarray, s: np.ndarray, t: float) -> float:
    tp, fp, tn, fn = confusion(y, s, t)
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / den) if den else 0.0


def oracle_fpr_at_dr(y: np.ndarray, s: np.ndarray, dr: float = 0.95) -> float:
    """DIAGNOSTIC: FPR when the threshold is re-picked on (y, s) itself to reach `dr`."""
    return rates(y, s, threshold_at_dr(y, s, dr))[0]


def evaluate(y: np.ndarray, s: np.ndarray, t: float, target_dr: float = 0.95) -> dict[str, float]:
    """All reported metrics for one (model, source, target) at the frozen source threshold t."""
    y = np.asarray(y)
    fpr, dr = rates(y, s, t)
    both = 0 < y.sum() < len(y)
    return {
        "fpr_at_thr": fpr,                       # headline, always next to dr_at_thr
        "dr_at_thr": dr,
        "pr_auc": float(average_precision_score(y, s)) if both else float("nan"),
        "mcc_at_thr": mcc(y, s, t),
        "roc_auc": float(roc_auc_score(y, s)) if both else float("nan"),
        "oracle_fpr_at_dr": oracle_fpr_at_dr(y, s, target_dr) if y.sum() else float("nan"),   # diagnostic
        "n": int(len(y)),
        "n_attack": int(y.sum()),
    }
