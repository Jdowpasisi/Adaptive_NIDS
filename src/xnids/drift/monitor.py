"""C8 drift monitor: one DriftReport per window of incoming flows, without labels.

    mon = DriftMonitor(bundle, ref_X, ref_y, cfg)      # reference = a fixed sample of SOURCE validation flows
    reports = mon.run(windows)                         # iterable of polars DataFrames (the track's feature columns)

Per window: per-feature KS (Bonferroni, optional effect-size floor), MMD with a permutation p-value, ADWIN on the
model-confidence stream (stateful across windows), mean confidence / entropy, ATC-estimated error rise, a
plain-language explanation of the top-5 shifted features, and a recommendation that requires both statistical
drift and the cost trigger. ref_y is used only to fit ATC on source validation; incoming windows are never labelled.
"""

import json
from dataclasses import asdict, dataclass, field

import numpy as np
import polars as pl

from xnids.drift import explain, ks, trigger
from xnids.drift.adwin import ConfidenceADWIN, confidence, entropy
from xnids.drift.mmd import MMDTest
from xnids.models.bundle import Bundle

DEFAULTS = {
    "reference": {"n": 20_000, "seed": 0},
    "ks": {"alpha": 0.05, "min_stat": 0.1},
    "mmd": {"n_ref": 2000, "n_win": 2000, "n_perm": 200, "alpha": 0.05, "pca_dims": 20},
    "adwin": {"delta": 0.002},
    "combine": {"min_votes": 2},
    "cost": {},
    # C15: "calibrated" = alert on the novelty share over a threshold calibrated on time-ordered source traffic,
    # held for `persist` windows (live traffic). "c8" = the C8 rule (2 of {KS effect, MMD, ADWIN} + cost trigger).
    "mode": "c8",
    "persist": 2,
    "novelty": {"flow_quantile": 0.99},
}


@dataclass
class DriftReport:
    window_id: int
    n: int
    ks: dict[str, tuple[float, float]]
    mmd: tuple[float, float]
    adwin_change: bool
    mean_conf: float
    entropy: float
    est_fpr_rise: float             # ATC-estimated rise in error rate vs source val (label-free)
    attack_share_pred: float        # share of the window flagged at the frozen threshold
    top_features: list[str]
    message: str
    recommend: bool
    detectors: dict[str, bool] = field(default_factory=dict)   # ks, ks_effect, mmd, adwin, combined
    extra_wrong_per_hour: float = 0.0
    explanations: list[str] = field(default_factory=list)
    novelty_share: float | None = None     # C15 calibrated mode: share of flows unlike any reference flow
    streak: int = 0                        # C15 calibrated mode: consecutive windows over the threshold

    def features(self) -> dict[str, float]:
        """The label-free drift features the C11 selector consumes."""
        d = np.array([v[0] for v in self.ks.values()])
        ks_alpha = 0.05 / max(len(self.ks), 1)          # Bonferroni at the default alpha
        return {"ks_max": float(d.max()), "ks_mean": float(d.mean()),
                "n_ks_sig": int(sum(p < ks_alpha for _, p in self.ks.values())),
                "mmd_stat": self.mmd[0], "mmd_p": self.mmd[1], "adwin_flag": float(self.adwin_change),
                "mean_conf": self.mean_conf, "entropy": self.entropy, "est_fpr_rise": self.est_fpr_rise,
                "attack_share_pred": self.attack_share_pred}

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=float)


def window_stats(rep: "DriftReport") -> dict[str, float]:
    """Scalar statistics of one report, the candidates for calibrated thresholds (C15)."""
    d = np.sort(np.array([v[0] for v in rep.ks.values()]))[::-1]
    return {"ks_max": float(d[0]), "ks_top5": float(d[:5].mean()), "ks_mean": float(d.mean()),
            "mmd_stat": float(rep.mmd[0]), "mean_conf": rep.mean_conf, "entropy": rep.entropy,
            "est_fpr_rise": rep.est_fpr_rise, "attack_share_pred": rep.attack_share_pred,
            "adwin": float(rep.adwin_change), "novelty_share": float("nan") if rep.novelty_share is None else rep.novelty_share}


def _deep(base: dict, over: dict) -> dict:
    out = {k: dict(v) if isinstance(v, dict) else v for k, v in base.items()}
    for k, v in (over or {}).items():
        out[k] = {**out.get(k, {}), **v} if isinstance(v, dict) else v
    return out


class DriftMonitor:
    def __init__(self, bundle: Bundle, ref_X: pl.DataFrame, ref_y: np.ndarray, cfg: dict | None = None) -> None:
        self.cfg = _deep(DEFAULTS, cfg or {})
        self.bundle = bundle
        rc = self.cfg["reference"]
        rng = np.random.default_rng(rc["seed"])
        idx = np.sort(rng.choice(ref_X.height, min(rc["n"], ref_X.height), replace=False))
        self.ref_idx = idx                                          # rows of ref_X used; hold the rest out
        Xr, yr = ref_X[idx], np.asarray(ref_y)[idx]
        self.names = bundle.preprocess.features_out
        self.ref = bundle.preprocess.transform(Xr)                  # imputed: KS / MMD never see inf or NaN
        s_ref = bundle.model.score(self.ref)
        self.ref_attack_share = float(np.mean(s_ref >= bundle.threshold))
        self.atc = trigger.ATC().fit(confidence(s_ref), (s_ref >= bundle.threshold) == (yr == 1))
        m = self.cfg["mmd"]
        self.mmd = MMDTest(m["n_ref"], m["n_win"], m["n_perm"], m["pca_dims"], rc["seed"]).fit(self.ref)
        self.adwin = ConfidenceADWIN(self.cfg["adwin"]["delta"])
        self.cost = trigger.CostConfig(**self.cfg["cost"])
        self.window_id = 0
        self.streak = 0
        self.thresholds: dict[str, float] = dict(self.cfg.get("thresholds", {}))
        self.novelty = None
        if self.cfg["mode"] == "calibrated":
            from xnids.drift.novelty import Novelty

            self.novelty = Novelty(bundle).fit(self.ref)
            if "novelty_tau" in self.cfg:
                self.novelty.tau = float(self.cfg["novelty_tau"])

    def process(self, X: pl.DataFrame) -> DriftReport:
        W = self.bundle.preprocess.transform(X)
        s = self.bundle.model.score(W)
        conf = confidence(s)
        kc, mc = self.cfg["ks"], self.cfg["mmd"]
        k = ks.ks_features(self.ref, W, self.names)
        mmd_stat, mmd_p = self.mmd.test(W)
        cuts = self.adwin.update(conf)
        det = {
            "ks": bool(ks.significant(k, kc["alpha"])),
            "ks_effect": bool(ks.significant(k, kc["alpha"], kc["min_stat"])),
            "mmd": mmd_p < mc["alpha"],
            "adwin": bool(cuts),
        }
        det["combined"] = sum((det["ks_effect"], det["mmd"], det["adwin"])) >= self.cfg["combine"]["min_votes"]
        rise = self.atc.error_rise(conf)
        worth, extra = trigger.worth_acting(rise, self.cost)
        nov = None
        if self.novelty is not None and self.novelty.tau is not None:
            nov = self.novelty.share(W)
            if "novelty_share" in self.thresholds:
                det["novelty"] = nov > self.thresholds["novelty_share"]
                det["combined"] = det["novelty"]
        if self.cfg["mode"] == "calibrated":
            # the ATC cost trigger is not used: on the demo it estimated a larger error rise for the held-out
            # in-distribution segment (0.04-0.10) than for the new network (0.03-0.07)
            self.streak = self.streak + 1 if det["combined"] else 0
            worth = self.streak >= self.cfg["persist"]
        top = ks.ranked(k)[:5]
        sentences = [explain.describe(self.ref[:, self.names.index(f)], W[:, self.names.index(f)], f, k[f][0])
                     for f in top]
        rep = DriftReport(
            window_id=self.window_id, n=len(W), ks=k, mmd=(mmd_stat, mmd_p), adwin_change=det["adwin"],
            mean_conf=float(conf.mean()), entropy=float(entropy(s).mean()), est_fpr_rise=rise,
            attack_share_pred=float(np.mean(s >= self.bundle.threshold)), top_features=top,
            message=explain.message(sentences, det["combined"]), recommend=bool(det["combined"] and worth),
            detectors=det, extra_wrong_per_hour=extra, explanations=sentences, novelty_share=nov,
            streak=self.streak)
        self.window_id += 1
        return rep

    def run(self, stream) -> list[DriftReport]:
        return [self.process(w) for w in stream]
