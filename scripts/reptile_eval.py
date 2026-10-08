"""C12 / H6: Reptile, leave one target out (resumable per track x target x seed).

    python scripts/reptile_eval.py
    python scripts/reptile_eval.py --track nf43 --targets nf_ton_v2 --seeds 0 --iters 200     # quick check

For held-out target T: theta_0 = an MLP trained on the pooled OTHER datasets of the track; theta_meta = Reptile from
theta_0 on tasks drawn from those datasets (+ synthetic rescaled domains). On T (never seen in training):
  pooled-none        theta_0, threshold from pooled validation
  pooled-adabn       theta_0 + AdaBN on T's unlabelled pool
  reptile-adabn      theta_meta + AdaBN (zero-shot)
  finetune-{n}       theta_0 + AdaBN + k steps on a labelled support set of n target flows  (fair baseline)
  reptile-{n}        theta_meta + AdaBN + the same k steps on the same support set
Support sets come from T's TRAIN split (random selection, `repeats` draws); scores on T's TEST split.
Outputs: reports/tables/c12_long.csv, c12_summary.csv, reports/figures/c12_h6.png; models/reptile/<track>/<T>-s<seed>/.
"""

import argparse
import copy
import json
import logging
import time

import mlflow
import numpy as np
import pandas as pd
import polars as pl
import torch

from xnids.adapt.adabn import adabn_
from xnids.adapt.reptile import Task, inner_adapt, reptile
from xnids.eval import harness, metrics
from xnids.models import zoo
from xnids.models.preprocess import Preprocessor
from xnids.utils import config, log, paths, seed

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
LOG = logging.getLogger("c12")


class Domain:
    """One dataset of the track, preprocessed with the pooled Preprocessor (raw scale, float32)."""

    def __init__(self, name: str, X: np.ndarray, y: np.ndarray, family: np.ndarray) -> None:
        self.name, self.X, self.y, self.family = name, X, y, family
        self.fams = [f for f in np.unique(family) if f != "Benign"]
        self.by_fam = {f: np.flatnonzero(family == f) for f in np.unique(family)}

    def window(self, rng: np.random.Generator, n: int, benign_share: tuple[float, float]) -> np.ndarray:
        b = rng.uniform(*benign_share)
        w = rng.dirichlet(np.ones(len(self.fams))) if self.fams else np.array([])
        counts = {"Benign": int(round(b * n)), **{f: int(round((1 - b) * n * x)) for f, x in zip(self.fams, w,
                                                                                                strict=True)}}
        idx = [rng.choice(self.by_fam[f], c, replace=c > len(self.by_fam[f])) for f, c in counts.items()
               if c > 0 and f in self.by_fam]
        return rng.permutation(np.concatenate(idx))


def to_net(model, X: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(model.scaler.transform(X)).to(model.device)


def score(model, net, X: np.ndarray) -> np.ndarray:
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 65536):
            out.append(torch.sigmoid(net(to_net(model, X[i:i + 65536]))).cpu().numpy())
    return np.concatenate(out)


def evaluate(y: np.ndarray, s: np.ndarray, thr: float) -> dict:
    ev = metrics.evaluate(y, s, thr)
    return {k: ev[k] for k in ("fpr_at_thr", "dr_at_thr", "mcc_at_thr", "pr_auc")}


def run_target(cfg: dict, track: str, target: str, s: int, iters: int | None) -> pd.DataFrame:
    d, tc, ec = cfg["data"], cfg["tasks"], cfg["eval"]
    rcfg = {k: v for k, v in cfg.items() if k not in ("tracks", "seeds")} | {"track": track, "target": target}
    if iters:
        rcfg["reptile"] = {**rcfg["reptile"], "iters": iters}
    h = config.cfg_hash(rcfg)
    runs = log.find_runs(experiment=cfg["experiment"], finished_only=True, cfg_hash=h, seed=str(s))
    if not runs.empty:
        rid = runs.sort_values("start_time").run_id.iloc[-1]
        return pd.read_parquet(mlflow.artifacts.download_artifacts(run_id=rid, artifact_path="c12_rows.parquet"))
    seed.set_seed(s)
    rng = np.random.default_rng(s)
    t0 = time.time()
    others = [x for x in cfg["tracks"][track] if x != target]
    load = lambda ds, sp: harness.load_split(ds, track, sp, d["max_rows_per_split"], d["min_per_family"],  # noqa: E731
                                              d["dev_seed"])
    n_tr = cfg["pretrain"]["rows_per_dataset"]
    trs, vas = {}, {}
    for ds in others:
        tr, va = load(ds, "train"), load(ds, "val")
        it = np.sort(rng.choice(tr.X.height, min(n_tr, tr.X.height), replace=False))
        iv = np.sort(rng.choice(va.X.height, min(n_tr // 3, va.X.height), replace=False))
        trs[ds], vas[ds] = (tr.X[it], tr.y[it], tr.family[it]), (va.X[iv], va.y[iv])
    pre = Preprocessor().fit(pl.concat([v[0] for v in trs.values()]))
    Xtr = np.vstack([pre.transform(v[0]) for v in trs.values()])
    ytr = np.concatenate([v[1] for v in trs.values()])
    Xva = np.vstack([pre.transform(v[0]) for v in vas.values()])
    yva = np.concatenate([v[1] for v in vas.values()])
    model = zoo.build("mlp", cfg["pretrain"]["params"], seed=s).fit(Xtr, ytr, Xva, yva)
    thr0 = metrics.threshold_at_dr(yva, model.score(Xva), 0.95)
    theta0 = copy.deepcopy(model.net)
    domains = [Domain(ds, pre.transform(trs[ds][0]), trs[ds][1], trs[ds][2]) for ds in others]

    def sample_task(r: np.random.Generator) -> Task:
        dom = domains[r.integers(len(domains))]
        idx = dom.window(r, tc["window"], tuple(tc["benign_share"]))
        X = dom.X[idx]
        if r.random() < tc["synthetic_prob"]:
            # float64 and clipped: NF-v2 rate fields hold values near the float32 maximum (rates over the broken
            # zero durations), and x2 overflowed to inf -> NaN weights (first full C12 run, NF-CSE18 held out)
            lim = np.finfo(np.float32).max
            X = np.clip(X.astype(np.float64) * r.uniform(*tc["scale_range"], size=X.shape[1]), -lim, lim).astype(
                np.float32)
        n = int(r.choice(tc["support_sizes"]))
        T = to_net(model, X)
        return Task(T, T[:n], torch.as_tensor(dom.y[idx][:n], dtype=torch.float32, device=model.device))

    rp = rcfg["reptile"]
    meta = reptile(copy.deepcopy(theta0), sample_task, iters=rp["iters"], k=rp["k"], inner_lr=rp["inner_lr"],
                   eps0=rp["eps0"], meta_batch=rp["meta_batch"], batch=rp["batch"], seed=s,
                   log_every=max(1, rp["iters"] // 5), logger=lambda i, nd: LOG.info("%s/%s s%d meta-iter %d |delta| %.4f",
                                                                                      track, target, s, i, nd))
    thr_meta = metrics.threshold_at_dr(yva, score(model, meta, Xva), 0.95)
    train_min = (time.time() - t0) / 60

    tgt_tr, tgt_te = load(target, "train"), load(target, "test")
    Ztr, Zte = pre.transform(tgt_tr.X), pre.transform(tgt_te.X)
    pool = to_net(model, Ztr[rng.choice(len(Ztr), min(ec["pool_rows"], len(Ztr)), replace=False)])
    rows = []

    def add(method, budget, rep, net, thr, extra=None):
        rows.append({"track": track, "target": target, "seed": s, "method": method, "budget": budget, "repeat": rep,
                     **evaluate(tgt_te.y, score(model, net, Zte), thr), **(extra or {})})

    add("pooled-none", 0, 0, theta0, thr0)
    a0 = copy.deepcopy(theta0)
    adabn_(a0, pool)
    add("pooled-adabn", 0, 0, a0, thr0)
    am = copy.deepcopy(meta)
    adabn_(am, pool)
    add("reptile-adabn", 0, 0, am, thr_meta)
    gen_seed = 1000 * s
    for n in ec["budgets"]:
        for rep in range(ec["repeats"]):
            sup = np.random.default_rng(gen_seed + 17 * rep + n).choice(len(Ztr), n, replace=False)
            task = Task(pool, to_net(model, Ztr[sup]), torch.as_tensor(tgt_tr.y[sup], dtype=torch.float32,
                                                                       device=model.device))
            n_att = int(tgt_tr.y[sup].sum())
            for name, start, fallback in (("finetune", theta0, thr0), ("reptile", meta, thr_meta)):
                f = inner_adapt(copy.deepcopy(start), task, rp["k"], rp["inner_lr"], rp["batch"],
                                torch.Generator().manual_seed(gen_seed + rep))
                if n_att >= ec["min_attacks"]:
                    thr = metrics.threshold_at_dr(tgt_tr.y[sup], score(model, f, Ztr[sup]), 0.95)
                else:
                    thr = fallback
                add(f"{name}-{n}", n, rep, f, thr, {"support_attacks": n_att,
                                                    "threshold_on_support": n_att >= ec["min_attacks"]})
    df = pd.DataFrame(rows)
    out = paths.MODELS / "reptile" / track / f"{target}-s{s}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(meta.state_dict(), out / "meta.pt")
    torch.save(theta0.state_dict(), out / "theta0.pt")
    pre.save(out / "preprocess.json")
    (out / "scaler.json").write_text(json.dumps(model.scaler.to_dict()))
    (out / "meta.json").write_text(json.dumps({"track": track, "held_out": target, "trained_on": others, "seed": s,
                                               "thr_meta": thr_meta, "thr0": thr0, "cfg_hash": h,
                                               "d_in": Xtr.shape[1]}, indent=1))
    with log.start_run(rcfg | {"run": {"name": "c12"}}, seed=s, experiment=cfg["experiment"],
                       run_name=f"c12-{track}-{target}", tags={"track": track, "target": target}) as run:
        mlflow.set_tag("cfg_hash", h)
        mlflow.log_metric("train_minutes", train_min)
        for m, v in df.groupby("method").mcc_at_thr.mean().items():
            mlflow.log_metric(f"mcc/{m}", v)
        df["run_id"] = run.info.run_id
        log.log_df_artifact(df, "c12_rows.parquet")
    LOG.info("%s/%s s%d: %s", track, target, s, df.groupby("method").mcc_at_thr.mean().round(3).to_dict())
    return df


def c9_reference() -> pd.DataFrame:
    """C9 numbers on the same nf43 targets (single-source MLPs; different starting point, for context only)."""
    p = paths.TABLES / "c9_long.csv"
    if not p.exists():
        return pd.DataFrame()
    c9 = pd.read_csv(p)
    c9 = c9[(c9.part == "mlp") & (c9.track == "nf43")]
    return c9.groupby(["target", "adapter"]).tgt_mcc_at_thr.mean().rename("mcc").reset_index()


def plot(summ: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tracks = list(dict.fromkeys(summ.track))
    fig, axes = plt.subplots(1, len(tracks), figsize=(5.6 * len(tracks), 3.8), squeeze=False, sharey=True,
                             facecolor=SURFACE, layout="constrained")
    for ax, t in zip(axes[0], tracks, strict=True):
        g = summ[summ.track == t]
        ax.set_facecolor(SURFACE)
        for name, col in (("finetune", "#eb6834"), ("reptile", "#2a78d6")):
            gg = g[g.method.str.startswith(name + "-") & (g.budget > 0)].sort_values("budget")
            ax.plot(gg.budget, gg.mcc, color=col, lw=2, marker="o", ms=5, markeredgecolor=SURFACE,
                    label=f"{name} (+AdaBN, 8 steps)")
        for name, ls in (("pooled-adabn", (0, (4, 3))), ("reptile-adabn", (0, (1, 2)))):
            v = g[g.method == name].mcc
            if len(v):
                ax.axhline(float(v.iloc[0]), color=MUTED, lw=1.2, ls=ls, label=f"{name} (0 labels)")
        ax.set_xscale("log")
        ax.set_xticks([50, 200, 1000], ["50", "200", "1000"])
        ax.set_xlabel("labelled target flows (support set)", fontsize=8, color=INK2)
        ax.set_title(f"track {t}: mean over held-out targets and seeds", fontsize=9, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED, labelsize=7.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(AXIS)
    axes[0, 0].set_ylabel("target MCC", fontsize=8, color=INK2)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=4, frameon=False,
               fontsize=7.5, labelcolor=INK2)
    fig.suptitle("C12 / H6: Reptile vs plain fine-tuning from the same pooled initialisation", x=0.01, ha="left",
                 fontsize=10, color=INK)
    fig.savefig(paths.FIGURES / "c12_h6.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/adapt/c12.yaml")
    ap.add_argument("--track", default="all")
    ap.add_argument("--targets", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--iters", type=int, default=None, help="override meta-iterations (quick checks only)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("mlflow", "alembic", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = config.load(args.config)
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else cfg["seeds"]
    parts = []
    for track, dsets in cfg["tracks"].items():
        if args.track not in ("all", track):
            continue
        for tgt in (args.targets.split(",") if args.targets else dsets):
            for s in seeds:
                parts.append(run_target(cfg, track, tgt, s, args.iters))
    df = pd.concat(parts, ignore_index=True)
    if args.iters is None and args.targets is None:
        df.to_csv(paths.TABLES / "c12_long.csv", index=False)
    summ = df.groupby(["track", "method", "budget"]).agg(
        mcc=("mcc_at_thr", "mean"), fpr=("fpr_at_thr", "mean"), dr=("dr_at_thr", "mean"),
        pr_auc=("pr_auc", "mean"), runs=("mcc_at_thr", "size")).reset_index()
    paired = df[df.budget > 0].assign(kind=lambda x: x.method.str.split("-").str[0]).pivot_table(
        index=["track", "target", "seed", "budget", "repeat"], columns="kind", values="mcc_at_thr").reset_index()
    paired["reptile_minus_finetune"] = paired.reptile - paired.finetune
    wins = paired.groupby(["track", "budget"]).agg(mean_gain=("reptile_minus_finetune", "mean"),
                                                   reptile_wins=("reptile_minus_finetune", lambda x: float((x > 0).mean())))
    if args.iters is None and args.targets is None:
        summ.to_csv(paths.TABLES / "c12_summary.csv", index=False)
        wins.reset_index().to_csv(paths.TABLES / "c12_paired.csv", index=False)
        plot(summ)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(summ.round(3).to_string(index=False))
        print("\nreptile vs fine-tune, paired (same support set):\n" + wins.round(3).to_string())
        ref = c9_reference()
        if len(ref):
            print("\nC9 reference on nf43 targets (single-source MLPs, mean MCC):\n" +
                  ref.pivot_table(index="adapter", columns="target", values="mcc").round(3).to_string())


if __name__ == "__main__":
    main()
