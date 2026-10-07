"""C9: apply every adapter to source-trained bundles and measure before / after (resumable per pair x seed x model).

    python scripts/adapt_eval.py                         # everything in configs/adapt/c9.yaml
    python scripts/adapt_eval.py --part mlp --pairs 0 --seeds 0
    python scripts/adapt_eval.py --summary-only

Per (pair, seed, model): one MLflow run (experiment c9) holding a table with the unadapted bundle ("none") and
every adapter: metrics on the TARGET test split at the adapted bundle's threshold, the same on the SOURCE test split
(forgetting), labels used, compute seconds, threshold source.
Outputs: reports/tables/c9_long.csv, c9_summary.csv; figure reports/figures/c9_before_after.png
"""

import argparse
import json
import logging
import time

import mlflow
import numpy as np
import pandas as pd

from xnids import adapt
from xnids.adapt.base import AdaptContext
from xnids.eval import harness, metrics
from xnids.models.bundle import bundle_for
from xnids.utils import config, log, paths, seed

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
KEEP = ("fpr_at_thr", "dr_at_thr", "pr_auc", "mcc_at_thr", "oracle_fpr_at_dr")


def label(a: dict) -> str:
    rest = {k: v for k, v in a.items() if k != "name"}
    return a["name"] + ("(" + ",".join(f"{k}={v}" for k, v in rest.items()) + ")" if rest else "")


def evaluate(b, te_t, te_s, dr: float) -> dict:
    out = {}
    for tag, te in (("tgt", te_t), ("src", te_s)):
        ev = metrics.evaluate(te.y, b.score(te.X), b.threshold, dr)
        out.update({f"{tag}_{k}": ev[k] for k in KEEP})
    return out


def run_pair(cfg: dict, part: str, track: str, src: str, tgt: str, s: int) -> pd.DataFrame:
    pc = cfg[part]
    model = pc.get("model", "mlp")
    d = cfg["data"]
    rcfg = {"experiment": cfg["experiment"], "version": cfg.get("version", 1), "part": part, "model": model, "pair": [track, src, tgt],
            "adapters": pc["adapters"], "data": d, "pool_rows": cfg["pool_rows"], "src_rows": cfg["src_rows"]}
    h = config.cfg_hash(rcfg)
    runs = log.find_runs(experiment=cfg["experiment"], finished_only=True, cfg_hash=h, seed=str(s))
    if not runs.empty:
        rid = runs.sort_values("start_time").run_id.iloc[-1]
        return pd.read_parquet(mlflow.artifacts.download_artifacts(run_id=rid, artifact_path="c9_table.parquet"))
    seed.set_seed(s)
    load = lambda ds, sp: harness.load_split(ds, track, sp, d["max_rows_per_split"], d["min_per_family"],  # noqa: E731
                                              d["dev_seed"])
    b = bundle_for(model, track, src, s)
    st, sv, tp, te_t, te_s = load(src, "train"), load(src, "val"), load(tgt, "train"), load(tgt, "test"), \
        load(src, "test")
    rng = np.random.default_rng(s)
    pi = np.sort(rng.choice(tp.X.height, min(cfg["pool_rows"], tp.X.height), replace=False))
    si = np.sort(rng.choice(st.X.height, min(cfg["src_rows"], st.X.height), replace=False))
    ctx = AdaptContext((st.X[si], st.y[si]), (sv.X, sv.y), tp.X[pi], seed=s, params={"pool_labels": tp.y[pi]})
    dr = 0.95
    rows = [{"adapter": "none", "labels_used": 0, "compute_s": 0.0, "threshold_source": "frozen source threshold",
             **evaluate(b, te_t, te_s, dr)}]
    with log.start_run(rcfg | {"run": {"name": f"c9-{part}"}}, seed=s, experiment=cfg["experiment"],
                       run_name=f"c9-{model}-{src}-{tgt}", tags={"part": part, "model": model, "track": track,
                                                                  "source": src, "target": tgt}) as run:
        mlflow.set_tag("cfg_hash", h)                       # the hash of rcfg without the run block
        for a in pc["adapters"]:
            t0 = time.time()
            nb = adapt.get(a["name"], **{k: v for k, v in a.items() if k != "name"}).adapt(b, ctx)
            dt = time.time() - t0
            r = {"adapter": label(a), "labels_used": nb.meta.get("labels_used", 0), "compute_s": dt,
                 "threshold_source": nb.meta.get("threshold_source", ""), **evaluate(nb, te_t, te_s, dr),
                 "meta": json.dumps({k: v for k, v in nb.meta.items() if k.startswith(("tent", "fewshot", "budget"))})}
            rows.append(r)
            logging.info("%s %s->%s s%d %-32s tgt FPR %.4f DR %.4f MCC %.3f | src MCC %.3f | %.0fs", model, src, tgt,
                         s, r["adapter"], r["tgt_fpr_at_thr"], r["tgt_dr_at_thr"], r["tgt_mcc_at_thr"],
                         r["src_mcc_at_thr"], dt)
        table = pd.DataFrame(rows).assign(part=part, model=model, track=track, source=src, target=tgt, seed=s,
                                          run_id=run.info.run_id)
        log.log_df_artifact(table, "c9_table.parquet")
    return table


def summarise(long: pd.DataFrame) -> pd.DataFrame:
    base = long[long.adapter == "none"].set_index(["part", "model", "source", "target", "seed"])
    rows = []
    for _, r in long[long.adapter != "none"].iterrows():
        b0 = base.loc[(r.part, r.model, r.source, r.target, r.seed)]
        rows.append({"part": r.part, "model": r.model, "track": r.track, "source": r.source, "target": r.target,
                     "seed": r.seed, "adapter": r.adapter, "labels_used": r.labels_used, "compute_s": r.compute_s,
                     **{f"d_{k}": r[f"tgt_{k}"] - b0[f"tgt_{k}"] for k in KEEP},
                     **{f"after_{k}": r[f"tgt_{k}"] for k in KEEP},
                     **{f"before_{k}": b0[f"tgt_{k}"] for k in KEEP},
                     "d_src_mcc": r.src_mcc_at_thr - b0.src_mcc_at_thr})
    d = pd.DataFrame(rows)
    return d.groupby(["part", "model", "track", "adapter"]).agg(
        pairs_x_seeds=("seed", "size"), labels_used=("labels_used", "mean"), compute_s=("compute_s", "mean"),
        before_mcc=("before_mcc_at_thr", "mean"), after_mcc=("after_mcc_at_thr", "mean"),
        d_mcc=("d_mcc_at_thr", "mean"), d_dr=("d_dr_at_thr", "mean"), d_fpr=("d_fpr_at_thr", "mean"),
        d_pr_auc=("d_pr_auc", "mean"), d_oracle_fpr=("d_oracle_fpr_at_dr", "mean"),
        share_improved_mcc=("d_mcc_at_thr", lambda x: float((x > 0.01).mean())),
        d_src_mcc=("d_src_mcc", "mean")).reset_index()


def plot(summ: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = summ[(summ.part == "mlp")]
    if m.empty:
        return
    tracks = list(dict.fromkeys(m.track))
    fig, axes = plt.subplots(1, len(tracks), figsize=(5.2 * len(tracks), 4.6), squeeze=False, sharey=True,
                             facecolor=SURFACE, layout="constrained")
    order = list(dict.fromkeys(m.sort_values("adapter").adapter))
    for ax, t in zip(axes[0], tracks, strict=True):
        g = m[m.track == t].set_index("adapter").reindex(order)
        y = np.arange(len(order))
        ax.set_facecolor(SURFACE)
        for yi, (b0, a1) in enumerate(zip(g.before_mcc, g.after_mcc, strict=True)):
            ax.plot([b0, a1], [yi, yi], color=AXIS, lw=1.5, zorder=1)
        ax.scatter(g.before_mcc, y, color=MUTED, s=28, zorder=2, label="before (unadapted)")
        ax.scatter(g.after_mcc, y, color="#2a78d6", s=34, zorder=3, edgecolor=SURFACE, linewidth=1,
                   label="after adaptation")
        ax.axvline(0, color=AXIS, lw=1)
        ax.set_yticks(y, order, fontsize=7.5, color=INK2)
        ax.set_title(f"MLP, track {t} (mean over pairs and seeds)", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("target MCC at the adapted model's threshold", fontsize=8, color=INK2)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED, labelsize=7.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(AXIS)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False,
               fontsize=8, labelcolor=INK2)
    fig.suptitle("C9: target MCC before and after each adapter", x=0.01, ha="left", fontsize=10, color=INK)
    fig.savefig(paths.FIGURES / "c9_before_after.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/adapt/c9.yaml")
    ap.add_argument("--part", default="all", choices=["all", "mlp", "trees"])
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("mlflow", "alembic", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = config.load(args.config)
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else cfg["seeds"]
    tables = []
    for part in (["mlp", "trees"] if args.part == "all" else [args.part]):
        pairs = cfg[part]["pairs"]
        idx = [int(i) for i in args.pairs.split(",")] if args.pairs else range(len(pairs))
        for i in idx:
            for s in seeds:
                h = config.cfg_hash({"experiment": cfg["experiment"], "version": cfg.get("version", 1), "part": part, "model": cfg[part].get("model", "mlp"),
                                     "pair": pairs[i], "adapters": cfg[part]["adapters"], "data": cfg["data"],
                                     "pool_rows": cfg["pool_rows"], "src_rows": cfg["src_rows"]})
                runs = log.find_runs(experiment=cfg["experiment"], finished_only=True, cfg_hash=h, seed=str(s))
                if args.summary_only and runs.empty:
                    continue
                tables.append(run_pair(cfg, part, *pairs[i], s))
    if not tables:
        print("no C9 runs yet")
        return
    long = pd.concat(tables, ignore_index=True)
    long.to_csv(paths.TABLES / "c9_long.csv", index=False)
    summ = summarise(long)
    summ.to_csv(paths.TABLES / "c9_summary.csv", index=False)
    plot(summ)
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_rows", 100):
        print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
