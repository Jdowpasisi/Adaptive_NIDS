"""C11: evaluate the adapter selector leave-one-pair-out on the C10 logs and fit the deployable selector.

    python scripts/selector_eval.py

For each held-out pair the selector is trained on the other pairs' windows and, on every held-out window, picks one
action (or waits). Compared with: always-Tent, the best single fixed action (chosen on the TRAINING pairs), a
random action, and the oracle (best action per window, known only in hindsight).
Metrics on held-out pairs: mean regret (oracle utility - chosen utility), mean realised utility, top-1 agreement
with the oracle, Spearman correlation of predicted vs actual gain.

Outputs: reports/tables/c11_lopo.csv (per held-out pair x method), c11_summary.csv (per utility/label cost x method),
reports/figures/c11_regret.png; models/selector/ (deployable selector, trained on all pairs, MCC utility); MLflow c11.
"""

import json
import logging
import pickle

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from xnids.select.selector import DRIFT_COLS, Selector, decision_table
from xnids.utils import config, log, paths

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
CFG = {"experiment": "c11", "deployed": "mlp", "utilities": ["mcc", "fpr"], "label_costs": [0.0, 1e-4, 2e-4, 5e-4],
       "headline": {"utility": "mcc", "label_cost": 2e-4}, "selector": {"n_models": 5, "margin": 0.005, "seed": 0},
       "tent_action": "tent", "seeds": [0, 1, 2]}


def lopo(table: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    for pair in sorted(table.pair.unique()):
        tr, te = table[table.pair != pair], table[table.pair == pair]
        sel = Selector(DRIFT_COLS, **cfg["selector"]).fit(tr)
        chosen = sel.choose_table(te)
        g = te.pivot_table(index="scenario_id", columns="action", values="gain")
        oracle_a, oracle_u = g.idxmax(axis=1), g.max(axis=1)
        fixed = tr.groupby("action").gain.mean().idxmax()          # best single action on the TRAINING pairs
        rng = np.random.default_rng(0)
        rand = pd.Series(rng.choice(g.columns, len(g)), index=g.index)
        pred = te[te.action != "wait"].assign(pred=sel.predict(te[te.action != "wait"])[0])
        rho = spearmanr(pred.pred, pred.gain).statistic
        methods = {"selector": chosen.action, "always-tent": pd.Series(cfg["tent_action"], index=g.index),
                   f"best-fixed ({fixed})": pd.Series(fixed, index=g.index), "random": rand, "oracle": oracle_a}
        for name, acts in methods.items():
            acts = acts.reindex(g.index)
            u = np.array([g.loc[s, a] for s, a in acts.items()])
            rows.append({"pair": pair, "method": name.split(" (")[0], "fixed_action": fixed if "fixed" in name else "",
                         "windows": len(g), "utility": u.mean(), "regret": (oracle_u.to_numpy() - u).mean(),
                         "top1": float((acts == oracle_a).mean()),
                         "wait_share": float((acts == "wait").mean()),
                         "spearman": rho if name == "selector" else np.nan})
    return pd.DataFrame(rows)


def plot(res: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["selector", "best-fixed", "always-tent", "random", "oracle"]
    cols = {"selector": "#2a78d6", "best-fixed": "#eb6834", "always-tent": "#1baf7a", "random": "#eda100",
            "oracle": MUTED}
    pairs = list(dict.fromkeys(res.pair))
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=SURFACE, layout="constrained")
    ax.set_facecolor(SURFACE)
    x = np.arange(len(pairs))
    w = 0.16
    for k, m in enumerate(order):
        g = res[res.method == m].set_index("pair").reindex(pairs)
        ax.bar(x + (k - 2) * (w + 0.01), g.regret, width=w, color=cols[m], label=m, edgecolor=SURFACE, linewidth=1,
               zorder=2)
    ax.set_xticks(x, [p.replace("->", " →\n") for p in pairs], fontsize=7, color=INK2)
    ax.set_ylabel("mean regret vs oracle (MCC units)\nlower is better", fontsize=8, color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=7.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=5, frameon=False, fontsize=8,
               labelcolor=INK2)
    ax.set_title("C11: leave-one-pair-out regret of the adapter selector (utility = MCC gain - 2e-4 x labels)",
                 fontsize=9.5, color=INK, loc="left")
    fig.savefig(paths.FIGURES / "c11_regret.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("mlflow", "alembic", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logdf = pd.read_parquet(paths.LOGS / "adapt_log.parquet")
    cfg = CFG
    all_res, summ = [], []
    with log.start_run(cfg | {"run": {"name": "c11"}, "log_rows": len(logdf)}, experiment=cfg["experiment"],
                       run_name="c11-lopo") as run:
        for util in cfg["utilities"]:
            for lc in cfg["label_costs"]:
                if util == "fpr" and lc not in (0.0, cfg["headline"]["label_cost"]):
                    continue
                t = decision_table(logdf, cfg["deployed"], util, lc)
                res = lopo(t, cfg).assign(utility_metric=util, label_cost=lc)
                all_res.append(res)
                s = res.groupby("method").agg(utility=("utility", "mean"), regret=("regret", "mean"),
                                              top1=("top1", "mean"), wait_share=("wait_share", "mean"),
                                              spearman=("spearman", "mean")).reset_index()
                summ.append(s.assign(utility_metric=util, label_cost=lc))
                logging.info("utility=%s label_cost=%g: %s", util, lc,
                             s.set_index("method").regret.round(4).to_dict())
        res, summ = pd.concat(all_res), pd.concat(summ)
        res.to_csv(paths.TABLES / "c11_lopo.csv", index=False)
        summ.to_csv(paths.TABLES / "c11_summary.csv", index=False)
        h = cfg["headline"]
        head = res[(res.utility_metric == h["utility"]) & (res.label_cost == h["label_cost"])]
        plot(head)
        for _, r in summ[(summ.utility_metric == h["utility"]) & (summ.label_cost == h["label_cost"])].iterrows():
            mlflow.log_metric(f"regret/{r.method}", r.regret)
        # the deployable selector: all pairs, headline utility
        t = decision_table(logdf, cfg["deployed"], h["utility"], h["label_cost"])
        sel = Selector(DRIFT_COLS, **cfg["selector"]).fit(t)
        out = paths.MODELS / "selector"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "selector.pkl", "wb") as f:
            pickle.dump(sel, f)
        (out / "meta.json").write_text(json.dumps({"run_id": run.info.run_id, "cfg_hash": config.cfg_hash(cfg),
                                                   "actions": sel.actions_, "drift_cols": DRIFT_COLS,
                                                   "deployed": cfg["deployed"], **h}, indent=1))
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(summ.round(4).to_string(index=False))
        print("\nheadline, per held-out pair:")
        print(head.pivot_table(index="pair", columns="method", values="regret").round(4).to_string())


if __name__ == "__main__":
    main()
