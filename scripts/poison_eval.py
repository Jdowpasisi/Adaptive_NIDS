"""C17: poisoning the self-updating adaptation loop (H7). Offline, replayed nfs17 flows, 3 seeds.

    python scripts/poison_eval.py [--config configs/poison.yaml] [--seeds 0]

Source TEST is split in two halves by family: half A feeds the clean adaptation pools, half B is the evaluation set
(the target family's DR, the benign FPR, the other attack families' DR). The attacker's material comes from source
TRAIN. Every condition sees the same clean pools and the same poison flows per (seed, round): differences between
guard sets are the guards' doing. Round 0 = before any adaptation.

Outputs: reports/tables/c17_rounds.csv (every round of every condition), c17_summary.csv (round 20, mean +- std over
seeds), reports/figures/c17_<target>_<adapter>.png (target DR vs round: rows = attack, columns = rho, lines = guard set).
"""

import argparse
import copy
import logging
import time

import mlflow
import numpy as np
import polars as pl

from xnids.attack import guards as G
from xnids.attack.poison import ATTACKS
from xnids.eval.harness import load_split
from xnids.models.bundle import bundle_for
from xnids.utils import config, log, paths

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
COLORS = {"none": "#eb6834", "canary": "#2a78d6", "canary_step": "#7fb2ea", "clip": "#eda100", "trimmed": "#9b59b6",
          "gate": "#1baf7a", "rollback": "#898781", "all": INK}


class Data:
    def __init__(self, cfg: dict, seed: int, target: str) -> None:
        m = cfg["model"]
        rng = np.random.default_rng(seed)
        te = load_split(m["source"], m["track"], "test")
        half = np.zeros(len(te.y), bool)
        for fam in np.unique(te.family):                       # stratified 50/50 split of TEST
            idx = np.flatnonzero(te.family == fam)
            half[rng.choice(idx, len(idx) // 2, replace=False)] = True
        self.pool_src = te.X[np.flatnonzero(~half)]
        ev = np.flatnonzero(half)
        fam_ev = te.family[ev]
        ben = ev[fam_ev == "Benign"]
        ben = rng.choice(ben, min(cfg["eval"]["benign"], len(ben)), replace=False)
        self.ev_target = te.X[ev[fam_ev == target]]
        self.ev_benign = te.X[np.sort(ben)]
        self.ev_other = te.X[ev[(fam_ev != "Benign") & (fam_ev != target)]]
        tr = load_split(m["source"], m["track"], "train")
        self.atk_target = tr.X[np.flatnonzero(tr.family == target)]
        b = np.flatnonzero(tr.family == "Benign")
        self.atk_benign = tr.X[np.sort(rng.choice(b, min(50_000, len(b)), replace=False))]
        va = load_split(m["source"], m["track"], "val")
        c = cfg["canary"]
        r2 = np.random.default_rng(c["seed"])
        a, bn = np.flatnonzero(va.y == 1), np.flatnonzero(va.y == 0)
        idx = np.sort(np.concatenate([r2.choice(a, min(c["attacks"], len(a)), replace=False),
                                      r2.choice(bn, min(c["benign"], len(bn)), replace=False)]))
        self.canary = (va.X[idx], va.y[idx])
        self.target = target

    def pool(self, cfg: dict, seed: int, r: int):
        rng = np.random.default_rng([seed, r, 7])
        return self.pool_src[np.sort(rng.choice(self.pool_src.height, cfg["pool_rows"], replace=False))]


def evaluate(b, d: Data) -> dict:
    return {"dr_target": float(b.predict(d.ev_target).mean()), "fpr": float(b.predict(d.ev_benign).mean()),
            "dr_other": float(b.predict(d.ev_other).mean()), "canary_dr": G.canary_dr(b, d.canary)}


def adapt_once(adapter: str, b, pool, cfg: dict, d: Data, seed: int, r: int, trimmed: bool):
    a = cfg["adapters"][adapter]
    if adapter == "adabn":
        return G.continual_adabn(b, pool, a["momentum"], cfg["trim"] if trimmed else 0.0)
    return G.continual_tent(b, pool, d.canary, a, seed * 1000 + r)


def run_condition(cfg, d: Data, base, seed, adapter, attack, rho, gset, radius, gate_cfg) -> list[dict]:
    on = set(G.GUARDS) - {"canary_step"} if gset == "all" else ({gset} - {"none"})
    rounds, n_poison = cfg["rounds"], int(round(rho * cfg["pool_rows"]))
    gen = None
    if attack and n_poison:
        p = cfg["attacks"][attack]
        gen = ATTACKS[attack](d.atk_target, d.atk_benign, rounds=rounds, **p) if attack == "frog" else \
            ATTACKS[attack](d.atk_target, d.atk_benign, **(p or {}))
    cur = copy.deepcopy(base)
    base_canary = G.canary_dr(base, d.canary)
    good = copy.deepcopy(base)                                 # last audited good model (rollback)
    rows = [{"round": 0, **evaluate(cur, d), "accepted": True, "clipped": False, "rolled_back": False,
             "update_norm": 0.0}]
    for r in range(1, rounds + 1):
        pool = d.pool(cfg, seed, r)
        if gen is not None:
            pool = pl.concat([pool, gen.flows(n_poison, r, np.random.default_rng([seed, r, 11]), model=cur)])
        cand = adapt_once(adapter, cur, pool, cfg, d, seed, r, "trimmed" in on)
        clipped, norm = False, float((G.update_vector(adapter, cand.model.net)
                                      - G.update_vector(adapter, cur.model.net)).norm())
        if "clip" in on and radius:
            cand, norm = G.clip_update(adapter, cur, cand, radius)
            clipped = norm > radius
        ok = True
        if "canary" in on and G.canary_dr(cand, d.canary) < base_canary - cfg["canary_drop"]:
            ok = False
        if ok and "canary_step" in on and G.canary_dr(cand, d.canary) < G.canary_dr(cur, d.canary) - cfg["canary_drop"]:
            ok = False
        if ok and "gate" in on and not G.passes_gate(cur, cand, d.canary, pool, gate_cfg):
            ok = False
        if ok:
            cur = cand
        rolled = False
        if "rollback" in on and r % cfg["audit_every"] == 0:
            if G.canary_dr(cur, d.canary) < base_canary - cfg["canary_drop"]:
                cur, rolled = copy.deepcopy(good), True
            else:
                good = copy.deepcopy(cur)
        rows.append({"round": r, **evaluate(cur, d), "accepted": ok, "clipped": clipped, "rolled_back": rolled,
                     "update_norm": norm})
    return [{"seed": seed, "target": d.target, "adapter": adapter, "attack": attack or "clean", "rho": rho,
             "guards": gset, **x} for x in rows]


def plot(df: pl.DataFrame, cfg: dict, target: str, adapter: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    attacks, rhos = list(cfg["attacks"]), cfg["rhos"]
    fig, axes = plt.subplots(len(attacks), len(rhos), figsize=(12, 6.6), sharex=True, sharey=True, squeeze=False,
                             facecolor=SURFACE, layout="constrained")
    df = df.filter(pl.col("target") == target)
    clean = df.filter((pl.col("adapter") == adapter) & (pl.col("attack") == "clean") & (pl.col("guards") == "none"))
    clean = clean.group_by("round").agg(pl.col("dr_target").mean()).sort("round")
    for i, att in enumerate(attacks):
        for j, rho in enumerate(rhos):
            ax = axes[i][j]
            ax.set_facecolor(SURFACE)
            ax.plot(clean["round"], clean["dr_target"] * 100, color=MUTED, lw=1.2, ls=":", label="no poison")
            sub = df.filter((pl.col("adapter") == adapter) & (pl.col("attack") == att) & (pl.col("rho") == rho))
            for gset in cfg["guard_sets"]:
                g = sub.filter(pl.col("guards") == gset)
                if g.is_empty():
                    continue
                s = g.group_by("round").agg(pl.col("dr_target").mean().alias("m"),
                                            pl.col("dr_target").std().fill_null(0).alias("s")).sort("round")
                ax.plot(s["round"], s["m"] * 100, color=COLORS[gset], lw=2.2 if gset in ("none", "all") else 1.3,
                        label=gset, zorder=3 if gset in ("none", "all") else 2)
                ax.fill_between(s["round"], ((s["m"] - s["s"]) * 100).clip(0, 100),
                                ((s["m"] + s["s"]) * 100).clip(0, 100), color=COLORS[gset], alpha=0.12, lw=0)
            ax.set_title(f"{att}, rho = {rho:.0%}", fontsize=9, color=INK, loc="left")
            ax.set_ylim(-3, 103)
            fin = sub.filter(pl.col("round") == sub["round"].max()).group_by("guards").agg(pl.col("dr_target").mean())
            if not fin.is_empty() and fin["dr_target"].max() - fin["dr_target"].min() < 0.01:
                ax.text(0.98, 0.06, "all guard sets overlap", transform=ax.transAxes, ha="right", fontsize=7.5,
                        color=MUTED)
            ax.grid(color=GRID, lw=0.8)
            ax.tick_params(colors=MUTED, labelsize=7.5)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(AXIS)
            if j == 0:
                ax.set_ylabel(f"{target} detection rate (%)", fontsize=8, color=INK2)
            if i == len(attacks) - 1:
                ax.set_xlabel("adaptation round", fontsize=8, color=INK2)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=len(lab), frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle(f"C17: poisoning a self-updating {adapter} loop, target {target} (mean over seeds; band = +-1 std)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.savefig(paths.FIGURES / f"c17_{target.lower()}_{adapter}.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("mlflow", "alembic", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/poison.yaml")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--plot-only", action="store_true", help="redraw the figures from c17_rounds.csv")
    args = ap.parse_args()
    cfg = config.load(args.config)
    if args.plot_only:
        df = pl.read_csv(paths.TABLES / "c17_rounds.csv")
        for target in cfg["targets"]:
            for adapter in cfg["adapters"]:
                plot(df, cfg, target, adapter)
        return
    seeds = args.seeds if args.seeds is not None else cfg["seeds"]
    gate_cfg = config.load(paths.CONFIGS / "live.yaml")["gates"]
    out = paths.TABLES / "c17_rounds.csv"
    rows = []
    with log.start_run(cfg | {"run": {"name": "c17"}}, experiment=cfg["experiment"], run_name="c17-poison") as run:
        for seed, target in [(s_, t_) for s_ in seeds for t_ in cfg["targets"]]:
            m = cfg["model"]
            base = bundle_for(m["name"], m["track"], m["source"], seed)
            d = Data(cfg, seed, target)
            logging.info("seed %d target %s: pool %d, eval target %d / benign %d / other %d, attacker target %d",
                         seed, target, d.pool_src.height, d.ev_target.height, d.ev_benign.height, d.ev_other.height,
                         d.atk_target.height)
            for adapter in cfg["adapters"]:
                t0 = time.time()
                clean = run_condition(cfg, d, base, seed, adapter, None, 0.0, "none", None, gate_cfg)
                radius = cfg["clip_multiplier"] * float(np.median([x["update_norm"] for x in clean[1:]]))
                rows += clean
                rows += run_condition(cfg, d, base, seed, adapter, None, 0.0, "all", radius, gate_cfg)
                for attack in cfg["attacks"]:
                    for rho in cfg["rhos"]:
                        for gset in cfg["guard_sets"]:
                            if gset == "trimmed" and adapter != "adabn":
                                continue
                            rows += run_condition(cfg, d, base, seed, adapter, attack, rho, gset, radius, gate_cfg)
                last = [x for x in rows if x["seed"] == seed and x["target"] == target and x["adapter"] == adapter
                        and x["round"] == cfg["rounds"] and x["guards"] == "none" and x["rho"] == max(cfg["rhos"])]
                logging.info("seed %d %s done in %.0f s (clip radius %.4f); unguarded rho=%.2f final target DR: %s",
                             seed, adapter, time.time() - t0, radius, max(cfg["rhos"]),
                             {x["attack"]: round(x["dr_target"], 3) for x in last})
                pl.DataFrame(rows).write_csv(out)               # checkpoint
        df = pl.DataFrame(rows)
        df.write_csv(out)
        fin = df.filter(pl.col("round") == cfg["rounds"])
        summ = (fin.group_by("target", "adapter", "attack", "rho", "guards")
                .agg(*[pl.col(c).mean().alias(c) for c in ("dr_target", "fpr", "dr_other", "canary_dr")],
                     pl.col("dr_target").std().alias("dr_target_std"), pl.len().alias("seeds"))
                .sort("target", "adapter", "attack", "rho", "guards"))
        acc = df.filter(pl.col("round") > 0).group_by("target", "adapter", "attack", "rho", "guards").agg(
            pl.col("accepted").mean().alias("accept_rate"), pl.col("rolled_back").sum().alias("rollbacks"),
            pl.col("clipped").mean().alias("clip_rate"))
        summ = summ.join(acc, on=["target", "adapter", "attack", "rho", "guards"])
        summ.write_csv(paths.TABLES / "c17_summary.csv")
        for target in cfg["targets"]:
            for adapter in cfg["adapters"]:
                plot(df, cfg, target, adapter)
        mlflow.log_artifact(str(out))
        mlflow.log_artifact(str(paths.TABLES / "c17_summary.csv"))
        logging.info("MLflow run %s", run.info.run_id)
    with pl.Config(tbl_rows=200, tbl_width_chars=200, float_precision=3):
        print(summ)


if __name__ == "__main__":
    main()
