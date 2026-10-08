"""C15: calibrate the drift monitor for TIME-ORDERED real traffic (fixes the C14 finding that the C8 monitor flags
34 of 35 in-distribution demo windows).

    python scripts/calibrate_monitor.py [--config configs/replay.yaml]

Reference: a random sample of RAW (not deduplicated) source flows. The C8 reference came from the deduplicated
split, which lacks the repeated short flows live traffic is full of.
Null: every other raw source flow, in time order, in the monitor's window size. Attack bursts of the source
are part of the null: they are traffic the model was trained on, not drift. The demo replay's segment-A window
(the nfs17 split holdout) is excluded from both.
Thresholds: the `quantile` of each statistic over the null windows. Writes models/monitor/<track>-<source>.json
(reference sample row ids + thresholds) and reports/tables/c15_monitor_{null,demo}.csv (per-window statistics,
for the record and the C15 figure).
"""

import argparse
import datetime as dt
import json

import polars as pl

from xnids.data import schema
from xnids.drift.monitor import DriftMonitor, window_stats
from xnids.live import replay_labels
from xnids.models.bundle import bundle_for
from xnids.utils import config, paths


def raw_source(track: str, source: str, holdout: list[str] | None) -> pl.DataFrame:
    df = pl.read_parquet(paths.PROCESSED / source / f"{track}.parquet")
    if holdout:
        lo, hi = (int(dt.datetime.fromisoformat(t).replace(tzinfo=dt.UTC).timestamp() * 1e6) for t in holdout)
        df = df.filter(~pl.col("ts").is_between(lo, hi, closed="left"))
    return df.sort("ts", maintain_order=True)


def stream_stats(mon: DriftMonitor, X: pl.DataFrame, extra: pl.DataFrame, window: int) -> pl.DataFrame:
    rows = []
    for w0 in range(0, X.height - window // 2, window):
        rep = mon.process(X.slice(w0, window))
        e = extra.slice(w0, window)
        rows.append({"window": w0 // window, "n": rep.n, **window_stats(rep),
                     **{c: (e[c].mode()[0] if e[c].dtype == pl.String else float(e[c].mean())) for c in e.columns}})
    return pl.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/replay.yaml")
    args = ap.parse_args()
    cfg = config.load(args.config)
    m, mc = cfg["model"], cfg["monitor"]
    holdout = config.load(paths.CONFIGS / "split.yaml")["datasets"][m["source"]].get("holdout")
    b = bundle_for(m["name"], m["track"], m["source"], m["seed"])
    feats = schema.features(m["track"])
    raw = raw_source(m["track"], m["source"], holdout)
    mon = DriftMonitor(b, raw.select(feats), raw["y"].to_numpy(), mc)
    ref_rows = raw["row_id"].to_numpy()[mon.ref_idx]
    rest = raw.filter(~pl.col("row_id").is_in(ref_rows))
    # flow-level novelty cutoff: the flow_quantile of nearest-reference distances of held-out source flows
    hold = rest.sample(mc["novelty"]["holdout"], seed=mc["reference"]["seed"] + 1)
    tau = mon.novelty.calibrate_tau(b.preprocess.transform(hold.select(feats)), mc["novelty"]["flow_quantile"])
    null = rest.filter(~pl.col("row_id").is_in(hold["row_id"]))
    window = mc["window"]
    print(f"reference {len(ref_rows)} raw source flows; novelty tau {tau:.4f} from {hold.height} held-out flows; "
          f"null {null.height} flows in time order ({null.height // window} windows)", flush=True)
    ns = stream_stats(mon, null.select(feats), null.select(pl.col("y").cast(pl.Float64).alias("attack_share")),
                      window)
    ns.write_csv(paths.TABLES / "c15_monitor_null.csv")
    q = mc["calibration"]["quantile"]
    thr = {s: float(ns[s].quantile(q)) for s in mc["calibration"]["stats"]}
    out = paths.MODELS / "monitor"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{m['track']}-{m['source']}.json").write_text(json.dumps({
        "model": b.meta["version"], "reference_row_ids": ref_rows.tolist(), "thresholds": thr, "quantile": q,
        "novelty_tau": tau, "novelty_holdout_row_ids": hold["row_id"].to_list(),
        "null_windows": ns.height, "window": window, "holdout_excluded": holdout}, indent=1))
    print("thresholds:", {k: round(v, 4) for k, v in thr.items()}, flush=True)
    # the demo replay, for the record (calibration never sees it)
    from xnids.live.monitoring import build_monitor

    mon2 = build_monitor(cfg)
    demo = replay_labels.attach(pl.read_parquet(paths.REPLAY / "demo_flows.parquet")).sort("timestamp",
                                                                                          maintain_order=True)
    ds = stream_stats(mon2, demo.select(feats), demo.select("segment", pl.col("y").cast(pl.Float64)
                                                            .alias("attack_share")), window)
    ds.write_csv(paths.TABLES / "c15_monitor_demo.csv")
    with pl.Config(tbl_rows=80, float_precision=3, tbl_width_chars=220, tbl_cols=20):
        print(ns.select(list(thr)).describe())
        print(ds.group_by("segment", maintain_order=True).agg(
            *[(pl.col(s) > thr[s]).mean().alias(f">{s}") for s in thr], *[pl.col(s).median() for s in thr]))


if __name__ == "__main__":
    main()
