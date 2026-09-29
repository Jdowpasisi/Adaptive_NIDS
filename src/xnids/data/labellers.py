"""Per-file labelling for datasets that ship unlabelled.

LycoS-IDS2017 publishes LycoSTand CSVs without labels plus a pandas script (labelling.py in
https://maupiti-git.univ-lemans.fr/lycos/lycos-ids2017) that assigns them by attacker/victim address and
attack time window. `lycos17` is a line-for-line port of that script to polars expressions so it runs lazily
on the full files. Rules are applied in the original order, so a later rule overwrites an earlier one exactly
as the original `.loc` assignments did. Timestamps are epoch microseconds.
"""

from collections.abc import Callable

import polars as pl

ATTACKER, VICTIM = "172.16.0.1", "192.168.10.50"


def _t(lo: int, hi: int) -> pl.Expr:
    return pl.col("timestamp").is_between(lo, hi)


def _pair(src: str = ATTACKER, dst: str = VICTIM) -> pl.Expr:
    return (pl.col("src_addr") == src) & (pl.col("dst_addr") == dst)


_TCP = pl.col("ip_prot") == 6

# day -> (rows to drop, ordered [(condition, label)])
_LYCOS17_RULES: dict[str, tuple[list[pl.Expr], list[tuple[pl.Expr, str]]]] = {
    "Monday": ([], []),
    "Tuesday": ([], [
        (_pair() & _t(1499170620000000, 1499175000000000), "ftp_patator"),
        (_pair() & _t(1499188140000000, 1499191860000000), "ssh_patator"),
    ]),
    "Wednesday": ([], [
        (_pair(dst="192.168.10.51") & (pl.col("src_port") == 45022) & (pl.col("dst_port") == 444), "heartbleed"),
        (_pair() & (pl.col("dst_port") == 80)
         & (_t(1499256060000000, 1499260260000000) | _t(1499275440000000, 1499275500000000)), "dos_slowloris"),
        (_pair() & (pl.col("dst_port") == 80) & _t(1499260500000000, 1499261820000000), "dos_slowhttptest"),
        (_pair() & (pl.col("dst_port") == 80) & _t(1499262180000000, 1499263620000000), "dos_hulk"),
        (_pair() & (pl.col("dst_port") == 80) & _t(1499263800000000, 1499264340000000), "dos_goldeneye"),
    ]),
    "Thursday": (
        # the original drops all of Thursday afternoon (infiltration, not reliably labelled)
        [pl.col("timestamp") >= 1499353200000000],
        [
            (_pair() & _TCP & _t(1499343300000000, 1499346000000000), "webattack_bruteforce"),
            (_pair() & _TCP & _t(1499346935000000, 1499348100000000), "webattack_xss"),
            (_pair() & _TCP & _t(1499348400000000, 1499348576000000), "webattack_sql_injection"),
        ],
    ),
    "Friday": (
        [
            # portscan traffic to the bot C&C address after the bot window was wrongly labelled upstream
            (pl.col("dst_addr") == "205.174.165.73") & (pl.col("timestamp") > 1499436193000000),
            # labelled Bot in ISCX but not listed on the CIC site; the authors drop them
            pl.col("dst_addr").is_in(["52.6.13.28", "52.7.235.158"]),
        ],
        [
            ((pl.col("dst_addr") == "205.174.165.73") & _t(1499430840000000, 1499436122000000), "bot"),
            (_pair() & _TCP & _t(1499453791000000, 1499454973000000), "ddos"),
            (_pair() & _TCP & _t(1499443530000000, 1499451842000000), "portscan"),
        ],
    ),
}


def lycos17(lf: pl.LazyFrame, source_file: str, label_col: str = "label") -> pl.LazyFrame:
    day = next((d for d in _LYCOS17_RULES if source_file.startswith(d) or f"/{d}-" in source_file), None)
    if day is None:
        raise ValueError(f"cannot infer weekday from LycoS17 file name: {source_file}")
    drops, rules = _LYCOS17_RULES[day]
    for cond in drops:
        # fill_null(False) keeps rows whose condition can't be evaluated, matching pandas .loc semantics
        lf = lf.filter(~cond.fill_null(False))
    label = pl.lit("benign")
    for cond, name in rules:
        label = pl.when(cond.fill_null(False)).then(pl.lit(name)).otherwise(label)
    return lf.with_columns(label.alias(label_col))


LABELLERS: dict[str, Callable[..., pl.LazyFrame]] = {"lycos17": lycos17}
