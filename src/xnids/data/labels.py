"""Raw label strings -> 8 attack families -> binary y (benign = 0).

The map lives in configs/labels.yaml and is keyed by a normalised form of the raw string (lowercase,
every run of non-alphanumerics -> one space), so 'Web Attack – XSS', 'Web Attack � XSS' and
'webattack_xss' variants only need to be listed once each. Any label that is not in the map raises:
nothing silently becomes 'Other'.
"""

import re
from functools import cache
from pathlib import Path

import polars as pl

from xnids.utils import config, paths

FAMILIES = ("Benign", "DoS", "DDoS", "Recon", "BruteForce", "WebAttack", "Bot/Backdoor", "Other")
BENIGN = "Benign"


class UnmappedLabelError(KeyError):
    pass


def norm(label: str) -> str:
    return re.sub(r"[^0-9a-z]+", " ", str(label).lower()).strip()


@cache
def load_map(path: str | Path = paths.CONFIGS / "labels.yaml") -> dict[str, str]:
    """{normalised raw label: family}; validates that every family is one of FAMILIES."""
    raw = config.load(path)["families"]
    out: dict[str, str] = {}
    for fam, labels in raw.items():
        if fam not in FAMILIES:
            raise ValueError(f"labels.yaml: unknown family {fam!r}")
        for lab in labels:
            k = norm(lab)
            if k in out and out[k] != fam:
                raise ValueError(f"labels.yaml: {lab!r} mapped to both {out[k]} and {fam}")
            out[k] = fam
    return out


def check_mapped(raw_labels: list[str], mapping: dict[str, str] | None = None) -> None:
    mapping = mapping or load_map()
    missing = sorted({lab for lab in raw_labels if norm(lab) not in mapping})
    if missing:
        raise UnmappedLabelError(f"labels not in configs/labels.yaml: {missing}")


def to_family(label: pl.Expr, mapping: dict[str, str] | None = None) -> pl.Expr:
    """Polars expression: raw label -> family. Call check_mapped first for a readable error."""
    mapping = mapping or load_map()
    key = label.cast(pl.String).str.to_lowercase().str.replace_all(r"[^0-9a-z]+", " ").str.strip_chars()
    return key.replace_strict(mapping, return_dtype=pl.String)


def to_binary(family: pl.Expr) -> pl.Expr:
    return (family != BENIGN).cast(pl.Int8)
