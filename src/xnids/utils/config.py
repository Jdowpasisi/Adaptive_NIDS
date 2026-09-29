"""YAML config loading and hashing.

A config may name a parent with `_base: relative/or/absolute.yaml`; the child is deep-merged
on top of it. The hash is computed on the fully resolved dict, so two configs that resolve to
the same settings share a hash regardless of how they were written.
"""

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open() as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base", None)
    if base is not None:
        cfg = deep_merge(load(path.parent / base), cfg)
    return cfg


def cfg_hash(cfg: dict) -> str:
    """sha1 of the canonical JSON form, first 10 hex chars (Build Guide rule 2)."""
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]


def flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    """{'a': {'b': 1}} -> {'a.b': 1}; lists are JSON-encoded so they fit in MLflow params."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key))
        elif isinstance(v, list | tuple):
            out[key] = json.dumps(v, default=str)
        else:
            out[key] = v
    return out
