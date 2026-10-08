"""C17 self-updating adapters (continual AdaBN, continual Tent) and the safeguards (Build Guide C17).

One ROUND: candidate = adapter(current model, this round's pool); each guard may then shrink (clip) or refuse
(keep the current model) the update; a periodic audit may roll back. Guards:
  canary    refuse if canary DR < baseline canary DR - 5 points. Anchored to the ORIGINAL model; the per-step
            variant (vs the current model) is 'canary_step', for the frog-boiling question
  clip      shrink the update so its L2 norm (BN affine params for Tent, BN running stats for AdaBN) <= radius
  trimmed   AdaBN statistics from the 1%-trimmed mean / variance per channel (AdaBN only)
  gate      the C14 promotion checks vs the current model (canary DR, canary FPR, attack rate): the human reviewer's
            evidence, applied automatically here
  rollback  every `audit_every` rounds: canary DR < baseline - 5 points -> restore the last audited good model
"""

import copy

import numpy as np
import torch

from xnids.adapt.base import AdaptContext
from xnids.adapt.tent import TentAdapter
from xnids.adapt.torchutil import bn_layers, tensor
from xnids.live.registry import gate_checks
from xnids.models.bundle import Bundle

GUARDS = ("canary", "canary_step", "clip", "trimmed", "gate", "rollback")


# ---------------------------------------------------------------- continual adapters
def _bn_state(net) -> torch.Tensor:
    return torch.cat([torch.cat([m.running_mean, m.running_var]) for m in bn_layers(net)]).detach().clone()


def _affine(net) -> torch.Tensor:
    return torch.cat([torch.cat([m.weight, m.bias]) for m in bn_layers(net)]).detach().clone()


@torch.no_grad()
def pool_bn_stats(net, T: torch.Tensor, trim: float = 0.0) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Per BN layer: the (optionally trimmed) mean / variance of its inputs on the pool, layer by layer (layer i is
    measured with layers < i already set to their pool statistics, as AdaBN does)."""
    net.eval()
    layers = bn_layers(net)
    saved = [(m.running_mean.clone(), m.running_var.clone()) for m in layers]
    out = []
    for m in layers:
        cap = []
        h = m.register_forward_pre_hook(lambda _m, inp, cap=cap: cap.append(inp[0].detach()))
        for i in range(0, len(T), 8192):
            net(T[i:i + 8192])
        h.remove()
        x = torch.cat(cap)
        if trim:
            lo, hi = torch.quantile(x, trim, dim=0), torch.quantile(x, 1 - trim, dim=0)
            keep = (x >= lo) & (x <= hi)
            cnt = keep.sum(0).clamp(min=1)
            mean = (x * keep).sum(0) / cnt
            var = (((x - mean) ** 2) * keep).sum(0) / cnt
        else:
            mean, var = x.mean(0), x.var(0, unbiased=False)
        m.running_mean.copy_(mean)
        m.running_var.copy_(var)
        out.append((mean, var))
    for m, (mu, v) in zip(layers, saved, strict=True):
        m.running_mean.copy_(mu)
        m.running_var.copy_(v)
    return out


def continual_adabn(b: Bundle, pool, momentum: float, trim: float = 0.0) -> Bundle:
    nb = copy.deepcopy(b)
    net = nb.model.net
    T = tensor(nb, pool)
    with torch.no_grad():
        for m, (mu, v) in zip(bn_layers(net), pool_bn_stats(net, T, trim), strict=True):
            m.running_mean.mul_(1 - momentum).add_(momentum * mu)
            m.running_var.mul_(1 - momentum).add_(momentum * v)
    return nb


def continual_tent(b: Bundle, pool, src_val, params: dict, seed: int) -> Bundle:
    ctx = AdaptContext(src_val, src_val, pool, seed=seed)
    nb = TentAdapter(**params).adapt(b, ctx)
    nb.meta = b.meta                                   # keep the version chain short over many rounds
    return nb


# ---------------------------------------------------------------- guards
def update_vector(adapter: str, net) -> torch.Tensor:
    return _bn_state(net) if adapter == "adabn" else _affine(net)


def clip_update(adapter: str, old: Bundle, new: Bundle, radius: float) -> tuple[Bundle, float]:
    """Project the update onto the L2 ball of `radius` around the current model. Returns (bundle, raw norm)."""
    a, b = update_vector(adapter, old.model.net), update_vector(adapter, new.model.net)
    norm = float((b - a).norm())
    if norm <= radius or norm == 0:
        return new, norm
    f = radius / norm
    with torch.no_grad():
        for mo, mn in zip(bn_layers(old.model.net), bn_layers(new.model.net), strict=True):
            names = ("running_mean", "running_var") if adapter == "adabn" else ("weight", "bias")
            for n in names:
                to, tn = getattr(mo, n), getattr(mn, n)
                tn.copy_(to + f * (tn - to))
    return new, norm


def canary_dr(b: Bundle, canary) -> float:
    X, y = canary
    return float(np.mean(b.predict(X[np.flatnonzero(y == 1)])))


def passes_gate(current: Bundle, cand: Bundle, canary, recent, cfg: dict) -> bool:
    g = gate_checks(current, cand, canary, recent, cfg)
    return bool(g["all_passed"])
