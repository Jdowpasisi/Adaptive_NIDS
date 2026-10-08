"""C12 Reptile (Nichol et al. 2018): meta-train the MLP so that a few gradient steps on a SMALL LABELLED SUPPORT SET
from a new target adapt it well.

Correction to the project-guide sketch (Build Guide C12): the inner steps run on the TARGET task's support set, not
on the source task. A task is (dataset, window, labelled support set S, query set Q).

Inner loop: k optimiser steps (plain SGD, as in the Build Guide sketch, or Adam) on mini-batches of S, after AdaBN on
the task's unlabelled pool, because BatchNorm statistics are not meta-learned (the same happens at deployment).
Meta-training starts from a good pretrained theta_0, so the outer step must stay small: Adam's fixed-size inner
steps with eps0 0.1 over 3,000 iterations moved theta further from theta_0 than theta_0's own norm and destroyed
the network (first full C12 run). The setting is now selected per fold by meta-validation (scripts/reptile_eval.py).
Outer loop: theta <- theta + eps * mean_task(theta_task - theta), eps decayed linearly; BN running statistics of
theta are left as pretrained (only parameters are meta-updated).
"""

import copy
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from xnids.adapt.adabn import adabn_


@dataclass
class Task:
    pool: torch.Tensor          # unlabelled window traffic (network input space), for AdaBN
    Xs: torch.Tensor            # labelled support set
    ys: torch.Tensor


def inner_adapt(net: nn.Module, task: Task, k: int, lr: float, batch: int, gen: torch.Generator,
                do_adabn: bool = True, opt_name: str = "adam") -> nn.Module:
    """AdaBN on the pool, then k optimiser steps (Adam or plain SGD) of class-weighted BCE on the support set.
    Modifies net in place."""
    if do_adabn and len(task.pool) > 1:
        adabn_(net, task.pool, batch_size=4096)
    pos = (task.ys == 0).sum() / (task.ys == 1).sum().clamp(min=1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos)
    opt = torch.optim.Adam(net.parameters(), lr=lr) if opt_name == "adam" else torch.optim.SGD(net.parameters(), lr=lr)
    net.eval()                                    # BN uses the (AdaBN) running stats; support batches can be tiny
    n = len(task.Xs)
    for _ in range(k):
        idx = torch.randint(n, (min(batch, n),), generator=gen).to(task.Xs.device)
        loss = loss_fn(net(task.Xs[idx]).squeeze(-1), task.ys[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


def reptile(net: nn.Module, sample_task: Callable[[np.random.Generator], Task], iters: int = 3000, k: int = 8,
            inner_lr: float = 1e-3, eps0: float = 0.1, meta_batch: int = 4, batch: int = 64, seed: int = 0,
            log_every: int = 0, logger=None, inner_opt: str = "adam") -> nn.Module:
    """In place: theta <- theta + eps * mean(theta_task - theta) over meta_batch tasks per iteration."""
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    params = [n for n, _ in net.named_parameters()]
    for it in range(iters):
        eps = eps0 * (1 - it / iters)
        base = {n: p.detach().clone() for n, p in net.named_parameters()}
        delta = {n: torch.zeros_like(p) for n, p in base.items()}
        for _ in range(meta_batch):
            fast = copy.deepcopy(net)
            inner_adapt(fast, sample_task(rng), k, inner_lr, batch, gen, opt_name=inner_opt)
            for n, p in fast.named_parameters():
                delta[n] += (p.detach() - base[n]) / meta_batch
        with torch.no_grad():
            for n, p in net.named_parameters():
                if n in params:
                    p += eps * delta[n]
        if log_every and logger and (it + 1) % log_every == 0:
            logger(it + 1, float(sum(d.norm() for d in delta.values())))
    return net


# ---------------------------------------------------------------- C10 / C11 / C14 action

def load_meta_bundle(track: str, held_out: str, seed: int, device: str = "auto"):
    """The Reptile meta-model trained WITHOUT `held_out` (models/reptile/<track>/<held_out>-s<seed>/) as a Bundle."""
    import json

    from xnids.models import zoo
    from xnids.models.bundle import Bundle
    from xnids.models.mlp import InputScaler, MLPNet
    from xnids.models.preprocess import Preprocessor
    from xnids.utils import paths

    d = paths.MODELS / "reptile" / track / f"{held_out}-s{seed}"
    meta = json.loads((d / "meta.json").read_text())
    m = zoo.build("mlp", {}, seed=seed, device=device)
    m.scaler = InputScaler.from_dict(json.loads((d / "scaler.json").read_text()))
    m.net = MLPNet(meta["d_in"]).to(m.device)
    m.net.load_state_dict(torch.load(d / "meta.pt", map_location=m.device))
    m.net.eval()
    pre = Preprocessor.load(d / "preprocess.json")
    return Bundle(m, pre, float(meta["thr_meta"]), pre.features_in,
                  {"version": f"reptile-{track}-{held_out}-s{seed}", **meta})


class ReptileAdapter:
    """Switch to the meta-model trained without this target, AdaBN on the unlabelled pool, k Adam steps on `budget`
    randomly bought labels, threshold re-picked on them when they hold >= min_attacks attacks (as few-shot)."""
    name = "reptile"
    needs_labels = True

    def __init__(self, budget: int = 200, track: str = "nf43", held_out: str = "", seed: int = 0, k: int = 8,
                 batch: int = 64, min_attacks: int = 5) -> None:
        self.p = dict(budget=budget, track=track, held_out=held_out, seed=seed, k=k, batch=batch,
                      min_attacks=min_attacks)

    def adapt(self, _deployed, ctx):
        from xnids.eval import metrics

        p = self.p
        b = load_meta_bundle(p["track"], p["held_out"], p["seed"])
        m = b.model
        to = lambda X: torch.from_numpy(m.scaler.transform(b.model_input(X))).to(m.device)  # noqa: E731
        rng = np.random.default_rng(ctx.seed)
        idx = np.sort(rng.choice(ctx.tgt_pool.height, min(p["budget"], ctx.tgt_pool.height), replace=False))
        oracle = ctx.params["pool_labels"]
        yl = np.asarray(oracle(idx) if callable(oracle) else np.asarray(oracle)[idx])
        task = Task(to(ctx.tgt_pool), to(ctx.tgt_pool[idx]), torch.as_tensor(yl, dtype=torch.float32, device=m.device))
        inner_adapt(m.net, task, p["k"], b.meta["inner_lr"], p["batch"], torch.Generator().manual_seed(ctx.seed),
                    opt_name=b.meta["inner_opt"])                       # the inner settings selected in C12
        b.meta.update({"adapter": f"reptile_{p['budget']}", "labels_used": int(len(idx)),
                       "budget_attacks": int(yl.sum())})
        if yl.sum() >= p["min_attacks"]:
            b.threshold = metrics.threshold_at_dr(yl, b.score(ctx.tgt_pool[idx]), 0.95)
            b.meta["threshold_source"] = f"re-picked on the {len(idx)}-flow labelled budget"
        else:
            b.meta["threshold_source"] = "meta-model's pooled-validation threshold (too few attacks in the budget)"
        return b
