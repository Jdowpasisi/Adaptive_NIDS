"""C12 Reptile (Nichol et al. 2018): meta-train the MLP so that a few gradient steps on a SMALL LABELLED SUPPORT SET
from a new target adapt it well.

Correction to the project-guide sketch (Build Guide C12): the inner steps run on the TARGET task's support set, not
on the source task. A task is (dataset, window, labelled support set S, query set Q).

Inner loop: k Adam steps (lr 1e-3, as in the Reptile paper's experiments) on mini-batches of S, after AdaBN on the
task's unlabelled pool, because BatchNorm statistics are not meta-learned (the same happens at deployment).
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
                do_adabn: bool = True) -> nn.Module:
    """AdaBN on the pool, then k Adam steps of class-weighted BCE on the support set. Modifies net in place."""
    if do_adabn and len(task.pool) > 1:
        adabn_(net, task.pool, batch_size=4096)
    pos = (task.ys == 0).sum() / (task.ys == 1).sum().clamp(min=1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
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
            log_every: int = 0, logger=None) -> nn.Module:
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
            inner_adapt(fast, sample_task(rng), k, inner_lr, batch, gen)
            for n, p in fast.named_parameters():
                delta[n] += (p.detach() - base[n]) / meta_batch
        with torch.no_grad():
            for n, p in net.named_parameters():
                if n in params:
                    p += eps * delta[n]
        if log_every and logger and (it + 1) % log_every == 0:
            logger(it + 1, float(sum(d.norm() for d in delta.values())))
    return net
