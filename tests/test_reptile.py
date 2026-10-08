"""C12: Reptile mechanics (Build Guide: one meta-iteration moves the parameters toward the fast weights)."""

import copy

import numpy as np
import torch

from xnids.adapt.reptile import Task, inner_adapt, reptile
from xnids.models.mlp import MLPNet


def _task(rng: np.random.Generator, shift: float = 0.0) -> Task:
    X = rng.normal(size=(300, 6)).astype(np.float32)
    y = (X[:, 0] + shift > 0).astype(np.float32)
    return Task(torch.from_numpy(X), torch.from_numpy(X[:64]), torch.from_numpy(y[:64]))


def test_one_meta_iteration_moves_toward_fast_weights():
    torch.manual_seed(0)
    net = MLPNet(6, (16, 8), 0.0)
    theta0 = {n: p.detach().clone() for n, p in net.named_parameters()}
    fixed = _task(np.random.default_rng(1))
    # the fast weights reached from theta0 on the same task with the same generator seed
    fast = inner_adapt(copy.deepcopy(net), fixed, k=8, lr=1e-2, batch=64, gen=torch.Generator().manual_seed(0))
    reptile(net, lambda rng: fixed, iters=1, k=8, inner_lr=1e-2, eps0=0.5, meta_batch=1, batch=64, seed=0)
    for n, p in net.named_parameters():
        step, target = p.detach() - theta0[n], dict(fast.named_parameters())[n].detach() - theta0[n]
        if target.norm() > 0:
            # theta1 - theta0 = 0.5 * (fast - theta0): same direction, half the length
            assert torch.allclose(step, 0.5 * target, atol=1e-6), n


def test_inner_adapt_lowers_support_loss():
    torch.manual_seed(0)
    net = MLPNet(6, (16, 8), 0.0)
    t = _task(np.random.default_rng(2))
    lossf = torch.nn.BCEWithLogitsLoss()
    net.eval()
    with torch.no_grad():
        before = lossf(net(t.Xs).squeeze(-1), t.ys).item()
    inner_adapt(net, t, k=50, lr=1e-2, batch=64, gen=torch.Generator().manual_seed(0))
    with torch.no_grad():
        after = lossf(net(t.Xs).squeeze(-1), t.ys).item()
    assert after < before * 0.7


def test_meta_training_helps_few_step_adaptation_to_a_task_family():
    """Tasks differ by a shift of the decision boundary. After meta-training, 4 steps on a new task's support set
    give a lower loss on that task's held-out flows than 4 steps from the untrained initialisation."""
    torch.manual_seed(0)
    init = MLPNet(6, (32, 16), 0.0)
    meta = reptile(copy.deepcopy(init), lambda rng: _task(rng, float(rng.uniform(-1, 1))), iters=150, k=8,
                   inner_lr=1e-2, eps0=0.5, meta_batch=4, batch=64, seed=0)
    rng = np.random.default_rng(99)
    lossf = torch.nn.BCEWithLogitsLoss()
    wins = 0
    for _ in range(10):
        shift = float(rng.uniform(-1, 1))
        t = _task(rng, shift)
        q_y = (t.pool[64:, 0] + shift > 0).float()                # the task's own labels, on unseen flows
        res = []
        for start in (init, meta):
            f = inner_adapt(copy.deepcopy(start), t, k=4, lr=1e-2, batch=64, gen=torch.Generator().manual_seed(0),
                            do_adabn=False)
            with torch.no_grad():
                res.append(lossf(f(t.pool[64:]).squeeze(-1), q_y).item())
        wins += res[1] < res[0]
    assert wins >= 7
