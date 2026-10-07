"""Tent (Wang et al., ICLR 2021): test-time entropy minimisation, training ONLY the BatchNorm affine parameters
(weight, bias) on unlabelled target batches; BN normalises with the target batch statistics during adaptation and
the running statistics follow them. Dropout stays off.

Guard (Build Guide: "stop if the predicted attack rate moves more than 2x from source"): entropy minimisation can
collapse to "everything benign" (or run away to "everything attack"). The predicted attack rate at the frozen
threshold is tracked on every batch against r_src, the model's predicted attack rate on SOURCE validation. Adaptation
stops, keeping the last parameters before the violation, when the rate exceeds guard_factor * r_src, or falls below
r_src / guard_factor while still falling. A model that starts silent on the target (rate ~0, the C6 failure) may
therefore climb back toward r_src; a first version measured the band around that silent rate and so reverted
exactly the useful first step (C9 smoke test, 7 Oct 2026).
"""

import copy

import torch

from xnids.adapt.base import AdaptContext, Adapter
from xnids.adapt.torchutil import batches, bn_layers, predicted_attack_share, require_mlp, tensor
from xnids.models.bundle import Bundle


def guard_violation(rate: float, prev: float, r_src: float, factor: float) -> bool:
    """True when the predicted attack rate runs away (> factor * r_src) or collapses (< r_src / factor while still
    falling). Climbing back from a silent start is allowed."""
    return rate > factor * r_src or (rate < r_src / factor and rate < prev)


def tent_step(net: torch.nn.Module, opt: torch.optim.Optimizer, x: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(net(x)).clamp(1e-6, 1 - 1e-6)
    ent = -(p * p.log() + (1 - p) * (1 - p).log()).mean()
    opt.zero_grad()
    ent.backward()
    opt.step()
    return p.detach()


class TentAdapter(Adapter):
    name = "tent"
    needs_gradients = True

    def adapt(self, bundle: Bundle, ctx: AdaptContext) -> Bundle:
        require_mlp(bundle, self.name)
        p = {"lr": 1e-3, "epochs": 1, "batch_size": 4096, "pool_rows": 200_000, "guard_factor": 2.0,
             "min_rate": 0.005} | self.params
        b = self.clone(bundle, "tent")
        net = b.model.net
        T = tensor(b, ctx.tgt_pool, p["pool_rows"], ctx.seed)
        r_src = max(predicted_attack_share(bundle, tensor(b, ctx.src_val[0], p["pool_rows"], ctx.seed)), p["min_rate"])
        prev = predicted_attack_share(bundle, T)
        for prm in net.parameters():
            prm.requires_grad_(False)
        bn = bn_layers(net)
        affine = [t for m in bn for t in (m.weight, m.bias)]
        for t in affine:
            t.requires_grad_(True)
        opt = torch.optim.Adam(affine, lr=p["lr"])
        net.train()
        for m in net.modules():
            if isinstance(m, torch.nn.Dropout):
                m.eval()
        steps, stopped, good = 0, False, copy.deepcopy(net.state_dict())
        for _ in range(p["epochs"]):
            for xb in batches(T, p["batch_size"], ctx.seed):
                probs = tent_step(net, opt, xb)
                rate = float((probs >= b.threshold).float().mean())
                if guard_violation(rate, prev, r_src, p["guard_factor"]):
                    net.load_state_dict(good)
                    stopped = True
                    break
                good, prev = copy.deepcopy(net.state_dict()), rate
                steps += 1
            if stopped:
                break
        for prm in net.parameters():
            prm.requires_grad_(True)
        net.eval()
        b.meta.update({"threshold_source": "frozen source threshold", "tent_steps": steps, "tent_guard_stop": stopped,
                       "tent_source_attack_rate": r_src})
        return b
