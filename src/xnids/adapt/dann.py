"""DANN (Ganin et al. 2016): a domain classifier on the 64-d penultimate layer, connected through a gradient reversal
layer, so the feature extractor learns to make source and target indistinguishable while the label head keeps
classifying labelled SOURCE flows. GRL coefficient follows the paper's schedule 2 / (1 + exp(-10 p)) - 1 over
training progress p. Fine-tuned from the source MLP; threshold re-picked on source validation."""

import math

import torch
from torch import nn

from xnids.adapt.base import AdaptContext, Adapter
from xnids.adapt.coral import src_tensors
from xnids.adapt.torchutil import batches, require_mlp, tensor
from xnids.models.bundle import Bundle


class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def grl(x: torch.Tensor, lam: float) -> torch.Tensor:
    return _GRL.apply(x, lam)


class DANNAdapter(Adapter):
    name = "dann"
    needs_gradients = True

    def adapt(self, bundle: Bundle, ctx: AdaptContext) -> Bundle:
        require_mlp(bundle, self.name)
        p = {"epochs": 3, "lr": 1e-4, "batch_size": 2048, "src_rows": 300_000, "pool_rows": 200_000,
             "max_lambda": 1.0} | self.params
        b = self.clone(bundle, "dann")
        net = b.model.net
        dev = b.model.device
        torch.manual_seed(ctx.seed)
        head = nn.Sequential(nn.Linear(net.head.in_features, 32), nn.ReLU(), nn.Linear(32, 1)).to(dev)
        Ts, ys = src_tensors(b, ctx, p["src_rows"])
        Tt = tensor(b, ctx.tgt_pool, p["pool_rows"], ctx.seed)
        pos = (ys == 0).sum() / (ys == 1).sum().clamp(min=1)
        bce, dom = nn.BCEWithLogitsLoss(pos_weight=pos), nn.BCEWithLogitsLoss()
        opt = torch.optim.Adam(list(net.parameters()) + list(head.parameters()), lr=p["lr"])
        g = torch.Generator().manual_seed(ctx.seed)
        steps = p["epochs"] * math.ceil(len(Ts) / p["batch_size"])
        k = 0
        net.train()
        for ep in range(p["epochs"]):
            for idx in batches(torch.arange(len(Ts)), p["batch_size"], ctx.seed + ep):
                lam = p["max_lambda"] * (2 / (1 + math.exp(-10 * k / steps)) - 1)
                xs, yb = Ts[idx.to(dev)], ys[idx.to(dev)]
                xt = Tt[torch.randint(len(Tt), (len(idx),), generator=g).to(dev)]
                hs, ht = net.features(xs), net.features(xt)
                d_logits = head(grl(torch.cat([hs, ht]), lam)).squeeze(-1)
                d_true = torch.cat([torch.zeros(len(hs), device=dev), torch.ones(len(ht), device=dev)])
                loss = bce(net.head(hs).squeeze(-1), yb) + dom(d_logits, d_true)
                opt.zero_grad()
                loss.backward()
                opt.step()
                k += 1
        net.eval()
        b.meta["dann_max_lambda"] = p["max_lambda"]
        return self.rethreshold_on_source_val(b, ctx)
