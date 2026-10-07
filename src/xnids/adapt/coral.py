"""Deep CORAL (Sun & Saenko 2016): fine-tune the source MLP with BCE on labelled SOURCE batches plus
lambda * CORAL(h_s, h_t), the distance between source and target covariances of the 64-d penultimate layer.
Target batches are unlabelled. lambda is fixed in advance (Build Guide: report 0.1, 1 and 10; never pick by target
test). The threshold is re-picked on source validation with the adapted model."""

import torch
from torch import nn

from xnids.adapt.base import AdaptContext, Adapter
from xnids.adapt.torchutil import batches, require_mlp, tensor
from xnids.models.bundle import Bundle


def coral(hs: torch.Tensor, ht: torch.Tensor) -> torch.Tensor:
    d = hs.size(1)
    return ((torch.cov(hs.T) - torch.cov(ht.T)) ** 2).sum() / (4 * d * d)


def src_tensors(b: Bundle, ctx: AdaptContext, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    import numpy as np

    X, y = ctx.src_train
    if X.height > n:
        idx = np.sort(np.random.default_rng(ctx.seed).choice(X.height, n, replace=False))
        X, y = X[idx], y[idx]
    T = torch.from_numpy(b.model.scaler.transform(b.model_input(X))).to(b.model.device)
    return T, torch.as_tensor(y, dtype=torch.float32, device=b.model.device)


class CORALAdapter(Adapter):
    name = "coral"
    needs_gradients = True

    def adapt(self, bundle: Bundle, ctx: AdaptContext) -> Bundle:
        require_mlp(bundle, self.name)
        p = {"lam": 1.0, "epochs": 3, "lr": 1e-4, "batch_size": 2048, "src_rows": 300_000, "pool_rows": 200_000} \
            | self.params
        b = self.clone(bundle, f"coral_l{p['lam']}")
        net = b.model.net
        Ts, ys = src_tensors(b, ctx, p["src_rows"])
        Tt = tensor(b, ctx.tgt_pool, p["pool_rows"], ctx.seed)
        pos = (ys == 0).sum() / (ys == 1).sum().clamp(min=1)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos)
        opt = torch.optim.Adam(net.parameters(), lr=p["lr"])
        g = torch.Generator().manual_seed(ctx.seed)
        net.train()
        for ep in range(p["epochs"]):
            for idx in batches(torch.arange(len(Ts)), p["batch_size"], ctx.seed + ep):
                xs, yb = Ts[idx.to(Ts.device)], ys[idx.to(ys.device)]
                xt = Tt[torch.randint(len(Tt), (len(idx),), generator=g).to(Tt.device)]
                hs, ht = net.features(xs), net.features(xt)
                loss = bce(net.head(hs).squeeze(-1), yb) + p["lam"] * coral(hs, ht)
                opt.zero_grad()
                loss.backward()
                opt.step()
        net.eval()
        b.meta["coral_lambda"] = p["lam"]
        return self.rethreshold_on_source_val(b, ctx)
