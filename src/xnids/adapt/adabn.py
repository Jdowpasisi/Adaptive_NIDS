"""AdaBN (Li et al. 2016): replace the BatchNorm running statistics with ones estimated on target traffic.
No gradients, no labels; every weight stays as trained on the source. The frozen threshold is kept."""

import torch

from xnids.adapt.base import AdaptContext, Adapter
from xnids.adapt.torchutil import batches, bn_layers, require_mlp, tensor
from xnids.models.bundle import Bundle


@torch.no_grad()
def adabn_(net: torch.nn.Module, T: torch.Tensor, batch_size: int = 4096, seed: int = 0) -> None:
    """In place: cumulative-average BN statistics over the target batches (momentum None), other layers in eval."""
    net.eval()
    for m in bn_layers(net):
        m.reset_running_stats()
        m.momentum = None
        m.train()
    for xb in batches(T, batch_size, seed):
        net(xb)
    for m in bn_layers(net):
        m.momentum = 0.1
    net.eval()


class AdaBNAdapter(Adapter):
    name = "adabn"
    needs_gradients = True

    def adapt(self, bundle: Bundle, ctx: AdaptContext) -> Bundle:
        require_mlp(bundle, self.name)
        b = self.clone(bundle, "adabn")
        adabn_(b.model.net, tensor(b, ctx.tgt_pool, self.params.get("pool_rows", 200_000), ctx.seed),
               self.params.get("batch_size", 4096), ctx.seed)
        b.meta["threshold_source"] = "frozen source threshold"
        return b
