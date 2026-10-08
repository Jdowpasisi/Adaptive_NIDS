"""Few-shot adaptation: buy labels for a small budget of target flows, then adapt with them.

Selection rules (label-free, applied to the target's unlabelled pool):
  random       uniform sample
  uncertainty  flows whose score is closest to the frozen threshold (in logit space)
  drift        flows that look most unlike the source in the 5 most-shifted features (largest |z| w.r.t. source
               signed-log statistics), i.e. what the drift monitor points at
Adaptation:
  MLP          fine-tune the last two layers (last hidden block + head) on the budget, class-weighted BCE
  other models retrain on source train + the budget, the budget repeated so it carries `target_weight` of the total
Threshold: re-picked on the budget when it holds at least `min_attacks` attacks (logged), else the source threshold.
labels_used is recorded; the oracle labels of the pool are read ONLY at the selected indices.
"""

import numpy as np
import polars as pl
import torch
from torch import nn

from xnids.adapt.base import AdaptContext, Adapter
from xnids.drift import ks
from xnids.eval import metrics
from xnids.models import zoo
from xnids.models.bundle import Bundle
from xnids.models.mlp import MLP


def select(bundle: Bundle, pool: pl.DataFrame, n: int, rule: str, seed: int, src_X: pl.DataFrame) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = min(n, pool.height)
    if rule == "random":
        return np.sort(rng.choice(pool.height, n, replace=False))
    if rule == "uncertainty":
        s = np.clip(bundle.score(pool), 1e-7, 1 - 1e-7)
        t = np.clip(bundle.threshold, 1e-7, 1 - 1e-7)
        dist = np.abs(np.log(s / (1 - s)) - np.log(t / (1 - t))) + rng.random(len(s)) * 1e-9
        return np.sort(np.argsort(dist)[:n])
    if rule == "drift":
        Zs, Zp = bundle.model_input(src_X), bundle.model_input(pool)
        names = bundle.preprocess.features_out
        k = ks.ks_features(Zs[rng.choice(len(Zs), min(20000, len(Zs)), replace=False)],
                           Zp[rng.choice(len(Zp), min(20000, len(Zp)), replace=False)], names)
        top = [names.index(f) for f in ks.ranked(k)[:5]]
        ls, lp = (np.sign(z[:, top]) * np.log1p(np.abs(z[:, top])) for z in (Zs, Zp))
        sd = ls.std(0)
        sd[sd == 0] = 1
        out = np.abs((lp - ls.mean(0)) / sd).sum(1) + rng.random(len(lp)) * 1e-9
        return np.sort(np.argsort(-out)[:n])
    raise ValueError(f"unknown selection rule {rule!r}")


class FewShotAdapter(Adapter):
    name = "fewshot"
    needs_labels = True

    def adapt(self, bundle: Bundle, ctx: AdaptContext) -> Bundle:
        p = {"budget": 200, "rule": "random", "epochs": 50, "lr": 1e-3, "target_weight": 0.2, "min_attacks": 5} \
            | self.params
        if ctx.params.get("pool_labels") is None:
            raise ValueError("few-shot needs ctx.params['pool_labels'] (the oracle; read only at selected rows)")
        idx = select(bundle, ctx.tgt_pool, p["budget"], p["rule"], ctx.seed, ctx.src_train[0])
        oracle = ctx.params["pool_labels"]               # an array, or a callable idx -> labels (an analyst)
        Xl, yl = ctx.tgt_pool[idx], np.asarray(oracle(idx) if callable(oracle) else np.asarray(oracle)[idx])
        b = self.clone(bundle, f"fewshot_{p['rule']}_{p['budget']}", labels_used=int(len(idx)),
                       budget_attacks=int(yl.sum()))
        if isinstance(b.model, MLP):
            self._finetune_mlp(b, Xl, yl, p, ctx.seed)
        else:
            self._retrain(b, ctx, Xl, yl, p)
        if yl.sum() >= p["min_attacks"]:
            b.threshold = metrics.threshold_at_dr(yl, b.score(Xl), ctx.target_dr)
            b.meta["threshold_source"] = f"re-picked on the {len(idx)}-flow labelled budget ({int(yl.sum())} attacks)"
        else:
            # the model was fine-tuned / retrained, so the frozen source threshold no longer matches its score scale
            # (C14: a retrained XGBoost alerted on every flow); re-pick on SOURCE validation, as CORAL / DANN do.
            # Before 9 Oct 2026 the frozen threshold was kept here (C10 v2 rows with < min_attacks budget attacks).
            self.rethreshold_on_source_val(b, ctx)
            b.meta["threshold_source"] = "source validation (re-picked after adaptation: too few attacks in budget)"
        return b

    @staticmethod
    def _finetune_mlp(b: Bundle, Xl, yl, p: dict, seed: int) -> None:
        net, dev = b.model.net, b.model.device
        torch.manual_seed(seed)
        T = torch.from_numpy(b.model.scaler.transform(b.model_input(Xl))).to(dev)
        y = torch.as_tensor(yl, dtype=torch.float32, device=dev)
        for prm in net.parameters():
            prm.requires_grad_(False)
        last = list(net.body.children())[-4:]                 # Linear, BatchNorm1d, ReLU, Dropout of the last block
        train = [t for m in last + [net.head] for t in m.parameters()]
        for t in train:
            t.requires_grad_(True)
        pos = (y == 0).sum() / (y == 1).sum().clamp(min=1)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos)
        opt = torch.optim.Adam(train, lr=p["lr"])
        net.eval()                                            # BN uses its running stats: budgets can be tiny
        for _ in range(p["epochs"]):
            loss = loss_fn(net(T), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        for prm in net.parameters():
            prm.requires_grad_(True)
        net.eval()

    @staticmethod
    def _retrain(b: Bundle, ctx: AdaptContext, Xl, yl, p: dict) -> None:
        Xs, ys = ctx.src_train
        Zs, Zl = b.model_input(Xs), b.model_input(Xl)
        rep = max(1, int(round(p["target_weight"] / (1 - p["target_weight"]) * len(Zs) / max(len(Zl), 1))))
        X = np.vstack([Zs, np.repeat(Zl, rep, axis=0)])
        y = np.concatenate([ys, np.repeat(yl, rep)])
        Zv = b.model_input(ctx.src_val[0])
        b.model = zoo.build(b.model.name, b.model.params, ctx.seed, b.model.device).fit(X, y, Zv, ctx.src_val[1])
        b.meta["fewshot_repeat"] = rep
