"""Label-free cost trigger: is the drift worth acting on?

Average Thresholded Confidence (ATC; Garg et al., ICLR 2022): on SOURCE validation, pick tau so that the share of
flows with confidence below tau equals the model's error rate there (at the frozen detection threshold). On a new
window, the share with confidence below tau estimates the error rate without labels. The estimated rise in error,
times the flow rate, gives expected extra wrong decisions per hour, which are compared with the cost of adapting.

Known limitation: a model that becomes confidently wrong on a new network (the "silent" failure seen in C6) keeps its
confidence high, so ATC underestimates the error there. The H5 evaluation reports how often that happens.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class CostConfig:
    flows_per_hour: float = 100_000.0       # expected traffic volume
    cost_per_wrong_decision: float = 1.0    # analyst cost units per extra false alarm / missed attack
    horizon_hours: float = 24.0             # how long an unadapted model would keep running
    adaptation_cost: float = 2_000.0        # labels + compute + review, in the same units


class ATC:
    def fit(self, conf_val: np.ndarray, correct_val: np.ndarray) -> "ATC":
        self.source_err = float(1 - np.mean(correct_val))
        self.tau = float(np.quantile(conf_val, self.source_err)) if self.source_err > 0 else float(conf_val.min())
        return self

    def estimate_error(self, conf: np.ndarray) -> float:
        return float(np.mean(conf < self.tau))

    def error_rise(self, conf: np.ndarray) -> float:
        return self.estimate_error(conf) - self.source_err


def worth_acting(err_rise: float, cost: CostConfig) -> tuple[bool, float]:
    """(recommend?, expected extra wrong decisions per hour)."""
    extra_per_hour = max(err_rise, 0.0) * cost.flows_per_hour
    return extra_per_hour * cost.cost_per_wrong_decision * cost.horizon_hours > cost.adaptation_cost, extra_per_hour
