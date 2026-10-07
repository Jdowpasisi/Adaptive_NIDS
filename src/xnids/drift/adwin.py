"""ADWIN (river) on the stream of model confidence max(p, 1 - p), one update per flow."""

import numpy as np
from river.drift import ADWIN


class ConfidenceADWIN:
    def __init__(self, delta: float = 0.002) -> None:
        self.delta = delta
        self.adwin = ADWIN(delta=delta)
        self.seen = 0

    def update(self, conf: np.ndarray) -> list[int]:
        """Feed one window of confidences; return the stream positions (flow index) where ADWIN cut its window."""
        cuts = []
        for c in conf:
            self.adwin.update(float(c))
            if self.adwin.drift_detected:
                cuts.append(self.seen)
            self.seen += 1
        return cuts


def confidence(scores: np.ndarray) -> np.ndarray:
    return np.maximum(scores, 1 - scores)


def entropy(scores: np.ndarray) -> np.ndarray:
    p = np.clip(scores, 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))
