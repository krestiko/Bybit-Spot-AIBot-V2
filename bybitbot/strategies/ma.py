from __future__ import annotations

import numpy as np

from .base import ITradingStrategy


class MaCrossStrategy(ITradingStrategy):
    """Simple moving average crossover strategy."""

    def __init__(self, fast_index: int = 0, slow_index: int = 1) -> None:
        self.fast_index = fast_index
        self.slow_index = slow_index

    def decide(self, features: np.ndarray, price: float) -> str | None:
        try:
            fast_ma = features[self.fast_index]
            slow_ma = features[self.slow_index]
        except IndexError:
            return None
        if fast_ma > slow_ma:
            return "long"
        if fast_ma < slow_ma:
            return "short"
        return None
