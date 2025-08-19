from __future__ import annotations

import numpy as np

from .base import ITradingStrategy


class BreakoutStrategy(ITradingStrategy):
    """Breakout strategy comparing price to recent high/low features."""

    def __init__(self, high_index: int = 0, low_index: int = 1) -> None:
        self.high_index = high_index
        self.low_index = low_index

    def decide(self, features: np.ndarray, price: float) -> str | None:
        try:
            recent_high = features[self.high_index]
            recent_low = features[self.low_index]
        except IndexError:
            return None
        if price > recent_high:
            return "long"
        if price < recent_low:
            return "short"
        return None
