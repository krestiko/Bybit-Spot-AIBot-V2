from __future__ import annotations

import numpy as np

from .base import ITradingStrategy


class RSIStrategy(ITradingStrategy):
    """RSI threshold strategy."""

    def __init__(self, rsi_index: int = 0, low: float = 30, high: float = 70) -> None:
        self.rsi_index = rsi_index
        self.low = low
        self.high = high

    def decide(self, features: np.ndarray, price: float) -> str | None:
        try:
            rsi = features[self.rsi_index]
        except IndexError:
            return None
        if rsi < self.low:
            return "long"
        if rsi > self.high:
            return "short"
        return None
