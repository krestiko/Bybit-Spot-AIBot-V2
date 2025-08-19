from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class ITradingStrategy(ABC):
    """Interface for trading strategies."""

    @abstractmethod
    def decide(self, features: np.ndarray, price: float) -> str | None:
        """Return 'long', 'short' or None based on features and current price."""
        raise NotImplementedError
