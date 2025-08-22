from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class ITradingStrategy(ABC):
    """Interface for trading strategies."""

    @abstractmethod
    def decide(self, features: np.ndarray, price: float) -> str | None:
        """Return 'long', 'short' or None based on features and current price."""
        raise NotImplementedError

    def set_model(self, model: Any) -> None:  # pragma: no cover - default no-op
        """Attach an ML model to the strategy if needed."""
        return None

    def decide_with_prob(
        self, features: np.ndarray, price: float, prob: float
    ) -> str | None:
        """Optional decision hook that also receives model probability."""
        return self.decide(features, price)
