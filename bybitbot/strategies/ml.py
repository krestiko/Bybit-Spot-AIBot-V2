from __future__ import annotations

import numpy as np

from .base import ITradingStrategy


class MLProbabilityStrategy(ITradingStrategy):
    """Strategy that delegates decision to the model's probability."""

    def __init__(self, long_threshold: float, short_threshold: float) -> None:
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.model = None

    def set_model(self, model) -> None:
        self.model = model

    def decide(self, features: np.ndarray, price: float) -> str | None:
        if self.model is None:
            return None
        prob = self.model.predict_proba(features)[0][1]
        return self.decide_with_prob(features, price, prob)

    def decide_with_prob(
        self, features: np.ndarray, price: float, prob: float
    ) -> str | None:
        if prob > self.long_threshold:
            return "long"
        if prob < self.short_threshold:
            return "short"
        return None
