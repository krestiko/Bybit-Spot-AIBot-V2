from __future__ import annotations

import time


class RiskManager:
    """Basic risk management handling daily loss/profit limits."""

    def __init__(
        self,
        daily_loss_limit: float = 0.0,
        daily_profit_limit: float = 0.0,
    ) -> None:
        self.daily_loss_limit = daily_loss_limit
        self.daily_profit_limit = daily_profit_limit
        self.daily_pnl = 0.0
        self.daily_date = time.strftime("%Y-%m-%d")

    def _reset_if_new_day(self) -> None:
        today = time.strftime("%Y-%m-%d")
        if today != self.daily_date:
            self.daily_date = today
            self.daily_pnl = 0.0

    def reset_if_new_day(self) -> None:
        """Public wrapper to reset PnL counters when day changes."""
        self._reset_if_new_day()

    def can_trade(self) -> bool:
        """Check whether trading is allowed based on daily limits."""
        self._reset_if_new_day()
        if self.daily_loss_limit and self.daily_pnl <= -self.daily_loss_limit:
            return False
        if self.daily_profit_limit and self.daily_pnl >= self.daily_profit_limit:
            return False
        return True

    def update_pnl(self, pnl: float) -> None:
        self._reset_if_new_day()
        self.daily_pnl += pnl
