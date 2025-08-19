import pytest

from bybitbot import TradingBot
import asyncio


def test_daily_loss_limit_stops_trading(monkeypatch):
    bot = TradingBot()
    bot.daily_loss_limit = 100
    bot.update_pnl(-30)
    bot.update_pnl(-40)
    bot.update_pnl(-50)  # total -120 <= -100
    monkeypatch.setattr(bot, "get_market_data", lambda: pytest.fail("should not fetch data"))
    monkeypatch.setattr(bot, "update_model", lambda *a, **k: pytest.fail("should not update model"))
    asyncio.run(bot.trade_cycle(max_iterations=1))


def test_daily_profit_limit_stops_trading(monkeypatch):
    bot = TradingBot()
    bot.daily_profit_limit = 100
    bot.update_pnl(60)
    bot.update_pnl(50)  # total 110 >= 100
    monkeypatch.setattr(bot, "get_market_data", lambda: pytest.fail("should not fetch data"))
    monkeypatch.setattr(bot, "update_model", lambda *a, **k: pytest.fail("should not update model"))
    asyncio.run(bot.trade_cycle(max_iterations=1))

