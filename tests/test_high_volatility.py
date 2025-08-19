import pytest
from bybitbot import TradingBot


def test_compute_trade_amount_high_volatility(monkeypatch):
    monkeypatch.setenv("MIN_TRADE_AMOUNT_USDT", "5")
    bot = TradingBot()
    for i in range(20):
        price = 100 if i % 2 else 1
        bot.append_market_data(
            price=price,
            high=price + 0.5,
            low=price - 0.5,
            volume=1,
            bid=1,
            ask=1,
        )
    amt = bot.compute_trade_amount()
    assert amt == pytest.approx(bot.min_trade_amount)
