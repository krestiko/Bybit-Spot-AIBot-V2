import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from bybitbot import TradingBot


def test_compute_trade_amount_basic():
    bot = TradingBot()
    for i in range(12):
        bot.append_market_data(
            price=1 + i,
            high=1 + i + 0.5,
            low=1 + i - 0.5,
            volume=1,
            bid=1,
            ask=1,
        )
    price = bot.history_df["close"].iloc[-1]
    qty = bot.compute_trade_amount(price)
    assert qty <= bot.trade_amount / price
    assert qty > 0
