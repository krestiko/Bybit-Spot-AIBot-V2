from bybitbot import TradingBot


def test_compute_trade_amount_basic():
    bot = TradingBot()
    for i in range(12):
        bot.append_market_data(price=1 + i, volume=1, bid=1, ask=1)
    amt = bot.compute_trade_amount()
    assert amt <= bot.trade_amount
    assert amt > 0
