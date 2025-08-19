from bybitbot import TradingBot


def test_handle_trailing_long(tmp_path):
    bot = TradingBot()
    bot.trade_file = tmp_path / "trades.csv"
    bot.place_order = lambda side, qty, tp=None, sl=None: {"filledQty": qty}
    bot.send_telegram = lambda msg: None
    bot.position_price = 100.0
    bot.position_amount = 1.0
    bot.trailing_percent = 1.0
    bot.trailing_price = 99.0
    # price rises, trailing adjusted
    assert not bot.handle_trailing(102.0)
    assert bot.trailing_price == 102.0 * 0.99
    # price falls, trailing triggers close
    assert bot.handle_trailing(100.5)
    assert bot.position_amount == 0
