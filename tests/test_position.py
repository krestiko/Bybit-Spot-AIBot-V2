import pytest

from bybitbot import TradingBot


def test_open_short_and_close_position(tmp_path):
    """Verify opening a short position and closing it updates state and PnL."""

    bot = TradingBot()
    bot.trade_file = tmp_path / "trades.csv"
    bot.place_order = lambda side, qty, tp=None, sl=None: {"filledQty": qty}
    bot.send_telegram = lambda msg: None
    bot._round_qty = lambda x: x
    bot.trailing_percent = 1.0
    bot.qty_step = 0.01

    open_price = 100.0
    bot.open_short(open_price)

    expected_qty = bot.compute_trade_amount() / open_price
    assert bot.position_price == open_price
    assert bot.position_amount == -expected_qty
    assert bot.trailing_price == pytest.approx(open_price * 1.01)

    close_price = 90.0
    bot.close_position(close_price)

    assert bot.position_amount == 0
    assert bot.position_price is None
    assert bot.trailing_price is None
    expected_pnl = (close_price - open_price) * (-expected_qty)
    assert bot.daily_pnl == pytest.approx(expected_pnl)

    lines = bot.trade_file.read_text().strip().splitlines()
    assert lines[1].split(",")[1] == "sell"
    assert lines[2].split(",")[1] == "close"

