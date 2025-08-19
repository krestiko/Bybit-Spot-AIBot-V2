import asyncio

from bybitbot import TradingBot


def test_open_short(tmp_path):
    bot = TradingBot()
    bot.trade_file = tmp_path / "trades.csv"
    messages = []
    async def fake_place_order(side, qty, tp=None, sl=None):
        return {"filledQty": qty}
    bot.place_order = fake_place_order

    async def fake_send(msg):
        messages.append(msg)
    bot.send_telegram = fake_send
    bot.trailing_percent = 1.0
    bot.qty_step = 0.01
    expected_qty = bot._round_qty(bot.compute_trade_amount() / 100.0)
    asyncio.run(bot.open_short(100.0))
    assert bot.position_price == 100.0
    assert bot.position_amount == -expected_qty
    assert bot.trailing_price == 101.0
    assert messages and "Opened short" in messages[0]
    lines = bot.trade_file.read_text().strip().splitlines()
    assert lines[0] == "time,side,price,qty,pnl,tp,sl"
    row = lines[1].split(",")
    assert row[1] == "sell"
    assert float(row[2]) == 100.0
    assert float(row[3]) == expected_qty
    assert float(row[5]) == 100.0 * (1 - bot.tp_percent / 100)
    assert float(row[6]) == 100.0 * (1 + bot.sl_percent / 100)
