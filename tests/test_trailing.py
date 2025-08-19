import asyncio

from bybitbot import TradingBot


def test_handle_trailing_long(tmp_path):
    bot = TradingBot()
    bot.trade_file = tmp_path / "trades.csv"

    async def fake_place_order(side, qty, tp=None, sl=None):
        return {"filledQty": qty}

    async def fake_send(msg):
        pass

    bot.place_order = fake_place_order
    bot.send_telegram = fake_send
    bot.position_price = 100.0
    bot.position_amount = 1.0
    bot.trailing_percent = 1.0
    bot.trailing_price = 99.0

    async def run() -> None:
        assert not await bot.handle_trailing(102.0)
        assert bot.trailing_price == 102.0 * 0.99
        assert await bot.handle_trailing(100.5)

    asyncio.run(run())
    assert bot.position_amount == 0
