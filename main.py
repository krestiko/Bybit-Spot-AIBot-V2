from bybitbot import TradingBot
import asyncio


if __name__ == "__main__":
    bot = TradingBot()
    asyncio.run(bot.trade_cycle())
