from bybitbot import TradingBot


def test_training_continues_after_indicator_change():
    bot = TradingBot()
    bot.history_len = 10
    bot.indicators = ["ema"]
    for i in range(11):
        bot.update_model(i + 1, i + 1.5, i + 0.5, 1, 1, 1)
    assert len(bot.features_list) > 0
    bot.indicators = ["ema", "sma"]
    assert len(bot.features_list) == 0
    result = bot.update_model(12, 12.5, 11.5, 1, 1, 1)
    assert result is not None
