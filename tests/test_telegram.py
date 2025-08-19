import logging
import requests

from bybitbot import TradingBot


def test_send_telegram_timeout(monkeypatch, caplog):
    bot = TradingBot()
    bot.telegram_token = "token"
    bot.telegram_chat_id = "chat"

    captured = {}

    def fake_post(url, data=None, timeout=None):
        captured["timeout"] = timeout
        raise requests.Timeout

    monkeypatch.setattr(requests, "post", fake_post)

    with caplog.at_level(logging.WARNING):
        bot.send_telegram("hi")

    assert captured["timeout"] == 5
    assert "Telegram send timeout" in caplog.text
