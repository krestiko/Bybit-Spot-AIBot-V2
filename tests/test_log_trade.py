import builtins
import logging

from bybitbot import TradingBot


def test_log_trade_creates_file(tmp_path):
    bot = TradingBot()
    bot.trade_file = tmp_path / "trades.csv"
    bot.log_trade("buy", 100.0, 1.0, pnl=0.1, tp=1.5, sl=1.0)
    bot.log_trade("sell", 110.0, 0.5, pnl=-0.2, tp=1.5, sl=1.0)
    lines = bot.trade_file.read_text().strip().splitlines()
    assert lines[0] == "time,side,price,qty,pnl,tp,sl"
    assert len(lines) == 3
    row1 = lines[1].split(",")
    row2 = lines[2].split(",")
    assert row1[1:] == ["buy", "100.0", "1.0", "0.1", "1.5", "1.0"]
    assert row2[1:] == ["sell", "110.0", "0.5", "-0.2", "1.5", "1.0"]


def test_log_trade_handles_oserror(monkeypatch, tmp_path, caplog):
    bot = TradingBot()
    bot.trade_file = tmp_path / "trades.csv"

    orig_open = builtins.open
    def fake_open(*args, **kwargs):
        raise OSError("disk error")

    monkeypatch.setattr(builtins, "open", fake_open)
    with caplog.at_level(logging.WARNING):
        bot.log_trade("buy", 100.0, 1.0)
    assert "Unable to write trade log" in caplog.text
    assert len(bot._trade_buffer) == 1

    monkeypatch.setattr(builtins, "open", orig_open)
    bot.log_trade("sell", 110.0, 0.5)
    lines = bot.trade_file.read_text().strip().splitlines()
    assert lines[0] == "time,side,price,qty,pnl,tp,sl"
    assert len(lines) == 3
    row1 = lines[1].split(",")
    row2 = lines[2].split(",")
    assert row1[1] == "buy"
    assert row2[1] == "sell"
