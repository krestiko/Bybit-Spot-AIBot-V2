import pandas as pd
import pytest
from bybitbot import TradingBot


def test_compute_features_shape():
    data = {
        "close": list(range(1, 61)),
        "high": [x + 0.5 for x in range(1, 61)],
        "low": [x - 0.5 for x in range(1, 61)],
        "volume": [1] * 60,
        "bid_qty": [1] * 60,
        "ask_qty": [1] * 60,
    }
    df = pd.DataFrame(data)
    bot = TradingBot()
    bot.indicators = ["rsi"]
    features = bot.compute_features(df)
    assert features is not None
    assert features.shape == (1, len(features.flatten()))


def test_ema_indicator():
    data = {
        "close": list(range(1, 61)),
        "high": [x + 0.5 for x in range(1, 61)],
        "low": [x - 0.5 for x in range(1, 61)],
        "volume": [1] * 60,
        "bid_qty": [1] * 60,
        "ask_qty": [1] * 60,
    }
    df = pd.DataFrame(data)
    bot = TradingBot()
    bot.indicators = ["ema"]
    feats = bot.compute_features(df)
    assert feats is not None
    assert feats.shape == (1, 3)


def test_adx_indicator():
    data = {
        "close": list(range(1, 61)),
        "high": [x + 0.5 for x in range(1, 61)],
        "low": [x - 0.5 for x in range(1, 61)],
        "volume": [1] * 60,
        "bid_qty": [1] * 60,
        "ask_qty": [1] * 60,
    }
    df = pd.DataFrame(data)
    bot = TradingBot()
    bot.indicators = ["adx"]
    feats = bot.compute_features(df)
    assert feats is not None
    assert feats.shape == (1, 3)

def test_bid_ask_zero_division():
    data = {
        "close": [1, 2],
        "high": [1.5, 2.5],
        "low": [0.5, 1.5],
        "volume": [1, 1],
        "bid_qty": [0, 0],
        "ask_qty": [0, 0],
    }
    df = pd.DataFrame(data)
    bot = TradingBot()
    bot.indicators = []
    feats = bot.compute_features(df)
    assert feats is not None
    assert not pd.isna(feats).any()


def _make_dataframe(size: int = 60) -> pd.DataFrame:
    data = {
        "close": list(range(1, size + 1)),
        "high": [x + 0.5 for x in range(1, size + 1)],
        "low": [x - 0.5 for x in range(1, size + 1)],
        "volume": [1] * size,
        "bid_qty": [1] * size,
        "ask_qty": [1] * size,
    }
    return pd.DataFrame(data)


@pytest.mark.parametrize(
    "inds, expected_len, has_news",
    [(["ema"], 3, False), (["ema", "news"], 4, True), (["ema", "adx", "news"], 5, True)],
)
def test_compute_features_combinations(inds, expected_len, has_news, monkeypatch):
    df = _make_dataframe()
    bot = TradingBot()
    bot.indicators = inds
    monkeypatch.setattr(bot, "fetch_news_sentiment", lambda: 0.5)
    feats = bot.compute_features(df)
    assert feats is not None
    assert feats.shape == (1, expected_len)
    if has_news:
        assert feats[0, -2] == 0.5
