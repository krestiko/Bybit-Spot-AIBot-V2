import pandas as pd
from bybitbot.bot import compute_features, indicators


def test_compute_features_shape():
    # Use minimal data for indicators
    data = {
        "close": list(range(1, 61)),
        "volume": [1]*60,
        "bid_qty": [1]*60,
        "ask_qty": [1]*60,
    }
    df = pd.DataFrame(data)
    indicators.clear()
    indicators.append("rsi")
    features = compute_features(df)
    assert features is not None
    assert features.shape[0] == 1
    assert features.shape[1] == len(features.flatten())
