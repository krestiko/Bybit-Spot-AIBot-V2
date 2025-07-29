import pandas as pd
from bybitbot.bot import (
    compute_trade_amount,
    append_market_data,
    history_df,
    trade_amount,
)


def test_compute_trade_amount_basic():
    # Ensure history_df has at least 10 entries
    for i in range(12):
        append_market_data(price=1+i, volume=1, bid=1, ask=1)
    amt = compute_trade_amount()
    assert amt <= trade_amount
    assert amt > 0
