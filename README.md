# Bybit Spot AI Bot V2

This project contains an experimental trading bot for Bybit Spot markets. The bot collects market data, trains a machine-learning model and can automatically trade based on predictions.

## Installation

1. Install Python 3.12 and `pip`.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file or set environment variables with your Bybit API credentials and other options. Example variables:

```
BYBIT_API_KEY=your_key
BYBIT_API_SECRET=your_secret
SYMBOL=BTCUSDT
TRADE_AMOUNT_USDT=10
TP_PERCENT=1.5
SL_PERCENT=1.0
MODEL_SAVE_INTERVAL=10
LONG_THRESHOLD=0.55
SHORT_THRESHOLD=0.45
```

## Running tests

Install the requirements and run:

```bash
PYTHONPATH=. pytest
```

## Usage

To start the trading loop run:

```bash
python -m bybitbot.bot
```

Read `config.yaml` and the environment variables in the code for additional options such as stop‑loss, take‑profit and indicator settings.
