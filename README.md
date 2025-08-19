# Bybit Spot AI Bot V2

This project contains an experimental trading bot for Bybit Spot markets. The bot collects market data, trains a machine-learning model and can automatically trade based on predictions. The bot now supports pluggable strategies, typed configuration and basic risk management.

## Installation

1. Install Python 3.12 and `pip`.
2. Install dependencies:

```bash
pip install -e .[dev]
pre-commit install
```

3. Create a `.env` file or set environment variables with your Bybit API credentials and other options. See `.env.example` for the full list. At minimum set:

```
BYBIT_API_KEY=your_key
BYBIT_API_SECRET=your_secret
```

## Running tests

Install the requirements and run:

```bash
PYTHONPATH=. pytest
```

## Usage

To start the trading loop run:

```bash
python -m bybitbot.bot  # live trading
```

To run in Docker:

```bash
docker-compose up -d --build
```

Read `config.yaml` and `.env.example` for additional options such as stop-loss, take-profit and indicator settings.
