import os
import time
import logging
from typing import List, Optional

import joblib
import numpy as np
np.NaN = np.nan  # compatibility for pandas_ta
import pandas as pd
import pandas_ta as ta
import requests
import yaml
from dotenv import load_dotenv
from pybit.unified_trading import HTTP
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

load_dotenv()


class TradingBot:
    """Simple trading bot encapsulating state and trading logic."""

    def __init__(self, config_file: str = "config.yaml") -> None:
        self.config = self._load_config(config_file)
        self.api_key = os.getenv("BYBIT_API_KEY")
        self.api_secret = os.getenv("BYBIT_API_SECRET")
        self.symbol = os.getenv("SYMBOL", "BTCUSDT")
        self.trade_amount = float(os.getenv("TRADE_AMOUNT_USDT", 10))
        self.tp_percent = float(
            self.config.get("trade", {}).get("tp_percent", os.getenv("TP_PERCENT", 1.5))
        )
        self.sl_percent = float(
            self.config.get("trade", {}).get("sl_percent", os.getenv("SL_PERCENT", 1.0))
        )
        self.interval = int(os.getenv("INTERVAL", 5)) * 60
        self.max_retries = int(os.getenv("MAX_RETRIES", 3))
        self.retrain_interval = int(os.getenv("RETRAIN_INTERVAL", 50))
        self.history_len = int(os.getenv("HISTORY_LENGTH", 100))
        self.price_file = os.getenv("PRICE_HISTORY_FILE", "price_history.csv")
        self.trade_file = os.getenv("TRADE_HISTORY_FILE", "trade_history.csv")
        self.save_interval = int(os.getenv("SAVE_INTERVAL", 10))
        self.trailing_percent = float(os.getenv("TRAILING_PERCENT", 0))
        self.model_type = self.config.get("trade", {}).get("model_type", os.getenv("MODEL_TYPE", "gb"))
        self.daily_loss_limit = float(os.getenv("DAILY_STOP_LOSS", 0))
        self.daily_profit_limit = float(os.getenv("DAILY_TAKE_PROFIT", 0))
        self.indicators: List[str] = self.config.get(
            "indicators",
            ["rsi", "macd", "bb", "sma", "ema", "adx", "stoch", "obv"],
        )
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.model_file = os.getenv("MODEL_FILE", "model.pkl")

        self.history_df = pd.DataFrame(columns=["close", "volume", "bid_qty", "ask_qty"])
        self._write_counter = 0
        self.features_list: List[np.ndarray] = []
        self.labels_list: List[int] = []
        self.scaler = StandardScaler()
        self.train_counter = 0
        self.model_initialized = False
        self._init_model()

        self.position_price: Optional[float] = None
        self.position_amount: float = 0.0
        self.trailing_price: Optional[float] = None
        self.daily_pnl: float = 0.0
        self.daily_date: str = time.strftime("%Y-%m-%d")

        self.session = HTTP(api_key=self.api_key, api_secret=self.api_secret)
        self.last_price: Optional[float] = None

    @staticmethod
    def _load_config(path: str) -> dict:
        if os.path.exists(path):
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _init_model(self) -> None:
        if os.path.exists(self.model_file):
@@ -160,130 +161,117 @@ class TradingBot:
            stoch = ta.stoch(df["close"])
            df["stoch_k"] = stoch["STOCHk_14_3_3"]
            df["stoch_d"] = stoch["STOCHd_14_3_3"]
            required.extend(["stoch_k", "stoch_d"])
        if "obv" in self.indicators:
            df["obv"] = ta.obv(df["close"], df["volume"])
            required.append("obv")
        denom = df["bid_qty"] + df["ask_qty"]
        df["bid_ask_ratio"] = np.where(denom == 0, 0, (df["bid_qty"] - df["ask_qty"]) / denom)
        required.append("bid_ask_ratio")
        if "news" in self.indicators:
            df["sentiment"] = self.fetch_news_sentiment()
            required.append("sentiment")
        row = df.iloc[-1]
        if row[required].isna().any():
            return None
        feats = row[required + ["volume"]].values
        return np.array(feats, dtype=float).reshape(1, -1)

    # ------------------------------------------------------------------
    def append_market_data(self, price: float, volume: float, bid: float, ask: float) -> None:
        """Append market data to internal history."""
        self.history_df.loc[len(self.history_df)] = [price, volume, bid, ask]
        if len(self.history_df) > self.history_len + 50:
            self.history_df = self.history_df.iloc[-(self.history_len + 50) :]
        self._write_counter += 1
        if self._write_counter % self.save_interval == 0:
            self.history_df.to_csv(self.price_file, index=False)

    def compute_trade_amount(self) -> float:
        """Return position size based on recent volatility."""
        if len(self.history_df) < 10:
            return self.trade_amount
        vol = self.history_df["close"].pct_change().rolling(10).std().iloc[-1]
        if pd.isna(vol) or vol == 0:
            return self.trade_amount
        factor = min(1.0, 0.02 / vol)
        return self.trade_amount * factor

    # ------------------------------------------------------------------
    def update_model(self, price: float, volume: float, bid: float, ask: float) -> Optional[np.ndarray]:
        """Update ML model with new market observation."""
        self.append_market_data(price, volume, bid, ask)
        if len(self.history_df) < self.history_len + 1:
            return None
        df = self.history_df.iloc[-(self.history_len + 1) :]
        features = self.compute_features(df.iloc[:-1])
        if features is None:
            return None
        label = 1 if df["close"].iloc[-1] > df["close"].iloc[-2] else 0
        self.features_list.append(features.flatten())
        self.labels_list.append(label)
        if len(self.features_list) > self.history_len:
            self.features_list.pop(0)
            self.labels_list.pop(0)
        X = np.array(self.features_list)
        y = np.array(self.labels_list)
        self.scaler.fit(X)
        Xs = self.scaler.transform(X)
        try:
            if not self.model_initialized:
                self.model.fit(Xs, y)
            elif hasattr(self.model, "partial_fit"):
                self.model.partial_fit(
                    self.scaler.transform(features), [label], classes=np.array([0, 1])
                )
            elif self.train_counter % self.retrain_interval == 0:
                self.model.fit(Xs, y)
        except Exception as exc:
            logging.warning("Model train error: %s", exc)
        self.train_counter += 1
        joblib.dump({"model": self.model, "scaler": self.scaler}, self.model_file)
        self.model_initialized = True
        latest = self.compute_features(df.iloc[-self.history_len :])
        return self.scaler.transform(latest) if latest is not None else None

    # ------------------------------------------------------------------
    def get_market_data(self) -> tuple[float, float, float, float]:
        """Fetch latest market data."""
        for _ in range(self.max_retries):
            try:
                tick = self.session.get_tickers(category="spot", symbol=self.symbol)
                ticker = tick["result"]["list"][0]
                price = float(ticker["lastPrice"])
                try:
                    kl = self.session.get_kline(
                        category="spot", symbol=self.symbol, interval="1", limit=1
                    )
                    kitem = kl["result"]["list"][0]
                    volume = float(kitem[5])
                except Exception:
                    volume = float(ticker.get("volume24h", ticker.get("turnover24h", 0)))
                ob = self.session.get_orderbook(category="spot", symbol=self.symbol, limit=1)
                bid_qty = float(ob["result"]["b"][0][1])
                ask_qty = float(ob["result"]["a"][0][1])
                return price, volume, bid_qty, ask_qty
            except Exception as exc:
                logging.warning("Market data error: %s", exc)
                time.sleep(1)
        raise RuntimeError("Failed to fetch market data")

    def place_order(self, side: str, qty: float, tp: Optional[float] = None, sl: Optional[float] = None):
        simulate = not (self.api_key and self.api_secret)
        if simulate:
            logging.info("Simulated order: %s %s", side, qty)
            return {"price": self.last_price, "qty": qty}
        for _ in range(self.max_retries):
            try:
                return self.session.place_order(
                    category="spot",
                    symbol=self.symbol,
                    side="Buy" if side.lower() == "buy" else "Sell",
                    orderType="Market",
                    qty=qty,
                    takeProfit=tp,
                    stopLoss=sl,
                )
@@ -393,53 +381,52 @@ class TradingBot:
        if use_websocket:
            from pybit.unified_trading import WebSocket

            def _cb(msg):
                data = msg.get("data")
                if isinstance(data, list):
                    data = data[0]
                lp = data.get("lastPrice") or data.get("lp")
                if lp:
                    self.last_price = float(lp)

            ws = WebSocket("spot")
            ws.ticker_stream(self.symbol, _cb)

        while True:
            self.reset_daily_pnl()
            if self.daily_loss_limit and self.daily_pnl <= -self.daily_loss_limit:
                logging.warning("Daily loss limit reached")
                break
            if self.daily_profit_limit and self.daily_pnl >= self.daily_profit_limit:
                logging.info("Daily profit target reached")
                break
            if self.last_price is None:
                price, vol, bid, ask = self.get_market_data()
            else:
                _, vol, bid, ask = self.get_market_data()
                price = self.last_price
            features = self.update_model(price, vol, bid, ask)
            if features is None:
                time.sleep(self.interval)
                continue
            prob = self.model.predict_proba(features)[0][1]

            if self.position_amount > 0:  # long
                if (
                    price >= self.position_price * (1 + self.tp_percent / 100)
                    or price <= self.position_price * (1 - self.sl_percent / 100)
                    or self.handle_trailing(price)
                ):
                    self.close_position(price)
            elif self.position_amount < 0:  # short
                if (
                    price <= self.position_price * (1 - self.tp_percent / 100)
                    or price >= self.position_price * (1 + self.sl_percent / 100)
                    or self.handle_trailing(price)
                ):
                    self.close_position(price)
            if self.position_amount == 0:
                if prob > 0.55:
                    self.open_long(price)
                elif prob < 0.45:
                    self.open_short(price)
