import os
import time
import logging
from typing import List, Optional

import joblib
import numpy as np
import math

# Compatibility for pandas_ta with NumPy >=2 where `NaN` alias was removed
if not hasattr(np, "NaN"):
    np.NaN = np.nan  # type: ignore[attr-defined]

import pandas as pd
import pandas_ta as ta
import requests
import yaml
from dotenv import load_dotenv
from pybit.unified_trading import HTTP, WebSocket
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
        self.model_save_interval = int(os.getenv("MODEL_SAVE_INTERVAL", 10))
        self.daily_loss_limit = float(os.getenv("DAILY_STOP_LOSS", 0))
        self.daily_profit_limit = float(os.getenv("DAILY_TAKE_PROFIT", 0))
        self.indicators: List[str] = self.config.get(
            "indicators",
            ["rsi", "macd", "bb", "sma", "ema", "adx", "stoch", "obv"],
        )
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
         self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.model_file = os.getenv("MODEL_FILE", "model.pkl")

        self.history_df = pd.DataFrame(
            columns=["close", "high", "low", "volume", "bid_qty", "ask_qty"]
        )
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

        self.news_sentiment: Optional[float] = None
        self.news_timestamp: float = 0.0
        self.news_refresh_interval = int(os.getenv("NEWS_INTERVAL", 30)) * 60

        self.session = HTTP(api_key=self.api_key, api_secret=self.api_secret)
        self.ws: Optional[WebSocket] = None
        self._ws_active: bool = False
        self.last_price: Optional[float] = None
        self.last_high: Optional[float] = None
        self.last_low: Optional[float] = None
        self.last_volume: Optional[float] = None
        self.last_bid_qty: float = 0.0
        self.last_ask_qty: float = 0.0

        try:
            info = self.session.get_instruments_info(
                category="spot", symbol=self.symbol
            )
            self.lot_size = float(info["result"]["list"][0].get("lotSize", 1e-8))
        except Exception:
            self.lot_size = 1e-8

        self.long_threshold = float(os.getenv("LONG_THRESHOLD", 0.55))
        self.short_threshold = float(os.getenv("SHORT_THRESHOLD", 0.45))

    @staticmethod
    def _load_config(path: str) -> dict:
        if os.path.exists(path):
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

        if os.path.exists(self.model_file):
            saved = joblib.load(self.model_file)
            self.model = saved.get("model")
            self.scaler = saved.get("scaler", self.scaler)
            self.model_initialized = True
        else:
            if self.model_type == "xgb":
                self.model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
            elif self.model_type == "gb":
                self.model = GradientBoostingClassifier()
            else:
                self.model = SGDClassifier(loss="log_loss")
            self.model_initialized = False

    def send_telegram(self, msg: str) -> None:
        """Send a message via Telegram if credentials are provided."""
        if not (self.telegram_token and self.telegram_chat_id):
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{self.telegram_token}/sendMessage",
                json={"chat_id": self.telegram_chat_id, "text": msg},
                timeout=5,
            )
        except Exception as exc:
            logging.warning("Telegram send error: %s", exc)

    def fetch_news_sentiment(self) -> float:
        """Fetch news sentiment and cache the result and timestamp."""
        sentiment = 0.0  # placeholder implementation
        self.news_sentiment = sentiment
        self.news_timestamp = time.time()
        return sentiment

    def compute_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Compute technical indicators from a price history frame."""
        df = df.copy()
        required: List[str] = []
        if "rsi" in self.indicators:
            df["rsi"] = ta.rsi(df["close"], length=14)
            required.append("rsi")
        if "macd" in self.indicators:
            macd = ta.macd(df["close"])
            df["macd"] = macd["MACD_12_26_9"]
            df["macd_signal"] = macd["MACDs_12_26_9"]
            required.extend(["macd", "macd_signal"])
        if "bb" in self.indicators:
            bb = ta.bbands(df["close"])
            df["bb_low"] = bb["BBL_5_2.0"]
            df["bb_high"] = bb["BBU_5_2.0"]
            required.extend(["bb_low", "bb_high"])
        if "sma" in self.indicators:
            df["sma"] = ta.sma(df["close"], length=10)
            required.append("sma")
        if "ema" in self.indicators:
            df["ema"] = ta.ema(df["close"], length=10)
            required.append("ema")
        if "adx" in self.indicators:
            adx = ta.adx(df["high"], df["low"], df["close"])
            df["adx"] = adx["ADX_14"]
            required.append("adx")
        if "stoch" in self.indicators:
            stoch = ta.stoch(df["high"], df["low"], df["close"])
            df["stoch_k"] = stoch["STOCHk_14_3_3"]
            df["stoch_d"] = stoch["STOCHd_14_3_3"]
            required.extend(["stoch_k", "stoch_d"])
        if "obv" in self.indicators:
            df["obv"] = ta.obv(df["close"], df["volume"])
            required.append("obv")
        denom = df["bid_qty"] + df["ask_qty"]
        df["bid_ask_ratio"] = np.where(
            denom == 0, 0, (df["bid_qty"] - df["ask_qty"]) / denom
        )
        required.append("bid_ask_ratio")
        if "news" in self.indicators:
            now = time.time()
            if (
                self.news_sentiment is None
                or now - self.news_timestamp > self.news_refresh_interval
            ):
                sentiment = self.fetch_news_sentiment()
            else:
                sentiment = self.news_sentiment
            df["sentiment"] = sentiment
            required.append("sentiment")
         required.append("sentiment")
        row = df.iloc[-1]
        if row[required].isna().any():
            return None
        feats = row[required + ["volume"]].values
        return np.array(feats, dtype=float).reshape(1, -1)

    # ------------------------------------------------------------------
    def append_market_data(
        self, price: float, high: float, low: float, volume: float, bid: float, ask: float
    ) -> None:
        """Append market data to internal history."""
        self.history_df.loc[len(self.history_df)] = [price, high, low, volume, bid, ask]
        if len(self.history_df) > self.history_len + 50:
            self.history_df = self.history_df.iloc[-(self.history_len + 50) :]
        self._write_counter += 1
        if self._write_counter % self.save_interval == 0:
            self.history_df.to_csv(self.price_file, index=False)

    def _round_qty(self, qty: float) -> float:
        if self.lot_size <= 0:
            return qty
        return math.floor(qty / self.lot_size) * self.lot_size

    def compute_trade_amount(self, price: float) -> float:
        """Return quantity to trade based on configured USDT amount."""
        if price <= 0:
            return 0.0
        qty = self.trade_amount / price
        return self._round_qty(qty)
    # ------------------------------------------------------------------
    def get_market_data(self) -> tuple[float, float, float, float, float, float]:
        """Fetch latest market data.

        This placeholder implementation avoids network calls in tests.
        """
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    def place_order(
        self, side: str, qty: float, tp: Optional[float] = None, sl: Optional[float] = None
    ):
        """Simplified order placement used for testing."""
        logging.info("Simulated order: %s %s", side, qty)
        return {"price": self.last_price, "qty": qty}

    def open_long(self, price: float) -> None:
        """Placeholder for opening a long position."""
        pass

    def close_position(self, price: float) -> None:
        """Placeholder for closing an open position."""
        pass

    def log_trade(
        self,
        side: str,
        price: float,
        qty: float,
        pnl: float = 0.0,
        tp: Optional[float] = None,
        sl: Optional[float] = None,
    ) -> None:
        """Placeholder for recording trade information."""
        pass

    # ------------------------------------------------------------------
    # WebSocket handling
    def _handle_ticker(self, msg: dict) -> None:
        """Process incoming ticker messages to update last known values."""
        try:
            data = msg.get("data", [{}])[0]
            self.last_price = float(data.get("lastPrice", self.last_price or 0))
            self.last_high = float(data.get("highPrice24h", self.last_high or 0))
            self.last_low = float(data.get("lowPrice24h", self.last_low or 0))
            self.last_volume = float(data.get("volume24h", self.last_volume or 0))
            self.last_bid_qty = float(data.get("bid1Size", self.last_bid_qty))
            self.last_ask_qty = float(data.get("ask1Size", self.last_ask_qty))
        except Exception as exc:
            logging.debug("Ticker parse error: %s", exc)

    def _subscribe_streams(self) -> None:
        """Subscribe to required WebSocket streams."""
        if self.ws is None:
            return
        try:
            self.ws.ticker_stream(symbol=self.symbol, callback=self._handle_ticker)
        except Exception as exc:
            logging.warning("WebSocket subscribe error: %s", exc)

    def _on_ws_close(self, ws, *args, **kwargs) -> None:
        """Callback for WebSocket close events."""
        logging.warning("WebSocket closed")
        self._ws_active = False

    def _on_ws_error(self, ws, error=None, *args, **kwargs) -> None:
        """Callback for WebSocket errors."""
        logging.warning("WebSocket error: %s", error)
        self._ws_active = False

    def run_websocket(self) -> None:
        """Run WebSocket connection with automatic reconnection."""
        while True:
            try:
                self.ws = WebSocket(
                    channel_type="spot",
                    on_close=self._on_ws_close,
                    on_error=self._on_ws_error,
                )
                self._subscribe_streams()
                self._ws_active = True
                while self._ws_active:
                    time.sleep(1)
            except Exception as exc:
                logging.warning("WebSocket connection error: %s", exc)
            finally:
                if self.ws is not None:
                    try:
                        self.ws.exit()
                    except Exception:
                        pass
                    self.ws = None
            time.sleep(5)
