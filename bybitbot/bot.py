import os
import time
import logging
from typing import List, Optional

import joblib
import numpy as np

# Compatibility for pandas_ta with NumPy >=2 where `NaN` alias was removed
if not hasattr(np, "NaN"):
    np.NaN = np.nan  # type: ignore[attr-defined]

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

        self.session = HTTP(api_key=self.api_key, api_secret=self.api_secret)
        self.last_price: Optional[float] = None
        self.last_high: Optional[float] = None
        self.last_low: Optional[float] = None
        self.last_volume: Optional[float] = None
        self.last_bid_qty: float = 0.0
        self.last_ask_qty: float = 0.0

        self.long_threshold = float(os.getenv("LONG_THRESHOLD", 0.55))
        self.short_threshold = float(os.getenv("SHORT_THRESHOLD", 0.45))

    @staticmethod
    def _load_config(path: str) -> dict:
        if os.path.exists(path):
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _init_model(self) -> None:
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

    # ------------------------------------------------------------------
    def send_telegram(self, msg: str) -> None:
        """Send a message via Telegram if credentials are set."""
        if not (self.telegram_token and self.telegram_chat_id):
            return
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        try:
            requests.post(
                url,
                json={"chat_id": self.telegram_chat_id, "text": msg},
                timeout=5,
            )
        except Exception as exc:
            logging.warning("Telegram send error: %s", exc)

    def fetch_news_sentiment(self) -> float:
        """Placeholder for news sentiment. Returns neutral sentiment."""
        return 0.0

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
            df["sentiment"] = self.fetch_news_sentiment()
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
    def update_model(
        self,
        price: float,
        high: float,
        low: float,
        volume: float,
        bid: float,
        ask: float,
    ) -> Optional[np.ndarray]:
        """Update ML model with new market observation."""
        self.append_market_data(price, high, low, volume, bid, ask)
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
        self.scaler.partial_fit(features)
        X = np.array(self.features_list)
        y = np.array(self.labels_list)
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
        if len(Xs) and hasattr(self.model, "predict"):
            try:
                preds = self.model.predict(Xs)
                acc = float((preds == y).mean())
                logging.info("Training accuracy: %.3f", acc)
            except Exception:
                pass
        self.train_counter += 1
        if self.train_counter % self.model_save_interval == 0:
            joblib.dump({"model": self.model, "scaler": self.scaler}, self.model_file)
        self.model_initialized = True
        latest = self.compute_features(df.iloc[-self.history_len :])
        return self.scaler.transform(latest) if latest is not None else None

    # ------------------------------------------------------------------
    def get_market_data(self) -> tuple[float, float, float, float, float, float]:
        """Fetch latest market data."""
        for _ in range(self.max_retries):
            try:
                tick = self.session.get_tickers(category="spot", symbol=self.symbol)
                ticker = tick["result"]["list"][0]
                price = float(ticker["lastPrice"])
                high = price
                    low = price
                try:
                    kl = self.session.get_kline(
                        category="spot", symbol=self.symbol, interval="1", limit=1
                    )
                    kitem = kl["result"]["list"][0]
                    price = float(kitem[4])
                    high = float(kitem[2])
                    low = float(kitem[3])
                    volume = float(kitem[5])
                except Exception:
                    volume = float(
                        ticker.get("volume24h", ticker.get("turnover24h", 0))
                    )
                ob = self.session.get_orderbook(
                    category="spot", symbol=self.symbol, limit=1
                )
                bid_qty = float(ob["result"]["b"][0][1])
                ask_qty = float(ob["result"]["a"][0][1])
                return price, high, low, volume, bid_qty, ask_qty
            except Exception as exc:
                logging.warning("Market data error: %s", exc)
                time.sleep(1)
        raise RuntimeError("Failed to fetch market data")

    def _get_base_quote(self) -> tuple[str, str]:
        """Determine base and quote currencies for the current symbol.

        Tries to fetch instrument information from the exchange and falls
        back to splitting by common quote suffixes if that fails.
        """
        try:
            info = self.session.get_instruments_info(
                category="spot", symbol=self.symbol
            )
            items = info.get("result", {}).get("list", [])
            if items:
                inst = items[0]
                base = inst.get("baseCoin")
                quote = inst.get("quoteCoin")
                if base and quote:
                    return base, quote
        except Exception as exc:
            logging.warning("Instrument info error: %s", exc)

        known_suffixes = ["USDT", "USDC", "BTC", "ETH", "EUR", "GBP"]
        for suffix in known_suffixes:
            if self.symbol.endswith(suffix):
                return self.symbol[: -len(suffix)], suffix
        raise RuntimeError(f"Unable to determine base/quote for {self.symbol}")

    def place_order(self, side: str, qty: float, tp: Optional[float] = None, sl: Optional[float] = None):
        simulate = not (self.api_key and self.api_secret)
        if simulate:
            logging.info("Simulated order: %s %s", side, qty)
            return {"price": self.last_price, "qty": qty}
        base, quote = self._get_base_quote()
        price = self.last_price or 0
        try:
            bal = self.session.get_wallet_balance(accountType="spot", coin=quote if side.lower() == "buy" else base)
            avail = float(bal["result"]["list"][0]["availableBalance"])
            needed = qty * price if side.lower() == "buy" else qty
            if avail < needed:
                logging.warning("Insufficient balance: have %s need %s", avail, needed)
                return None
        except Exception as exc:
            logging.warning("Balance check error: %s", exc)

        for _ in range(self.max_retries):
            try:
                params = {
                    "category": "spot",
                    "symbol": self.symbol,
                    "side": side.upper(),
                    "orderType": "Market",
                    "qty": str(qty),
                }
                if tp is not None:
                    params["takeProfit"] = str(tp)
                if sl is not None:
                    params["stopLoss"] = str(sl)
                return self.session.place_order(**params)
            except Exception as exc:
                logging.warning("Order error: %s", exc)
                time.sleep(1)
        raise RuntimeError("Failed to place order")

    # ------------------------------------------------------------------
    def open_long(self, price: float) -> None:
        qty = self.compute_trade_amount() / price
        self.place_order(
            "buy",
            qty,
            tp=price * (1 + self.tp_percent / 100),
            sl=price * (1 - self.sl_percent / 100),
        )
        self.position_price = price
        self.position_amount = qty
        self.trailing_price = (
            price * (1 - self.trailing_percent / 100) if self.trailing_percent else None
        )
        self.log_trade("buy", price, qty, tp=self.tp_percent, sl=self.sl_percent)
        self.send_telegram(f"Opened long {qty:.6f} {self.symbol} @ {price}")

    def open_short(self, price: float) -> None:
        qty = self.compute_trade_amount() / price
        self.place_order(
