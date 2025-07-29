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
from sklearn.model_selection import GridSearchCV, cross_val_score
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
        """Send message to telegram if credentials configured."""
        if not self.telegram_token or not self.telegram_chat_id:
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{self.telegram_token}/sendMessage",
                data={"chat_id": self.telegram_chat_id, "text": msg},
            )
        except Exception as exc:  # pragma: no cover - network issues
            logging.warning("Telegram error: %s", exc)

    def fetch_news_sentiment(self) -> float:
        token = os.getenv("CRYPTO_NEWS_TOKEN")
        if not token:
            return 0.0
        try:
            resp = requests.get(
                "https://cryptopanic.com/api/v1/posts/",
                params={"auth_token": token, "kind": "news", "public": "true"},
                timeout=10,
            )
            data = resp.json()
            posts = data.get("results", [])[:10]
            score = 0.0
            for post in posts:
                if post.get("positive_votes", 0) >= post.get("negative_votes", 0):
                    score += 1
                else:
                    score -= 1
            return score / max(len(posts), 1)
        except Exception as exc:  # pragma: no cover - network issues
            logging.warning("News fetch error: %s", exc)
            return 0.0

    # ------------------------------------------------------------------
    def compute_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Compute indicator features for the given dataframe."""
        df = df.copy()
        required: List[str] = []
        if "rsi" in self.indicators:
            df["rsi"] = ta.rsi(df["close"], length=14)
            required.append("rsi")
        if "macd" in self.indicators:
            macd = ta.macd(df["close"])
            df["macd"] = macd["MACD_12_26_9"]
            required.append("macd")
        if "bb" in self.indicators:
            bb = ta.bbands(df["close"], length=5)
            df["bb_upper"] = bb["BBU_5_2.0"]
            df["bb_lower"] = bb["BBL_5_2.0"]
            required.extend(["bb_upper", "bb_lower"])
        if "sma" in self.indicators:
            df["sma"] = ta.sma(df["close"], length=10)
            required.append("sma")
        if "ema" in self.indicators:
            df["ema"] = ta.ema(df["close"], length=10)
            required.append("ema")
        if "adx" in self.indicators:
            adx = ta.adx(df["close"], length=14)
            df["adx"] = adx["ADX_14"]
            required.append("adx")
        if "stoch" in self.indicators:
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
            do_full = (not self.model_initialized) or (
                self.train_counter % self.retrain_interval == 0
            )
            if do_full:
                if len(y) >= 20:
                    if self.model_type == "xgb":
                        params = {
                            "n_estimators": [50, 100],
                            "max_depth": [3, 5],
                            "learning_rate": [0.05, 0.1],
                        }
                    elif self.model_type == "gb":
                        params = {"n_estimators": [50, 100], "max_depth": [3, 5]}
                    else:
                        params = {"alpha": [0.0001, 0.001]}
                    grid = GridSearchCV(self.model, params, cv=3, n_jobs=-1)
                    grid.fit(Xs, y)
                    self.model = grid.best_estimator_
                    score = grid.best_score_
                else:
                    self.model.fit(Xs, y)
                    score = (
                        cross_val_score(self.model, Xs, y, cv=3).mean() if len(y) >= 3 else None
                    )
                if score is not None:
                    logging.info("CV score: %.3f", score)
            elif hasattr(self.model, "partial_fit"):
                self.model.partial_fit(
                    self.scaler.transform(features), [label], classes=np.array([0, 1])
                )
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
                volume = float(ticker.get("turnover24h", 0))
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
            "sell",
            qty,
            tp=price * (1 - self.tp_percent / 100),
            sl=price * (1 + self.sl_percent / 100),
        )
        self.position_price = price
        self.position_amount = -qty
        self.trailing_price = (
            price * (1 + self.trailing_percent / 100) if self.trailing_percent else None
        )
        self.log_trade("sell", price, qty, tp=self.tp_percent, sl=self.sl_percent)
        self.send_telegram(f"Opened short {qty:.6f} {self.symbol} @ {price}")

    def close_position(self, price: float) -> None:
        if self.position_amount == 0:
            return
        side = "sell" if self.position_amount > 0 else "buy"
        qty = abs(self.position_amount)
        self.place_order(side, qty)
        pnl = (price - self.position_price) * self.position_amount
        self.update_pnl(pnl)
        self.log_trade("close", price, qty, pnl)
        self.send_telegram(f"Closed position {side} pnl={pnl:.2f}")
        self.position_amount = 0
        self.position_price = None
        self.trailing_price = None

    def handle_trailing(self, price: float) -> bool:
        if self.trailing_price is None:
            return False
        if self.position_amount > 0:
            if (
                price > self.position_price
                and price * (1 - self.trailing_percent / 100) > self.trailing_price
            ):
                self.trailing_price = price * (1 - self.trailing_percent / 100)
            if price <= self.trailing_price:
                return True
        else:
            if (
                price < self.position_price
                and price * (1 + self.trailing_percent / 100) < self.trailing_price
            ):
                self.trailing_price = price * (1 + self.trailing_percent / 100)
            if price >= self.trailing_price:
                return True
        return False

    # ------------------------------------------------------------------
    def log_trade(
        self,
        side: str,
        price: float,
        qty: float,
        pnl: float = 0.0,
        tp: Optional[float] = None,
        sl: Optional[float] = None,
    ) -> None:
        header = not os.path.exists(self.trade_file)
        with open(self.trade_file, "a") as f:
            if header:
                f.write("time,side,price,qty,pnl,tp,sl\n")
            f.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')},{side},{price},{qty},{pnl},{tp},{sl}\n"
            )

    def reset_daily_pnl(self) -> None:
        today = time.strftime("%Y-%m-%d")
        if today != self.daily_date:
            self.daily_date = today
            self.daily_pnl = 0.0

    def update_pnl(self, profit: float) -> None:
        self.daily_pnl += profit

    # ------------------------------------------------------------------
    def trade_cycle(self) -> None:
        use_websocket = os.getenv("USE_WEBSOCKET", "0") == "1"
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
                price = self.last_price
                vol = 0
                bid = ask = 0
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
            time.sleep(self.interval)


if __name__ == "__main__":
    bot = TradingBot()
    bot.trade_cycle()
