import os
import time
import logging
import asyncio
from typing import List, Optional, Any

# mypy: ignore-errors

import joblib
import numpy as np

try:  # pragma: no cover - needed for pandas_ta on NumPy>=2
    from numpy import NaN  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover
    np.NaN = np.nan
import pandas as pd
import pandas_ta as ta
import math
import requests
import yaml
from dotenv import load_dotenv
from pybit.unified_trading import HTTP
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .config import Settings
from .strategies import ITradingStrategy, MLProbabilityStrategy
from .risk import RiskManager

logger = logging.getLogger(__name__)

load_dotenv()


class TradingBot:
    """Simple trading bot encapsulating state and trading logic."""

    def __init__(
        self,
        settings: Settings | None = None,
        strategy: ITradingStrategy | None = None,
        config_file: str = "config.yaml",
        risk_manager: RiskManager | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.config = self._load_config(config_file)
        log_level_str = self.settings.log_level or self.config.get("logging", {}).get(
            "level", "INFO"
        )
        log_level = getattr(logging, log_level_str.upper(), logging.INFO)
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=log_level, format="%(asctime)s %(levelname)s %(message)s"
            )
        else:
            logging.getLogger().setLevel(log_level)
        self.api_key = self.settings.api_key
        self.api_secret = self.settings.api_secret
        self.symbol = self.settings.symbol
        self.trade_amount = self.settings.trade_amount
        self.min_trade_amount = self.config.get("trade", {}).get(
            "min_trade_amount", self.settings.min_trade_amount
        )
        self.tp_percent = self.config.get("trade", {}).get(
            "tp_percent", self.settings.tp_percent
        )
        self.sl_percent = self.config.get("trade", {}).get(
            "sl_percent", self.settings.sl_percent
        )
        self.interval = self._get_env_int("INTERVAL", 5) * 60
        self.max_retries = self._get_env_int("MAX_RETRIES", 3)
        self.retrain_interval = self._get_env_int("RETRAIN_INTERVAL", 50)
        self.history_len = self._get_env_int("HISTORY_LENGTH", 100)
        self.price_file = os.getenv("PRICE_HISTORY_FILE", "price_history.csv")
        self.trade_file = os.getenv("TRADE_HISTORY_FILE", "trade_history.csv")
        self.save_interval = self._get_env_int("SAVE_INTERVAL", 10)
        self.trailing_percent = self._get_env_float("TRAILING_PERCENT", 0)
        self.model_type = self.config.get("trade", {}).get(
            "model_type", os.getenv("MODEL_TYPE", "gb")
        )
        self.model_save_interval = self._get_env_int("MODEL_SAVE_INTERVAL", 10)
        d_loss = self.settings.daily_loss_limit
        d_profit = self.settings.daily_profit_limit
        self._indicators: List[str] = self.config.get(
            "indicators",
            ["rsi", "macd", "bb", "sma", "ema", "adx", "stoch", "obv"],
        )
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.model_file = os.getenv("MODEL_FILE", "model.pkl")

        self.risk_manager = risk_manager or RiskManager(
            d_loss,
            d_profit,
        )
        self.daily_pnl = self.risk_manager.daily_pnl
        self.daily_date = self.risk_manager.daily_date
        self.daily_loss_limit = d_loss
        self.daily_profit_limit = d_profit

        self.history_df = pd.DataFrame(
            columns=["close", "high", "low", "volume", "bid_qty", "ask_qty"]
        )
        self._write_counter = 0
        self._pending_price_save = False
        self._trade_buffer: List[str] = []
        self.features_list: List[np.ndarray] = []
        self.labels_list: List[int] = []
        self.scaler = StandardScaler()
        self.train_counter = 0
        self.model_initialized = False
        self._init_model()

        self.position_price: Optional[float] = None
        self.position_amount: float = 0.0
        self.trailing_price: Optional[float] = None

        self.session = HTTP(api_key=self.api_key, api_secret=self.api_secret)
        self.base_asset, self.quote_asset = self._split_symbol(self.symbol)
        self.qty_step = self._get_qty_step()
        self.last_price: Optional[float] = None
        self.last_high: Optional[float] = None
        self.last_low: Optional[float] = None
        self.last_volume: Optional[float] = None
        self.last_bid_qty: float = 0.0
        self.last_ask_qty: float = 0.0

        self._news_cache: float = 0.0
        self._news_time: float = 0.0
        self.news_cache_interval = self._get_env_int("NEWS_CACHE_MINUTES", 30) * 60

        self.long_threshold = self._get_env_float("LONG_THRESHOLD", 0.55)
        self.short_threshold = self._get_env_float("SHORT_THRESHOLD", 0.45)
        self.strategy = strategy or MLProbabilityStrategy(
            self.long_threshold, self.short_threshold
        )
        if hasattr(self.strategy, "set_model"):
            # type: ignore[call-arg]
            self.strategy.set_model(self.model)

    @property
    def indicators(self) -> List[str]:
        return self._indicators

    @indicators.setter
    def indicators(self, value: List[str]) -> None:
        self._indicators = list(dict.fromkeys(value))
        self.features_list.clear()
        self.labels_list.clear()
        self.scaler = StandardScaler()

    @staticmethod
    def _load_config(path: str) -> dict:
        if os.path.exists(path):
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    @staticmethod
    def _get_env_float(name: str, default: float) -> float:
        val = os.getenv(name)
        if val is None:
            return default
        try:
            return float(val)
        except ValueError:
            logger.warning("Invalid value for %s: %s", name, val)
            return default

    @staticmethod
    def _get_env_int(name: str, default: int) -> int:
        val = os.getenv(name)
        if val is None:
            return default
        try:
            return int(val)
        except ValueError:
            logger.warning("Invalid value for %s: %s", name, val)
            return default

    def _split_symbol(self, symbol: str) -> tuple[str, str]:
        quotes = ["USDT", "USDC", "BTC", "ETH", "EUR", "USD"]
        for q in quotes:
            if symbol.endswith(q):
                return symbol[: -len(q)], q
        try:
            info = self.session.get_instruments_info(category="spot", symbol=symbol)
            item = info.get("result", {}).get("list", [{}])[0]
            return item.get("baseCoin", symbol[:-4]), item.get("quoteCoin", symbol[-4:])
        except Exception:
            return symbol[:-4], symbol[-4:]

    def _get_qty_step(self) -> float:
        env_step = os.getenv("QTY_STEP")
        if env_step:
            try:
                step = float(env_step)
                return step if step > 0 else 1.0
            except ValueError:
                pass
        try:
            info = self.session.get_instruments_info(
                category="spot", symbol=self.symbol
            )
            item = info.get("result", {}).get("list", [{}])[0]
            lot = item.get("lotSizeFilter", {})
            step = float(lot.get("qtyStep", 1))
            return step if step > 0 else 1.0
        except Exception:
            return 1.0

    def _round_qty(self, qty: float) -> float:
        step = self.qty_step
        return math.floor(qty / step) * step

    def _init_model(self) -> None:
        if os.path.exists(self.model_file):
            try:
                saved = joblib.load(self.model_file)
                self.model = saved.get("model")
                self.scaler = saved.get("scaler", self.scaler)
                self.model_initialized = True
                return
            except Exception as exc:
                logger.warning("Failed to load model file: %s", exc)

        if self.model_type == "xgb":
            self.model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        elif self.model_type == "gb":
            self.model = GradientBoostingClassifier()
        elif self.model_type == "sgd":
            self.model = SGDClassifier(loss="log_loss")
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
        self.model_initialized = False

    # ------------------------------------------------------------------
    async def send_telegram(self, msg: str, timeout: int = 5) -> None:
        """Send message to telegram if credentials configured."""
        if not self.telegram_token or not self.telegram_chat_id:
            return
        try:
            await asyncio.to_thread(
                requests.post,
                f"https://api.telegram.org/bot{self.telegram_token}/sendMessage",
                data={"chat_id": self.telegram_chat_id, "text": msg},
                timeout=timeout,
            )
        except requests.Timeout:
            logger.warning("Telegram send timeout")
        except requests.RequestException as exc:  # pragma: no cover - network issues
            logger.warning("Telegram error: %s", exc)

    def fetch_news_sentiment(self) -> float:
        token = os.getenv("CRYPTO_NEWS_TOKEN")
        if not token:
            logger.warning("CRYPTO_NEWS_TOKEN not set")
            return 0.0
        now = time.time()
        if now - self._news_time < self.news_cache_interval:
            return self._news_cache
        try:
            resp = requests.get(
                "https://cryptopanic.com/api/v1/posts/",
                params={"auth_token": token, "kind": "news", "public": "true"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            posts = data.get("results", [])[:10]
            score = 0.0
            for post in posts:
                if post.get("positive_votes", 0) >= post.get("negative_votes", 0):
                    score += 1
                else:
                    score -= 1
            self._news_cache = score / max(len(posts), 1)
            self._news_time = now
            return self._news_cache
        except requests.RequestException as exc:  # pragma: no cover - network issues
            logger.warning("News fetch error: %s", exc)
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
            adx = ta.adx(df["high"], df["low"], df["close"], length=14)
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
        feature_counts = {
            "rsi": 1,
            "macd": 1,
            "bb": 2,
            "sma": 1,
            "ema": 1,
            "adx": 1,
            "stoch": 2,
            "obv": 1,
            "news": 1,
        }
        expected = 1
        for ind in self.indicators:
            if ind not in feature_counts:
                logger.warning("Unknown indicator: %s", ind)
                return None
            expected += feature_counts[ind]
        if len(required) != expected:
            logger.warning(
                "Indicator feature mismatch: expected %d, got %d",
                expected,
                len(required),
            )
            return None
        row = df.iloc[-1]
        if row[required + ["volume"]].isna().any():
            return None
        feats = row[required + ["volume"]].values
        return np.array(feats, dtype=float).reshape(1, -1)

    # ------------------------------------------------------------------
    def append_market_data(
        self,
        price: float,
        high: float,
        low: float,
        volume: float,
        bid: float,
        ask: float,
    ) -> None:
        """Append market data to internal history."""
        self.history_df.loc[len(self.history_df)] = [price, high, low, volume, bid, ask]
        if len(self.history_df) > self.history_len + 50:
            self.history_df = self.history_df.iloc[-(self.history_len + 50) :]
        self._write_counter += 1
        if self._write_counter % self.save_interval == 0 or self._pending_price_save:
            try:
                self.history_df.to_csv(self.price_file, index=False)
                self._pending_price_save = False
            except OSError as exc:
                logger.warning("Unable to save price history: %s", exc)
                self._pending_price_save = True

    def compute_trade_amount(self) -> float:
        """Return position size based on recent volatility."""
        if len(self.history_df) < 10:
            return max(self.min_trade_amount, self.trade_amount)
        vol = self.history_df["close"].pct_change().rolling(10).std().iloc[-1]
        if pd.isna(vol) or vol == 0:
            return max(self.min_trade_amount, self.trade_amount)
        factor = min(1.0, 0.02 / vol)
        return max(self.min_trade_amount, self.trade_amount * factor)

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
            logger.warning("Model train error: %s", exc)
        if len(Xs) and hasattr(self.model, "predict"):
            try:
                preds = self.model.predict(Xs)
                acc = float((preds == y).mean())
                logger.info("Training accuracy: %.3f", acc)
            except Exception:
                pass
        self.train_counter += 1
        if self.train_counter % self.model_save_interval == 0:
            try:
                joblib.dump(
                    {"model": self.model, "scaler": self.scaler}, self.model_file
                )
            except Exception as exc:
                logger.warning("Failed to save model: %s", exc)
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
                try:
                    kl = self.session.get_kline(
                        category="spot", symbol=self.symbol, interval="1", limit=1
                    )
                    kitem = kl["result"]["list"][0]
                    high = float(kitem[2])
                    low = float(kitem[3])
                    volume = float(kitem[5])
                except Exception:
                    high = low = price
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
                logger.warning("Market data error: %s", exc)
                time.sleep(1)
        raise RuntimeError("Failed to fetch market data")

    async def place_order(
        self,
        side: str,
        qty: float,
        tp: Optional[float] = None,
        sl: Optional[float] = None,
    ) -> Optional[dict[str, Any]]:
        simulate = not (self.api_key and self.api_secret)
        if simulate:
            logger.info("Simulated order: %s %s", side, qty)
            return {"response": None, "filledQty": qty}

        price = self.last_price
        if price is None:
            logger.error("Cannot place order without last price")
            return None
        try:
            coin = self.quote_asset if side.lower() == "buy" else self.base_asset
            bal = await asyncio.to_thread(
                self.session.get_wallet_balance, accountType="spot", coin=coin
            )
            avail = float(bal["result"]["list"][0]["availableBalance"])
            needed = qty * price if side.lower() == "buy" else qty
            if avail < needed:
                logger.warning("Insufficient balance: have %s need %s", avail, needed)
                return None
        except Exception as exc:
            logger.warning("Balance check error: %s", exc)

        for _ in range(self.max_retries):
            try:
                params = {
                    "category": "spot",
                    "symbol": self.symbol,
                    "side": "Buy" if side.lower() == "buy" else "Sell",
                    "orderType": "Market",
                    "qty": qty,
                }
                if tp is not None:
                    params["takeProfit"] = tp
                if sl is not None:
                    params["stopLoss"] = sl

                order = await asyncio.to_thread(self.session.place_order, **params)
                order_id = order.get("result", {}).get("orderId")
                filled = qty
                if order_id:
                    try:
                        status = await asyncio.to_thread(
                            self.session.get_order,
                            category="spot",
                            symbol=self.symbol,
                            orderId=order_id,
                        )
                        ord_info = status.get("result", {}).get("list", [{}])[0]
                        filled = float(ord_info.get("cumExecQty", filled))
                        logger.info("Order status: %s", ord_info.get("orderStatus"))
                    except Exception as exc:
                        logger.warning("Order status error: %s", exc)
                return {"response": order, "filledQty": filled}
            except Exception as exc:
                logger.warning("Order error: %s", exc)
                await asyncio.sleep(1)
        raise RuntimeError("Failed to place order")

    # ------------------------------------------------------------------
    async def open_long(self, price: float) -> None:
        if not self.risk_manager.can_trade():
            logger.info("Risk limits prevent opening long position")
            return
        qty = self._round_qty(self.compute_trade_amount() / price)
        if qty < self.qty_step:
            logger.warning("Computed quantity %s below qty_step %s", qty, self.qty_step)
            return
        tp_price = price * (1 + self.tp_percent / 100)
        sl_price = price * (1 - self.sl_percent / 100)
        order = await self.place_order("buy", qty, tp=tp_price, sl=sl_price)
        filled = order.get("filledQty", qty) if order else qty
        self.position_price = price
        self.position_amount = filled
        self.trailing_price = (
            price * (1 - self.trailing_percent / 100) if self.trailing_percent else None
        )
        self.log_trade("buy", price, filled, tp=tp_price, sl=sl_price)
        await self.send_telegram(f"Opened long {filled:.6f} {self.symbol} @ {price}")

    async def open_short(self, price: float) -> None:
        if not self.risk_manager.can_trade():
            logger.info("Risk limits prevent opening short position")
            return
        qty = self._round_qty(self.compute_trade_amount() / price)
        if qty < self.qty_step:
            logger.warning("Computed quantity %s below qty_step %s", qty, self.qty_step)
            return
        tp_price = price * (1 - self.tp_percent / 100)
        sl_price = price * (1 + self.sl_percent / 100)
        order = await self.place_order("sell", qty, tp=tp_price, sl=sl_price)
        filled = order.get("filledQty", qty) if order else qty
        self.position_price = price
        self.position_amount = -filled
        self.trailing_price = (
            price * (1 + self.trailing_percent / 100) if self.trailing_percent else None
        )
        self.log_trade("sell", price, filled, tp=tp_price, sl=sl_price)
        await self.send_telegram(f"Opened short {filled:.6f} {self.symbol} @ {price}")

    async def close_position(self, price: float) -> None:
        if self.position_amount == 0:
            return
        side = "sell" if self.position_amount > 0 else "buy"
        qty = abs(self.position_amount)
        try:
            order = await self.place_order(side, qty)
        except Exception as exc:
            logger.warning("Close position error: %s", exc)
            return
        if not order:
            logger.warning("Close order returned no data")
            return
        filled = order.get("filledQty", 0)
        if filled < qty:
            logger.warning("Partial close: %s/%s", filled, qty)
            pnl = (price - self.position_price) * (
                filled if self.position_amount > 0 else -filled
            )
            self.update_pnl(pnl)
            self.log_trade("close_partial", price, filled, pnl)
            await self.send_telegram(
                f"Partially closed {filled:.6f} {self.symbol} pnl={pnl:.2f}"
            )
            if self.position_amount > 0:
                self.position_amount -= filled
            else:
                self.position_amount += filled
            if self.position_amount == 0:
                self.position_price = None
                self.trailing_price = None
            return
        pnl = (price - self.position_price) * self.position_amount
        self.update_pnl(pnl)
        self.log_trade("close", price, qty, pnl)
        await self.send_telegram(f"Closed position {side} pnl={pnl:.2f}")
        self.position_amount = 0
        self.position_price = None
        self.trailing_price = None

    async def handle_trailing(self, price: float) -> bool:
        """Update trailing stop and close position if hit.

        Returns ``True`` if the trailing stop triggered and the position was
        closed, otherwise ``False``.
        """
        if self.trailing_price is None or self.position_amount == 0:
            return False

        if self.position_amount > 0:  # long position
            # move trailing price upwards as the market moves in our favour
            if (
                price > self.position_price
                and price * (1 - self.trailing_percent / 100) > self.trailing_price
            ):
                self.trailing_price = price * (1 - self.trailing_percent / 100)
            if price <= self.trailing_price:
                await self.close_position(price)
                return True
        else:  # short position
            if (
                price < self.position_price
                and price * (1 + self.trailing_percent / 100) < self.trailing_price
            ):
                self.trailing_price = price * (1 + self.trailing_percent / 100)
            if price >= self.trailing_price:
                await self.close_position(price)
                return True

        return False

    def log_trade(
        self,
        side: str,
        price: float,
        qty: float,
        pnl: float = 0.0,
        tp: Optional[float] = None,
        sl: Optional[float] = None,
    ) -> None:
        """Append trade information to the trade history file.

        If the file cannot be written (e.g. due to ``OSError``), the trade
        is buffered and a warning is logged. Buffered trades are retried on
        subsequent calls.
        """

        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')},{side},{price},{qty},{pnl},{tp},{sl}\n"
        records = self._trade_buffer + [line]
        try:
            header = not os.path.exists(self.trade_file)
            with open(self.trade_file, "a") as f:
                if header:
                    f.write("time,side,price,qty,pnl,tp,sl\n")
                for rec in records:
                    f.write(rec)
            self._trade_buffer.clear()
        except OSError as exc:
            logger.warning("Unable to write trade log: %s", exc)
            self._trade_buffer = records

    def reset_daily_pnl(self) -> None:
        self.risk_manager._reset_if_new_day()
        self.daily_pnl = self.risk_manager.daily_pnl
        self.daily_date = self.risk_manager.daily_date

    def update_pnl(self, profit: float) -> None:
        self.risk_manager.update_pnl(profit)
        self.daily_pnl = self.risk_manager.daily_pnl

    @property
    def daily_loss_limit(self) -> float:  # type: ignore[override]
        return self.risk_manager.daily_loss_limit

    @daily_loss_limit.setter
    def daily_loss_limit(self, value: float) -> None:
        self.risk_manager.daily_loss_limit = value

    @property
    def daily_profit_limit(self) -> float:  # type: ignore[override]
        return self.risk_manager.daily_profit_limit

    @daily_profit_limit.setter
    def daily_profit_limit(self, value: float) -> None:
        self.risk_manager.daily_profit_limit = value

    # ------------------------------------------------------------------
    async def trade_cycle(self, max_iterations: Optional[int] = None) -> None:
        use_websocket = os.getenv("USE_WEBSOCKET", "0") == "1"
        ws = None
        if use_websocket:
            from pybit.unified_trading import WebSocket

            def _ticker_cb(msg):
                data = msg.get("data")
                if isinstance(data, list):
                    data = data[0]
                lp = data.get("lastPrice") or data.get("lp")
                if lp:
                    self.last_price = float(lp)

            def _kline_cb(msg):
                data = msg.get("data")
                if isinstance(data, list):
                    data = data[0]
                vol = data.get("volume") or data.get("v")
                high = data.get("high") or data.get("h")
                low = data.get("low") or data.get("l")
                close = data.get("close") or data.get("c")
                if vol:
                    self.last_volume = float(vol)
                if high:
                    self.last_high = float(high)
                if low:
                    self.last_low = float(low)
                if close:
                    self.last_price = float(close)

            def _order_cb(msg):
                data = msg.get("data")
                if isinstance(data, list):
                    data = data[0]
                bids = data.get("b") or data.get("bid")
                asks = data.get("a") or data.get("ask")
                try:
                    if bids:
                        self.last_bid_qty = float(bids[0][1])
                    if asks:
                        self.last_ask_qty = float(asks[0][1])
                except Exception:
                    pass

            ws = WebSocket("spot")
            ws.ticker_stream(self.symbol, _ticker_cb)
            ws.kline_stream(self.symbol, "1", _kline_cb)
            ws.orderbook_stream(self.symbol, 1, _order_cb)

        iteration = 0
        try:
            while True:
                self.reset_daily_pnl()
                if not self.risk_manager.can_trade():
                    logger.warning("Daily trading limit reached")
                    break
                if use_websocket and None not in (
                    self.last_price,
                    self.last_high,
                    self.last_low,
                    self.last_volume,
                ):
                    price = self.last_price
                    high = self.last_high
                    low = self.last_low
                    vol = self.last_volume
                    bid = self.last_bid_qty
                    ask = self.last_ask_qty
                else:
                    price, high, low, vol, bid, ask = await asyncio.to_thread(
                        self.get_market_data
                    )
                features = await asyncio.to_thread(
                    self.update_model, price, high, low, vol, bid, ask
                )
                if features is None:
                    await asyncio.sleep(self.interval)
                    continue
                prob = self.model.predict_proba(features)[0][1]
                if hasattr(self.strategy, "decide_with_prob"):
                    action = self.strategy.decide_with_prob(features, price, prob)
                else:
                    action = self.strategy.decide(features, price)

                if self.position_amount > 0:  # long
                    if price >= self.position_price * (
                        1 + self.tp_percent / 100
                    ) or price <= self.position_price * (1 - self.sl_percent / 100):
                        await self.close_position(price)
                    else:
                        await self.handle_trailing(price)
                elif self.position_amount < 0:  # short
                    if price <= self.position_price * (
                        1 - self.tp_percent / 100
                    ) or price >= self.position_price * (1 + self.sl_percent / 100):
                        await self.close_position(price)
                    else:
                        await self.handle_trailing(price)
                if self.position_amount == 0:
                    if action == "long" or (
                        action is None and prob > self.long_threshold
                    ):
                        await self.open_long(price)
                    elif action == "short" or (
                        action is None and prob < self.short_threshold
                    ):
                        await self.open_short(price)
                await asyncio.sleep(self.interval)
                iteration += 1
                if max_iterations is not None and iteration >= max_iterations:
                    break
        except asyncio.CancelledError:
            logger.info("Trading loop cancelled")
            raise
        except KeyboardInterrupt:
            logger.info("Trading loop interrupted by user")
        except Exception as exc:
            logger.exception("Error in trading loop: %s", exc)
        finally:
            if use_websocket and ws is not None:
                try:
                    close = getattr(ws, "close", None) or getattr(ws, "exit", None)
                    if close:
                        close()
                except Exception as exc:
                    logger.warning("Failed to close WebSocket: %s", exc)
