from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# mypy: ignore-errors


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    api_key: str = Field("", alias="BYBIT_API_KEY")
    api_secret: str = Field("", alias="BYBIT_API_SECRET")
    symbol: str = Field("BTCUSDT", alias="SYMBOL")
    trade_amount: float = Field(10.0, alias="TRADE_AMOUNT_USDT")
    min_trade_amount: float = Field(1.0, alias="MIN_TRADE_AMOUNT_USDT")
    tp_percent: float = Field(1.5, alias="TP_PERCENT")
    sl_percent: float = Field(1.0, alias="SL_PERCENT")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    daily_loss_limit: float = Field(0.0, alias="DAILY_STOP_LOSS")
    daily_profit_limit: float = Field(0.0, alias="DAILY_TAKE_PROFIT")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
