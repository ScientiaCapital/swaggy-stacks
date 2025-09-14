"""
Application configuration settings
"""

import os
from typing import List, Union

from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # Project
    PROJECT_NAME: str = "Swaggy Stacks - Advanced Markov Trading System"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days

    # CORS
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://localhost:8000",
    ]

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    # Allowed hosts
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1", "0.0.0.0"]

    # Database
    POSTGRES_SERVER: str = os.getenv("POSTGRES_SERVER", "localhost")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "password")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "trading_system")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")

    # Alpaca API
    ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
    ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
    ALPACA_BASE_URL: str = os.getenv(
        "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
    )
    ALPACA_DATA_URL: str = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")

    # Trading Settings
    DEFAULT_POSITION_SIZE: float = 1000.0  # Default position size in dollars
    MAX_POSITION_SIZE: float = 10000.0  # Maximum position size
    MAX_DAILY_LOSS: float = 500.0  # Maximum daily loss limit
    RISK_FREE_RATE: float = 0.02  # Risk-free rate for Sharpe ratio

    # Market Data
    MARKET_DATA_UPDATE_INTERVAL: int = 1  # Seconds between market data updates
    MAX_SYMBOLS: int = 50  # Maximum number of symbols to track

    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090

    # Logging
    LOG_LEVEL: str = "INFO"

    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"

    # ML Features
    ML_FEATURES_ENABLED: bool = (
        os.getenv("ML_FEATURES_ENABLED", "false").lower() == "true"
    )
    EMBEDDING_SERVICE_TYPE: str = os.getenv(
        "EMBEDDING_SERVICE_TYPE", "auto"
    )  # "local", "mock", "auto"
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    USE_MPS_DEVICE: bool = os.getenv("USE_MPS_DEVICE", "true").lower() == "true"

    # Markov System Settings (Optional ML)
    MARKOV_LOOKBACK_PERIOD: int = int(os.getenv("MARKOV_LOOKBACK_PERIOD", "100"))
    MARKOV_N_STATES: int = int(os.getenv("MARKOV_N_STATES", "5"))

    # Email Notification Settings
    EMAIL_HOST: str = os.getenv("EMAIL_HOST", "smtp.gmail.com")
    EMAIL_PORT: int = int(os.getenv("EMAIL_PORT", "587"))
    EMAIL_USERNAME: str = os.getenv("EMAIL_USERNAME", "")
    EMAIL_PASSWORD: str = os.getenv("EMAIL_PASSWORD", "")
    EMAIL_FROM: str = os.getenv("EMAIL_FROM", "alerts@swaggy-stacks.com")
    EMAIL_USE_TLS: bool = os.getenv("EMAIL_USE_TLS", "true").lower() == "true"
    ALERT_EMAIL_TO: str = os.getenv("ALERT_EMAIL_TO", "tkipper@gmail.com")

    # SMS Notification Settings (Twilio) - Commented out (no Twilio account)
    # TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    # TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    # TWILIO_FROM_NUMBER: str = os.getenv("TWILIO_FROM_NUMBER", "")
    # ALERT_SMS_TO: str = os.getenv("ALERT_SMS_TO", "+14773535838")

    class Config:
        case_sensitive = True
        env_file = ".env"


# Global settings instance
settings = Settings()


def get_settings():
    """Get settings instance - for dependency injection"""
    return settings
