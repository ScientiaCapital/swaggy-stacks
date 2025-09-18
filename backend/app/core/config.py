"""
Application configuration settings
"""

import os
from typing import List, Union

from pydantic import AnyHttpUrl, field_validator, ConfigDict
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
    # NATS Configuration - Ultra-low latency messaging
    NATS_URL: str = os.getenv("NATS_URL", "nats://localhost:4222")
    NATS_MONITORING_URL: str = os.getenv("NATS_MONITORING_URL", "http://localhost:8222")
    ENABLE_NATS_MESSAGING: bool = os.getenv("ENABLE_NATS_MESSAGING", "true").lower() == "true"
    # NATS Security Configuration
    NATS_SECURITY_ENABLED: bool = os.getenv("NATS_SECURITY_ENABLED", "true").lower() == "true"
    NATS_TLS_ENABLED: bool = os.getenv("NATS_TLS_ENABLED", "false").lower() == "true"
    NATS_TLS_CERT_FILE: str = os.getenv("NATS_TLS_CERT_FILE", "")
    NATS_TLS_KEY_FILE: str = os.getenv("NATS_TLS_KEY_FILE", "")
    NATS_TLS_CA_FILE: str = os.getenv("NATS_TLS_CA_FILE", "")
    NATS_TLS_VERIFY_HOSTNAME: bool = os.getenv("NATS_TLS_VERIFY_HOSTNAME", "true").lower() == "true"
    
    # NATS Performance Optimization Settings
    NATS_CONNECTION_POOL_SIZE: int = int(os.getenv("NATS_CONNECTION_POOL_SIZE", "5"))
    NATS_MAX_RECONNECT_ATTEMPTS: int = int(os.getenv("NATS_MAX_RECONNECT_ATTEMPTS", "60"))
    NATS_RECONNECT_TIME_WAIT: int = int(os.getenv("NATS_RECONNECT_TIME_WAIT", "2"))  # seconds
    NATS_MAX_OUTSTANDING_PINGS: int = int(os.getenv("NATS_MAX_OUTSTANDING_PINGS", "2"))
    NATS_PING_INTERVAL: int = int(os.getenv("NATS_PING_INTERVAL", "120"))  # seconds
    NATS_MAX_PAYLOAD: int = int(os.getenv("NATS_MAX_PAYLOAD", "1048576"))  # 1MB
    NATS_SEND_BUFFER_SIZE: int = int(os.getenv("NATS_SEND_BUFFER_SIZE", "2097152"))  # 2MB
    NATS_RECEIVE_BUFFER_SIZE: int = int(os.getenv("NATS_RECEIVE_BUFFER_SIZE", "2097152"))  # 2MB
    NATS_FLUSH_TIMEOUT: float = float(os.getenv("NATS_FLUSH_TIMEOUT", "1.0"))  # seconds
    
    # Message Batching Configuration
    NATS_MESSAGE_BATCH_SIZE: int = int(os.getenv("NATS_MESSAGE_BATCH_SIZE", "100"))
    NATS_BATCH_TIMEOUT: float = float(os.getenv("NATS_BATCH_TIMEOUT", "0.005"))  # 5ms
    NATS_HIGH_THROUGHPUT_MODE: bool = os.getenv("NATS_HIGH_THROUGHPUT_MODE", "true").lower() == "true"
    
    # NATS Authentication & Encryption
    NATS_MASTER_KEY: str = os.getenv("NATS_MASTER_KEY", "")
    NATS_JWT_SECRET: str = os.getenv("NATS_JWT_SECRET", "")
    NATS_JWT_EXPIRY_HOURS: int = int(os.getenv("NATS_JWT_EXPIRY_HOURS", "24"))
    
    # NATS Security Policies
    NATS_ENCRYPTION_ENABLED: bool = os.getenv("NATS_ENCRYPTION_ENABLED", "true").lower() == "true"
    NATS_MESSAGE_SIGNING_ENABLED: bool = os.getenv("NATS_MESSAGE_SIGNING_ENABLED", "true").lower() == "true"
    NATS_RATE_LIMITING_ENABLED: bool = os.getenv("NATS_RATE_LIMITING_ENABLED", "true").lower() == "true"
    
    # NATS Rate Limits (messages per minute by agent type)
    NATS_RATE_LIMIT_MARKET_ANALYST: int = int(os.getenv("NATS_RATE_LIMIT_MARKET_ANALYST", "1000"))
    NATS_RATE_LIMIT_RISK_ADVISOR: int = int(os.getenv("NATS_RATE_LIMIT_RISK_ADVISOR", "500"))
    NATS_RATE_LIMIT_STRATEGY_OPTIMIZER: int = int(os.getenv("NATS_RATE_LIMIT_STRATEGY_OPTIMIZER", "300"))
    NATS_RATE_LIMIT_PERFORMANCE_COACH: int = int(os.getenv("NATS_RATE_LIMIT_PERFORMANCE_COACH", "200"))
    NATS_RATE_LIMIT_DEFAULT: int = int(os.getenv("NATS_RATE_LIMIT_DEFAULT", "100"))
    
    # NATS Audit Configuration
    NATS_MAX_AUDIT_ENTRIES: int = int(os.getenv("NATS_MAX_AUDIT_ENTRIES", "10000"))

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

    # Streaming Configuration
    ALPACA_DATA_FEED: str = os.getenv("ALPACA_DATA_FEED", "iex")  # "iex" for free, "sip" for pro
    STREAMING_ENABLED: bool = os.getenv("STREAMING_ENABLED", "true").lower() == "true"
    STREAMING_BUFFER_SIZE: int = int(os.getenv("STREAMING_BUFFER_SIZE", "10000"))
    STREAMING_RECONNECT_ATTEMPTS: int = int(os.getenv("STREAMING_RECONNECT_ATTEMPTS", "10"))
    STREAMING_RECONNECT_DELAY: int = int(os.getenv("STREAMING_RECONNECT_DELAY", "5"))

    # Event Trigger Configuration
    EVENT_TRIGGERS_ENABLED: bool = os.getenv("EVENT_TRIGGERS_ENABLED", "true").lower() == "true"
    PRICE_MOVE_THRESHOLD: float = float(os.getenv("PRICE_MOVE_THRESHOLD", "2.0"))  # 2% price move
    VOLUME_SPIKE_THRESHOLD: float = float(os.getenv("VOLUME_SPIKE_THRESHOLD", "3.0"))  # 3x volume
    VOLATILITY_SPIKE_THRESHOLD: float = float(os.getenv("VOLATILITY_SPIKE_THRESHOLD", "20.0"))  # 20% IV spike

    # SMS Notification Settings (Twilio) - Commented out (no Twilio account)
    # TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    # TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    # TWILIO_FROM_NUMBER: str = os.getenv("TWILIO_FROM_NUMBER", "")
    # ALERT_SMS_TO: str = os.getenv("ALERT_SMS_TO", "+14773535838")

    model_config = ConfigDict(
        case_sensitive=True,
        env_file=".env",
        extra="ignore"  # Ignore extra environment variables
    )


# Global settings instance
settings = Settings()


def get_settings():
    """Get settings instance - for dependency injection"""
    return settings
