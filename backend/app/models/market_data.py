"""
Market data models for storing price and volume data
"""

from sqlalchemy import Boolean, Column, DateTime, Index, Integer, Numeric, String
from sqlalchemy.sql import func

from app.core.database import Base


class MarketData(Base):
    """Market data model for storing OHLCV data"""

    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)

    # OHLCV data
    open_price = Column(Numeric(10, 4), nullable=False)
    high_price = Column(Numeric(10, 4), nullable=False)
    low_price = Column(Numeric(10, 4), nullable=False)
    close_price = Column(Numeric(10, 4), nullable=False)
    volume = Column(Integer, nullable=False)

    # Additional data
    vwap = Column(Numeric(10, 4), nullable=True)  # Volume Weighted Average Price
    trade_count = Column(Integer, nullable=True)

    # Data quality flags
    is_complete = Column(Boolean, default=True)
    data_source = Column(String(50), default="ALPACA")

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=func.now())

    # Composite index for efficient queries
    __table_args__ = (Index("ix_market_data_symbol_timestamp", "symbol", "timestamp"),)

    def __repr__(self):
        return f"<MarketData(symbol={self.symbol}, timestamp={self.timestamp}, close={self.close_price})>"


class TechnicalIndicator(Base):
    """Technical indicator model for storing calculated indicators"""

    __tablename__ = "technical_indicators"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    indicator_type = Column(String(50), nullable=False)  # RSI, MACD, SMA, etc.

    # Indicator values
    value = Column(Numeric(15, 6), nullable=False)
    signal = Column(String(20), nullable=True)  # BUY, SELL, HOLD

    # Parameters used for calculation
    parameters = Column(String(255), nullable=True)  # JSON string of parameters

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=func.now())

    # Composite index for efficient queries
    __table_args__ = (
        Index(
            "ix_technical_indicators_symbol_timestamp_type",
            "symbol",
            "timestamp",
            "indicator_type",
        ),
    )

    def __repr__(self):
        return f"<TechnicalIndicator(symbol={self.symbol}, type={self.indicator_type}, value={self.value})>"
