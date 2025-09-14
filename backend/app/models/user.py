"""
User model for authentication and user management
"""

from sqlalchemy import Boolean, Column, DateTime, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base


class User(Base):
    """User model for authentication and user management"""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(50), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=True)

    # User status
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    is_superuser = Column(Boolean, default=False)

    # Trading preferences
    default_position_size = Column(
        Integer, default=1000
    )  # Default position size in dollars
    max_position_size = Column(Integer, default=10000)  # Maximum position size
    max_daily_loss = Column(Integer, default=500)  # Maximum daily loss limit
    risk_tolerance = Column(String(20), default="MODERATE")  # LOW, MODERATE, HIGH

    # Alpaca account info
    alpaca_api_key = Column(String(255), nullable=True)
    alpaca_secret_key = Column(String(255), nullable=True)
    alpaca_account_id = Column(String(50), nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now()
    )
    last_login = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    trades = relationship("Trade", back_populates="user")
    strategies = relationship("Strategy", back_populates="user")
    backtest_runs = relationship(
        "BacktestRun", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, email={self.email})>"

    @property
    def is_trading_enabled(self) -> bool:
        """Check if user has trading enabled (has Alpaca credentials)"""
        return bool(self.alpaca_api_key and self.alpaca_secret_key)
