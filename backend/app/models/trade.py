"""
Trade model for the trading system
"""

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base


class Trade(Base):
    """Trade model representing individual trades"""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    quantity = Column(Numeric(10, 2), nullable=False)
    entry_price = Column(Numeric(10, 4), nullable=False)
    exit_price = Column(Numeric(10, 4), nullable=True)
    entry_time = Column(DateTime(timezone=True), nullable=False, default=func.now())
    exit_time = Column(DateTime(timezone=True), nullable=True)
    pnl = Column(Numeric(10, 2), nullable=True)
    status = Column(
        String(20), nullable=False, default="OPEN"
    )  # OPEN, CLOSED, CANCELLED
    side = Column(String(10), nullable=False)  # BUY, SELL
    order_type = Column(
        String(20), nullable=False, default="MARKET"
    )  # MARKET, LIMIT, STOP
    time_in_force = Column(String(10), nullable=False, default="GTC")  # GTC, IOC, FOK
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Alpaca specific fields
    alpaca_order_id = Column(String(50), nullable=True, unique=True)
    alpaca_client_order_id = Column(String(50), nullable=True)

    # Risk management fields
    stop_loss = Column(Numeric(10, 4), nullable=True)
    take_profit = Column(Numeric(10, 4), nullable=True)
    max_loss = Column(Numeric(10, 2), nullable=True)

    # Metadata
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now()
    )

    # Relationships
    strategy = relationship("Strategy", back_populates="trades")
    user = relationship("User", back_populates="trades")

    def __repr__(self):
        return f"<Trade(id={self.id}, symbol={self.symbol}, side={self.side}, status={self.status})>"

    @property
    def is_open(self) -> bool:
        """Check if trade is open"""
        return self.status == "OPEN"

    @property
    def is_closed(self) -> bool:
        """Check if trade is closed"""
        return self.status == "CLOSED"

    @property
    def current_pnl(self) -> float:
        """Calculate current P&L if trade is open"""
        if self.is_closed:
            return float(self.pnl or 0)
        # For open trades, would need current market price
        return 0.0
