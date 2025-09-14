"""
Strategy model for trading strategies
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


class Strategy(Base):
    """Strategy model for trading strategies"""

    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    strategy_type = Column(
        String(50), nullable=False
    )  # MARKOV, FIBONACCI, ELLIOTT_WAVE, WYCKOFF

    # Strategy parameters
    parameters = Column(Text, nullable=True)  # JSON string of strategy parameters
    is_active = Column(Boolean, default=True)

    # Performance metrics
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    total_pnl = Column(Numeric(15, 2), default=0)
    max_drawdown = Column(Numeric(10, 4), default=0)
    sharpe_ratio = Column(Numeric(10, 4), default=0)
    win_rate = Column(Numeric(5, 2), default=0)

    # Risk management
    max_position_size = Column(Numeric(10, 2), nullable=True)
    stop_loss_percentage = Column(Numeric(5, 2), nullable=True)
    take_profit_percentage = Column(Numeric(5, 2), nullable=True)

    # User association
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now()
    )

    # Relationships
    user = relationship("User", back_populates="strategies")
    trades = relationship("Trade", back_populates="strategy")
    backtest_runs = relationship("BacktestRun", back_populates="strategy", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Strategy(id={self.id}, name={self.name}, type={self.strategy_type})>"

    @property
    def profit_factor(self) -> float:
        """Calculate profit factor"""
        if self.losing_trades == 0:
            return float("inf") if self.winning_trades > 0 else 0
        return (
            float(self.winning_trades / self.losing_trades)
            if self.losing_trades > 0
            else 0
        )

    @property
    def average_win(self) -> float:
        """Calculate average win amount"""
        return (
            float(self.total_pnl / self.winning_trades)
            if self.winning_trades > 0
            else 0
        )

    @property
    def average_loss(self) -> float:
        """Calculate average loss amount"""
        return (
            float(self.total_pnl / self.losing_trades) if self.losing_trades > 0 else 0
        )
