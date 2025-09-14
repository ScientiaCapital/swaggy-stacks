"""
Base Database Models
Common patterns and mixins for database models
"""

from datetime import datetime
from sqlalchemy import Column, DateTime, String, Float, Integer, Boolean, JSON, Text, Index
from sqlalchemy.dialects.postgresql import UUID
import uuid

from app.core.database import Base


class TimestampMixin:
    """Mixin for created_at and updated_at timestamps"""

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class UUIDMixin:
    """Mixin for UUID primary key"""

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class PerformanceMetricsMixin:
    """Common performance tracking fields"""

    alpha_generated = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    success_rate = Column(Float, nullable=False, default=0.0)


class SymbolMixin:
    """Common symbol and market data fields"""

    symbol = Column(String(20), nullable=False, index=True)
    sector = Column(String(50), nullable=True)
    asset_class = Column(String(20), nullable=False, default='equity')


class LLMTrackingMixin:
    """Common LLM tracking fields"""

    llm_model = Column(String(50), nullable=False, index=True)
    confidence_score = Column(Float, nullable=False)
    generation_timestamp = Column(DateTime, default=datetime.utcnow, index=True)


class MarketConditionsMixin:
    """Common market conditions tracking"""

    market_volatility = Column(Float, nullable=True)
    market_regime = Column(String(20), nullable=True)  # bull, bear, sideways, volatile
    volume_profile = Column(String(20), nullable=True)  # high, medium, low
    market_conditions = Column(JSON, nullable=True)


class SignalMixin:
    """Common signal fields"""

    direction = Column(String(10), nullable=False)  # long, short, up, down
    strength = Column(Float, nullable=False)  # 0.0 to 1.0
    time_horizon = Column(String(20), nullable=False)  # short, medium, long
    reasoning = Column(Text, nullable=True)


class BasePerformanceModel(Base, UUIDMixin, TimestampMixin, PerformanceMetricsMixin):
    """Abstract base for performance tracking models"""

    __abstract__ = True


class BaseSignalModel(Base, UUIDMixin, TimestampMixin, SymbolMixin, LLMTrackingMixin, SignalMixin):
    """Abstract base for signal models"""

    __abstract__ = True


class BaseLearningModel(Base, UUIDMixin, TimestampMixin, PerformanceMetricsMixin):
    """Abstract base for learning/optimization models"""

    __abstract__ = True