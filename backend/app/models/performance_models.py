"""
Performance Tracking Models
Models for tracking pattern and LLM performance metrics
"""

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
)

from .base_models import (
    BasePerformanceModel,
    MarketConditionsMixin,
    SymbolMixin,
)


class PatternPerformance(BasePerformanceModel, SymbolMixin, MarketConditionsMixin):
    """Track performance of trading patterns identified by LLMs"""

    __tablename__ = "pattern_performance"

    # Pattern identification
    pattern_type = Column(String(100), nullable=False, index=True)
    pattern_subtype = Column(String(100), nullable=True, index=True)
    pattern_signature = Column(String(500), nullable=False)

    # Pattern detection details
    timeframe = Column(String(20), nullable=False)
    detected_by_llm = Column(String(50), nullable=False, index=True)
    detection_confidence = Column(Float, nullable=False)
    detection_timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Pattern prediction and outcome
    predicted_direction = Column(String(10), nullable=False)
    predicted_magnitude = Column(Float, nullable=True)
    confidence_level = Column(Float, nullable=False)
    time_horizon = Column(String(20), nullable=False)

    # Actual outcome tracking
    outcome_verified = Column(Boolean, default=False)
    actual_direction = Column(String(10), nullable=True)
    actual_magnitude = Column(Float, nullable=True)
    verification_timestamp = Column(DateTime, nullable=True)

    # Performance scoring
    pattern_score = Column(Float, nullable=True)
    prediction_accuracy = Column(Float, nullable=True)
    magnitude_accuracy = Column(Float, nullable=True)
    win_rate = Column(Float, nullable=True)

    # Additional metadata
    technical_indicators = Column(JSON, nullable=True)
    market_context = Column(JSON, nullable=True)
    notes = Column(Text, nullable=True)

    # Indexes for performance
    __table_args__ = (
        Index(
            "idx_pattern_symbol_time", "pattern_type", "symbol", "detection_timestamp"
        ),
        Index(
            "idx_llm_performance", "detected_by_llm", "pattern_score", "alpha_generated"
        ),
        Index("idx_alpha_tracking", "alpha_generated", "verification_timestamp"),
        Index(
            "idx_pattern_success",
            "pattern_type",
            "prediction_accuracy",
            "alpha_generated",
        ),
    )


class LLMPerformanceMetrics(BasePerformanceModel):
    """Track LLM performance for alpha generation tasks"""

    __tablename__ = "llm_performance_metrics"

    # LLM identification
    llm_model = Column(String(50), nullable=False, index=True)
    task_type = Column(String(50), nullable=False, index=True)

    # Time period
    measurement_date = Column(DateTime, nullable=False, index=True)
    measurement_period = Column(String(20), nullable=False)

    # Performance metrics
    total_predictions = Column(Integer, default=0)
    successful_predictions = Column(Integer, default=0)

    # Alpha generation performance
    total_alpha_generated = Column(Float, default=0.0)
    avg_alpha_per_prediction = Column(Float, default=0.0)
    alpha_consistency = Column(Float, default=0.0)

    # Accuracy metrics
    direction_accuracy = Column(Float, default=0.0)
    magnitude_accuracy = Column(Float, default=0.0)
    confidence_calibration = Column(Float, default=0.0)

    # Risk metrics
    avg_max_drawdown = Column(Float, default=0.0)
    volatility_adjusted_return = Column(Float, default=0.0)
    avg_sharpe_ratio = Column(Float, default=0.0)

    # Operational metrics
    avg_execution_time = Column(Float, default=0.0)
    error_rate = Column(Float, default=0.0)
    availability = Column(Float, default=1.0)

    # Market condition performance
    bull_market_performance = Column(Float, nullable=True)
    bear_market_performance = Column(Float, nullable=True)
    volatile_market_performance = Column(Float, nullable=True)

    # Pattern specialization scores
    pattern_specialization = Column(JSON, nullable=True)

    # Indexes
    __table_args__ = (
        Index("idx_llm_task_date", "llm_model", "task_type", "measurement_date"),
        Index("idx_alpha_performance", "total_alpha_generated", "success_rate"),
        Index(
            "idx_routing_metrics",
            "llm_model",
            "task_type",
            "success_rate",
            "avg_execution_time",
        ),
    )
