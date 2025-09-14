"""
Database models for indicator performance tracking and ML model management
"""

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base


class IndicatorPerformance(Base):
    """Track performance metrics for each indicator"""

    __tablename__ = "indicator_performance"

    id = Column(Integer, primary_key=True, index=True)
    indicator_name = Column(String(100), nullable=False, index=True)
    indicator_type = Column(String(20), nullable=False)  # TRADITIONAL, MODERN, LLM

    # Performance metrics
    total_signals = Column(Integer, default=0)
    correct_signals = Column(Integer, default=0)
    win_rate = Column(Numeric(5, 4), default=0.0)
    total_return = Column(Numeric(12, 4), default=0.0)
    sharpe_ratio = Column(Numeric(8, 4), default=0.0)
    max_drawdown = Column(Numeric(8, 4), default=0.0)
    avg_signal_strength = Column(Numeric(5, 4), default=0.0)

    # Market condition analysis
    market_condition = Column(
        String(20), nullable=True
    )  # volatile, trending_up, trending_down, sideways
    condition_win_rate = Column(Numeric(5, 4), nullable=True)
    condition_avg_return = Column(Numeric(10, 4), nullable=True)

    # Time period tracking
    period_start = Column(DateTime(timezone=True), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=False)

    # Parameters used
    parameters = Column(JSON, nullable=True)  # Store indicator-specific parameters
    optimization_version = Column(Integer, default=1)

    # Relationships
    backtest_run_id = Column(Integer, ForeignKey("backtest_runs.id"), nullable=True)
    backtest_run = relationship("BacktestRun", back_populates="indicator_performances")

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now()
    )

    # Indexes for performance
    __table_args__ = (
        Index("ix_indicator_performance_name_type", "indicator_name", "indicator_type"),
        Index("ix_indicator_performance_period", "period_start", "period_end"),
        Index("ix_indicator_performance_market_condition", "market_condition"),
        UniqueConstraint(
            "indicator_name",
            "indicator_type",
            "period_start",
            "period_end",
            "market_condition",
            name="uq_indicator_performance_unique",
        ),
    )

    def __repr__(self):
        return f"<IndicatorPerformance(indicator={self.indicator_name}, type={self.indicator_type}, win_rate={self.win_rate})>"

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage"""
        if self.total_signals > 0:
            return float(self.correct_signals / self.total_signals * 100)
        return 0.0

    @property
    def is_profitable(self) -> bool:
        """Check if indicator is profitable"""
        return self.total_return > 0 and self.win_rate > 0.5


class MLModelVersion(Base):
    """Track versions of ML models including LLM configurations"""

    __tablename__ = "ml_model_versions"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False, index=True)
    model_type = Column(
        String(50), nullable=False
    )  # LLM, LSTM, GRU, TRANSFORMER, ENSEMBLE

    # Version information
    version = Column(String(20), nullable=False)
    is_active = Column(Boolean, default=False)
    is_production = Column(Boolean, default=False)

    # Model configuration
    model_config = Column(
        JSON, nullable=False
    )  # Store model architecture, hyperparameters
    training_config = Column(
        JSON, nullable=True
    )  # Training parameters, data preprocessing

    # For LLM models
    llm_provider = Column(String(50), nullable=True)  # ollama, openai, anthropic
    llm_model_id = Column(
        String(100), nullable=True
    )  # qwen2.5, yi-6b, glm-4-9b, deepseek
    prompt_template = Column(Text, nullable=True)
    context_window = Column(Integer, nullable=True)

    # Performance metrics
    training_metrics = Column(JSON, nullable=True)  # Loss, accuracy, validation metrics
    validation_metrics = Column(JSON, nullable=True)
    production_metrics = Column(JSON, nullable=True)  # Real-world performance

    # Training data info
    training_start_date = Column(DateTime(timezone=True), nullable=True)
    training_end_date = Column(DateTime(timezone=True), nullable=True)
    training_samples = Column(Integer, nullable=True)
    features_used = Column(JSON, nullable=True)  # List of feature names

    # Deployment info
    deployed_at = Column(DateTime(timezone=True), nullable=True)
    retired_at = Column(DateTime(timezone=True), nullable=True)
    deployment_notes = Column(Text, nullable=True)

    # Relationships
    predictions = relationship("MLPrediction", back_populates="model_version")
    optimization_results = relationship(
        "ParameterOptimization", back_populates="model_version"
    )

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now()
    )

    # Indexes
    __table_args__ = (
        Index("ix_ml_model_versions_active", "is_active", "is_production"),
        Index("ix_ml_model_versions_model_type", "model_type"),
        UniqueConstraint("model_name", "version", name="uq_ml_model_version"),
    )

    def __repr__(self):
        return f"<MLModelVersion(name={self.model_name}, version={self.version}, active={self.is_active})>"


class MLPrediction(Base):
    """Track individual ML predictions for analysis and feedback learning"""

    __tablename__ = "ml_predictions"

    id = Column(Integer, primary_key=True, index=True)
    model_version_id = Column(
        Integer, ForeignKey("ml_model_versions.id"), nullable=False
    )

    # Prediction details
    symbol = Column(String(10), nullable=False, index=True)
    prediction_time = Column(DateTime(timezone=True), nullable=False, index=True)
    prediction_horizon = Column(Integer, nullable=False)  # Days ahead

    # Prediction values
    predicted_direction = Column(
        String(10), nullable=False
    )  # BULLISH, BEARISH, NEUTRAL
    predicted_return = Column(Numeric(10, 6), nullable=True)
    confidence_score = Column(Numeric(5, 4), nullable=False)

    # Ensemble details (for multi-model predictions)
    individual_predictions = Column(
        JSON, nullable=True
    )  # Store each model's prediction
    ensemble_weights = Column(JSON, nullable=True)

    # Market context at prediction
    market_conditions = Column(JSON, nullable=True)
    technical_indicators = Column(JSON, nullable=True)

    # Actual outcome (for learning)
    actual_direction = Column(String(10), nullable=True)
    actual_return = Column(Numeric(10, 6), nullable=True)
    outcome_recorded_at = Column(DateTime(timezone=True), nullable=True)
    prediction_error = Column(Numeric(10, 6), nullable=True)

    # Feedback learning
    was_correct = Column(Boolean, nullable=True)
    feedback_processed = Column(Boolean, default=False)
    learning_weight = Column(Numeric(5, 4), default=1.0)  # Weight for retraining

    # Relationships
    model_version = relationship("MLModelVersion", back_populates="predictions")
    signal_id = Column(Integer, ForeignKey("signal_history.id"), nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now()
    )

    # Indexes
    __table_args__ = (
        Index("ix_ml_predictions_symbol_time", "symbol", "prediction_time"),
        Index("ix_ml_predictions_outcome", "was_correct", "feedback_processed"),
    )

    def __repr__(self):
        return f"<MLPrediction(symbol={self.symbol}, direction={self.predicted_direction}, confidence={self.confidence_score})>"


class IndicatorParameters(Base):
    """Store optimal parameters for indicators under different market conditions"""

    __tablename__ = "indicator_parameters"

    id = Column(Integer, primary_key=True, index=True)
    indicator_name = Column(String(100), nullable=False, index=True)
    indicator_type = Column(String(20), nullable=False)

    # Market condition for these parameters
    market_condition = Column(
        String(20), nullable=True
    )  # Can be null for general parameters

    # Parameter values
    parameters = Column(JSON, nullable=False)  # {period: 14, multiplier: 2.0, etc.}
    parameter_hash = Column(String(64), nullable=False)  # Hash for quick lookup

    # Performance with these parameters
    backtest_performance = Column(JSON, nullable=True)  # Sharpe, return, drawdown, etc.
    live_performance = Column(JSON, nullable=True)

    # Optimization details
    optimization_method = Column(
        String(50), nullable=True
    )  # grid_search, bayesian, genetic
    optimization_metric = Column(
        String(50), nullable=True
    )  # sharpe_ratio, total_return, win_rate
    optimization_score = Column(Numeric(10, 6), nullable=True)

    # Usage tracking
    times_used = Column(Integer, default=0)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    is_default = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)

    # Version control
    version = Column(Integer, default=1)
    previous_version_id = Column(
        Integer, ForeignKey("indicator_parameters.id"), nullable=True
    )

    # Notes and metadata
    notes = Column(Text, nullable=True)
    metadata_json = Column(JSON, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now()
    )

    # Indexes
    __table_args__ = (
        Index(
            "ix_indicator_parameters_lookup",
            "indicator_name",
            "market_condition",
            "is_active",
        ),
        Index("ix_indicator_parameters_hash", "parameter_hash"),
        UniqueConstraint(
            "indicator_name",
            "market_condition",
            "parameter_hash",
            name="uq_indicator_parameters_unique",
        ),
    )

    def __repr__(self):
        return f"<IndicatorParameters(indicator={self.indicator_name}, condition={self.market_condition})>"


class ParameterOptimization(Base):
    """Track parameter optimization runs and results"""

    __tablename__ = "parameter_optimizations"

    id = Column(Integer, primary_key=True, index=True)
    optimization_id = Column(String(36), nullable=False, unique=True)  # UUID

    # Optimization target
    indicator_name = Column(
        String(100), nullable=True
    )  # Can optimize multiple indicators
    model_version_id = Column(
        Integer, ForeignKey("ml_model_versions.id"), nullable=True
    )
    optimization_type = Column(String(50), nullable=False)  # indicator, model, ensemble

    # Optimization configuration
    method = Column(
        String(50), nullable=False
    )  # grid_search, bayesian, genetic, random
    metric = Column(String(50), nullable=False)  # sharpe_ratio, total_return, win_rate
    parameter_space = Column(JSON, nullable=False)  # Search space definition
    constraints = Column(JSON, nullable=True)  # Any constraints on parameters

    # Execution details
    status = Column(
        String(20), nullable=False, default="PENDING"
    )  # PENDING, RUNNING, COMPLETED, FAILED
    progress = Column(Numeric(5, 2), default=0.0)
    iterations_completed = Column(Integer, default=0)
    total_iterations = Column(Integer, nullable=True)

    # Results
    best_parameters = Column(JSON, nullable=True)
    best_score = Column(Numeric(10, 6), nullable=True)
    all_results = Column(JSON, nullable=True)  # Top N results
    convergence_history = Column(JSON, nullable=True)  # Score over iterations

    # Performance metrics
    improvement_percentage = Column(Numeric(8, 4), nullable=True)  # vs baseline
    baseline_score = Column(Numeric(10, 6), nullable=True)

    # Time and resources
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    execution_time_seconds = Column(Integer, nullable=True)

    # Market context
    market_condition = Column(String(20), nullable=True)
    backtest_period_start = Column(DateTime(timezone=True), nullable=True)
    backtest_period_end = Column(DateTime(timezone=True), nullable=True)

    # Error handling
    error_message = Column(Text, nullable=True)
    warnings = Column(JSON, nullable=True)

    # Relationships
    model_version = relationship(
        "MLModelVersion", back_populates="optimization_results"
    )
    created_parameters = relationship(
        "IndicatorParameters",
        foreign_keys="IndicatorParameters.previous_version_id",
        backref="optimization_run",
    )

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now()
    )

    # Indexes
    __table_args__ = (
        Index("ix_parameter_optimizations_status", "status", "optimization_type"),
        Index("ix_parameter_optimizations_indicator", "indicator_name"),
    )

    def __repr__(self):
        return f"<ParameterOptimization(id={self.optimization_id}, status={self.status}, best_score={self.best_score})>"

    @property
    def duration_minutes(self) -> float:
        """Calculate optimization duration in minutes"""
        if self.execution_time_seconds:
            return self.execution_time_seconds / 60.0
        return 0.0

    @property
    def is_completed(self) -> bool:
        """Check if optimization is completed"""
        return self.status == "COMPLETED"

    @property
    def has_improvement(self) -> bool:
        """Check if optimization improved the baseline"""
        if self.improvement_percentage:
            return self.improvement_percentage > 0
        return False
