"""
Consolidated API Models
Main entry point for all API models - imports from specialized modules
"""

# Import base models for extension
from .base_models import (
    BaseErrorModel,
    BaseFilterModel,
    BaseMetricsModel,
    BasePaginationModel,
    BaseResponseModel,
    BaseSymbolModel,
    BaseTimeRangeModel,
    BaseTimestampedModel,
)

# Import all enums
from .enums import (
    OrderSide,
    OrderStatus,
    OrderType,
    RiskLevel,
    TimeInForce,
    TradingAction,
)

# Import all request models
from .request_models import (
    FilterParams,
    MarketAnalysisRequest,
    OrderRequest,
    PaginationParams,
    PortfolioRequest,
    PositionCloseRequest,
    RiskAssessmentRequest,
    StrategySignalRequest,
    TimeRangeParams,
)

# Import all response models
from .response_models import (
    ErrorResponse,
    HealthCheckResponse,
    MarketAnalysisResponse,
    OrderResponse,
    PortfolioResponse,
    PositionResponse,
    RiskAssessmentResponse,
    StrategySignalResponse,
    TradeResponse,
)

# Export all models for easy importing
__all__ = [
    # Enums
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "TimeInForce",
    "TradingAction",
    "RiskLevel",
    # Request models
    "OrderRequest",
    "MarketAnalysisRequest",
    "RiskAssessmentRequest",
    "PortfolioRequest",
    "StrategySignalRequest",
    "PositionCloseRequest",
    # Response models
    "OrderResponse",
    "TradeResponse",
    "PositionResponse",
    "MarketAnalysisResponse",
    "RiskAssessmentResponse",
    "PortfolioResponse",
    "StrategySignalResponse",
    "HealthCheckResponse",
    "ErrorResponse",
    # Utility models
    "PaginationParams",
    "TimeRangeParams",
    "FilterParams",
    # Base models for extension
    "BaseSymbolModel",
    "BaseTimestampedModel",
    "BaseResponseModel",
    "BaseMetricsModel",
    "BasePaginationModel",
    "BaseFilterModel",
    "BaseTimeRangeModel",
    "BaseErrorModel",
]
