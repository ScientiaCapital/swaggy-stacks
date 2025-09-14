"""
Consolidated API Models
Main entry point for all API models - imports from specialized modules
"""

# Import all enums
from .enums import (
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
    TradingAction,
    RiskLevel,
)

# Import all request models
from .request_models import (
    OrderRequest,
    MarketAnalysisRequest,
    RiskAssessmentRequest,
    PortfolioRequest,
    StrategySignalRequest,
    PositionCloseRequest,
    PaginationParams,
    TimeRangeParams,
    FilterParams,
)

# Import all response models
from .response_models import (
    OrderResponse,
    TradeResponse,
    PositionResponse,
    MarketAnalysisResponse,
    RiskAssessmentResponse,
    PortfolioResponse,
    StrategySignalResponse,
    HealthCheckResponse,
    ErrorResponse,
)

# Import base models for extension
from .base_models import (
    BaseSymbolModel,
    BaseTimestampedModel,
    BaseResponseModel,
    BaseMetricsModel,
    BasePaginationModel,
    BaseFilterModel,
    BaseTimeRangeModel,
    BaseErrorModel,
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
