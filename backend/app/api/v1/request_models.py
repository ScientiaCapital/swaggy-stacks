"""
API Request Models
Request models for all endpoints
"""

from typing import Any, Dict, List, Optional

from pydantic import Field, validator

from .base_models import (
    BaseFilterModel,
    BasePaginationModel,
    BaseSymbolModel,
    BaseTimeRangeModel,
)
from .enums import OrderSide, OrderType, TimeInForce


class OrderRequest(BaseSymbolModel):
    """Common order request model"""

    quantity: float = Field(..., gt=0, description="Number of shares to trade")
    side: OrderSide = Field(..., description="Order side (BUY or SELL)")
    order_type: OrderType = Field(OrderType.MARKET, description="Type of order")
    time_in_force: TimeInForce = Field(TimeInForce.GTC, description="Time in force")
    limit_price: Optional[float] = Field(
        None, gt=0, description="Limit price for limit orders"
    )
    stop_price: Optional[float] = Field(
        None, gt=0, description="Stop price for stop orders"
    )
    strategy_id: Optional[int] = Field(None, description="Associated strategy ID")

    @validator("limit_price", "stop_price")
    def validate_conditional_prices(cls, v, values):
        return v


class MarketAnalysisRequest(BaseSymbolModel):
    """Market analysis request model"""

    timeframe: Optional[str] = Field("1D", description="Analysis timeframe")
    lookback_periods: Optional[int] = Field(
        100, ge=10, le=500, description="Number of periods to analyze"
    )
    strategies: Optional[List[str]] = Field(
        None, description="Specific strategies to use"
    )
    context: Optional[str] = Field(None, description="Additional context for analysis")


class RiskAssessmentRequest(BaseSymbolModel):
    """Risk assessment request model"""

    position_size: float = Field(..., description="Proposed position size")
    account_value: float = Field(..., gt=0, description="Total account value")
    current_positions: List[Dict[str, Any]] = Field(
        default_factory=list, description="Current positions"
    )
    proposed_trade: Dict[str, Any] = Field(
        default_factory=dict, description="Proposed trade details"
    )


class PortfolioRequest(BasePaginationModel):
    """Portfolio information request"""

    include_positions: bool = Field(True, description="Include position details")
    include_performance: bool = Field(True, description="Include performance metrics")
    include_risk_metrics: bool = Field(True, description="Include risk analysis")


class StrategySignalRequest(BaseSymbolModel):
    """Strategy signal generation request"""

    strategies: List[str] = Field(default_factory=list, description="Strategies to use")
    market_data: Dict[str, Any] = Field(
        default_factory=dict, description="Market data for analysis"
    )
    technical_indicators: Dict[str, Any] = Field(
        default_factory=dict, description="Technical indicators"
    )


class PositionCloseRequest(BaseSymbolModel):
    """Position closing request"""

    quantity: Optional[float] = Field(
        None, gt=0, description="Quantity to close (None = full position)"
    )
    order_type: OrderType = Field(
        OrderType.MARKET, description="Order type for closing"
    )


# Utility request models
class PaginationParams(BasePaginationModel):
    """Pagination parameters"""


class TimeRangeParams(BaseTimeRangeModel):
    """Time range parameters"""


class FilterParams(BaseFilterModel):
    """Common filter parameters"""
