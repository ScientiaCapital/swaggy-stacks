"""
API Response Models
Response models for all endpoints
"""

from typing import Any, Dict, List, Optional

from pydantic import Field

from .base_models import (
    BaseErrorModel,
    BaseMetricsModel,
    BaseResponseModel,
    BaseTimestampedModel,
)
from .enums import OrderSide, OrderStatus, OrderType, RiskLevel, TradingAction


class OrderResponse(BaseTimestampedModel):
    """Common order response model"""

    order_id: str = Field(..., description="Unique order identifier")
    symbol: str = Field(..., description="Stock symbol")
    quantity: float = Field(..., description="Order quantity")
    side: OrderSide = Field(..., description="Order side")
    status: OrderStatus = Field(..., description="Order status")
    order_type: OrderType = Field(..., description="Order type")
    submitted_at: str = Field(..., description="Submission timestamp")
    filled_at: Optional[str] = Field(None, description="Fill timestamp")
    filled_price: Optional[float] = Field(None, description="Average fill price")
    filled_quantity: Optional[float] = Field(None, description="Filled quantity")
    message: Optional[str] = Field(None, description="Status message")


class TradeResponse(BaseResponseModel):
    """Trade information response"""

    id: int = Field(..., description="Trade ID")
    symbol: str = Field(..., description="Stock symbol")
    quantity: float = Field(..., description="Trade quantity")
    side: OrderSide = Field(..., description="Trade side")
    entry_price: float = Field(..., description="Entry price")
    exit_price: Optional[float] = Field(None, description="Exit price")
    entry_time: str = Field(..., description="Entry timestamp")
    exit_time: Optional[str] = Field(None, description="Exit timestamp")
    pnl: Optional[float] = Field(None, description="Profit/Loss")
    status: str = Field(..., description="Trade status")
    strategy: Optional[str] = Field(None, description="Strategy used")


class PositionResponse(BaseResponseModel):
    """Position information response"""

    symbol: str = Field(..., description="Stock symbol")
    quantity: float = Field(..., description="Position size")
    side: str = Field(..., description="Position side (long/short)")
    entry_price: float = Field(..., description="Average entry price")
    current_price: float = Field(..., description="Current market price")
    unrealized_pnl: float = Field(..., description="Unrealized P&L")
    unrealized_pnl_pct: float = Field(..., description="Unrealized P&L percentage")
    market_value: float = Field(..., description="Current market value")
    cost_basis: float = Field(..., description="Cost basis")


class MarketAnalysisResponse(BaseMetricsModel, BaseResponseModel):
    """Market analysis response model"""

    symbol: str = Field(..., description="Analyzed symbol")
    sentiment: str = Field(..., description="Market sentiment")
    action: TradingAction = Field(..., description="Recommended action")
    entry_price: Optional[float] = Field(None, description="Suggested entry price")
    stop_loss: Optional[float] = Field(None, description="Suggested stop loss")
    take_profit: Optional[float] = Field(None, description="Suggested take profit")
    key_factors: List[str] = Field(
        default_factory=list, description="Key analysis factors"
    )
    technical_indicators: Dict[str, Any] = Field(
        default_factory=dict, description="Technical indicator values"
    )
    strategy_signals: Dict[str, Any] = Field(
        default_factory=dict, description="Individual strategy signals"
    )
    risk_level: RiskLevel = Field(..., description="Risk assessment")


class RiskAssessmentResponse(BaseResponseModel):
    """Risk assessment response model"""

    symbol: str = Field(..., description="Assessed symbol")
    risk_level: RiskLevel = Field(..., description="Overall risk level")
    portfolio_heat: float = Field(
        ..., ge=0, le=1, description="Portfolio heat percentage"
    )
    recommended_position_size: float = Field(
        ..., description="Recommended position size"
    )
    max_position_risk: float = Field(..., description="Maximum position risk")
    position_concentration: float = Field(
        ..., description="Position concentration risk"
    )
    correlation_risk: float = Field(
        ..., description="Correlation risk with other positions"
    )
    volatility_risk: float = Field(..., description="Volatility-based risk")
    liquidity_risk: float = Field(..., description="Liquidity risk assessment")
    key_risk_factors: List[str] = Field(
        default_factory=list, description="Key risk factors identified"
    )
    mitigation_strategies: List[str] = Field(
        default_factory=list, description="Risk mitigation suggestions"
    )
    exit_conditions: List[str] = Field(
        default_factory=list, description="Recommended exit conditions"
    )


class PortfolioResponse(BaseResponseModel):
    """Portfolio status response model"""

    account: Dict[str, Any] = Field(..., description="Account information")
    positions: List[PositionResponse] = Field(
        default_factory=list, description="Current positions"
    )
    performance: Dict[str, Any] = Field(
        default_factory=dict, description="Performance metrics"
    )
    risk_metrics: Dict[str, Any] = Field(
        default_factory=dict, description="Risk metrics"
    )
    summary: Dict[str, Any] = Field(
        default_factory=dict, description="Portfolio summary"
    )


class StrategySignalResponse(BaseMetricsModel, BaseResponseModel):
    """Strategy signal response model"""

    symbol: str = Field(..., description="Symbol analyzed")
    action: TradingAction = Field(..., description="Recommended action")
    entry_price: Optional[float] = Field(None, description="Entry price")
    stop_loss: Optional[float] = Field(None, description="Stop loss price")
    take_profit: Optional[float] = Field(None, description="Take profit price")
    position_size: Optional[float] = Field(
        None, description="Recommended position size"
    )
    strategy_breakdown: Dict[str, Any] = Field(
        default_factory=dict, description="Individual strategy results"
    )
    consensus_method: str = Field(..., description="Method used for consensus")
    risk_reward_ratio: Optional[float] = Field(None, description="Risk-reward ratio")
    expected_return: Optional[float] = Field(None, description="Expected return")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class HealthCheckResponse(BaseResponseModel):
    """Health check response model"""

    status: str = Field(..., description="Overall system status")
    components: Dict[str, str] = Field(
        default_factory=dict, description="Component statuses"
    )
    uptime: Optional[float] = Field(None, description="System uptime in seconds")
    version: Optional[str] = Field(None, description="API version")


class ErrorResponse(BaseErrorModel):
    """Standard error response model"""
