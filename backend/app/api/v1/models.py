"""
Shared API Models
Consolidates Pydantic models used across all API endpoints to eliminate redundancy
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class OrderSide(str, Enum):
    """Order side enumeration"""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    """Order status enumeration"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class TimeInForce(str, Enum):
    """Time in force enumeration"""
    GTC = "gtc"  # Good till canceled
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill
    DAY = "day"  # Day order


class TradingAction(str, Enum):
    """Trading action enumeration"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class RiskLevel(str, Enum):
    """Risk level enumeration"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ============================================================================
# REQUEST MODELS
# ============================================================================

class OrderRequest(BaseModel):
    """Common order request model"""
    symbol: str = Field(..., description="Stock symbol to trade")
    quantity: float = Field(..., gt=0, description="Number of shares to trade")
    side: OrderSide = Field(..., description="Order side (BUY or SELL)")
    order_type: OrderType = Field(OrderType.MARKET, description="Type of order")
    time_in_force: TimeInForce = Field(TimeInForce.GTC, description="Time in force")
    limit_price: Optional[float] = Field(None, gt=0, description="Limit price for limit orders")
    stop_price: Optional[float] = Field(None, gt=0, description="Stop price for stop orders")
    strategy_id: Optional[int] = Field(None, description="Associated strategy ID")
    
    @validator('symbol')
    def symbol_must_be_uppercase(cls, v):
        return v.strip().upper() if isinstance(v, str) else v
    
    @validator('limit_price', 'stop_price')
    def validate_conditional_prices(cls, v, values):
        # Additional validation can be added here
        return v


class MarketAnalysisRequest(BaseModel):
    """Market analysis request model"""
    symbol: str = Field(..., description="Stock symbol to analyze")
    timeframe: Optional[str] = Field("1D", description="Analysis timeframe")
    lookback_periods: Optional[int] = Field(100, ge=10, le=500, description="Number of periods to analyze")
    strategies: Optional[List[str]] = Field(None, description="Specific strategies to use")
    context: Optional[str] = Field(None, description="Additional context for analysis")


class RiskAssessmentRequest(BaseModel):
    """Risk assessment request model"""
    symbol: str = Field(..., description="Symbol to assess")
    position_size: float = Field(..., description="Proposed position size")
    account_value: float = Field(..., gt=0, description="Total account value")
    current_positions: List[Dict[str, Any]] = Field(default_factory=list, description="Current positions")
    proposed_trade: Dict[str, Any] = Field(default_factory=dict, description="Proposed trade details")
    
    @validator('symbol')
    def symbol_must_be_uppercase(cls, v):
        return v.strip().upper() if isinstance(v, str) else v


class PortfolioRequest(BaseModel):
    """Portfolio information request"""
    include_positions: bool = Field(True, description="Include position details")
    include_performance: bool = Field(True, description="Include performance metrics")
    include_risk_metrics: bool = Field(True, description="Include risk analysis")


class StrategySignalRequest(BaseModel):
    """Strategy signal generation request"""
    symbol: str = Field(..., description="Symbol to analyze")
    strategies: List[str] = Field(default_factory=list, description="Strategies to use")
    market_data: Dict[str, Any] = Field(default_factory=dict, description="Market data for analysis")
    technical_indicators: Dict[str, Any] = Field(default_factory=dict, description="Technical indicators")
    
    @validator('symbol')
    def symbol_must_be_uppercase(cls, v):
        return v.strip().upper() if isinstance(v, str) else v


class PositionCloseRequest(BaseModel):
    """Position closing request"""
    symbol: str = Field(..., description="Symbol to close position for")
    quantity: Optional[float] = Field(None, gt=0, description="Quantity to close (None = full position)")
    order_type: OrderType = Field(OrderType.MARKET, description="Order type for closing")
    
    @validator('symbol')
    def symbol_must_be_uppercase(cls, v):
        return v.strip().upper() if isinstance(v, str) else v


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class OrderResponse(BaseModel):
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


class TradeResponse(BaseModel):
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


class PositionResponse(BaseModel):
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


class MarketAnalysisResponse(BaseModel):
    """Market analysis response model"""
    symbol: str = Field(..., description="Analyzed symbol")
    sentiment: str = Field(..., description="Market sentiment")
    confidence: float = Field(..., ge=0, le=1, description="Analysis confidence")
    action: TradingAction = Field(..., description="Recommended action")
    entry_price: Optional[float] = Field(None, description="Suggested entry price")
    stop_loss: Optional[float] = Field(None, description="Suggested stop loss")
    take_profit: Optional[float] = Field(None, description="Suggested take profit")
    key_factors: List[str] = Field(default_factory=list, description="Key analysis factors")
    technical_indicators: Dict[str, Any] = Field(default_factory=dict, description="Technical indicator values")
    strategy_signals: Dict[str, Any] = Field(default_factory=dict, description="Individual strategy signals")
    risk_level: RiskLevel = Field(..., description="Risk assessment")
    reasoning: str = Field(..., description="Analysis reasoning")
    timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")


class RiskAssessmentResponse(BaseModel):
    """Risk assessment response model"""
    symbol: str = Field(..., description="Assessed symbol")
    risk_level: RiskLevel = Field(..., description="Overall risk level")
    portfolio_heat: float = Field(..., ge=0, le=1, description="Portfolio heat percentage")
    recommended_position_size: float = Field(..., description="Recommended position size")
    max_position_risk: float = Field(..., description="Maximum position risk")
    position_concentration: float = Field(..., description="Position concentration risk")
    correlation_risk: float = Field(..., description="Correlation risk with other positions")
    volatility_risk: float = Field(..., description="Volatility-based risk")
    liquidity_risk: float = Field(..., description="Liquidity risk assessment")
    key_risk_factors: List[str] = Field(default_factory=list, description="Key risk factors identified")
    mitigation_strategies: List[str] = Field(default_factory=list, description="Risk mitigation suggestions")
    exit_conditions: List[str] = Field(default_factory=list, description="Recommended exit conditions")
    timestamp: datetime = Field(default_factory=datetime.now, description="Assessment timestamp")


class PortfolioResponse(BaseModel):
    """Portfolio status response model"""
    account: Dict[str, Any] = Field(..., description="Account information")
    positions: List[PositionResponse] = Field(default_factory=list, description="Current positions")
    performance: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    risk_metrics: Dict[str, Any] = Field(default_factory=dict, description="Risk metrics")
    summary: Dict[str, Any] = Field(default_factory=dict, description="Portfolio summary")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")


class StrategySignalResponse(BaseModel):
    """Strategy signal response model"""
    symbol: str = Field(..., description="Symbol analyzed")
    action: TradingAction = Field(..., description="Recommended action")
    confidence: float = Field(..., ge=0, le=1, description="Signal confidence")
    entry_price: Optional[float] = Field(None, description="Entry price")
    stop_loss: Optional[float] = Field(None, description="Stop loss price")
    take_profit: Optional[float] = Field(None, description="Take profit price")
    position_size: Optional[float] = Field(None, description="Recommended position size")
    strategy_breakdown: Dict[str, Any] = Field(default_factory=dict, description="Individual strategy results")
    consensus_method: str = Field(..., description="Method used for consensus")
    risk_reward_ratio: Optional[float] = Field(None, description="Risk-reward ratio")
    expected_return: Optional[float] = Field(None, description="Expected return")
    reasoning: str = Field(..., description="Signal reasoning")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: datetime = Field(default_factory=datetime.now, description="Signal timestamp")


class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Overall system status")
    components: Dict[str, str] = Field(default_factory=dict, description="Component statuses")
    uptime: Optional[float] = Field(None, description="System uptime in seconds")
    version: Optional[str] = Field(None, description="API version")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")


class ErrorResponse(BaseModel):
    """Standard error response model"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    code: Optional[str] = Field(None, description="Error code")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")


# ============================================================================
# UTILITY MODELS
# ============================================================================

class PaginationParams(BaseModel):
    """Pagination parameters"""
    page: int = Field(1, ge=1, description="Page number")
    per_page: int = Field(20, ge=1, le=100, description="Items per page")


class TimeRangeParams(BaseModel):
    """Time range parameters"""
    start_date: Optional[datetime] = Field(None, description="Start date")
    end_date: Optional[datetime] = Field(None, description="End date")
    
    @validator('end_date')
    def end_date_must_be_after_start_date(cls, v, values):
        if v and 'start_date' in values and values['start_date']:
            if v <= values['start_date']:
                raise ValueError('end_date must be after start_date')
        return v


class FilterParams(BaseModel):
    """Common filter parameters"""
    symbol: Optional[str] = Field(None, description="Filter by symbol")
    strategy: Optional[str] = Field(None, description="Filter by strategy")
    status: Optional[str] = Field(None, description="Filter by status")
    
    @validator('symbol')
    def symbol_must_be_uppercase(cls, v):
        return v.strip().upper() if isinstance(v, str) and v else v


# Export all models for easy importing
__all__ = [
    # Enums
    'OrderSide', 'OrderType', 'OrderStatus', 'TimeInForce', 'TradingAction', 'RiskLevel',
    
    # Request models
    'OrderRequest', 'MarketAnalysisRequest', 'RiskAssessmentRequest', 'PortfolioRequest',
    'StrategySignalRequest', 'PositionCloseRequest',
    
    # Response models
    'OrderResponse', 'TradeResponse', 'PositionResponse', 'MarketAnalysisResponse',
    'RiskAssessmentResponse', 'PortfolioResponse', 'StrategySignalResponse',
    'HealthCheckResponse', 'ErrorResponse',
    
    # Utility models
    'PaginationParams', 'TimeRangeParams', 'FilterParams'
]