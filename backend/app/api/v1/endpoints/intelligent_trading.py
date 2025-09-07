"""
Intelligent Trading API endpoints
Chinese LLM routing system for advanced trading decisions
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session

from app.api.v1.dependencies import get_current_user
from app.core.database import get_db
from app.models.user import User
from app.ai.trading_intelligence_hub import (
    TradingIntelligenceHub,
    TradingIntelligenceRequest,
    TradingIntelligenceResponse
)
from app.ai.deepseek_trade_orchestrator import TaskType
from app.ai.llm_router import RoutingStrategy
from app.trading.alpaca_client import AlpacaClient

logger = structlog.get_logger()
router = APIRouter()

# Global intelligence hub instance (would be better injected via DI)
intelligence_hub: Optional[TradingIntelligenceHub] = None


def get_intelligence_hub() -> TradingIntelligenceHub:
    """Get or create intelligence hub instance"""
    global intelligence_hub
    if intelligence_hub is None:
        intelligence_hub = TradingIntelligenceHub(
            routing_strategy=RoutingStrategy.ADAPTIVE,
            enable_ensemble_mode=True
        )
    return intelligence_hub


class IntelligentTradingRequest(BaseModel):
    """Request for intelligent trading analysis"""
    
    symbol: str = Field(..., description="Trading symbol (e.g., AAPL, TSLA)")
    task_type: str = Field(..., description="Type of analysis required")
    time_horizon: str = Field("short", description="Time horizon: short, medium, long")
    priority: str = Field("normal", description="Priority: low, normal, high, critical")
    
    # Market data
    current_price: Optional[float] = None
    volume: Optional[int] = None
    volatility: Optional[float] = None
    
    # Portfolio context
    portfolio_value: Optional[float] = None
    current_position: Optional[float] = None
    
    # Risk parameters
    risk_tolerance: str = Field("moderate", description="Risk tolerance: conservative, moderate, aggressive")
    max_position_size: Optional[float] = None
    
    # Additional context
    custom_context: Optional[Dict[str, Any]] = None
    
    @validator('task_type')
    def validate_task_type(cls, v):
        valid_types = [task.value for task in TaskType]
        if v not in valid_types:
            raise ValueError(f"Task type must be one of: {valid_types}")
        return v
    
    @validator('time_horizon')
    def validate_time_horizon(cls, v):
        if v not in ["short", "medium", "long"]:
            raise ValueError("Time horizon must be: short, medium, or long")
        return v
    
    @validator('priority')
    def validate_priority(cls, v):
        if v not in ["low", "normal", "high", "critical"]:
            raise ValueError("Priority must be: low, normal, high, or critical")
        return v


class IntelligentTradingResponse(BaseModel):
    """Response from intelligent trading analysis"""
    
    # Trading decision
    action: str = Field(..., description="Recommended action: buy, sell, hold")
    confidence: float = Field(..., description="Confidence level (0.0 to 1.0)")
    reasoning: str = Field(..., description="Detailed reasoning for the decision")
    
    # Routing information
    selected_llm: str = Field(..., description="LLM model used for analysis")
    routing_confidence: float = Field(..., description="Confidence in model selection")
    routing_reasoning: str = Field(..., description="Why this model was selected")
    
    # Execution metrics
    execution_time: float = Field(..., description="Analysis execution time in seconds")
    efficiency_ratio: float = Field(..., description="Actual vs expected time ratio")
    
    # Confidence factors
    confidence_factors: Dict[str, float] = Field(..., description="Detailed confidence breakdown")
    
    # Optional ensemble data
    ensemble_used: bool = Field(False, description="Whether ensemble analysis was used")
    alternative_scenarios: Optional[List[Dict[str, Any]]] = None
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: str = Field(..., description="Unique request identifier")


class SystemStatusResponse(BaseModel):
    """System status and performance metrics"""
    
    timestamp: datetime
    routing_strategy: str
    ensemble_enabled: bool
    available_models: List[str]
    execution_history_size: int
    performance_metrics: Dict[str, Any]


class ModelPerformanceResponse(BaseModel):
    """Model performance report"""
    
    timestamp: datetime
    total_metrics: int
    llm_performance: Dict[str, Dict[str, Any]]
    task_performance: Dict[str, Dict[str, Any]]
    top_performers: Dict[str, Any]


@router.post("/analyze", response_model=IntelligentTradingResponse)
async def analyze_trading_opportunity(
    request: IntelligentTradingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    hub: TradingIntelligenceHub = Depends(get_intelligence_hub)
):
    """
    Analyze trading opportunity using intelligent Chinese LLM routing
    
    This endpoint leverages DeepSeek and other Chinese LLMs trained on financial data
    to provide sophisticated trading analysis with optimal model selection.
    """
    try:
        # Generate unique request ID
        request_id = f"req_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{current_user.id}"
        
        # Get current market data if not provided
        market_data = await _get_market_data(request.symbol, request)
        
        # Build portfolio context
        portfolio_context = {
            "total_value": request.portfolio_value or 100000,  # Default portfolio value
            "current_position": request.current_position or 0,
            "user_id": current_user.id
        }
        
        # Build risk parameters
        risk_parameters = {
            "tolerance": request.risk_tolerance,
            "max_position_size": request.max_position_size or 10000,
            "user_risk_profile": {
                "max_position_size": getattr(current_user, 'max_position_size', 10000),
                "max_daily_loss": getattr(current_user, 'max_daily_loss', 5000),
            }
        }
        
        # Create intelligence request
        intelligence_request = TradingIntelligenceRequest(
            task_type=TaskType(request.task_type),
            symbol=request.symbol,
            market_data=market_data,
            portfolio_context=portfolio_context,
            risk_parameters=risk_parameters,
            time_horizon=request.time_horizon,
            priority=request.priority,
            custom_context=request.custom_context
        )
        
        # Process with intelligent routing
        intelligence_response = await hub.process_trading_request(
            intelligence_request,
            force_ensemble=(request.priority == "critical")
        )
        
        # Schedule background performance tracking
        background_tasks.add_task(
            _track_request_performance,
            request_id,
            intelligence_request,
            intelligence_response
        )
        
        # Build response
        response = IntelligentTradingResponse(
            action=intelligence_response.decision.action,
            confidence=intelligence_response.decision.confidence,
            reasoning=intelligence_response.decision.reasoning,
            selected_llm=intelligence_response.routing_info["selected_llm"],
            routing_confidence=intelligence_response.routing_info["routing_confidence"],
            routing_reasoning=intelligence_response.routing_info["routing_reasoning"],
            execution_time=intelligence_response.execution_metrics["execution_time"],
            efficiency_ratio=intelligence_response.execution_metrics["efficiency_ratio"],
            confidence_factors=intelligence_response.confidence_factors,
            ensemble_used=intelligence_response.routing_info["selected_llm"].startswith("ensemble_"),
            alternative_scenarios=intelligence_response.alternative_scenarios,
            request_id=request_id
        )
        
        logger.info(
            "Intelligent trading analysis completed",
            request_id=request_id,
            symbol=request.symbol,
            action=response.action,
            selected_llm=response.selected_llm,
            execution_time=response.execution_time,
            user_id=current_user.id
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "Error in intelligent trading analysis",
            error=str(e),
            symbol=request.symbol,
            task_type=request.task_type,
            user_id=current_user.id
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


@router.get("/system/status", response_model=SystemStatusResponse)
async def get_system_status(
    hub: TradingIntelligenceHub = Depends(get_intelligence_hub)
):
    """Get current system status and performance metrics"""
    try:
        status_data = hub.get_system_status()
        
        return SystemStatusResponse(
            timestamp=datetime.fromisoformat(status_data["timestamp"]),
            routing_strategy=status_data["router_strategy"],
            ensemble_enabled=status_data["ensemble_enabled"],
            available_models=status_data["available_models"],
            execution_history_size=status_data["execution_history_size"],
            performance_metrics=status_data["performance_report"]
        )
        
    except Exception as e:
        logger.error("Error getting system status", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system status: {str(e)}"
        )


@router.get("/system/performance", response_model=ModelPerformanceResponse)
async def get_performance_report(
    hub: TradingIntelligenceHub = Depends(get_intelligence_hub)
):
    """Get detailed model performance report"""
    try:
        performance_data = hub.router.get_performance_report()
        
        return ModelPerformanceResponse(
            timestamp=datetime.fromisoformat(performance_data["timestamp"]),
            total_metrics=performance_data["total_metrics"],
            llm_performance=performance_data["llm_performance"],
            task_performance=performance_data["task_performance"],
            top_performers=performance_data["top_performers"]
        )
        
    except Exception as e:
        logger.error("Error getting performance report", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance report: {str(e)}"
        )


@router.post("/system/strategy")
async def update_routing_strategy(
    strategy: str,
    hub: TradingIntelligenceHub = Depends(get_intelligence_hub)
):
    """Update routing strategy"""
    try:
        valid_strategies = [s.value for s in RoutingStrategy]
        if strategy not in valid_strategies:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid strategy. Must be one of: {valid_strategies}"
            )
        
        hub.switch_routing_strategy(RoutingStrategy(strategy))
        
        return {
            "message": f"Routing strategy updated to {strategy}",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Error updating routing strategy", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update strategy: {str(e)}"
        )


@router.post("/system/warmup")
async def warmup_models(
    models: Optional[List[str]] = None,
    background_tasks: BackgroundTasks,
    hub: TradingIntelligenceHub = Depends(get_intelligence_hub)
):
    """Warm up Chinese LLM models for faster response times"""
    try:
        # Schedule warmup as background task
        background_tasks.add_task(_warmup_models, hub, models)
        
        return {
            "message": "Model warmup initiated",
            "models": models or "all available models",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Error initiating model warmup", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initiate warmup: {str(e)}"
        )


# Helper functions

async def _get_market_data(symbol: str, request: IntelligentTradingRequest) -> Dict[str, Any]:
    """Get current market data for the symbol"""
    # This would typically fetch from market data provider
    # For now, use provided data or defaults
    return {
        "symbol": symbol,
        "current_price": request.current_price or 100.0,
        "volume": request.volume or 1000000,
        "volatility": request.volatility or 0.3,
        "timestamp": datetime.utcnow().isoformat(),
        "conditions": {
            "market_open": True,
            "volatility_regime": "normal" if (request.volatility or 0.3) < 0.5 else "high"
        }
    }


async def _track_request_performance(
    request_id: str,
    intelligence_request: TradingIntelligenceRequest,
    intelligence_response: TradingIntelligenceResponse
):
    """Background task to track request performance"""
    try:
        # This would typically store detailed performance metrics in database
        logger.info(
            "Request performance tracked",
            request_id=request_id,
            symbol=intelligence_request.symbol,
            task_type=intelligence_request.task_type.value,
            selected_llm=intelligence_response.routing_info["selected_llm"],
            execution_time=intelligence_response.execution_metrics["execution_time"],
            confidence=intelligence_response.decision.confidence
        )
    except Exception as e:
        logger.warning(f"Failed to track request performance: {e}")


async def _warmup_models(hub: TradingIntelligenceHub, models: Optional[List[str]]):
    """Background task for model warmup"""
    try:
        await hub.warm_up_models(models)
        logger.info("Model warmup completed", models=models or "all")
    except Exception as e:
        logger.error(f"Model warmup failed: {e}")


# Startup event to initialize the intelligence hub
@router.on_event("startup")
async def startup_intelligence_hub():
    """Initialize intelligence hub on startup"""
    global intelligence_hub
    try:
        intelligence_hub = TradingIntelligenceHub(
            routing_strategy=RoutingStrategy.ADAPTIVE,
            enable_ensemble_mode=True
        )
        logger.info("Trading Intelligence Hub initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Trading Intelligence Hub: {e}")
        # Continue without intelligence hub - endpoints will fail gracefully