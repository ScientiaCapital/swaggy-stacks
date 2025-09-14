"""
AI Trading endpoints for Swaggy Stacks Trading System
"""

from datetime import datetime
from typing import Dict, List, Optional

import structlog
import yfinance as yf
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.ai.trading_agents import AITradingCoordinator

logger = structlog.get_logger()

router = APIRouter()

# Global AI coordinator instance
ai_coordinator = None


async def get_ai_coordinator():
    """Get or initialize AI coordinator"""
    global ai_coordinator
    if ai_coordinator is None:
        try:
            ai_coordinator = AITradingCoordinator()
            await ai_coordinator.health_check()  # Verify it's working
            logger.info("AI coordinator initialized")
        except Exception as e:
            logger.error("Failed to initialize AI coordinator", error=str(e))
            raise HTTPException(status_code=503, detail="AI services unavailable")
    return ai_coordinator


# Pydantic models for requests/responses
class MarketAnalysisRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol to analyze")
    context: Optional[str] = Field(None, description="Additional context for analysis")


class MarketAnalysisResponse(BaseModel):
    symbol: str
    sentiment: str
    confidence: float
    key_factors: List[str]
    recommendations: List[str]
    risk_level: str
    reasoning: str
    timestamp: datetime


class RiskAssessmentRequest(BaseModel):
    symbol: str
    position_size: float
    account_value: float
    current_positions: List[Dict] = Field(default_factory=list)
    proposed_trade: Dict = Field(default_factory=dict)


class RiskAssessmentResponse(BaseModel):
    symbol: str
    risk_level: str
    portfolio_heat: float
    recommended_position_size: float
    key_risk_factors: List[str]
    mitigation_strategies: List[str]
    exit_conditions: List[str]
    max_position_risk: float
    timestamp: datetime


class StrategySignalRequest(BaseModel):
    symbol: str
    markov_analysis: Dict = Field(default_factory=dict)
    technical_indicators: Dict = Field(default_factory=dict)
    market_context: Dict = Field(default_factory=dict)


class StrategySignalResponse(BaseModel):
    symbol: str
    action: str
    confidence: float
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    position_size: Optional[float]
    reasoning: str
    technical_factors: List[str]
    timestamp: datetime


class TradeReviewRequest(BaseModel):
    trade_data: Dict
    market_context: Dict = Field(default_factory=dict)
    system_performance: Dict = Field(default_factory=dict)


class TradeReviewResponse(BaseModel):
    trade_id: str
    symbol: str
    performance_grade: str
    execution_quality: str
    key_learnings: List[str]
    improvement_suggestions: List[str]
    pattern_insights: List[str]
    systematic_improvements: List[str]
    timestamp: datetime


class ComprehensiveAnalysisRequest(BaseModel):
    symbol: str
    account_value: float = Field(default=100000)
    current_positions: List[Dict] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str
    context: Optional[Dict] = None


class ChatResponse(BaseModel):
    response: str
    timestamp: datetime


@router.get("/health")
async def ai_health_check():
    """Check AI services health"""
    try:
        coordinator = await get_ai_coordinator()
        health_status = await coordinator.health_check()
        return {
            "status": "healthy",
            "ai_services": health_status,
            "timestamp": datetime.now(),
        }
    except Exception as e:
        logger.error("AI health check failed", error=str(e))
        return {"status": "unhealthy", "error": str(e), "timestamp": datetime.now()}


@router.post("/analyze/market", response_model=MarketAnalysisResponse)
async def analyze_market(request: MarketAnalysisRequest):
    """Get AI market analysis for a symbol"""
    try:
        coordinator = await get_ai_coordinator()

        # Get recent market data
        ticker = yf.Ticker(request.symbol)
        hist = ticker.history(period="3mo")

        if hist.empty:
            raise HTTPException(
                status_code=404, detail=f"No market data found for {request.symbol}"
            )

        # Prepare market data
        current_price = hist["Close"].iloc[-1]
        market_data = {
            "current_price": current_price,
            "volume": hist["Volume"].iloc[-1],
            "high_52w": hist["High"].max(),
            "low_52w": hist["Low"].min(),
            "volatility": hist["Close"].pct_change().std() * (252**0.5),
        }

        # Calculate technical indicators
        technical_indicators = _calculate_basic_indicators(hist)

        # Get AI analysis
        analysis = await coordinator.market_analyst.analyze_market(
            symbol=request.symbol,
            market_data=market_data,
            technical_indicators=technical_indicators,
            context=request.context or "",
        )

        return MarketAnalysisResponse(
            symbol=analysis.symbol,
            sentiment=analysis.sentiment,
            confidence=analysis.confidence,
            key_factors=analysis.key_factors,
            recommendations=analysis.recommendations,
            risk_level=analysis.risk_level,
            reasoning=analysis.reasoning,
            timestamp=analysis.timestamp,
        )

    except Exception as e:
        logger.error("Market analysis failed", symbol=request.symbol, error=str(e))
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/assess/risk", response_model=RiskAssessmentResponse)
async def assess_risk(request: RiskAssessmentRequest):
    """Get AI risk assessment for a proposed trade"""
    try:
        coordinator = await get_ai_coordinator()

        # Get market volatility data
        ticker = yf.Ticker(request.symbol)
        hist = ticker.history(period="3mo")

        market_volatility = {
            "hist_vol": (
                hist["Close"].pct_change().std() * (252**0.5) if not hist.empty else 0.2
            ),
            "atr": (
                _calculate_atr(hist)
                if not hist.empty
                else 0.02 * hist["Close"].iloc[-1] if not hist.empty else 0
            ),
        }

        # Get risk assessment
        assessment = await coordinator.risk_advisor.assess_risk(
            symbol=request.symbol,
            position_size=request.position_size,
            account_value=request.account_value,
            current_positions=request.current_positions,
            market_volatility=market_volatility,
            proposed_trade=request.proposed_trade,
        )

        return RiskAssessmentResponse(
            symbol=assessment.symbol,
            risk_level=assessment.risk_level,
            portfolio_heat=assessment.portfolio_heat,
            recommended_position_size=assessment.recommended_position_size,
            key_risk_factors=assessment.key_risk_factors,
            mitigation_strategies=assessment.mitigation_strategies,
            exit_conditions=assessment.exit_conditions,
            max_position_risk=assessment.max_position_risk,
            timestamp=assessment.timestamp,
        )

    except Exception as e:
        logger.error("Risk assessment failed", symbol=request.symbol, error=str(e))
        raise HTTPException(status_code=500, detail=f"Risk assessment failed: {str(e)}")


@router.post("/generate/signal", response_model=StrategySignalResponse)
async def generate_strategy_signal(request: StrategySignalRequest):
    """Generate AI-optimized trading signal"""
    try:
        coordinator = await get_ai_coordinator()

        # Generate signal
        signal = await coordinator.strategy_optimizer.generate_signal(
            symbol=request.symbol,
            markov_analysis=request.markov_analysis,
            technical_indicators=request.technical_indicators,
            market_context=request.market_context,
            performance_history=[],  # Could be populated from trading history
        )

        return StrategySignalResponse(
            symbol=signal.symbol,
            action=signal.action,
            confidence=signal.confidence,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            position_size=signal.position_size,
            reasoning=signal.reasoning,
            technical_factors=signal.technical_factors,
            timestamp=signal.timestamp,
        )

    except Exception as e:
        logger.error("Signal generation failed", symbol=request.symbol, error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Signal generation failed: {str(e)}"
        )


@router.post("/review/trade", response_model=TradeReviewResponse)
async def review_trade(request: TradeReviewRequest):
    """Get AI trade review and insights"""
    try:
        coordinator = await get_ai_coordinator()

        # Get trade review
        review = await coordinator.performance_coach.review_trade(
            trade_data=request.trade_data,
            market_context=request.market_context,
            system_performance=request.system_performance,
        )

        return TradeReviewResponse(
            trade_id=review.trade_id,
            symbol=review.symbol,
            performance_grade=review.performance_grade,
            execution_quality=review.execution_quality,
            key_learnings=review.key_learnings,
            improvement_suggestions=review.improvement_suggestions,
            pattern_insights=review.pattern_insights,
            systematic_improvements=review.systematic_improvements,
            timestamp=review.timestamp,
        )

    except Exception as e:
        logger.error("Trade review failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Trade review failed: {str(e)}")


@router.post("/analyze/comprehensive")
async def comprehensive_analysis(request: ComprehensiveAnalysisRequest):
    """Run comprehensive AI analysis using all agents"""
    try:
        coordinator = await get_ai_coordinator()

        # Get market data
        ticker = yf.Ticker(request.symbol)
        hist = ticker.history(period="3mo")

        if hist.empty:
            raise HTTPException(
                status_code=404, detail=f"No market data found for {request.symbol}"
            )

        # Prepare data
        current_price = hist["Close"].iloc[-1]
        market_data = {
            "current_price": current_price,
            "volume": hist["Volume"].iloc[-1],
            "high_52w": hist["High"].max(),
            "low_52w": hist["Low"].min(),
            "volatility": hist["Close"].pct_change().std() * (252**0.5),
        }

        technical_indicators = _calculate_basic_indicators(hist)
        markov_analysis = _calculate_simple_markov(hist)

        # Run comprehensive analysis
        analysis = await coordinator.comprehensive_analysis(
            symbol=request.symbol,
            market_data=market_data,
            technical_indicators=technical_indicators,
            account_info={
                "equity": request.account_value,
                "cash": request.account_value * 0.5,
            },
            current_positions=request.current_positions,
            markov_analysis=markov_analysis,
        )

        return analysis

    except Exception as e:
        logger.error(
            "Comprehensive analysis failed", symbol=request.symbol, error=str(e)
        )
        raise HTTPException(
            status_code=500, detail=f"Comprehensive analysis failed: {str(e)}"
        )


@router.post("/chat", response_model=ChatResponse)
async def ai_chat(request: ChatRequest):
    """Chat with AI trading assistant"""
    try:
        coordinator = await get_ai_coordinator()

        # Use the chat model for conversational response
        response = await coordinator.ollama_client.generate_response(
            prompt=request.message, model_key="chat", max_tokens=512
        )

        return ChatResponse(response=response, timestamp=datetime.now())

    except Exception as e:
        logger.error("AI chat failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


# Helper functions
def _calculate_basic_indicators(hist_data) -> Dict:
    """Calculate basic technical indicators"""
    try:
        close = hist_data["Close"]

        # RSI
        gains = close.diff().clip(lower=0)
        losses = (-1 * close.diff()).clip(lower=0)
        avg_gains = gains.rolling(14).mean()
        avg_losses = losses.rolling(14).mean()
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs)).iloc[-1]

        # Moving averages
        ma20 = close.rolling(20).mean().iloc[-1]
        ma50 = close.rolling(50).mean().iloc[-1]

        # MACD
        exp12 = close.ewm(span=12).mean()
        exp26 = close.ewm(span=26).mean()
        macd = (exp12 - exp26).iloc[-1]

        return {
            "rsi": rsi if not pd.isna(rsi) else 50.0,
            "ma20": ma20 if not pd.isna(ma20) else close.iloc[-1],
            "ma50": ma50 if not pd.isna(ma50) else close.iloc[-1],
            "macd": macd if not pd.isna(macd) else 0.0,
            "atr": _calculate_atr(hist_data),
        }

    except Exception:
        return {"rsi": 50.0, "ma20": 0, "ma50": 0, "macd": 0.0, "atr": 0.0}


def _calculate_atr(hist_data) -> float:
    """Calculate Average True Range"""
    try:
        high_low = hist_data["High"] - hist_data["Low"]
        high_close = (hist_data["High"] - hist_data["Close"].shift()).abs()
        low_close = (hist_data["Low"] - hist_data["Close"].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return ranges.rolling(14).mean().iloc[-1]
    except Exception:
        return 0.0


def _calculate_simple_markov(hist_data) -> Dict:
    """Calculate simplified Markov analysis"""
    try:
        close = hist_data["Close"]
        returns = close.pct_change().dropna()

        # Simple momentum-based state
        recent_momentum = returns.tail(5).mean()

        if recent_momentum > 0.01:
            state = "bullish"
            confidence = min(0.8, abs(recent_momentum) * 50)
        elif recent_momentum < -0.01:
            state = "bearish"
            confidence = min(0.8, abs(recent_momentum) * 50)
        else:
            state = "neutral"
            confidence = 0.3

        return {
            "current_state": state,
            "confidence": confidence,
            "direction": "up" if recent_momentum > 0 else "down",
            "transition_prob": confidence,
        }

    except Exception:
        return {
            "current_state": "neutral",
            "confidence": 0.5,
            "direction": "neutral",
            "transition_prob": 0.5,
        }


# Add pandas import that we're using
import pandas as pd
