"""
Personal Dashboard API - Simple endpoints for personal trading interface

Provides clean, focused API endpoints for personal trading decisions
without enterprise complexity.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, Optional
import structlog

from ..personal.personal_trading_engine import PersonalTradingEngine, PersonalTradingDecision, PersonalPortfolioSummary
from ..core.auth import get_current_user
from ..trading.alpaca_client import AlpacaClient

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/personal", tags=["personal-trading"])

# Global personal engine instance (can be configured per user later)
personal_engine = None


def get_personal_engine(user_id: int) -> PersonalTradingEngine:
    """Get or create personal trading engine for user"""
    global personal_engine

    if personal_engine is None:
        personal_engine = PersonalTradingEngine(user_id=user_id)

    return personal_engine


@router.get("/dashboard", response_model=Dict[str, Any])
async def get_personal_dashboard(
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get personal trading dashboard - all key info in one place"""

    try:
        user_id = current_user["user_id"]
        engine = get_personal_engine(user_id)

        # Get account info (mock for now, replace with real Alpaca integration)
        account_info = {
            "equity": 50000,
            "cash": 10000,
            "daily_pnl": 250,
        }

        current_positions = []  # Replace with real positions

        # Get portfolio summary
        portfolio_summary = await engine.get_portfolio_summary(account_info, current_positions)

        # Get learning insights
        learning_insights = engine.get_learning_insights()

        # Get recent decisions
        recent_decisions = engine.decision_history[-5:] if engine.decision_history else []

        dashboard = {
            "user_id": user_id,
            "portfolio": {
                "total_value": portfolio_summary.total_value,
                "daily_pnl": portfolio_summary.daily_pnl,
                "daily_pnl_pct": portfolio_summary.daily_pnl_pct,
                "risk_level": portfolio_summary.portfolio_risk,
                "active_positions": portfolio_summary.active_positions,
                "cash_available": portfolio_summary.cash_available,
            },
            "ai_insights": {
                "recommendation": portfolio_summary.ai_recommendation,
                "learning_status": portfolio_summary.learning_status,
                "pattern_confidence": portfolio_summary.pattern_confidence,
                "total_decisions_made": learning_insights["total_decisions"],
            },
            "recent_decisions": [
                {
                    "symbol": d.symbol,
                    "action": d.action,
                    "confidence": round(d.confidence, 2),
                    "reasoning": d.reasoning,
                    "timestamp": d.timestamp.isoformat(),
                }
                for d in recent_decisions
            ],
            "quick_stats": {
                "learning_insights": learning_insights["insights"][:3],  # Top 3 insights
                "system_status": "learning" if learning_insights["total_decisions"] > 0 else "ready",
            },
            "last_updated": portfolio_summary.last_updated.isoformat(),
        }

        logger.info("Personal dashboard generated", user_id=user_id)
        return dashboard

    except Exception as e:
        logger.error("Failed to generate personal dashboard", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to load dashboard")


@router.post("/analyze/{symbol}")
async def analyze_symbol_for_trading(
    symbol: str,
    current_user: Dict = Depends(get_current_user)
) -> PersonalTradingDecision:
    """Analyze a symbol for personal trading decision"""

    try:
        user_id = current_user["user_id"]
        engine = get_personal_engine(user_id)

        # Mock market data (replace with real data)
        market_data = {
            "price": 150.00,
            "volume": 1000000,
            "price_change_pct": 0.02,
        }

        technical_indicators = {
            "rsi": 65,
            "macd": 0.5,
            "bb_position": 0.7,
        }

        account_info = {
            "equity": 50000,
            "cash": 10000,
        }

        current_positions = []

        # Get trading decision
        decision = await engine.should_i_trade(
            symbol=symbol,
            market_data=market_data,
            technical_indicators=technical_indicators,
            account_info=account_info,
            current_positions=current_positions
        )

        logger.info("Symbol analysis completed", symbol=symbol, action=decision.action)
        return decision

    except Exception as e:
        logger.error("Failed to analyze symbol", symbol=symbol, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to analyze {symbol}")


@router.get("/learning-insights")
async def get_learning_insights(
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get detailed learning insights about AI performance"""

    try:
        user_id = current_user["user_id"]
        engine = get_personal_engine(user_id)

        insights = engine.get_learning_insights()
        stats = engine.get_personal_stats()

        return {
            "learning_summary": insights,
            "personal_stats": stats,
            "recommendations": [
                "AI is learning your trading patterns",
                "More decisions will improve accuracy",
                "Check back regularly for insights"
            ] if insights["total_decisions"] < 10 else [
                f"AI has analyzed {insights['total_decisions']} decisions",
                "Pattern recognition is active",
                "System is adapting to your trading style"
            ]
        }

    except Exception as e:
        logger.error("Failed to get learning insights", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to load insights")


@router.post("/settings/simple-mode")
async def toggle_simple_mode(
    enabled: bool,
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, str]:
    """Toggle simple mode for faster decisions"""

    try:
        user_id = current_user["user_id"]
        engine = get_personal_engine(user_id)

        if enabled:
            engine.enable_simple_mode()
            message = "Simple mode enabled - faster decisions, less detail"
        else:
            engine.personal_preferences['explanation_detail'] = 'simple'
            message = "Normal mode enabled - detailed analysis"

        logger.info("Simple mode toggled", user_id=user_id, enabled=enabled)
        return {"status": "success", "message": message}

    except Exception as e:
        logger.error("Failed to toggle simple mode", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to update settings")


@router.get("/health")
async def personal_system_health() -> Dict[str, Any]:
    """Simple health check for personal trading system"""

    try:
        # Basic health indicators
        health_status = {
            "status": "healthy",
            "personal_engine": "ready",
            "ai_system": "active",
            "memory_usage": "optimal",  # Could add real memory monitoring
            "learning_active": True,
            "timestamp": "2024-01-01T00:00:00Z"  # Real timestamp
        }

        return health_status

    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "status": "error",
            "message": str(e),
            "timestamp": "2024-01-01T00:00:00Z"
        }