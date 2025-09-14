"""
API v1 router configuration
"""

from fastapi import APIRouter

from app.api.v1.endpoints import (
    ai_trading,
    analysis,
    auth,
    backtesting,
    github,
    health,
    market_data,
    monitoring,
    portfolio,
    trading,
    websocket,
)

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(health.router, tags=["health"])  # Health checks at root level
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(trading.router, prefix="/trading", tags=["trading"])
api_router.include_router(analysis.router, prefix="/analysis", tags=["analysis"])
api_router.include_router(portfolio.router, prefix="/portfolio", tags=["portfolio"])
api_router.include_router(
    market_data.router, prefix="/market-data", tags=["market-data"]
)
api_router.include_router(ai_trading.router, prefix="/ai", tags=["ai-trading"])
api_router.include_router(
    backtesting.router, prefix="/backtesting", tags=["backtesting"]
)
api_router.include_router(github.router, prefix="/github", tags=["github", "ci-cd"])
api_router.include_router(
    monitoring.router, prefix="/monitoring", tags=["monitoring", "observability"]
)
api_router.include_router(websocket.router, tags=["websocket", "real-time"])
