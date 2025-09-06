"""
API v1 router configuration
"""

from fastapi import APIRouter
from app.api.v1.endpoints import auth, trading, analysis, portfolio, market_data

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(trading.router, prefix="/trading", tags=["trading"])
api_router.include_router(analysis.router, prefix="/analysis", tags=["analysis"])
api_router.include_router(portfolio.router, prefix="/portfolio", tags=["portfolio"])
api_router.include_router(market_data.router, prefix="/market-data", tags=["market-data"])
