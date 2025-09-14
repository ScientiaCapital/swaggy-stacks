"""
Market data endpoints
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def get_market_data():
    """Get market data overview"""
    return {"status": "market data endpoint ready"}
