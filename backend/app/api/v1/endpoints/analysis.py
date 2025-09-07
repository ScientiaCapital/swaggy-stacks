"""
Analysis endpoints for trading system
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def get_analysis():
    """Get analysis overview"""
    return {"status": "analysis endpoint ready"}