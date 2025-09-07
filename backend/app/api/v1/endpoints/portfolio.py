"""
Portfolio management endpoints
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def get_portfolio():
    """Get portfolio overview"""
    return {"status": "portfolio endpoint ready"}