"""
Trading API endpoints
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
import structlog

from app.core.database import get_db
from app.core.exceptions import TradingError, RiskManagementError
from app.trading.alpaca_client import AlpacaClient
from app.trading.risk_manager import RiskManager
from app.models.trade import Trade
from app.models.user import User

logger = structlog.get_logger()
router = APIRouter()


class OrderRequest(BaseModel):
    """Order request model"""
    symbol: str
    quantity: float
    side: str  # BUY or SELL
    order_type: str = "market"  # market, limit, stop
    time_in_force: str = "gtc"  # gtc, ioc, fok
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    strategy_id: Optional[int] = None


class OrderResponse(BaseModel):
    """Order response model"""
    order_id: str
    symbol: str
    quantity: float
    side: str
    status: str
    submitted_at: str
    message: str


class TradeResponse(BaseModel):
    """Trade response model"""
    id: int
    symbol: str
    quantity: float
    entry_price: float
    exit_price: Optional[float]
    entry_time: str
    exit_time: Optional[str]
    pnl: Optional[float]
    status: str
    side: str


@router.post("/orders", response_model=OrderResponse)
async def create_order(
    order_request: OrderRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)  # This would be implemented
):
    """Create a new trading order"""
    try:
        # Initialize Alpaca client
        alpaca_client = AlpacaClient(
            api_key=current_user.alpaca_api_key,
            secret_key=current_user.alpaca_secret_key,
            paper=True
        )
        
        # Get current account info
        account = await alpaca_client.get_account()
        positions = await alpaca_client.get_positions()
        
        # Initialize risk manager
        risk_manager = RiskManager(
            user_id=current_user.id,
            user_risk_params={
                'max_position_size': current_user.max_position_size,
                'max_daily_loss': current_user.max_daily_loss,
                'max_portfolio_exposure': 0.95,
                'max_single_stock_exposure': 0.20
            }
        )
        
        # Validate order against risk management
        is_valid, reason = risk_manager.validate_order(
            symbol=order_request.symbol,
            quantity=order_request.quantity,
            price=order_request.limit_price or 0,  # Would need current price
            side=order_request.side,
            current_positions=positions,
            account_value=float(account['portfolio_value']),
            daily_pnl=0  # Would need to calculate from trades
        )
        
        if not is_valid:
            raise RiskManagementError(f"Order rejected: {reason}")
        
        # Execute order with Alpaca
        order = await alpaca_client.execute_order(
            symbol=order_request.symbol,
            qty=order_request.quantity,
            side=order_request.side,
            order_type=order_request.order_type,
            time_in_force=order_request.time_in_force,
            limit_price=order_request.limit_price,
            stop_price=order_request.stop_price
        )
        
        # Save trade to database
        trade = Trade(
            symbol=order_request.symbol,
            quantity=order_request.quantity,
            entry_price=order_request.limit_price or 0,  # Would need actual fill price
            side=order_request.side,
            order_type=order_request.order_type.upper(),
            time_in_force=order_request.time_in_force.upper(),
            alpaca_order_id=order['id'],
            user_id=current_user.id,
            strategy_id=order_request.strategy_id
        )
        
        db.add(trade)
        db.commit()
        db.refresh(trade)
        
        logger.info(
            "Order created successfully",
            user_id=current_user.id,
            symbol=order_request.symbol,
            order_id=order['id']
        )
        
        return OrderResponse(
            order_id=order['id'],
            symbol=order['symbol'],
            quantity=order['qty'],
            side=order['side'],
            status=order['status'],
            submitted_at=order['submitted_at'],
            message="Order submitted successfully"
        )
        
    except TradingError as e:
        logger.error("Trading error", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=400, detail=str(e))
    except RiskManagementError as e:
        logger.error("Risk management error", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Unexpected error creating order", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/orders", response_model=List[OrderResponse])
async def get_orders(
    status: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get user's orders"""
    try:
        alpaca_client = AlpacaClient(
            api_key=current_user.alpaca_api_key,
            secret_key=current_user.alpaca_secret_key,
            paper=True
        )
        
        orders = await alpaca_client.get_orders(status=status, limit=limit)
        
        return [
            OrderResponse(
                order_id=order['id'],
                symbol=order['symbol'],
                quantity=order['qty'],
                side=order['side'],
                status=order['status'],
                submitted_at=order['submitted_at'],
                message=""
            )
            for order in orders
        ]
        
    except Exception as e:
        logger.error("Error fetching orders", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/trades", response_model=List[TradeResponse])
async def get_trades(
    symbol: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get user's trades"""
    try:
        query = db.query(Trade).filter(Trade.user_id == current_user.id)
        
        if symbol:
            query = query.filter(Trade.symbol == symbol)
        if status:
            query = query.filter(Trade.status == status)
        
        trades = query.order_by(Trade.created_at.desc()).limit(limit).all()
        
        return [
            TradeResponse(
                id=trade.id,
                symbol=trade.symbol,
                quantity=trade.quantity,
                entry_price=trade.entry_price,
                exit_price=trade.exit_price,
                entry_time=trade.entry_time.isoformat(),
                exit_time=trade.exit_time.isoformat() if trade.exit_time else None,
                pnl=trade.pnl,
                status=trade.status,
                side=trade.side
            )
            for trade in trades
        ]
        
    except Exception as e:
        logger.error("Error fetching trades", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/orders/{order_id}/cancel")
async def cancel_order(
    order_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Cancel an order"""
    try:
        alpaca_client = AlpacaClient(
            api_key=current_user.alpaca_api_key,
            secret_key=current_user.alpaca_secret_key,
            paper=True
        )
        
        success = await alpaca_client.cancel_order(order_id)
        
        if success:
            # Update trade status in database
            trade = db.query(Trade).filter(
                Trade.alpaca_order_id == order_id,
                Trade.user_id == current_user.id
            ).first()
            
            if trade:
                trade.status = "CANCELLED"
                db.commit()
            
            logger.info("Order cancelled", order_id=order_id, user_id=current_user.id)
            return {"message": "Order cancelled successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to cancel order")
            
    except Exception as e:
        logger.error("Error cancelling order", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=500, detail="Internal server error")


# Placeholder for authentication dependency
def get_current_user():
    """Placeholder for user authentication"""
    # This would be implemented with JWT token validation
    pass
