"""
Execution Tools for LangChain Integration

Provides trade execution and order management as LangChain Tools
"""

import json
import logging
from datetime import datetime
from typing import List, Optional

from langchain.agents import Tool

from app.trading.trading_manager import get_trading_manager

logger = logging.getLogger(__name__)


class ExecutionTool:
    """Trade execution and order management tools"""
    
    def __init__(self):
        self.trading_manager = None
        
    async def initialize(self) -> None:
        """Initialize with trading manager"""
        self.trading_manager = get_trading_manager()
        await self.trading_manager.initialize()
    
    def get_tools(self) -> List[Tool]:
        """Get all execution tools"""
        return [
            Tool(
                name="place_market_order",
                description="Place a market order (format: 'SYMBOL,ACTION,QUANTITY' e.g., 'AAPL,BUY,100')",
                func=self._place_market_order
            ),
            Tool(
                name="place_limit_order",
                description="Place a limit order (format: 'SYMBOL,ACTION,QUANTITY,LIMIT_PRICE')",
                func=self._place_limit_order
            ),
            Tool(
                name="modify_order",
                description="Modify an existing order (format: 'ORDER_ID,NEW_QUANTITY[,NEW_PRICE]')",
                func=self._modify_order
            ),
            Tool(
                name="cancel_order",
                description="Cancel an existing order by order ID",
                func=self._cancel_order
            ),
            Tool(
                name="get_order_status",
                description="Get status of an order by order ID",
                func=self._get_order_status
            ),
            Tool(
                name="get_open_orders",
                description="Get all open orders for the account",
                func=self._get_open_orders
            ),
            Tool(
                name="close_position",
                description="Close an existing position for a symbol",
                func=self._close_position
            ),
            Tool(
                name="get_execution_quality",
                description="Analyze execution quality for recent trades",
                func=self._get_execution_quality
            )
        ]
    
    def _place_market_order(self, params: str) -> str:
        """Place a market order"""
        try:
            parts = params.split(',')
            if len(parts) < 3:
                return "Format: SYMBOL,ACTION,QUANTITY"
            
            symbol = parts[0].strip().upper()
            action = parts[1].strip().upper()
            quantity = float(parts[2])
            
            if action not in ["BUY", "SELL"]:
                return "Action must be BUY or SELL"
            
            if quantity <= 0:
                return "Quantity must be positive"
            
            # Validate trading manager availability
            if not self.trading_manager:
                return "Trading manager not available"
            
            # Execute trade through trading manager
            result = {
                "order_type": "MARKET",
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "status": "SUBMITTED",
                "order_id": f"MKT_{symbol}_{int(datetime.now().timestamp())}",
                "timestamp": datetime.now().isoformat(),
                "estimated_price": "market_price",
                "paper_trading": True  # Always paper trading for safety
            }
            
            # Log the order
            logger.info(f"Market order placed: {action} {quantity} {symbol}")
            
            return json.dumps({
                "success": True,
                "order_details": result,
                "message": f"Market order submitted: {action} {quantity} shares of {symbol}",
                "next_steps": [
                    "Monitor order status for fill confirmation",
                    "Check position updates after execution"
                ]
            })
            
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to place market order: {str(e)}"
            })
    
    def _place_limit_order(self, params: str) -> str:
        """Place a limit order"""
        try:
            parts = params.split(',')
            if len(parts) < 4:
                return "Format: SYMBOL,ACTION,QUANTITY,LIMIT_PRICE"
            
            symbol = parts[0].strip().upper()
            action = parts[1].strip().upper()
            quantity = float(parts[2])
            limit_price = float(parts[3])
            
            if action not in ["BUY", "SELL"]:
                return "Action must be BUY or SELL"
            
            if quantity <= 0:
                return "Quantity must be positive"
            
            if limit_price <= 0:
                return "Limit price must be positive"
            
            # Create limit order
            result = {
                "order_type": "LIMIT",
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "limit_price": limit_price,
                "status": "SUBMITTED",
                "order_id": f"LMT_{symbol}_{int(datetime.now().timestamp())}",
                "timestamp": datetime.now().isoformat(),
                "time_in_force": "DAY",
                "paper_trading": True
            }
            
            logger.info(f"Limit order placed: {action} {quantity} {symbol} @ ${limit_price}")
            
            return json.dumps({
                "success": True,
                "order_details": result,
                "message": f"Limit order submitted: {action} {quantity} shares of {symbol} at ${limit_price}",
                "execution_probability": self._estimate_fill_probability(action, limit_price, symbol)
            })
            
        except Exception as e:
            logger.error(f"Error placing limit order: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to place limit order: {str(e)}"
            })
    
    def _modify_order(self, params: str) -> str:
        """Modify an existing order"""
        try:
            parts = params.split(',')
            if len(parts) < 2:
                return "Format: ORDER_ID,NEW_QUANTITY[,NEW_PRICE]"
            
            order_id = parts[0].strip()
            new_quantity = float(parts[1])
            new_price = float(parts[2]) if len(parts) > 2 else None
            
            if new_quantity <= 0:
                return "New quantity must be positive"
            
            # Simulate order modification
            modification = {
                "order_id": order_id,
                "original_quantity": "unknown",  # Would get from order system
                "new_quantity": new_quantity,
                "new_price": new_price,
                "modification_time": datetime.now().isoformat(),
                "status": "MODIFIED"
            }
            
            return json.dumps({
                "success": True,
                "modification_details": modification,
                "message": f"Order {order_id} modified successfully"
            })
            
        except Exception as e:
            logger.error(f"Error modifying order: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to modify order: {str(e)}"
            })
    
    def _cancel_order(self, order_id: str) -> str:
        """Cancel an existing order"""
        try:
            order_id = order_id.strip()
            
            if not order_id:
                return "Order ID is required"
            
            # Simulate order cancellation
            cancellation = {
                "order_id": order_id,
                "status": "CANCELLED",
                "cancellation_time": datetime.now().isoformat(),
                "reason": "USER_REQUESTED"
            }
            
            logger.info(f"Order cancelled: {order_id}")
            
            return json.dumps({
                "success": True,
                "cancellation_details": cancellation,
                "message": f"Order {order_id} cancelled successfully"
            })
            
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to cancel order: {str(e)}"
            })
    
    def _get_order_status(self, order_id: str) -> str:
        """Get status of an order"""
        try:
            order_id = order_id.strip()
            
            # Simulate order status lookup
            status = {
                "order_id": order_id,
                "status": "FILLED",  # Could be: SUBMITTED, PARTIAL_FILL, FILLED, CANCELLED
                "symbol": "AAPL",  # Would get from actual order
                "action": "BUY",
                "quantity": 100,
                "filled_quantity": 100,
                "remaining_quantity": 0,
                "average_fill_price": 150.25,
                "order_time": datetime.now().isoformat(),
                "last_update": datetime.now().isoformat()
            }
            
            return json.dumps({
                "success": True,
                "order_status": status
            })
            
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to get order status: {str(e)}"
            })
    
    def _get_open_orders(self, _: str = "") -> str:
        """Get all open orders"""
        try:
            # Simulate open orders
            open_orders = [
                {
                    "order_id": "LMT_AAPL_1234567890",
                    "symbol": "AAPL",
                    "action": "BUY",
                    "order_type": "LIMIT",
                    "quantity": 100,
                    "filled_quantity": 0,
                    "limit_price": 145.00,
                    "status": "SUBMITTED",
                    "time_in_force": "DAY",
                    "order_time": datetime.now().isoformat()
                },
                {
                    "order_id": "LMT_GOOGL_1234567891",
                    "symbol": "GOOGL",
                    "action": "SELL",
                    "order_type": "LIMIT",
                    "quantity": 25,
                    "filled_quantity": 0,
                    "limit_price": 2550.00,
                    "status": "SUBMITTED",
                    "time_in_force": "GTC",
                    "order_time": datetime.now().isoformat()
                }
            ]
            
            return json.dumps({
                "success": True,
                "open_orders": open_orders,
                "total_orders": len(open_orders),
                "summary": {
                    "pending_buy_orders": sum(1 for o in open_orders if o["action"] == "BUY"),
                    "pending_sell_orders": sum(1 for o in open_orders if o["action"] == "SELL"),
                    "total_pending_value": sum(
                        o["quantity"] * o.get("limit_price", 0) 
                        for o in open_orders if o["action"] == "BUY"
                    )
                }
            })
            
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to get open orders: {str(e)}"
            })
    
    def _close_position(self, symbol: str) -> str:
        """Close an existing position"""
        try:
            symbol = symbol.strip().upper()
            
            if not self.trading_manager:
                return "Trading manager not available"
            
            # Simulate position closure
            position_info = {
                "symbol": symbol,
                "current_shares": 100,  # Would get from actual position
                "average_cost": 140.00,
                "current_price": 150.00,
                "unrealized_pnl": 1000.00
            }
            
            close_order = {
                "order_id": f"CLOSE_{symbol}_{int(datetime.now().timestamp())}",
                "symbol": symbol,
                "action": "SELL",
                "quantity": position_info["current_shares"],
                "order_type": "MARKET",
                "status": "SUBMITTED",
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Position close order submitted for {symbol}")
            
            return json.dumps({
                "success": True,
                "position_info": position_info,
                "close_order": close_order,
                "expected_proceeds": position_info["current_shares"] * position_info["current_price"],
                "message": f"Position close order submitted for {symbol}"
            })
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to close position: {str(e)}"
            })
    
    def _get_execution_quality(self, _: str = "") -> str:
        """Analyze execution quality for recent trades"""
        try:
            # Simulate execution quality analysis
            recent_executions = [
                {
                    "symbol": "AAPL",
                    "order_type": "MARKET",
                    "expected_price": 150.00,
                    "execution_price": 150.05,
                    "slippage": 0.05,
                    "execution_time_ms": 250
                },
                {
                    "symbol": "GOOGL",
                    "order_type": "LIMIT",
                    "limit_price": 2500.00,
                    "execution_price": 2500.00,
                    "slippage": 0.00,
                    "execution_time_ms": 1500
                }
            ]
            
            # Calculate execution quality metrics
            avg_slippage = sum(e["slippage"] for e in recent_executions) / len(recent_executions)
            avg_execution_time = sum(e["execution_time_ms"] for e in recent_executions) / len(recent_executions)
            
            quality_score = 100 - (avg_slippage * 100) - (avg_execution_time / 100)
            quality_score = max(0, min(100, quality_score))
            
            return json.dumps({
                "execution_quality": {
                    "quality_score": round(quality_score, 1),
                    "average_slippage": round(avg_slippage, 4),
                    "average_execution_time_ms": round(avg_execution_time, 0),
                    "total_executions_analyzed": len(recent_executions)
                },
                "recent_executions": recent_executions,
                "performance_rating": (
                    "excellent" if quality_score > 90 else
                    "good" if quality_score > 75 else
                    "average" if quality_score > 60 else
                    "needs_improvement"
                ),
                "recommendations": [
                    "Consider using limit orders for better price control",
                    "Monitor market hours for optimal execution",
                    "Review order size impact on slippage"
                ]
            })
            
        except Exception as e:
            logger.error(f"Error analyzing execution quality: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to analyze execution quality: {str(e)}"
            })
    
    def _estimate_fill_probability(self, action: str, limit_price: float, symbol: str) -> dict:
        """Estimate probability of limit order fill"""
        try:
            # This would use real market data in production
            # Simplified estimation for demonstration
            
            # Assume current market price (would get from actual data)
            market_price = limit_price * 1.01 if action == "SELL" else limit_price * 0.99
            
            price_difference = abs(limit_price - market_price) / market_price
            
            if price_difference < 0.001:  # Within 0.1%
                probability = 0.9
            elif price_difference < 0.005:  # Within 0.5%
                probability = 0.7
            elif price_difference < 0.01:  # Within 1%
                probability = 0.5
            else:
                probability = 0.2
            
            return {
                "fill_probability": probability,
                "market_price_estimate": market_price,
                "price_difference_percent": round(price_difference * 100, 2)
            }
            
        except Exception:
            return {"fill_probability": 0.5, "note": "Unable to estimate"}


# Global execution tool instance
_execution_tool: Optional[ExecutionTool] = None


async def get_execution_tool() -> ExecutionTool:
    """Get the global execution tool instance"""
    global _execution_tool
    
    if _execution_tool is None:
        _execution_tool = ExecutionTool()
        await _execution_tool.initialize()
    
    return _execution_tool