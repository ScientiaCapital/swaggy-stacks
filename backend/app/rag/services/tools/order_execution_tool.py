"""
Order Execution Tool for managing trading orders
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import logging

from .base_tool import AgentTool, ToolResult, ToolParameter
from app.trading.alpaca_client import AlpacaClient
from app.trading.order_manager import OrderManager
from app.trading.risk_manager import RiskManager
from app.core.exceptions import TradingError

logger = logging.getLogger(__name__)


class OrderExecutionTool(AgentTool):
    """Tool for executing and managing trading orders"""
    
    def __init__(self):
        super().__init__(
            name="order_execution",
            description="Execute, monitor, and manage trading orders including bracket orders and trailing stops"
        )
        self.category = "order_execution"
        self.alpaca_client = None
        self.order_manager = None
    
    async def _get_clients(self, user_id: int = 1) -> tuple[AlpacaClient, OrderManager]:
        """Get or create Alpaca client and order manager"""
        if self.alpaca_client is None:
            self.alpaca_client = AlpacaClient()
        
        if self.order_manager is None:
            risk_manager = RiskManager(user_id)
            self.order_manager = OrderManager(self.alpaca_client, risk_manager)
        
        return self.alpaca_client, self.order_manager
    
    def get_parameters(self) -> List[ToolParameter]:
        """Get tool parameter definitions"""
        return [
            ToolParameter(
                name="action",
                type="str",
                description="Action: 'execute_order', 'bracket_order', 'trailing_stop', 'cancel_order', 'get_order', 'monitor_status'",
                required=True
            ),
            ToolParameter(
                name="symbol",
                type="str",
                description="Stock symbol (required for most actions)",
                required=False
            ),
            ToolParameter(
                name="quantity",
                type="float",
                description="Order quantity (required for order execution)",
                required=False
            ),
            ToolParameter(
                name="side",
                type="str",
                description="Order side: 'BUY' or 'SELL' (required for order execution)",
                required=False
            ),
            ToolParameter(
                name="order_type",
                type="str",
                description="Order type: 'market', 'limit', 'stop', 'stop_limit'",
                required=False,
                default="market"
            ),
            ToolParameter(
                name="limit_price",
                type="float",
                description="Limit price for limit orders",
                required=False
            ),
            ToolParameter(
                name="stop_price",
                type="float",
                description="Stop price for stop orders",
                required=False
            ),
            ToolParameter(
                name="time_in_force",
                type="str",
                description="Time in force: 'gtc', 'day', 'ioc', 'fok'",
                required=False,
                default="gtc"
            ),
            ToolParameter(
                name="stop_loss_price",
                type="float",
                description="Stop loss price for bracket orders",
                required=False
            ),
            ToolParameter(
                name="take_profit_price",
                type="float",
                description="Take profit price for bracket orders",
                required=False
            ),
            ToolParameter(
                name="trail_percent",
                type="float",
                description="Trail percentage for trailing stops (e.g., 0.05 for 5%)",
                required=False,
                default=0.05
            ),
            ToolParameter(
                name="order_id",
                type="str",
                description="Order ID for cancellation or retrieval",
                required=False
            ),
            ToolParameter(
                name="user_id",
                type="int",
                description="User ID for order management",
                required=False,
                default=1
            ),
            ToolParameter(
                name="atr",
                type="float",
                description="Average True Range for ATR-based stop losses",
                required=False
            )
        ]
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute order management action"""
        try:
            action = parameters["action"].lower()
            user_id = parameters.get("user_id", 1)
            
            client, order_manager = await self._get_clients(user_id)
            
            if action == "execute_order":
                return await self._execute_simple_order(client, parameters)
            elif action == "bracket_order":
                return await self._execute_bracket_order(order_manager, parameters)
            elif action == "trailing_stop":
                return await self._create_trailing_stop(order_manager, parameters)
            elif action == "cancel_order":
                return await self._cancel_order(client, parameters)
            elif action == "get_order":
                return await self._get_order(client, parameters)
            elif action == "monitor_status":
                return await self._get_monitor_status(order_manager, parameters)
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Unknown action: {action}. Supported: 'execute_order', 'bracket_order', 'trailing_stop', 'cancel_order', 'get_order', 'monitor_status'"
                )
                
        except TradingError as e:
            logger.error(f"Trading error in order execution: {e}")
            return ToolResult(success=False, data=None, error=str(e))
        except Exception as e:
            logger.error(f"Unexpected error in order execution: {e}")
            return ToolResult(success=False, data=None, error=f"Unexpected error: {str(e)}")
    
    async def _execute_simple_order(self, client: AlpacaClient, parameters: Dict[str, Any]) -> ToolResult:
        """Execute a simple order"""
        try:
            # Check required parameters
            required_params = ["symbol", "quantity", "side"]
            missing_params = [p for p in required_params if p not in parameters or parameters[p] is None]
            if missing_params:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Missing required parameters: {missing_params}"
                )
            
            symbol = parameters["symbol"]
            quantity = float(parameters["quantity"])
            side = parameters["side"].upper()
            order_type = parameters.get("order_type", "market")
            limit_price = parameters.get("limit_price")
            stop_price = parameters.get("stop_price")
            time_in_force = parameters.get("time_in_force", "gtc")
            
            # Convert types
            if limit_price is not None:
                limit_price = float(limit_price)
            if stop_price is not None:
                stop_price = float(stop_price)
            
            order_result = await client.execute_order(
                symbol=symbol,
                qty=quantity,
                side=side,
                order_type=order_type,
                time_in_force=time_in_force,
                limit_price=limit_price,
                stop_price=stop_price
            )
            
            metadata = {
                "action": "execute_order",
                "executed_at": datetime.now().isoformat(),
                "order_type": order_type
            }
            
            return ToolResult(success=True, data=order_result, metadata=metadata)
            
        except Exception as e:
            raise TradingError(f"Simple order execution failed: {str(e)}")
    
    async def _execute_bracket_order(self, order_manager: OrderManager, parameters: Dict[str, Any]) -> ToolResult:
        """Execute a bracket order with stop loss and take profit"""
        try:
            # Check required parameters
            required_params = ["symbol", "quantity", "side"]
            missing_params = [p for p in required_params if p not in parameters or parameters[p] is None]
            if missing_params:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Missing required parameters: {missing_params}"
                )
            
            symbol = parameters["symbol"]
            quantity = float(parameters["quantity"])
            side = parameters["side"].upper()
            entry_price = parameters.get("limit_price")
            stop_loss_price = parameters.get("stop_loss_price")
            take_profit_price = parameters.get("take_profit_price")
            atr = parameters.get("atr")
            
            # Convert types
            if entry_price is not None:
                entry_price = float(entry_price)
            if stop_loss_price is not None:
                stop_loss_price = float(stop_loss_price)
            if take_profit_price is not None:
                take_profit_price = float(take_profit_price)
            if atr is not None:
                atr = float(atr)
            
            bracket_result = await order_manager.create_bracket_order(
                symbol=symbol,
                quantity=quantity,
                side=side,
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                atr=atr
            )
            
            metadata = {
                "action": "bracket_order",
                "executed_at": datetime.now().isoformat(),
                "includes_stop_loss": stop_loss_price is not None or atr is not None,
                "includes_take_profit": take_profit_price is not None
            }
            
            return ToolResult(success=True, data=bracket_result, metadata=metadata)
            
        except Exception as e:
            raise TradingError(f"Bracket order execution failed: {str(e)}")
    
    async def _create_trailing_stop(self, order_manager: OrderManager, parameters: Dict[str, Any]) -> ToolResult:
        """Create a trailing stop order"""
        try:
            # Check required parameters
            required_params = ["symbol", "quantity"]
            missing_params = [p for p in required_params if p not in parameters or parameters[p] is None]
            if missing_params:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Missing required parameters: {missing_params}"
                )
            
            symbol = parameters["symbol"]
            quantity = float(parameters["quantity"])
            trail_percent = float(parameters.get("trail_percent", 0.05))
            atr = parameters.get("atr")
            
            if atr is not None:
                atr = float(atr)
            
            trailing_result = await order_manager.create_trailing_stop(
                symbol=symbol,
                quantity=quantity,
                trail_percent=trail_percent,
                atr=atr
            )
            
            metadata = {
                "action": "trailing_stop",
                "created_at": datetime.now().isoformat(),
                "trail_percent": trail_percent,
                "atr_based": atr is not None
            }
            
            return ToolResult(success=True, data=trailing_result, metadata=metadata)
            
        except Exception as e:
            raise TradingError(f"Trailing stop creation failed: {str(e)}")
    
    async def _cancel_order(self, client: AlpacaClient, parameters: Dict[str, Any]) -> ToolResult:
        """Cancel an order"""
        try:
            order_id = parameters.get("order_id")
            if not order_id:
                return ToolResult(
                    success=False,
                    data=None,
                    error="order_id is required for cancellation"
                )
            
            success = await client.cancel_order(order_id)
            
            result_data = {
                "order_id": order_id,
                "cancelled": success,
                "cancelled_at": datetime.now().isoformat()
            }
            
            metadata = {
                "action": "cancel_order",
                "cancelled_at": datetime.now().isoformat()
            }
            
            return ToolResult(success=success, data=result_data, metadata=metadata)
            
        except Exception as e:
            raise TradingError(f"Order cancellation failed: {str(e)}")
    
    async def _get_order(self, client: AlpacaClient, parameters: Dict[str, Any]) -> ToolResult:
        """Get order details"""
        try:
            order_id = parameters.get("order_id")
            if not order_id:
                return ToolResult(
                    success=False,
                    data=None,
                    error="order_id is required to retrieve order"
                )
            
            order_details = await client.get_order(order_id)
            
            metadata = {
                "action": "get_order",
                "retrieved_at": datetime.now().isoformat()
            }
            
            return ToolResult(success=True, data=order_details, metadata=metadata)
            
        except Exception as e:
            raise TradingError(f"Order retrieval failed: {str(e)}")
    
    async def _get_monitor_status(self, order_manager: OrderManager, parameters: Dict[str, Any]) -> ToolResult:
        """Get order monitoring status"""
        try:
            status = order_manager.get_monitoring_status()
            
            metadata = {
                "action": "monitor_status",
                "retrieved_at": datetime.now().isoformat()
            }
            
            return ToolResult(success=True, data=status, metadata=metadata)
            
        except Exception as e:
            raise TradingError(f"Monitor status retrieval failed: {str(e)}")