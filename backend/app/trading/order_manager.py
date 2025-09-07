"""
Order management system for stop-losses, take-profits, and trailing stops
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import structlog

from app.core.exceptions import TradingError
from app.trading.alpaca_client import AlpacaClient
from app.trading.risk_manager import RiskManager

logger = structlog.get_logger()


class OrderManager:
    """Manages stop-loss, take-profit, and trailing stop orders"""

    def __init__(self, alpaca_client: AlpacaClient, risk_manager: RiskManager):
        self.alpaca_client = alpaca_client
        self.risk_manager = risk_manager
        self.monitored_positions = {}  # {symbol: position_info}
        self.active_orders = {}  # {order_id: order_info}
        self.trailing_stops = {}  # {symbol: trailing_stop_info}

    async def create_bracket_order(
        self,
        symbol: str,
        quantity: float,
        side: str,
        entry_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        atr: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Create a bracket order with entry, stop-loss, and take-profit
        """
        try:
            # Get current market price if not provided
            if not entry_price:
                quotes = await self.alpaca_client.get_latest_quotes([symbol])
                if symbol in quotes:
                    bid = quotes[symbol].get("bid", 0)
                    ask = quotes[symbol].get("ask", 0)
                    entry_price = (bid + ask) / 2 if bid and ask else bid or ask
                else:
                    raise TradingError(f"Could not get market price for {symbol}")

            # Calculate stop-loss and take-profit if not provided
            if not stop_loss_price:
                if atr:
                    # Use 2x ATR for stop-loss
                    stop_loss_price = self._calculate_atr_stop_loss(
                        entry_price, side, atr, multiplier=2.0
                    )
                else:
                    # Use risk manager default
                    stop_loss_price = self.risk_manager.calculate_stop_loss(
                        entry_price, side
                    )

            if not take_profit_price:
                # Use risk manager default (15% take profit)
                take_profit_price = self.risk_manager.calculate_take_profit(
                    entry_price, side
                )

            logger.info(
                "Creating bracket order",
                symbol=symbol,
                quantity=quantity,
                side=side,
                entry_price=entry_price,
                stop_loss=stop_loss_price,
                take_profit=take_profit_price,
            )

            # Submit main order first
            main_order = await self.alpaca_client.execute_order(
                symbol=symbol,
                qty=quantity,
                side=side,
                order_type="market" if not entry_price else "limit",
                limit_price=entry_price if side.upper() == "BUY" else None,
            )

            # Store order for monitoring
            self.active_orders[main_order["id"]] = {
                "symbol": symbol,
                "quantity": quantity,
                "side": side,
                "entry_price": entry_price,
                "stop_loss_price": stop_loss_price,
                "take_profit_price": take_profit_price,
                "status": "pending",
                "created_at": datetime.now(),
            }

            # Set up position monitoring for stop-loss and take-profit
            await self._setup_position_monitoring(
                symbol, quantity, side, entry_price, stop_loss_price, take_profit_price
            )

            return {
                "main_order_id": main_order["id"],
                "symbol": symbol,
                "entry_price": entry_price,
                "stop_loss_price": stop_loss_price,
                "take_profit_price": take_profit_price,
                "status": "submitted",
            }

        except Exception as e:
            logger.error("Failed to create bracket order", error=str(e), symbol=symbol)
            raise TradingError(f"Failed to create bracket order: {e}")

    async def create_trailing_stop(
        self,
        symbol: str,
        quantity: float,
        trail_percent: float = 0.05,  # 5% trail
        atr: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Create a trailing stop order
        """
        try:
            # Get current market price
            quotes = await self.alpaca_client.get_latest_quotes([symbol])
            if symbol not in quotes:
                raise TradingError(f"Could not get market price for {symbol}")

            current_price = quotes[symbol].get("bid") or quotes[symbol].get("ask")
            if not current_price:
                raise TradingError(f"No valid price data for {symbol}")

            # Determine side based on current position
            positions = await self.alpaca_client.get_positions()
            position = next((p for p in positions if p["symbol"] == symbol), None)

            if not position:
                raise TradingError(f"No position found for {symbol}")

            position_qty = position["qty"]
            side = "sell" if position_qty > 0 else "buy"  # Opposite of position

            # Calculate initial trail distance
            if atr:
                trail_distance = atr * 2  # 2x ATR
                trail_percent = trail_distance / current_price

            # Submit trailing stop order
            trail_order = await self.alpaca_client.execute_order(
                symbol=symbol,
                qty=abs(quantity),
                side=side,
                order_type="trailing_stop",
                stop_price=current_price,
                # Note: Alpaca uses trail_percent parameter for trailing stops
            )

            # Store trailing stop info
            self.trailing_stops[symbol] = {
                "order_id": trail_order["id"],
                "symbol": symbol,
                "quantity": quantity,
                "trail_percent": trail_percent,
                "highest_price": current_price if position_qty > 0 else None,
                "lowest_price": current_price if position_qty < 0 else None,
                "created_at": datetime.now(),
            }

            logger.info(
                "Created trailing stop",
                symbol=symbol,
                quantity=quantity,
                trail_percent=trail_percent,
                current_price=current_price,
            )

            return {
                "order_id": trail_order["id"],
                "symbol": symbol,
                "trail_percent": trail_percent,
                "current_price": current_price,
                "status": "active",
            }

        except Exception as e:
            logger.error("Failed to create trailing stop", error=str(e), symbol=symbol)
            raise TradingError(f"Failed to create trailing stop: {e}")

    async def monitor_positions(self):
        """
        Monitor positions for stop-loss and take-profit triggers
        """
        try:
            # Get current positions
            positions = await self.alpaca_client.get_positions()

            # Get current quotes for all symbols
            symbols = [pos["symbol"] for pos in positions]
            if not symbols:
                return

            quotes = await self.alpaca_client.get_latest_quotes(symbols)

            for position in positions:
                symbol = position["symbol"]
                current_qty = position["qty"]
                current_price = quotes.get(symbol, {}).get("bid") or quotes.get(
                    symbol, {}
                ).get("ask")

                if not current_price or symbol not in self.monitored_positions:
                    continue

                monitor_info = self.monitored_positions[symbol]
                entry_price = monitor_info["entry_price"]
                stop_loss_price = monitor_info["stop_loss_price"]
                take_profit_price = monitor_info["take_profit_price"]
                side = monitor_info["side"]

                # Check stop-loss trigger
                stop_loss_triggered = False
                take_profit_triggered = False

                if side.upper() == "BUY":
                    if current_price <= stop_loss_price:
                        stop_loss_triggered = True
                    elif current_price >= take_profit_price:
                        take_profit_triggered = True
                else:  # SELL
                    if current_price >= stop_loss_price:
                        stop_loss_triggered = True
                    elif current_price <= take_profit_price:
                        take_profit_triggered = True

                # Execute stop-loss or take-profit
                if stop_loss_triggered:
                    await self._execute_stop_loss(
                        symbol, current_qty, current_price, "STOP_LOSS"
                    )
                elif take_profit_triggered:
                    await self._execute_take_profit(
                        symbol, current_qty, current_price, "TAKE_PROFIT"
                    )

                # Update trailing stops
                if symbol in self.trailing_stops:
                    await self._update_trailing_stop(symbol, current_price)

        except Exception as e:
            logger.error("Error monitoring positions", error=str(e))

    async def update_stop_loss(self, symbol: str, new_stop_price: float) -> bool:
        """
        Update stop-loss price for a position
        """
        try:
            if symbol not in self.monitored_positions:
                logger.warning("Symbol not found in monitored positions", symbol=symbol)
                return False

            self.monitored_positions[symbol]["stop_loss_price"] = new_stop_price

            logger.info(
                "Updated stop-loss price", symbol=symbol, new_stop_price=new_stop_price
            )

            return True

        except Exception as e:
            logger.error("Failed to update stop-loss", error=str(e), symbol=symbol)
            return False

    async def cancel_stop_orders(self, symbol: str) -> bool:
        """
        Cancel all stop orders for a symbol
        """
        try:
            # Remove from monitoring
            if symbol in self.monitored_positions:
                del self.monitored_positions[symbol]

            if symbol in self.trailing_stops:
                # Cancel trailing stop order
                trail_info = self.trailing_stops[symbol]
                await self.alpaca_client.cancel_order(trail_info["order_id"])
                del self.trailing_stops[symbol]

            logger.info("Cancelled stop orders", symbol=symbol)
            return True

        except Exception as e:
            logger.error("Failed to cancel stop orders", error=str(e), symbol=symbol)
            return False

    def _calculate_atr_stop_loss(
        self, entry_price: float, side: str, atr: float, multiplier: float = 2.0
    ) -> float:
        """Calculate stop-loss based on ATR"""
        stop_distance = atr * multiplier

        if side.upper() == "BUY":
            return entry_price - stop_distance
        else:  # SELL
            return entry_price + stop_distance

    async def _setup_position_monitoring(
        self,
        symbol: str,
        quantity: float,
        side: str,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
    ):
        """Set up monitoring for a new position"""
        self.monitored_positions[symbol] = {
            "symbol": symbol,
            "quantity": quantity,
            "side": side,
            "entry_price": entry_price,
            "stop_loss_price": stop_loss_price,
            "take_profit_price": take_profit_price,
            "created_at": datetime.now(),
            "status": "monitoring",
        }

        logger.info(
            "Set up position monitoring",
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss_price,
            take_profit=take_profit_price,
        )

    async def _execute_stop_loss(
        self, symbol: str, quantity: float, current_price: float, reason: str
    ):
        """Execute stop-loss order"""
        try:
            # Determine exit side (opposite of entry)
            exit_side = "sell" if quantity > 0 else "buy"

            # Submit market order to exit position
            exit_order = await self.alpaca_client.execute_order(
                symbol=symbol, qty=abs(quantity), side=exit_side, order_type="market"
            )

            # Remove from monitoring
            if symbol in self.monitored_positions:
                entry_price = self.monitored_positions[symbol]["entry_price"]
                pnl = (current_price - entry_price) * quantity

                logger.info(
                    "Stop-loss executed",
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=entry_price,
                    exit_price=current_price,
                    pnl=pnl,
                    reason=reason,
                    order_id=exit_order["id"],
                )

                del self.monitored_positions[symbol]

        except Exception as e:
            logger.error("Failed to execute stop-loss", error=str(e), symbol=symbol)

    async def _execute_take_profit(
        self, symbol: str, quantity: float, current_price: float, reason: str
    ):
        """Execute take-profit order"""
        try:
            # Determine exit side (opposite of entry)
            exit_side = "sell" if quantity > 0 else "buy"

            # Submit market order to exit position
            exit_order = await self.alpaca_client.execute_order(
                symbol=symbol, qty=abs(quantity), side=exit_side, order_type="market"
            )

            # Remove from monitoring
            if symbol in self.monitored_positions:
                entry_price = self.monitored_positions[symbol]["entry_price"]
                pnl = (current_price - entry_price) * quantity

                logger.info(
                    "Take-profit executed",
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=entry_price,
                    exit_price=current_price,
                    pnl=pnl,
                    reason=reason,
                    order_id=exit_order["id"],
                )

                del self.monitored_positions[symbol]

        except Exception as e:
            logger.error("Failed to execute take-profit", error=str(e), symbol=symbol)

    async def _update_trailing_stop(self, symbol: str, current_price: float):
        """Update trailing stop based on current price"""
        try:
            trail_info = self.trailing_stops[symbol]
            trail_percent = trail_info["trail_percent"]

            # Update highest/lowest price
            if trail_info.get("highest_price"):
                # Long position - track highest price
                if current_price > trail_info["highest_price"]:
                    trail_info["highest_price"] = current_price

                    # Calculate new stop price
                    new_stop_price = current_price * (1 - trail_percent)

                    # Update stop order (simplified - would need Alpaca API call)
                    logger.info(
                        "Updated trailing stop",
                        symbol=symbol,
                        current_price=current_price,
                        new_stop_price=new_stop_price,
                        trail_percent=trail_percent,
                    )

            elif trail_info.get("lowest_price"):
                # Short position - track lowest price
                if current_price < trail_info["lowest_price"]:
                    trail_info["lowest_price"] = current_price

                    # Calculate new stop price
                    new_stop_price = current_price * (1 + trail_percent)

                    logger.info(
                        "Updated trailing stop (short)",
                        symbol=symbol,
                        current_price=current_price,
                        new_stop_price=new_stop_price,
                        trail_percent=trail_percent,
                    )

        except Exception as e:
            logger.error("Failed to update trailing stop", error=str(e), symbol=symbol)

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get status of all monitored positions"""
        return {
            "monitored_positions": len(self.monitored_positions),
            "active_orders": len(self.active_orders),
            "trailing_stops": len(self.trailing_stops),
            "positions": list(self.monitored_positions.keys()),
            "details": {
                "monitored": self.monitored_positions,
                "trailing": self.trailing_stops,
            },
        }

    async def cleanup_expired_orders(self, max_age_hours: int = 24):
        """Clean up old orders and monitoring entries"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

        # Clean up old active orders
        expired_orders = [
            order_id
            for order_id, order_info in self.active_orders.items()
            if order_info["created_at"] < cutoff_time
        ]

        for order_id in expired_orders:
            del self.active_orders[order_id]

        # Clean up old monitoring entries
        expired_positions = [
            symbol
            for symbol, pos_info in self.monitored_positions.items()
            if pos_info["created_at"] < cutoff_time
        ]

        for symbol in expired_positions:
            del self.monitored_positions[symbol]

        if expired_orders or expired_positions:
            logger.info(
                "Cleaned up expired entries",
                expired_orders=len(expired_orders),
                expired_positions=len(expired_positions),
            )
