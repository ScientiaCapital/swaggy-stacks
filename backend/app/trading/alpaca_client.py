"""
Alpaca API client for trading operations
"""

import asyncio
from typing import Any, Dict, List, Optional

import alpaca_trade_api as tradeapi
import structlog
from alpaca_trade_api.rest import APIError

from app.core.config import settings
from app.core.exceptions import MarketDataError, TradingError

logger = structlog.get_logger()


class AlpacaClient:
    """Alpaca API client for trading operations"""

    def __init__(self, api_key: str = None, secret_key: str = None, paper: bool = True):
        self.api_key = api_key or settings.ALPACA_API_KEY
        self.secret_key = secret_key or settings.ALPACA_SECRET_KEY
        self.paper = paper

        if not self.api_key or not self.secret_key:
            raise TradingError("Alpaca API credentials not provided")

        self.api = tradeapi.REST(
            self.api_key,
            self.secret_key,
            base_url=settings.ALPACA_BASE_URL,
            api_version="v2",
        )

        self.data_api = tradeapi.REST(
            self.api_key,
            self.secret_key,
            base_url=settings.ALPACA_DATA_URL,
            api_version="v2",
        )

        logger.info("Alpaca client initialized", paper=paper)

    async def get_account(self) -> Dict[str, Any]:
        """Get account information"""
        try:
            account = self.api.get_account()
            return {
                "id": account.id,
                "account_number": account.account_number,
                "status": account.status,
                "currency": account.currency,
                "buying_power": float(account.buying_power),
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "equity": float(account.equity),
                "last_equity": float(account.last_equity),
                "long_market_value": float(account.long_market_value),
                "short_market_value": float(account.short_market_value),
                "initial_margin": float(account.initial_margin),
                "maintenance_margin": float(account.maintenance_margin),
                "daytrade_count": account.daytrade_count,
                "pattern_day_trader": account.pattern_day_trader,
            }
        except APIError as e:
            logger.error("Failed to get account info", error=str(e))
            raise TradingError(f"Failed to get account info: {e}")

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        try:
            positions = self.api.list_positions()
            return [
                {
                    "symbol": pos.symbol,
                    "qty": float(pos.qty),
                    "side": pos.side,
                    "market_value": float(pos.market_value),
                    "cost_basis": float(pos.cost_basis),
                    "unrealized_pl": float(pos.unrealized_pl),
                    "unrealized_plpc": float(pos.unrealized_plpc),
                    "current_price": float(pos.current_price),
                    "lastday_price": float(pos.lastday_price),
                    "change_today": float(pos.change_today),
                }
                for pos in positions
            ]
        except APIError as e:
            logger.error("Failed to get positions", error=str(e))
            raise TradingError(f"Failed to get positions: {e}")

    async def execute_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "gtc",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        trail_price: Optional[float] = None,
        trail_percent: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a trading order"""
        try:
            order_params = {
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "type": order_type,
                "time_in_force": time_in_force,
                "client_order_id": client_order_id,
            }

            # Add price parameters if provided
            if limit_price is not None:
                order_params["limit_price"] = limit_price
            if stop_price is not None:
                order_params["stop_price"] = stop_price
            if trail_price is not None:
                order_params["trail_price"] = trail_price
            if trail_percent is not None:
                order_params["trail_percent"] = trail_percent

            order = self.api.submit_order(**order_params)

            logger.info(
                "Order submitted",
                symbol=symbol,
                qty=qty,
                side=side,
                order_type=order_type,
                order_id=order.id,
            )

            return {
                "id": order.id,
                "client_order_id": order.client_order_id,
                "symbol": order.symbol,
                "qty": float(order.qty),
                "side": order.side,
                "type": order.type,
                "time_in_force": order.time_in_force,
                "status": order.status,
                "submitted_at": order.submitted_at,
                "limit_price": float(order.limit_price) if order.limit_price else None,
                "stop_price": float(order.stop_price) if order.stop_price else None,
            }
        except APIError as e:
            logger.error("Failed to execute order", error=str(e), symbol=symbol)
            raise TradingError(f"Failed to execute order: {e}")

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            self.api.cancel_order(order_id)
            logger.info("Order cancelled", order_id=order_id)
            return True
        except APIError as e:
            logger.error("Failed to cancel order", error=str(e), order_id=order_id)
            raise TradingError(f"Failed to cancel order: {e}")

    async def get_order(self, order_id: str) -> Dict[str, Any]:
        """Get order details"""
        try:
            order = self.api.get_order(order_id)
            return {
                "id": order.id,
                "client_order_id": order.client_order_id,
                "symbol": order.symbol,
                "qty": float(order.qty),
                "side": order.side,
                "type": order.type,
                "time_in_force": order.time_in_force,
                "status": order.status,
                "submitted_at": order.submitted_at,
                "filled_at": order.filled_at,
                "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
                "filled_avg_price": (
                    float(order.filled_avg_price) if order.filled_avg_price else None
                ),
                "limit_price": float(order.limit_price) if order.limit_price else None,
                "stop_price": float(order.stop_price) if order.stop_price else None,
            }
        except APIError as e:
            logger.error("Failed to get order", error=str(e), order_id=order_id)
            raise TradingError(f"Failed to get order: {e}")

    async def get_orders(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        after: Optional[str] = None,
        until: Optional[str] = None,
        direction: str = "desc",
    ) -> List[Dict[str, Any]]:
        """Get list of orders"""
        try:
            orders = self.api.list_orders(
                status=status,
                limit=limit,
                after=after,
                until=until,
                direction=direction,
            )

            return [
                {
                    "id": order.id,
                    "client_order_id": order.client_order_id,
                    "symbol": order.symbol,
                    "qty": float(order.qty),
                    "side": order.side,
                    "type": order.type,
                    "time_in_force": order.time_in_force,
                    "status": order.status,
                    "submitted_at": order.submitted_at,
                    "filled_at": order.filled_at,
                    "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
                    "filled_avg_price": (
                        float(order.filled_avg_price)
                        if order.filled_avg_price
                        else None
                    ),
                }
                for order in orders
            ]
        except APIError as e:
            logger.error("Failed to get orders", error=str(e))
            raise TradingError(f"Failed to get orders: {e}")

    async def get_market_data(
        self,
        symbols: List[str],
        timeframe: str = "1Min",
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 1000,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get historical market data"""
        try:
            data = {}
            for symbol in symbols:
                bars = self.data_api.get_bars(
                    symbol, timeframe, start=start, end=end, limit=limit
                )

                data[symbol] = [
                    {
                        "timestamp": bar.t,
                        "open": float(bar.o),
                        "high": float(bar.h),
                        "low": float(bar.l),
                        "close": float(bar.c),
                        "volume": int(bar.v),
                        "trade_count": int(bar.n) if hasattr(bar, "n") else None,
                        "vwap": float(bar.vw) if hasattr(bar, "vw") else None,
                    }
                    for bar in bars
                ]

            return data
        except APIError as e:
            logger.error("Failed to get market data", error=str(e), symbols=symbols)
            raise MarketDataError(f"Failed to get market data: {e}")

    async def get_latest_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get latest quotes for symbols"""
        try:
            quotes = self.data_api.get_latest_quotes(symbols)
            return {
                symbol: {
                    "bid": float(quote.bid_price) if quote.bid_price else None,
                    "ask": float(quote.ask_price) if quote.ask_price else None,
                    "bid_size": int(quote.bid_size) if quote.bid_size else None,
                    "ask_size": int(quote.ask_size) if quote.ask_size else None,
                    "timestamp": quote.timestamp,
                }
                for symbol, quote in quotes.items()
            }
        except APIError as e:
            logger.error("Failed to get latest quotes", error=str(e), symbols=symbols)
            raise MarketDataError(f"Failed to get latest quotes: {e}")

    async def get_latest_trades(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get latest trades for symbols"""
        try:
            trades = self.data_api.get_latest_trades(symbols)
            return {
                symbol: {
                    "price": float(trade.price),
                    "size": int(trade.size),
                    "timestamp": trade.timestamp,
                    "conditions": trade.conditions,
                }
                for symbol, trade in trades.items()
            }
        except APIError as e:
            logger.error("Failed to get latest trades", error=str(e), symbols=symbols)
            raise MarketDataError(f"Failed to get latest trades: {e}")

    async def submit_bracket_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        limit_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        time_in_force: str = "gtc",
    ) -> Dict[str, Any]:
        """
        Submit a bracket order (main + stop loss + take profit)
        """
        try:
            # Prepare bracket order parameters
            order_class = "bracket"

            # Main order parameters
            order_params = {
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "type": "market" if not limit_price else "limit",
                "time_in_force": time_in_force,
                "order_class": order_class,
            }

            if limit_price:
                order_params["limit_price"] = limit_price

            # Stop loss leg
            if stop_loss_price:
                order_params["stop_loss"] = {
                    "stop_price": stop_loss_price,
                    "limit_price": stop_loss_price,  # Stop limit order
                }

            # Take profit leg
            if take_profit_price:
                order_params["take_profit"] = {"limit_price": take_profit_price}

            # Submit bracket order
            order = self.api.submit_order(**order_params)

            logger.info(
                "Bracket order submitted",
                symbol=symbol,
                qty=qty,
                side=side,
                limit_price=limit_price,
                stop_loss=stop_loss_price,
                take_profit=take_profit_price,
                order_id=order.id,
            )

            return {
                "id": order.id,
                "symbol": order.symbol,
                "qty": float(order.qty),
                "side": order.side,
                "type": order.type,
                "order_class": order.order_class,
                "status": order.status,
                "submitted_at": order.submitted_at,
                "limit_price": float(order.limit_price) if order.limit_price else None,
                "stop_loss_price": stop_loss_price,
                "take_profit_price": take_profit_price,
                "legs": getattr(order, "legs", []),  # Child orders
            }

        except APIError as e:
            logger.error("Failed to submit bracket order", error=str(e), symbol=symbol)
            raise TradingError(f"Failed to submit bracket order: {e}")

    async def get_portfolio_history(
        self,
        period: str = "1M",  # 1D, 7D, 1M, 3M, 6M, 1Y, 2Y, 5Y
        timeframe: str = "1D",  # 1Min, 5Min, 15Min, 1H, 1D
    ) -> Dict[str, Any]:
        """Get portfolio performance history"""
        try:
            history = self.api.get_portfolio_history(
                period=period, timeframe=timeframe, extended_hours=False
            )

            return {
                "timestamp": history.timestamp,
                "equity": [float(val) for val in history.equity],
                "profit_loss": [float(val) for val in history.profit_loss],
                "profit_loss_pct": [float(val) for val in history.profit_loss_pct],
                "base_value": float(history.base_value),
                "timeframe": history.timeframe,
            }

        except APIError as e:
            logger.error("Failed to get portfolio history", error=str(e))
            raise TradingError(f"Failed to get portfolio history: {e}")

    async def calculate_atr(self, symbol: str, period: int = 14) -> Optional[float]:
        """
        Calculate Average True Range (ATR) for volatility-based stops
        """
        try:
            # Get recent daily bars
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period * 2)  # Extra buffer

            bars = self.data_api.get_bars(
                symbol,
                "1Day",
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                limit=period + 10,
            )

            if len(bars) < period:
                logger.warning("Insufficient data for ATR calculation", symbol=symbol)
                return None

            # Calculate True Range for each period
            true_ranges = []
            for i in range(1, len(bars)):
                current = bars[i]
                previous = bars[i - 1]

                high_low = current.h - current.l
                high_close = abs(current.h - previous.c)
                low_close = abs(current.l - previous.c)

                true_range = max(high_low, high_close, low_close)
                true_ranges.append(true_range)

            # Calculate ATR (simple moving average of True Range)
            if len(true_ranges) >= period:
                atr = sum(true_ranges[-period:]) / period
                logger.info("ATR calculated", symbol=symbol, atr=atr, period=period)
                return atr

            return None

        except Exception as e:
            logger.error("Failed to calculate ATR", error=str(e), symbol=symbol)
            return None
