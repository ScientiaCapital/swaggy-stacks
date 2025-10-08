"""
Alpaca API client for trading operations
"""

import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, List, Optional

import alpaca_trade_api as tradeapi
import structlog
from alpaca_trade_api.rest import APIError

from app.core.config import settings
from app.core.exceptions import MarketDataError, TradingError
from app.monitoring.metrics import PrometheusMetrics

logger = structlog.get_logger()


class AlpacaClient:
    """Alpaca API client for trading operations"""

    def __init__(self, api_key: str = None, secret_key: str = None, paper: bool = True):
        self.api_key = api_key or settings.ALPACA_API_KEY
        self.secret_key = secret_key or settings.ALPACA_SECRET_KEY
        self.paper = paper
        self.metrics = PrometheusMetrics()

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

    def _track_api_call(self, endpoint: str, method: str = "GET"):
        """Context manager for tracking Alpaca API calls with metrics."""
        class APICallTracker:
            def __init__(tracker_self, client, endpoint, method):
                tracker_self.client = client
                tracker_self.endpoint = endpoint
                tracker_self.method = method
                tracker_self.start_time = None
                
            def __enter__(tracker_self):
                tracker_self.start_time = time.time()
                return tracker_self
                
            def __exit__(tracker_self, exc_type, exc_val, exc_tb):
                duration = time.time() - tracker_self.start_time
                status_code = 500 if exc_type else 200
                
                # Track request metrics
                tracker_self.client.metrics.track_alpaca_request(
                    endpoint=tracker_self.endpoint,
                    method=tracker_self.method,
                    status_code=status_code,
                    duration=duration
                )
                
                # Track errors if present
                if exc_type:
                    error_type = exc_type.__name__ if exc_type else "unknown"
                    tracker_self.client.metrics.track_alpaca_error(
                        error_type=error_type,
                        endpoint=tracker_self.endpoint
                    )
                
                return False  # Don't suppress exceptions
        
        return APICallTracker(self, endpoint, method)

    async def get_account(self) -> Dict[str, Any]:
        """Get account information"""
        try:
            with self._track_api_call("/account", "GET"):
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

            with self._track_api_call("/orders", "POST"):
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

    # ==================== OPTIONS TRADING METHODS ====================

    async def get_option_chain(
        self,
        symbol: str,
        expiration_date: Optional[str] = None,
        strike_price_gte: Optional[float] = None,
        strike_price_lte: Optional[float] = None,
        option_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get option chain for a symbol

        Args:
            symbol: Underlying symbol (e.g., 'AAPL')
            expiration_date: Specific expiration date (YYYY-MM-DD)
            strike_price_gte: Minimum strike price
            strike_price_lte: Maximum strike price
            option_type: 'call' or 'put'
            limit: Maximum number of options to return
        """
        try:
            # Use Alpaca options data API
            # Note: This is based on Alpaca's options data structure
            params = {
                "symbols": symbol,
                "limit": limit,
            }

            if expiration_date:
                params["expiration_date"] = expiration_date
            if strike_price_gte:
                params["strike_price_gte"] = strike_price_gte
            if strike_price_lte:
                params["strike_price_lte"] = strike_price_lte
            if option_type:
                params["type"] = option_type

            # For now, return mock data structure that matches expected format
            # This will be replaced with actual Alpaca options API call
            logger.info("Retrieving option chain", symbol=symbol, params=params)

            # Mock response matching OptionContract structure
            mock_options = []

            # This would be replaced with actual API call:
            # options_data = self.api.get_options(**params)

            return mock_options

        except APIError as e:
            logger.error("Failed to get option chain", error=str(e), symbol=symbol)
            raise TradingError(f"Failed to get option chain: {e}")

    async def get_option_quote(self, option_symbol: str) -> Dict[str, Any]:
        """
        Get real-time quote for an option contract

        Args:
            option_symbol: Option symbol (e.g., 'AAPL230616C00150000')
        """
        try:
            # Use Alpaca options quote API
            # This would call the actual options quote endpoint
            logger.info("Getting option quote", option_symbol=option_symbol)

            # Mock response structure
            quote_data = {
                "symbol": option_symbol,
                "bid": 0.0,
                "ask": 0.0,
                "bid_size": 0,
                "ask_size": 0,
                "last_price": 0.0,
                "last_size": 0,
                "volume": 0,
                "open_interest": 0,
                "implied_volatility": 0.0,
                "delta": 0.0,
                "gamma": 0.0,
                "theta": 0.0,
                "vega": 0.0,
                "timestamp": datetime.now().isoformat()
            }

            return quote_data

        except APIError as e:
            logger.error("Failed to get option quote", error=str(e), option_symbol=option_symbol)
            raise TradingError(f"Failed to get option quote: {e}")

    async def get_options_positions(self) -> List[Dict[str, Any]]:
        """Get current options positions"""
        try:
            # Get all positions and filter for options
            all_positions = self.api.list_positions()

            options_positions = []
            for pos in all_positions:
                # Options symbols typically have specific format
                if self._is_option_symbol(pos.symbol):
                    position_data = {
                        "symbol": pos.symbol,
                        "underlying_symbol": self._extract_underlying_symbol(pos.symbol),
                        "qty": float(pos.qty),
                        "side": pos.side,
                        "market_value": float(pos.market_value),
                        "cost_basis": float(pos.cost_basis),
                        "unrealized_pl": float(pos.unrealized_pl),
                        "unrealized_plpc": float(pos.unrealized_plpc),
                        "current_price": float(pos.current_price),
                        "option_type": self._extract_option_type(pos.symbol),
                        "strike_price": self._extract_strike_price(pos.symbol),
                        "expiration_date": self._extract_expiration_date(pos.symbol),
                    }
                    options_positions.append(position_data)

            logger.info("Retrieved options positions", count=len(options_positions))
            return options_positions

        except APIError as e:
            logger.error("Failed to get options positions", error=str(e))
            raise TradingError(f"Failed to get options positions: {e}")

    async def execute_multi_leg_order(
        self,
        legs: List[Dict[str, Any]],
        order_class: str = "MLEG",
        time_in_force: str = "gtc",
        client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a multi-leg options order (e.g., spreads, straddles)

        Args:
            legs: List of order legs, each containing:
                - symbol: Option symbol
                - qty: Quantity (positive for buy, negative for sell)
                - side: 'buy' or 'sell'
            order_class: Order class ('MLEG' for multi-leg)
            time_in_force: Time in force
            client_order_id: Client order ID
        """
        try:
            if not legs or len(legs) < 2:
                raise TradingError("Multi-leg order requires at least 2 legs")

            if len(legs) > 4:
                raise TradingError("Multi-leg order supports maximum 4 legs")

            # Validate all legs have required fields
            for i, leg in enumerate(legs):
                required_fields = ['symbol', 'qty', 'side']
                for field in required_fields:
                    if field not in leg:
                        raise TradingError(f"Leg {i} missing required field: {field}")

            # Prepare multi-leg order parameters
            order_params = {
                "order_class": order_class,
                "time_in_force": time_in_force,
                "legs": []
            }

            if client_order_id:
                order_params["client_order_id"] = client_order_id

            # Process each leg
            for leg in legs:
                leg_params = {
                    "symbol": leg["symbol"],
                    "qty": abs(float(leg["qty"])),  # Quantity should be positive
                    "side": leg["side"],
                    "type": leg.get("type", "market"),
                }

                # Add limit price if provided
                if "limit_price" in leg:
                    leg_params["limit_price"] = leg["limit_price"]
                    leg_params["type"] = "limit"

                order_params["legs"].append(leg_params)

            # Submit multi-leg order
            # Note: This uses mock implementation - actual Alpaca API call would be:
            # order = self.api.submit_order(**order_params)

            logger.info(
                "Multi-leg order submitted",
                leg_count=len(legs),
                order_class=order_class,
                paper_trading=self.paper
            )

            # Mock response for now
            mock_order = {
                "id": f"mleg_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "order_class": order_class,
                "status": "accepted" if self.paper else "submitted",
                "legs": order_params["legs"],
                "submitted_at": datetime.now().isoformat(),
                "time_in_force": time_in_force,
                "paper_trading": self.paper
            }

            return mock_order

        except APIError as e:
            logger.error("Failed to execute multi-leg order", error=str(e))
            raise TradingError(f"Failed to execute multi-leg order: {e}")
        except Exception as e:
            logger.error("Multi-leg order validation failed", error=str(e))
            raise TradingError(f"Multi-leg order validation failed: {e}")

    async def get_latest_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest price for a symbol"""
        try:
            trades = await self.get_latest_trades([symbol])
            if symbol in trades:
                return {
                    "symbol": symbol,
                    "price": trades[symbol]["price"],
                    "timestamp": trades[symbol]["timestamp"]
                }
            return None
        except Exception as e:
            logger.error("Failed to get latest price", error=str(e), symbol=symbol)
            return None

    async def get_historical_data(self, symbol: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get historical daily data for volatility calculations"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            data = await self.get_market_data(
                symbols=[symbol],
                timeframe="1Day",
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                limit=days
            )

            return data.get(symbol, [])

        except Exception as e:
            logger.error("Failed to get historical data", error=str(e), symbol=symbol)
            return []

    # Helper methods for option symbol parsing
    def _is_option_symbol(self, symbol: str) -> bool:
        """Check if symbol is an option (basic heuristic)"""
        # Options symbols are typically longer and contain expiration/strike info
        return len(symbol) > 10 and any(char.isdigit() for char in symbol)

    def _extract_underlying_symbol(self, option_symbol: str) -> str:
        """Extract underlying symbol from option symbol"""
        # This is a simplified implementation
        # Actual implementation would need to handle various option symbol formats
        import re
        # Remove numbers and common option suffixes to get underlying
        underlying = re.sub(r'\d+', '', option_symbol)
        underlying = re.sub(r'[CP]$', '', underlying)  # Remove C/P suffix
        return underlying[:6]  # Take first 6 chars as approximation

    def _extract_option_type(self, option_symbol: str) -> str:
        """Extract option type (call/put) from option symbol"""
        if option_symbol.endswith('C') or 'C' in option_symbol[-3:]:
            return "call"
        elif option_symbol.endswith('P') or 'P' in option_symbol[-3:]:
            return "put"
        return "unknown"

    def _extract_strike_price(self, option_symbol: str) -> float:
        """Extract strike price from option symbol"""
        # Simplified implementation - actual would parse standard option symbol format
        import re
        numbers = re.findall(r'\d+', option_symbol)
        if numbers:
            # Typically the last or second-to-last number group is strike
            return float(numbers[-1]) / 1000 if len(numbers[-1]) > 3 else float(numbers[-1])
        return 0.0

    def _extract_expiration_date(self, option_symbol: str) -> str:
        """Extract expiration date from option symbol"""
        # Simplified implementation - would need proper option symbol parsing
        # For now, return placeholder
        return "2024-01-19"  # Placeholder
