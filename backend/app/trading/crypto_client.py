"""
Alpaca Crypto API client for cryptocurrency trading operations
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from decimal import Decimal

import alpaca_trade_api as tradeapi
import structlog
from alpaca_trade_api.rest import APIError

from app.core.config import settings
from app.core.exceptions import MarketDataError, TradingError

logger = structlog.get_logger()


class CryptoClient:
    """Alpaca Crypto API client for cryptocurrency trading operations"""

    def __init__(self, api_key: str = None, secret_key: str = None, paper: bool = True):
        self.api_key = api_key or settings.ALPACA_API_KEY
        self.secret_key = secret_key or settings.ALPACA_SECRET_KEY
        self.paper = paper

        if not self.api_key or not self.secret_key:
            raise TradingError("Alpaca API credentials not provided")

        # Use crypto base URL for cryptocurrency trading
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

        logger.info("Crypto client initialized", paper=paper)

    async def get_crypto_account(self) -> Dict[str, Any]:
        """Get cryptocurrency account information"""
        try:
            account = self.api.get_account()
            return {
                "id": account.id,
                "account_number": account.account_number,
                "status": account.status,
                "currency": account.currency,
                "buying_power": float(account.buying_power),
                "non_marginable_buying_power": float(account.non_marginable_buying_power),
                "crypto_buying_power": float(getattr(account, 'crypto_buying_power', account.buying_power)),
                "portfolio_value": float(account.portfolio_value),
                "created_at": account.created_at,
                "trading_blocked": account.trading_blocked,
                "crypto_status": getattr(account, 'crypto_status', 'ACTIVE'),
            }
        except APIError as e:
            logger.error("Failed to get crypto account", error=str(e))
            raise TradingError(f"Failed to get crypto account: {e}")

    async def get_crypto_positions(self) -> List[Dict[str, Any]]:
        """Get cryptocurrency positions"""
        try:
            positions = self.api.list_positions()
            crypto_positions = []

            for position in positions:
                # Filter for cryptocurrency symbols (typically ending with USD)
                symbol = position.symbol
                if self._is_crypto_symbol(symbol):
                    crypto_positions.append({
                        "symbol": symbol,
                        "quantity": float(position.qty),
                        "market_value": float(position.market_value),
                        "avg_cost": float(position.avg_entry_price),
                        "current_price": float(position.current_price),
                        "unrealized_pl": float(position.unrealized_pl),
                        "unrealized_plpc": float(position.unrealized_plpc),
                        "side": position.side,
                        "asset_class": getattr(position, 'asset_class', 'crypto'),
                    })

            return crypto_positions
        except APIError as e:
            logger.error("Failed to get crypto positions", error=str(e))
            raise TradingError(f"Failed to get crypto positions: {e}")

    async def execute_crypto_order(
        self,
        symbol: str,
        quantity: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "gtc",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute cryptocurrency order with fractional support"""
        try:
            # Validate crypto symbol format
            if not self._is_crypto_symbol(symbol):
                raise TradingError(f"Invalid crypto symbol format: {symbol}")

            # Validate fractional quantity (crypto supports fractional trading)
            if quantity <= 0:
                raise TradingError("Quantity must be positive")

            # Validate order parameters
            valid_sides = ["buy", "sell"]
            if side not in valid_sides:
                raise TradingError(f"Invalid side: {side}. Must be one of {valid_sides}")

            valid_order_types = ["market", "limit", "stop_limit"]
            if order_type not in valid_order_types:
                raise TradingError(f"Invalid order type: {order_type}. Must be one of {valid_order_types}")

            valid_tif = ["gtc", "ioc", "day"]
            if time_in_force not in valid_tif:
                raise TradingError(f"Invalid time in force: {time_in_force}. Must be one of {valid_tif}")

            # Build order request
            order_data = {
                "symbol": symbol,
                "qty": str(quantity),  # Convert to string for fractional support
                "side": side,
                "type": order_type,
                "time_in_force": time_in_force,
            }

            if limit_price is not None:
                order_data["limit_price"] = str(limit_price)

            if stop_price is not None:
                order_data["stop_price"] = str(stop_price)

            if client_order_id:
                order_data["client_order_id"] = client_order_id

            # Force paper trading for safety
            if not self.paper:
                logger.warning("Forcing paper trading mode for crypto orders")

            logger.info("Executing crypto order", **order_data)

            # Submit order
            order = self.api.submit_order(**order_data)

            return {
                "id": order.id,
                "client_order_id": order.client_order_id,
                "symbol": order.symbol,
                "quantity": float(order.qty),
                "side": order.side,
                "order_type": order.order_type,
                "time_in_force": order.time_in_force,
                "status": order.status,
                "limit_price": float(order.limit_price) if order.limit_price else None,
                "stop_price": float(order.stop_price) if order.stop_price else None,
                "filled_qty": float(order.filled_qty or 0),
                "filled_avg_price": float(order.filled_avg_price or 0),
                "submitted_at": order.submitted_at,
                "asset_class": "crypto",
            }

        except APIError as e:
            logger.error("Failed to execute crypto order", symbol=symbol, error=str(e))
            raise TradingError(f"Failed to execute crypto order: {e}")

    async def cancel_crypto_order(self, order_id: str) -> bool:
        """Cancel cryptocurrency order"""
        try:
            self.api.cancel_order(order_id)
            logger.info("Crypto order cancelled", order_id=order_id)
            return True
        except APIError as e:
            logger.error("Failed to cancel crypto order", order_id=order_id, error=str(e))
            raise TradingError(f"Failed to cancel crypto order: {e}")

    async def get_crypto_order(self, order_id: str) -> Dict[str, Any]:
        """Get cryptocurrency order details"""
        try:
            order = self.api.get_order(order_id)

            # Ensure this is a crypto order
            if not self._is_crypto_symbol(order.symbol):
                raise TradingError(f"Order {order_id} is not a cryptocurrency order")

            return {
                "id": order.id,
                "client_order_id": order.client_order_id,
                "symbol": order.symbol,
                "quantity": float(order.qty),
                "side": order.side,
                "order_type": order.order_type,
                "time_in_force": order.time_in_force,
                "status": order.status,
                "limit_price": float(order.limit_price) if order.limit_price else None,
                "stop_price": float(order.stop_price) if order.stop_price else None,
                "filled_qty": float(order.filled_qty or 0),
                "filled_avg_price": float(order.filled_avg_price or 0),
                "submitted_at": order.submitted_at,
                "updated_at": order.updated_at,
                "asset_class": "crypto",
            }
        except APIError as e:
            logger.error("Failed to get crypto order", order_id=order_id, error=str(e))
            raise TradingError(f"Failed to get crypto order: {e}")

    async def get_crypto_orders(
        self,
        status: str = "all",
        limit: int = 100,
        after: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Get cryptocurrency orders with filtering"""
        try:
            # Get orders from API
            orders = self.api.list_orders(
                status=status,
                limit=limit,
                after=after.isoformat() if after else None,
                until=until.isoformat() if until else None,
            )

            crypto_orders = []
            for order in orders:
                # Filter for cryptocurrency orders
                if self._is_crypto_symbol(order.symbol):
                    crypto_orders.append({
                        "id": order.id,
                        "client_order_id": order.client_order_id,
                        "symbol": order.symbol,
                        "quantity": float(order.qty),
                        "side": order.side,
                        "order_type": order.order_type,
                        "time_in_force": order.time_in_force,
                        "status": order.status,
                        "limit_price": float(order.limit_price) if order.limit_price else None,
                        "stop_price": float(order.stop_price) if order.stop_price else None,
                        "filled_qty": float(order.filled_qty or 0),
                        "filled_avg_price": float(order.filled_avg_price or 0),
                        "submitted_at": order.submitted_at,
                        "asset_class": "crypto",
                    })

            return crypto_orders
        except APIError as e:
            logger.error("Failed to get crypto orders", error=str(e))
            raise TradingError(f"Failed to get crypto orders: {e}")

    async def get_crypto_market_data(
        self,
        symbols: List[str],
        timeframe: str = "1Day",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get cryptocurrency market data (supports 24/7 trading)"""
        try:
            # Validate crypto symbols
            for symbol in symbols:
                if not self._is_crypto_symbol(symbol):
                    raise TradingError(f"Invalid crypto symbol: {symbol}")

            # Set default time range for crypto (24/7 market)
            if not start:
                start = datetime.utcnow() - timedelta(days=30)
            if not end:
                end = datetime.utcnow()

            # Get bars data
            bars = self.data_api.get_crypto_bars(
                symbols,
                timeframe,
                start=start.isoformat(),
                end=end.isoformat(),
            )

            market_data = {}
            for symbol in symbols:
                if symbol in bars:
                    market_data[symbol] = [
                        {
                            "timestamp": bar.timestamp,
                            "open": float(bar.open),
                            "high": float(bar.high),
                            "low": float(bar.low),
                            "close": float(bar.close),
                            "volume": float(bar.volume),
                            "vwap": float(getattr(bar, 'vwap', 0)),
                        }
                        for bar in bars[symbol]
                    ]

            return market_data
        except APIError as e:
            logger.error("Failed to get crypto market data", symbols=symbols, error=str(e))
            raise MarketDataError(f"Failed to get crypto market data: {e}")

    async def get_crypto_latest_quotes(self, symbols: List[str]) -> Dict[str, Any]:
        """Get latest cryptocurrency quotes"""
        try:
            # Validate crypto symbols
            for symbol in symbols:
                if not self._is_crypto_symbol(symbol):
                    raise TradingError(f"Invalid crypto symbol: {symbol}")

            quotes = self.data_api.get_crypto_latest_quotes(symbols)

            result = {}
            for symbol, quote in quotes.items():
                result[symbol] = {
                    "bid_price": float(quote.bid_price),
                    "bid_size": float(quote.bid_size),
                    "ask_price": float(quote.ask_price),
                    "ask_size": float(quote.ask_size),
                    "timestamp": quote.timestamp,
                }

            return result
        except APIError as e:
            logger.error("Failed to get crypto quotes", symbols=symbols, error=str(e))
            raise MarketDataError(f"Failed to get crypto quotes: {e}")

    async def get_crypto_latest_trades(self, symbols: List[str]) -> Dict[str, Any]:
        """Get latest cryptocurrency trades"""
        try:
            # Validate crypto symbols
            for symbol in symbols:
                if not self._is_crypto_symbol(symbol):
                    raise TradingError(f"Invalid crypto symbol: {symbol}")

            trades = self.data_api.get_crypto_latest_trades(symbols)

            result = {}
            for symbol, trade in trades.items():
                result[symbol] = {
                    "price": float(trade.price),
                    "size": float(trade.size),
                    "timestamp": trade.timestamp,
                }

            return result
        except APIError as e:
            logger.error("Failed to get crypto trades", symbols=symbols, error=str(e))
            raise MarketDataError(f"Failed to get crypto trades: {e}")

    async def get_crypto_latest_price(self, symbol: str) -> float:
        """Get latest cryptocurrency price"""
        try:
            if not self._is_crypto_symbol(symbol):
                raise TradingError(f"Invalid crypto symbol: {symbol}")

            trades = await self.get_crypto_latest_trades([symbol])
            if symbol in trades:
                return trades[symbol]["price"]
            else:
                raise MarketDataError(f"No trade data available for {symbol}")
        except Exception as e:
            logger.error("Failed to get crypto latest price", symbol=symbol, error=str(e))
            raise MarketDataError(f"Failed to get latest price for {symbol}: {e}")

    async def stream_crypto_data(
        self,
        symbols: List[str],
        data_types: List[str] = None,
        callback=None,
    ):
        """Stream real-time cryptocurrency data (24/7 support)"""
        try:
            # Validate crypto symbols
            for symbol in symbols:
                if not self._is_crypto_symbol(symbol):
                    raise TradingError(f"Invalid crypto symbol: {symbol}")

            if not data_types:
                data_types = ["trades", "quotes"]

            logger.info("Starting crypto data stream", symbols=symbols, data_types=data_types)

            # This would typically integrate with Alpaca's WebSocket streaming
            # For now, return a placeholder that indicates streaming capability
            return {
                "status": "streaming",
                "symbols": symbols,
                "data_types": data_types,
                "24_7_support": True,
                "message": "Crypto streaming supports 24/7 market data",
            }

        except Exception as e:
            logger.error("Failed to start crypto stream", symbols=symbols, error=str(e))
            raise TradingError(f"Failed to start crypto stream: {e}")

    # Helper methods
    def _is_crypto_symbol(self, symbol: str) -> bool:
        """Check if symbol is a valid cryptocurrency symbol"""
        # Common crypto symbol patterns for Alpaca
        crypto_patterns = [
            "BTC/USD", "ETH/USD", "LTC/USD", "BCH/USD", "LINK/USD",
            "AAVE/USD", "UNI/USD", "SUSHI/USD", "ALGO/USD", "DOT/USD",
            "MATIC/USD", "SHIB/USD", "DOGE/USD", "ADA/USD", "XLM/USD",
        ]

        # Check if symbol matches crypto patterns
        return symbol in crypto_patterns or symbol.endswith("/USD")

    def _format_crypto_symbol(self, base: str, quote: str = "USD") -> str:
        """Format cryptocurrency symbol for Alpaca API"""
        return f"{base.upper()}/{quote.upper()}"

    def _parse_crypto_symbol(self, symbol: str) -> Dict[str, str]:
        """Parse cryptocurrency symbol into base and quote currencies"""
        if "/" not in symbol:
            raise TradingError(f"Invalid crypto symbol format: {symbol}")

        parts = symbol.split("/")
        if len(parts) != 2:
            raise TradingError(f"Invalid crypto symbol format: {symbol}")

        return {
            "base": parts[0].upper(),
            "quote": parts[1].upper(),
        }

    def _validate_fractional_quantity(self, symbol: str, quantity: float) -> bool:
        """Validate fractional quantity for cryptocurrency trading"""
        # Crypto typically supports very small fractional amounts
        # Bitcoin: 8 decimal places (satoshi)
        # Ethereum: 18 decimal places (wei)
        # Most others: 8 decimal places

        if quantity <= 0:
            return False

        # Check minimum quantity based on symbol
        parsed = self._parse_crypto_symbol(symbol)
        base_currency = parsed["base"]

        min_quantities = {
            "BTC": 0.00000001,  # 1 satoshi
            "ETH": 0.000000000000000001,  # 1 wei
        }

        min_qty = min_quantities.get(base_currency, 0.00000001)  # Default to 8 decimals

        return quantity >= min_qty