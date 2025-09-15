"""
Alpaca WebSocket Stream Manager for real-time market data
"""

import asyncio
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set
from collections import deque, defaultdict
import structlog
from alpaca_trade_api.stream import Stream
from alpaca_trade_api.entity import Trade, Quote, Bar
from alpaca_trade_api.common import URL

from app.core.config import settings
from app.core.exceptions import TradingError

logger = structlog.get_logger(__name__)


class DataBuffer:
    """Thread-safe data buffer for high-frequency market data"""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.trades = deque(maxlen=max_size)
        self.quotes = deque(maxlen=max_size)
        self.bars = deque(maxlen=max_size)
        self.updated_bars = deque(maxlen=max_size)
        self._lock = asyncio.Lock()

    async def add_trade(self, trade: Trade):
        async with self._lock:
            self.trades.append({
                'symbol': trade.symbol,
                'price': float(trade.price),
                'size': int(trade.size),
                'timestamp': trade.timestamp,
                'exchange': getattr(trade, 'exchange', 'unknown'),
                'received_at': datetime.utcnow()
            })

    async def add_quote(self, quote: Quote):
        async with self._lock:
            self.quotes.append({
                'symbol': quote.symbol,
                'bid_price': float(quote.bid_price),
                'bid_size': int(quote.bid_size),
                'ask_price': float(quote.ask_price),
                'ask_size': int(quote.ask_size),
                'timestamp': quote.timestamp,
                'received_at': datetime.utcnow()
            })

    async def add_bar(self, bar: Bar, is_updated: bool = False):
        async with self._lock:
            bar_data = {
                'symbol': bar.symbol,
                'open': float(bar.open),
                'high': float(bar.high),
                'low': float(bar.low),
                'close': float(bar.close),
                'volume': int(bar.volume),
                'timestamp': bar.timestamp,
                'vwap': float(getattr(bar, 'vwap', 0)),
                'received_at': datetime.utcnow()
            }

            if is_updated:
                self.updated_bars.append(bar_data)
            else:
                self.bars.append(bar_data)

    async def get_latest_trades(self, symbol: str = None, limit: int = 100) -> List[Dict]:
        async with self._lock:
            trades = list(self.trades)
            if symbol:
                trades = [t for t in trades if t['symbol'] == symbol]
            return trades[-limit:] if trades else []

    async def get_latest_quotes(self, symbol: str = None, limit: int = 100) -> List[Dict]:
        async with self._lock:
            quotes = list(self.quotes)
            if symbol:
                quotes = [q for q in quotes if q['symbol'] == symbol]
            return quotes[-limit:] if quotes else []

    async def get_latest_bars(self, symbol: str = None, limit: int = 100) -> List[Dict]:
        async with self._lock:
            bars = list(self.bars)
            if symbol:
                bars = [b for b in bars if b['symbol'] == symbol]
            return bars[-limit:] if bars else []


class SubscriptionManager:
    """Manages dynamic symbol subscriptions"""

    def __init__(self):
        self.trade_symbols: Set[str] = set()
        self.quote_symbols: Set[str] = set()
        self.bar_symbols: Set[str] = set()
        self.daily_bar_symbols: Set[str] = set()
        self._lock = asyncio.Lock()

    async def add_trade_subscription(self, symbols: List[str]):
        async with self._lock:
            self.trade_symbols.update(symbols)
            logger.info("Added trade subscriptions", symbols=symbols, total=len(self.trade_symbols))

    async def add_quote_subscription(self, symbols: List[str]):
        async with self._lock:
            self.quote_symbols.update(symbols)
            logger.info("Added quote subscriptions", symbols=symbols, total=len(self.quote_symbols))

    async def add_bar_subscription(self, symbols: List[str]):
        async with self._lock:
            self.bar_symbols.update(symbols)
            logger.info("Added bar subscriptions", symbols=symbols, total=len(self.bar_symbols))

    async def add_daily_bar_subscription(self, symbols: List[str]):
        async with self._lock:
            self.daily_bar_symbols.update(symbols)
            logger.info("Added daily bar subscriptions", symbols=symbols, total=len(self.daily_bar_symbols))

    async def remove_subscription(self, symbols: List[str], data_type: str = "all"):
        async with self._lock:
            symbol_set = set(symbols)
            if data_type in ["all", "trades"]:
                self.trade_symbols -= symbol_set
            if data_type in ["all", "quotes"]:
                self.quote_symbols -= symbol_set
            if data_type in ["all", "bars"]:
                self.bar_symbols -= symbol_set
            if data_type in ["all", "daily_bars"]:
                self.daily_bar_symbols -= symbol_set
            logger.info("Removed subscriptions", symbols=symbols, data_type=data_type)

    async def get_all_symbols(self) -> Set[str]:
        async with self._lock:
            return self.trade_symbols | self.quote_symbols | self.bar_symbols | self.daily_bar_symbols


class AlpacaStreamManager:
    """
    Alpaca WebSocket Stream Manager with advanced features:
    - Auto-reconnection
    - Data buffering
    - Dynamic subscriptions
    - Connection health monitoring
    """

    def __init__(
        self,
        api_key: str = None,
        secret_key: str = None,
        paper: bool = True,
        data_feed: str = "iex"
    ):
        self.api_key = api_key or settings.ALPACA_API_KEY
        self.secret_key = secret_key or settings.ALPACA_SECRET_KEY
        self.paper = paper
        self.data_feed = data_feed  # 'iex' for free, 'sip' for pro

        if not self.api_key or not self.secret_key:
            raise TradingError("Alpaca API credentials not provided")

        # Connection management
        self.is_connected = False
        self.connection_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 5  # seconds
        self.last_heartbeat = None

        # Data management
        self.data_buffer = DataBuffer()
        self.subscription_manager = SubscriptionManager()

        # Callbacks
        self.custom_callbacks: Dict[str, List[Callable]] = defaultdict(list)

        # Stream instance
        self.stream: Optional[Stream] = None

        # Statistics
        self.stats = {
            'trades_received': 0,
            'quotes_received': 0,
            'bars_received': 0,
            'connection_time': None,
            'last_data_time': None
        }

        logger.info("AlpacaStreamManager initialized",
                   paper=paper, data_feed=data_feed)

    def _create_stream(self) -> Stream:
        """Create Alpaca Stream instance"""
        base_url = settings.ALPACA_BASE_URL if self.paper else settings.ALPACA_LIVE_URL

        return Stream(
            key_id=self.api_key,
            secret_key=self.secret_key,
            base_url=URL(base_url),
            data_feed=self.data_feed,
            websocket_params={
                'ping_interval': 10,
                'ping_timeout': 180,
                'max_queue': 1024
            }
        )

    async def initialize(self):
        """Initialize the stream manager"""
        try:
            self.stream = self._create_stream()

            # Register internal callbacks
            self.stream.subscribe_trades(self._handle_trade)
            self.stream.subscribe_quotes(self._handle_quote)
            self.stream.subscribe_bars(self._handle_bar)
            self.stream.subscribe_updated_bars(self._handle_updated_bar)
            self.stream.subscribe_daily_bars(self._handle_daily_bar)

            logger.info("Stream manager initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize stream manager", error=str(e))
            raise TradingError(f"Stream initialization failed: {e}")

    async def connect(self):
        """Connect to Alpaca stream with auto-retry"""
        if not self.stream:
            await self.initialize()

        for attempt in range(self.max_reconnect_attempts):
            try:
                self.connection_attempts = attempt + 1
                logger.info(f"Connecting to Alpaca stream (attempt {self.connection_attempts})")

                # Run stream in background task
                self.stream_task = asyncio.create_task(self._run_stream())

                # Wait a bit to check if connection succeeds
                await asyncio.sleep(2)

                if not self.stream_task.done():
                    self.is_connected = True
                    self.stats['connection_time'] = datetime.utcnow()
                    logger.info("Successfully connected to Alpaca stream")
                    return

            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1} failed", error=str(e))
                if attempt < self.max_reconnect_attempts - 1:
                    await asyncio.sleep(self.reconnect_delay)
                    self.reconnect_delay = min(self.reconnect_delay * 2, 60)  # Exponential backoff

        raise TradingError(f"Failed to connect after {self.max_reconnect_attempts} attempts")

    async def _run_stream(self):
        """Run the stream (blocking call)"""
        try:
            self.stream.run()
        except Exception as e:
            self.is_connected = False
            logger.error("Stream disconnected", error=str(e))
            # Attempt reconnection
            await asyncio.sleep(self.reconnect_delay)
            await self.connect()

    async def disconnect(self):
        """Disconnect from stream"""
        if self.stream_task and not self.stream_task.done():
            self.stream_task.cancel()

        self.is_connected = False
        logger.info("Disconnected from Alpaca stream")

    # Data handlers
    async def _handle_trade(self, trade: Trade):
        """Internal trade handler"""
        try:
            await self.data_buffer.add_trade(trade)
            self.stats['trades_received'] += 1
            self.stats['last_data_time'] = datetime.utcnow()
            self.last_heartbeat = time.time()

            # Call custom callbacks
            for callback in self.custom_callbacks['trade']:
                try:
                    await callback(trade)
                except Exception as e:
                    logger.error("Trade callback failed", error=str(e))

            logger.debug("Trade received",
                        symbol=trade.symbol,
                        price=trade.price,
                        size=trade.size)

        except Exception as e:
            logger.error("Failed to handle trade", error=str(e))

    async def _handle_quote(self, quote: Quote):
        """Internal quote handler"""
        try:
            await self.data_buffer.add_quote(quote)
            self.stats['quotes_received'] += 1
            self.stats['last_data_time'] = datetime.utcnow()
            self.last_heartbeat = time.time()

            # Call custom callbacks
            for callback in self.custom_callbacks['quote']:
                try:
                    await callback(quote)
                except Exception as e:
                    logger.error("Quote callback failed", error=str(e))

            logger.debug("Quote received",
                        symbol=quote.symbol,
                        bid=quote.bid_price,
                        ask=quote.ask_price)

        except Exception as e:
            logger.error("Failed to handle quote", error=str(e))

    async def _handle_bar(self, bar: Bar):
        """Internal bar handler"""
        try:
            await self.data_buffer.add_bar(bar)
            self.stats['bars_received'] += 1
            self.stats['last_data_time'] = datetime.utcnow()
            self.last_heartbeat = time.time()

            # Call custom callbacks
            for callback in self.custom_callbacks['bar']:
                try:
                    await callback(bar)
                except Exception as e:
                    logger.error("Bar callback failed", error=str(e))

            logger.debug("Bar received",
                        symbol=bar.symbol,
                        close=bar.close,
                        volume=bar.volume)

        except Exception as e:
            logger.error("Failed to handle bar", error=str(e))

    async def _handle_updated_bar(self, bar: Bar):
        """Internal updated bar handler"""
        try:
            await self.data_buffer.add_bar(bar, is_updated=True)
            self.last_heartbeat = time.time()

            # Call custom callbacks
            for callback in self.custom_callbacks['updated_bar']:
                try:
                    await callback(bar)
                except Exception as e:
                    logger.error("Updated bar callback failed", error=str(e))

            logger.debug("Updated bar received", symbol=bar.symbol)

        except Exception as e:
            logger.error("Failed to handle updated bar", error=str(e))

    async def _handle_daily_bar(self, bar: Bar):
        """Internal daily bar handler"""
        try:
            self.last_heartbeat = time.time()

            # Call custom callbacks
            for callback in self.custom_callbacks['daily_bar']:
                try:
                    await callback(bar)
                except Exception as e:
                    logger.error("Daily bar callback failed", error=str(e))

            logger.debug("Daily bar received", symbol=bar.symbol)

        except Exception as e:
            logger.error("Failed to handle daily bar", error=str(e))

    # Public API
    async def subscribe_trades(self, symbols: List[str], callback: Callable = None):
        """Subscribe to trade updates"""
        if not self.stream:
            raise TradingError("Stream not initialized")

        await self.subscription_manager.add_trade_subscription(symbols)

        # Add to stream
        for symbol in symbols:
            self.stream.subscribe_trades(self._handle_trade, symbol)

        if callback:
            self.custom_callbacks['trade'].append(callback)

        logger.info("Subscribed to trades", symbols=symbols)

    async def subscribe_quotes(self, symbols: List[str], callback: Callable = None):
        """Subscribe to quote updates"""
        if not self.stream:
            raise TradingError("Stream not initialized")

        await self.subscription_manager.add_quote_subscription(symbols)

        # Add to stream
        for symbol in symbols:
            self.stream.subscribe_quotes(self._handle_quote, symbol)

        if callback:
            self.custom_callbacks['quote'].append(callback)

        logger.info("Subscribed to quotes", symbols=symbols)

    async def subscribe_bars(self, symbols: List[str], callback: Callable = None):
        """Subscribe to bar updates"""
        if not self.stream:
            raise TradingError("Stream not initialized")

        await self.subscription_manager.add_bar_subscription(symbols)

        # Add to stream
        for symbol in symbols:
            self.stream.subscribe_bars(self._handle_bar, symbol)

        if callback:
            self.custom_callbacks['bar'].append(callback)

        logger.info("Subscribed to bars", symbols=symbols)

    async def subscribe_all_data(self, symbols: List[str], callbacks: Dict[str, Callable] = None):
        """Subscribe to all data types for symbols"""
        await self.subscribe_trades(symbols, callbacks.get('trade') if callbacks else None)
        await self.subscribe_quotes(symbols, callbacks.get('quote') if callbacks else None)
        await self.subscribe_bars(symbols, callbacks.get('bar') if callbacks else None)

        logger.info("Subscribed to all data types", symbols=symbols)

    async def get_connection_health(self) -> Dict[str, Any]:
        """Get connection health status"""
        now = time.time()
        time_since_heartbeat = now - self.last_heartbeat if self.last_heartbeat else None

        return {
            'is_connected': self.is_connected,
            'connection_attempts': self.connection_attempts,
            'last_heartbeat': self.last_heartbeat,
            'time_since_heartbeat': time_since_heartbeat,
            'healthy': self.is_connected and (time_since_heartbeat is None or time_since_heartbeat < 30),
            'stats': self.stats.copy(),
            'subscribed_symbols': len(await self.subscription_manager.get_all_symbols())
        }

    async def get_buffered_data(self, symbol: str = None, limit: int = 100) -> Dict[str, List]:
        """Get buffered market data"""
        return {
            'trades': await self.data_buffer.get_latest_trades(symbol, limit),
            'quotes': await self.data_buffer.get_latest_quotes(symbol, limit),
            'bars': await self.data_buffer.get_latest_bars(symbol, limit)
        }


# Global stream manager instance
stream_manager: Optional[AlpacaStreamManager] = None


async def get_stream_manager() -> AlpacaStreamManager:
    """Get or create global stream manager instance"""
    global stream_manager

    if stream_manager is None:
        stream_manager = AlpacaStreamManager(
            paper=settings.TRADING_PAPER_MODE,
            data_feed=settings.ALPACA_DATA_FEED
        )
        await stream_manager.initialize()

    return stream_manager


async def start_streaming(symbols: List[str] = None):
    """Start streaming for default symbols"""
    manager = await get_stream_manager()

    if not symbols:
        # Default symbols for testing
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "SPY", "QQQ"]

    await manager.connect()
    await manager.subscribe_all_data(symbols)

    logger.info("Started streaming for symbols", symbols=symbols)
    return manager