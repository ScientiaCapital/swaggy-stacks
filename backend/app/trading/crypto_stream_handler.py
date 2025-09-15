"""
24/7 Crypto Streaming Handler with continuous learning capabilities
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Callable
from collections import deque, defaultdict
from decimal import Decimal, ROUND_HALF_UP
import structlog

from app.trading.alpaca_stream_manager import AlpacaStreamManager, get_stream_manager
from app.core.config import settings
from app.core.exceptions import TradingError

logger = structlog.get_logger(__name__)


class CryptoMarketAnalyzer:
    """Real-time crypto market analysis and pattern detection"""

    def __init__(self):
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self.volatility_cache: Dict[str, float] = {}
        self.correlation_matrix: Dict[str, Dict[str, float]] = defaultdict(dict)

        # Learning patterns
        self.pattern_database: Dict[str, List[Dict]] = defaultdict(list)
        self.opportunity_scores: Dict[str, float] = defaultdict(float)

        # Performance tracking
        self.analysis_count = 0
        self.opportunities_found = 0
        self.last_analysis_time = None

    async def analyze_crypto_data(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze incoming crypto market data"""
        try:
            self.analysis_count += 1
            self.last_analysis_time = datetime.utcnow()

            # Store historical data
            price = float(market_data.get('price', 0))
            volume = float(market_data.get('volume', 0))
            timestamp = market_data.get('timestamp', datetime.utcnow())

            self.price_history[symbol].append({
                'price': price,
                'timestamp': timestamp,
                'volume': volume
            })

            if volume > 0:
                self.volume_history[symbol].append({
                    'volume': volume,
                    'timestamp': timestamp,
                    'price': price
                })

            # Perform analysis
            analysis = await self._perform_comprehensive_analysis(symbol, market_data)

            # Check for opportunities
            if analysis['opportunity_score'] > 0.7:
                self.opportunities_found += 1
                await self._log_opportunity(symbol, analysis)

            return analysis

        except Exception as e:
            logger.error("Failed to analyze crypto data", symbol=symbol, error=str(e))
            return {'error': str(e), 'opportunity_score': 0}

    async def _perform_comprehensive_analysis(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive crypto analysis"""
        analysis = {
            'symbol': symbol,
            'timestamp': datetime.utcnow(),
            'price': float(market_data.get('price', 0)),
            'volume': float(market_data.get('volume', 0)),
            'opportunity_score': 0.0,
            'signals': [],
            'volatility': 0.0,
            'trend': 'neutral',
            'support_resistance': {'support': 0, 'resistance': 0},
            'volume_profile': 'normal',
            'patterns_detected': []
        }

        # Calculate volatility
        volatility = await self._calculate_volatility(symbol)
        analysis['volatility'] = volatility
        self.volatility_cache[symbol] = volatility

        # Analyze trend
        trend = await self._analyze_trend(symbol)
        analysis['trend'] = trend

        # Detect patterns
        patterns = await self._detect_patterns(symbol)
        analysis['patterns_detected'] = patterns

        # Calculate support/resistance
        support_resistance = await self._calculate_support_resistance(symbol)
        analysis['support_resistance'] = support_resistance

        # Analyze volume profile
        volume_profile = await self._analyze_volume_profile(symbol)
        analysis['volume_profile'] = volume_profile

        # Generate signals
        signals = await self._generate_signals(symbol, analysis)
        analysis['signals'] = signals

        # Calculate opportunity score
        opportunity_score = await self._calculate_opportunity_score(analysis)
        analysis['opportunity_score'] = opportunity_score
        self.opportunity_scores[symbol] = opportunity_score

        return analysis

    async def _calculate_volatility(self, symbol: str) -> float:
        """Calculate real-time volatility"""
        if len(self.price_history[symbol]) < 20:
            return 0.0

        prices = [p['price'] for p in self.price_history[symbol]]

        # Calculate returns
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                returns.append((prices[i] - prices[i-1]) / prices[i-1])

        if len(returns) < 10:
            return 0.0

        # Calculate standard deviation
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        volatility = (variance ** 0.5) * 100  # Convert to percentage

        return round(volatility, 4)

    async def _analyze_trend(self, symbol: str) -> str:
        """Analyze price trend"""
        if len(self.price_history[symbol]) < 10:
            return 'neutral'

        prices = [p['price'] for p in self.price_history[symbol][-20:]]

        # Simple trend analysis
        short_avg = sum(prices[-5:]) / 5 if len(prices) >= 5 else prices[-1]
        long_avg = sum(prices[-20:]) / 20 if len(prices) >= 20 else sum(prices) / len(prices)

        if short_avg > long_avg * 1.02:
            return 'bullish'
        elif short_avg < long_avg * 0.98:
            return 'bearish'
        else:
            return 'neutral'

    async def _detect_patterns(self, symbol: str) -> List[str]:
        """Detect trading patterns"""
        if len(self.price_history[symbol]) < 50:
            return []

        patterns = []
        prices = [p['price'] for p in self.price_history[symbol][-50:]]

        # Simple pattern detection
        # Breakout pattern
        recent_high = max(prices[-20:])
        recent_low = min(prices[-20:])
        current_price = prices[-1]

        if current_price > recent_high * 1.05:
            patterns.append('breakout_up')
        elif current_price < recent_low * 0.95:
            patterns.append('breakout_down')

        # Support/resistance bounce
        support_level = min(prices[-30:-10])
        if abs(current_price - support_level) / support_level < 0.02:
            patterns.append('support_bounce')

        resistance_level = max(prices[-30:-10])
        if abs(current_price - resistance_level) / resistance_level < 0.02:
            patterns.append('resistance_test')

        return patterns

    async def _calculate_support_resistance(self, symbol: str) -> Dict[str, float]:
        """Calculate support and resistance levels"""
        if len(self.price_history[symbol]) < 100:
            return {'support': 0, 'resistance': 0}

        prices = [p['price'] for p in self.price_history[symbol][-100:]]

        # Simple support/resistance calculation
        sorted_prices = sorted(prices)
        support = sorted_prices[int(len(sorted_prices) * 0.1)]  # 10th percentile
        resistance = sorted_prices[int(len(sorted_prices) * 0.9)]  # 90th percentile

        return {
            'support': round(support, 8),
            'resistance': round(resistance, 8)
        }

    async def _analyze_volume_profile(self, symbol: str) -> str:
        """Analyze volume profile"""
        if len(self.volume_history[symbol]) < 20:
            return 'insufficient_data'

        volumes = [v['volume'] for v in self.volume_history[symbol]]
        avg_volume = sum(volumes) / len(volumes)
        current_volume = volumes[-1]

        if current_volume > avg_volume * 2:
            return 'high'
        elif current_volume < avg_volume * 0.5:
            return 'low'
        else:
            return 'normal'

    async def _generate_signals(self, symbol: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals"""
        signals = []

        # Volatility breakout signal
        if analysis['volatility'] > 15 and 'breakout_up' in analysis['patterns_detected']:
            signals.append({
                'type': 'volatility_breakout',
                'direction': 'long',
                'strength': 0.8,
                'reason': 'High volatility with upward breakout'
            })

        # Volume spike signal
        if analysis['volume_profile'] == 'high' and analysis['trend'] != 'neutral':
            signals.append({
                'type': 'volume_confirmation',
                'direction': 'long' if analysis['trend'] == 'bullish' else 'short',
                'strength': 0.7,
                'reason': f'High volume confirming {analysis["trend"]} trend'
            })

        # Support bounce signal
        if 'support_bounce' in analysis['patterns_detected']:
            signals.append({
                'type': 'support_bounce',
                'direction': 'long',
                'strength': 0.6,
                'reason': 'Price bouncing off support level'
            })

        return signals

    async def _calculate_opportunity_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall opportunity score (0-1)"""
        score = 0.0

        # Signal strength contribution
        for signal in analysis['signals']:
            score += signal['strength'] * 0.3

        # Volatility contribution (moderate volatility is good)
        volatility = analysis['volatility']
        if 5 < volatility < 25:
            score += 0.2
        elif volatility > 25:
            score += 0.1  # Too volatile

        # Pattern contribution
        pattern_bonus = len(analysis['patterns_detected']) * 0.1
        score += min(pattern_bonus, 0.3)

        # Volume contribution
        if analysis['volume_profile'] == 'high':
            score += 0.2

        return min(score, 1.0)

    async def _log_opportunity(self, symbol: str, analysis: Dict[str, Any]):
        """Log trading opportunity"""
        logger.info("Crypto opportunity detected",
                   symbol=symbol,
                   opportunity_score=analysis['opportunity_score'],
                   signals=len(analysis['signals']),
                   patterns=analysis['patterns_detected'],
                   volatility=analysis['volatility'])

    async def get_analysis_stats(self) -> Dict[str, Any]:
        """Get analysis statistics"""
        return {
            'total_analysis_count': self.analysis_count,
            'opportunities_found': self.opportunities_found,
            'symbols_tracked': len(self.price_history),
            'average_opportunity_score': sum(self.opportunity_scores.values()) / len(self.opportunity_scores) if self.opportunity_scores else 0,
            'last_analysis_time': self.last_analysis_time.isoformat() if self.last_analysis_time else None,
            'top_opportunities': sorted(
                [(symbol, score) for symbol, score in self.opportunity_scores.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }


class CryptoStreamHandler:
    """
    24/7 Crypto Streaming Handler with continuous learning:
    - Real-time crypto data processing
    - Pattern recognition and learning
    - Opportunity detection
    - Fractional trading support
    - 24/7 operation capability
    """

    def __init__(self):
        self.stream_manager: Optional[AlpacaStreamManager] = None
        self.analyzer = CryptoMarketAnalyzer()

        # Crypto symbols to monitor
        self.crypto_symbols = {
            # Major cryptos
            'BTC/USD', 'ETH/USD', 'BNB/USD', 'XRP/USD', 'ADA/USD',
            'DOGE/USD', 'MATIC/USD', 'SOL/USD', 'DOT/USD', 'AVAX/USD',
            # DeFi tokens
            'UNI/USD', 'LINK/USD', 'AAVE/USD', 'SUSHI/USD',
            # Layer 2
            'LTC/USD', 'BCH/USD', 'ETC/USD',
            # Meme coins (for volatility opportunities)
            'SHIB/USD'
        }

        # Learning and monitoring
        self.learning_callbacks: List[Callable] = []
        self.opportunity_callbacks: List[Callable] = []

        # Operation status
        self.is_streaming = False
        self.start_time: Optional[datetime] = None

        # Performance metrics
        self.metrics = {
            'data_points_processed': 0,
            'opportunities_detected': 0,
            'learning_events': 0,
            'uptime_hours': 0,
            'symbols_monitored': len(self.crypto_symbols)
        }

    async def initialize(self):
        """Initialize crypto stream handler"""
        try:
            self.stream_manager = await get_stream_manager()
            logger.info("CryptoStreamHandler initialized", symbols=len(self.crypto_symbols))

        except Exception as e:
            logger.error("Failed to initialize crypto stream handler", error=str(e))
            raise TradingError(f"Crypto stream initialization failed: {e}")

    async def start_streaming(self):
        """Start 24/7 crypto streaming"""
        if self.is_streaming:
            logger.warning("Crypto streaming already active")
            return

        try:
            if not self.stream_manager:
                await self.initialize()

            # Ensure stream manager is connected
            if not self.stream_manager.is_connected:
                await self.stream_manager.connect()

            # Subscribe to crypto data streams
            crypto_symbols_list = list(self.crypto_symbols)

            # Subscribe to trades and quotes
            await self.stream_manager.subscribe_trades(crypto_symbols_list, self._handle_crypto_trade)
            await self.stream_manager.subscribe_quotes(crypto_symbols_list, self._handle_crypto_quote)

            self.is_streaming = True
            self.start_time = datetime.utcnow()

            logger.info("24/7 Crypto streaming started",
                       symbols=len(crypto_symbols_list),
                       endpoint="wss://stream.data.alpaca.markets/v1beta3/crypto/us")

        except Exception as e:
            logger.error("Failed to start crypto streaming", error=str(e))
            raise TradingError(f"Failed to start crypto streaming: {e}")

    async def stop_streaming(self):
        """Stop crypto streaming"""
        self.is_streaming = False
        logger.info("Crypto streaming stopped")

    async def _handle_crypto_trade(self, trade):
        """Handle incoming crypto trade data"""
        try:
            if not self.is_streaming:
                return

            self.metrics['data_points_processed'] += 1

            # Convert trade to market data
            market_data = {
                'symbol': trade.symbol,
                'price': float(trade.price),
                'size': float(trade.size),
                'timestamp': trade.timestamp,
                'type': 'trade'
            }

            # Analyze the data
            analysis = await self.analyzer.analyze_crypto_data(trade.symbol, market_data)

            # Trigger learning if high opportunity score
            if analysis.get('opportunity_score', 0) > 0.7:
                await self._trigger_learning_event(trade.symbol, analysis)

            # Call learning callbacks
            for callback in self.learning_callbacks:
                try:
                    await callback(trade.symbol, market_data, analysis)
                except Exception as e:
                    logger.error("Learning callback failed", error=str(e))

            # Check for trading opportunities
            if analysis.get('opportunity_score', 0) > 0.8:
                await self._handle_trading_opportunity(trade.symbol, analysis)

        except Exception as e:
            logger.error("Failed to handle crypto trade", symbol=trade.symbol, error=str(e))

    async def _handle_crypto_quote(self, quote):
        """Handle incoming crypto quote data"""
        try:
            if not self.is_streaming:
                return

            self.metrics['data_points_processed'] += 1

            # Calculate mid price
            mid_price = (float(quote.bid_price) + float(quote.ask_price)) / 2

            market_data = {
                'symbol': quote.symbol,
                'price': mid_price,
                'bid_price': float(quote.bid_price),
                'ask_price': float(quote.ask_price),
                'bid_size': float(quote.bid_size),
                'ask_size': float(quote.ask_size),
                'spread': float(quote.ask_price) - float(quote.bid_price),
                'timestamp': quote.timestamp,
                'type': 'quote'
            }

            # Quick analysis for quotes (lighter than trades)
            if market_data['spread'] > 0:
                spread_pct = (market_data['spread'] / mid_price) * 100
                if spread_pct > 1.0:  # Wide spread opportunity
                    logger.debug("Wide spread detected",
                               symbol=quote.symbol,
                               spread_pct=round(spread_pct, 3))

        except Exception as e:
            logger.error("Failed to handle crypto quote", symbol=quote.symbol, error=str(e))

    async def _trigger_learning_event(self, symbol: str, analysis: Dict[str, Any]):
        """Trigger learning event for continuous improvement"""
        try:
            self.metrics['learning_events'] += 1

            # Import here to avoid circular imports
            from app.ml.crypto_learner import get_crypto_learner

            learner = await get_crypto_learner()
            await learner.process_learning_data(symbol, analysis)

            logger.info("Learning event triggered",
                       symbol=symbol,
                       opportunity_score=analysis.get('opportunity_score', 0))

        except Exception as e:
            logger.error("Failed to trigger learning event", symbol=symbol, error=str(e))

    async def _handle_trading_opportunity(self, symbol: str, analysis: Dict[str, Any]):
        """Handle high-confidence trading opportunities"""
        try:
            self.metrics['opportunities_detected'] += 1

            # Call opportunity callbacks
            for callback in self.opportunity_callbacks:
                try:
                    await callback(symbol, analysis)
                except Exception as e:
                    logger.error("Opportunity callback failed", error=str(e))

            # Wake up agents for crypto opportunity
            await self._wake_crypto_agents(symbol, analysis)

        except Exception as e:
            logger.error("Failed to handle trading opportunity", symbol=symbol, error=str(e))

    async def _wake_crypto_agents(self, symbol: str, analysis: Dict[str, Any]):
        """Wake up agents for crypto opportunity"""
        try:
            # Import here to avoid circular imports
            from app.events.trigger_engine import get_trigger_engine

            trigger_engine = await get_trigger_engine()

            # Process as market data to trigger events
            await trigger_engine.process_market_data(symbol, {
                'price': analysis['price'],
                'volume': analysis.get('volume', 0),
                'opportunity_score': analysis['opportunity_score'],
                'signals': analysis['signals'],
                'patterns': analysis['patterns_detected'],
                'crypto_opportunity': True
            })

        except Exception as e:
            logger.error("Failed to wake crypto agents", symbol=symbol, error=str(e))

    # Public API
    async def add_learning_callback(self, callback: Callable):
        """Add callback for learning events"""
        self.learning_callbacks.append(callback)

    async def add_opportunity_callback(self, callback: Callable):
        """Add callback for trading opportunities"""
        self.opportunity_callbacks.append(callback)

    async def add_crypto_symbol(self, symbol: str):
        """Add new crypto symbol to monitoring"""
        if symbol not in self.crypto_symbols:
            self.crypto_symbols.add(symbol)

            if self.is_streaming and self.stream_manager:
                await self.stream_manager.subscribe_trades([symbol], self._handle_crypto_trade)
                await self.stream_manager.subscribe_quotes([symbol], self._handle_crypto_quote)

            logger.info("Added crypto symbol", symbol=symbol)

    async def remove_crypto_symbol(self, symbol: str):
        """Remove crypto symbol from monitoring"""
        if symbol in self.crypto_symbols:
            self.crypto_symbols.discard(symbol)
            logger.info("Removed crypto symbol", symbol=symbol)

    async def get_streaming_status(self) -> Dict[str, Any]:
        """Get current streaming status"""
        uptime = 0
        if self.start_time:
            uptime = (datetime.utcnow() - self.start_time).total_seconds() / 3600

        self.metrics['uptime_hours'] = round(uptime, 2)

        analyzer_stats = await self.analyzer.get_analysis_stats()

        return {
            'is_streaming': self.is_streaming,
            'symbols_monitored': list(self.crypto_symbols),
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'uptime_hours': uptime,
            'metrics': self.metrics,
            'analyzer_stats': analyzer_stats,
            'connection_health': await self.stream_manager.get_connection_health() if self.stream_manager else None
        }

    async def get_crypto_opportunities(self, min_score: float = 0.6) -> List[Dict[str, Any]]:
        """Get current crypto opportunities above minimum score"""
        opportunities = []

        for symbol, score in self.analyzer.opportunity_scores.items():
            if score >= min_score:
                opportunities.append({
                    'symbol': symbol,
                    'opportunity_score': score,
                    'volatility': self.analyzer.volatility_cache.get(symbol, 0),
                    'last_analysis': datetime.utcnow().isoformat()
                })

        return sorted(opportunities, key=lambda x: x['opportunity_score'], reverse=True)

    def validate_fractional_quantity(self, symbol: str, quantity: float) -> bool:
        """Validate fractional quantity for crypto trading"""
        # Crypto supports very small fractional amounts
        if quantity <= 0:
            return False

        # Parse crypto symbol for minimum quantity rules
        base_currency = symbol.split('/')[0] if '/' in symbol else symbol.replace('USD', '')

        min_quantities = {
            'BTC': 0.00000001,  # 1 satoshi
            'ETH': 0.000000000000000001,  # 1 wei
            'DOGE': 0.00000001,
            'SHIB': 0.00000001,
        }

        min_qty = min_quantities.get(base_currency, 0.00000001)  # Default to 8 decimals
        return quantity >= min_qty

    def format_fractional_quantity(self, symbol: str, quantity: float) -> str:
        """Format fractional quantity for crypto trading"""
        base_currency = symbol.split('/')[0] if '/' in symbol else symbol.replace('USD', '')

        # Different precision for different cryptos
        if base_currency in ['BTC', 'ETH']:
            decimal_places = 8
        else:
            decimal_places = 6

        # Round to appropriate decimal places
        decimal_quantity = Decimal(str(quantity)).quantize(
            Decimal('0.' + '0' * decimal_places),
            rounding=ROUND_HALF_UP
        )

        return str(decimal_quantity)


# Global crypto stream handler instance
crypto_stream_handler: Optional[CryptoStreamHandler] = None


async def get_crypto_stream_handler() -> CryptoStreamHandler:
    """Get or create global crypto stream handler instance"""
    global crypto_stream_handler

    if crypto_stream_handler is None:
        crypto_stream_handler = CryptoStreamHandler()
        await crypto_stream_handler.initialize()

    return crypto_stream_handler


async def start_24_7_crypto_streaming():
    """Start 24/7 crypto streaming with learning"""
    handler = await get_crypto_stream_handler()
    await handler.start_streaming()
    return handler