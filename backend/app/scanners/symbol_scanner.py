"""
Multi-Symbol Scanner with concurrent analysis for thousands of symbols
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import time

from app.core.cache import market_data_cache
from app.core.config import settings
from app.trading.alpaca_client import AlpacaClient
from app.indicators.technical_indicators import TechnicalIndicators
from app.scanners.symbol_universe import SymbolUniverseManager, AssetClass
from app.scanners.opportunity_ranker import OpportunityRanker

logger = logging.getLogger(__name__)

class ScanResult(Enum):
    """Scan result status"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    ERROR = "error"
    NO_DATA = "no_data"

@dataclass
class SymbolAnalysis:
    """Analysis result for a single symbol"""
    symbol: str
    timestamp: datetime
    current_price: Optional[float] = None
    volume: Optional[int] = None
    price_change_24h: Optional[float] = None
    price_change_percent: Optional[float] = None

    # Technical indicators
    rsi: Optional[float] = None
    macd_signal: Optional[str] = None
    bollinger_position: Optional[str] = None
    volume_spike: bool = False

    # Opportunity scoring
    opportunity_score: float = 0.0
    scan_result: ScanResult = ScanResult.NEUTRAL
    confidence: float = 0.0

    # Risk metrics
    volatility: Optional[float] = None
    liquidity_score: float = 0.0

    # Trading signals
    entry_signal: bool = False
    exit_signal: bool = False
    position_size_suggestion: float = 0.0

    # Metadata
    scan_duration_ms: float = 0.0
    errors: List[str] = field(default_factory=list)

class SymbolScanner:
    """High-performance multi-symbol scanner with concurrent processing"""

    def __init__(self, max_concurrent: int = 50):
        self.max_concurrent = max_concurrent
        self.universe_manager = SymbolUniverseManager()
        self.opportunity_ranker = OpportunityRanker()
        self.technical_indicators = TechnicalIndicators()
        self.alpaca_client = AlpacaClient()

        # Scanning state
        self.is_scanning = False
        self.last_scan_time = None
        self.scan_statistics = {
            'total_scans': 0,
            'successful_scans': 0,
            'failed_scans': 0,
            'avg_scan_time_ms': 0.0,
            'opportunities_found': 0
        }

        # Performance tracking
        self.symbol_scan_times: Dict[str, float] = {}
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def initialize(self) -> Dict[str, any]:
        """Initialize the scanner with symbol universe"""
        logger.info("Initializing Symbol Scanner...")

        start_time = time.time()

        # Initialize universe
        universe_result = await self.universe_manager.initialize_universe()

        # Initialize opportunity ranker
        ranker_result = await self.opportunity_ranker.initialize()

        initialization_time = (time.time() - start_time) * 1000

        result = {
            'status': 'initialized',
            'universe_stats': universe_result,
            'ranker_initialized': ranker_result.get('status') == 'initialized',
            'initialization_time_ms': initialization_time,
            'max_concurrent': self.max_concurrent,
            'total_symbols': universe_result.get('total_symbols', 0)
        }

        logger.info(f"Scanner initialized: {result}")
        return result

    async def scan_symbol(self, symbol: str) -> SymbolAnalysis:
        """Scan a single symbol for trading opportunities"""
        async with self.semaphore:
            start_time = time.time()

            analysis = SymbolAnalysis(
                symbol=symbol,
                timestamp=datetime.utcnow()
            )

            try:
                # Get current market data
                await self._get_market_data(symbol, analysis)

                # Calculate technical indicators
                await self._calculate_technical_indicators(symbol, analysis)

                # Analyze volume and volatility
                await self._analyze_volume_and_volatility(symbol, analysis)

                # Generate opportunity score
                await self._calculate_opportunity_score(analysis)

                # Generate trading signals
                await self._generate_trading_signals(analysis)

                scan_duration = (time.time() - start_time) * 1000
                analysis.scan_duration_ms = scan_duration
                self.symbol_scan_times[symbol] = scan_duration

                logger.debug(f"Scanned {symbol}: score={analysis.opportunity_score:.2f}, "
                           f"signal={analysis.scan_result.value}, duration={scan_duration:.1f}ms")

                return analysis

            except Exception as e:
                scan_duration = (time.time() - start_time) * 1000
                analysis.scan_duration_ms = scan_duration
                analysis.scan_result = ScanResult.ERROR
                analysis.errors.append(str(e))

                logger.warning(f"Error scanning {symbol}: {e}")
                return analysis

    async def _get_market_data(self, symbol: str, analysis: SymbolAnalysis):
        """Get current market data for symbol"""
        try:
            # Try cache first
            cached_quote = market_data_cache.get_quotes([symbol]).get(symbol)

            if cached_quote and self._is_data_fresh(cached_quote, 60):  # 60 seconds
                analysis.current_price = cached_quote.get('price', cached_quote.get('ask_price'))
                analysis.volume = cached_quote.get('volume', 0)
            else:
                # Fetch fresh data
                quote = self.alpaca_client.get_latest_quote(symbol)
                if quote:
                    analysis.current_price = float(quote.ask_price)

                trade = self.alpaca_client.get_latest_trade(symbol)
                if trade:
                    analysis.current_price = float(trade.price)
                    analysis.volume = int(trade.size)

        except Exception as e:
            analysis.errors.append(f"Market data error: {str(e)}")

    async def _calculate_technical_indicators(self, symbol: str, analysis: SymbolAnalysis):
        """Calculate technical indicators for the symbol"""
        try:
            # Get historical data
            historical_data = market_data_cache.get_historical_data(symbol, '1Day', 30)
            if not historical_data or len(historical_data) < 14:
                return

            prices = [float(bar['close']) for bar in historical_data]
            volumes = [int(bar['volume']) for bar in historical_data]

            # Calculate RSI
            rsi_values = self.technical_indicators.rsi(prices, period=14)
            if rsi_values:
                analysis.rsi = rsi_values[-1]

            # Calculate MACD
            macd_line, macd_signal, _ = self.technical_indicators.macd(prices)
            if macd_line and macd_signal:
                latest_macd = macd_line[-1]
                latest_signal = macd_signal[-1]

                if latest_macd > latest_signal:
                    analysis.macd_signal = "bullish"
                elif latest_macd < latest_signal:
                    analysis.macd_signal = "bearish"
                else:
                    analysis.macd_signal = "neutral"

            # Calculate Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.technical_indicators.bollinger_bands(prices, period=20)
            if bb_upper and bb_middle and bb_lower and analysis.current_price:
                current_price = analysis.current_price
                latest_upper = bb_upper[-1]
                latest_lower = bb_lower[-1]
                latest_middle = bb_middle[-1]

                if current_price > latest_upper:
                    analysis.bollinger_position = "overbought"
                elif current_price < latest_lower:
                    analysis.bollinger_position = "oversold"
                elif current_price > latest_middle:
                    analysis.bollinger_position = "upper_half"
                else:
                    analysis.bollinger_position = "lower_half"

        except Exception as e:
            analysis.errors.append(f"Technical indicators error: {str(e)}")

    async def _analyze_volume_and_volatility(self, symbol: str, analysis: SymbolAnalysis):
        """Analyze volume patterns and volatility"""
        try:
            historical_data = market_data_cache.get_historical_data(symbol, '1Day', 20)
            if not historical_data or len(historical_data) < 10:
                return

            volumes = [int(bar['volume']) for bar in historical_data]
            prices = [float(bar['close']) for bar in historical_data]

            # Volume spike detection
            if len(volumes) >= 10:
                avg_volume = sum(volumes[-10:-1]) / 9  # Exclude today
                current_volume = analysis.volume or volumes[-1]

                if current_volume > avg_volume * 2.0:  # 2x average volume
                    analysis.volume_spike = True

            # Volatility calculation (20-day)
            if len(prices) >= 20:
                returns = []
                for i in range(1, len(prices)):
                    returns.append((prices[i] - prices[i-1]) / prices[i-1])

                if returns:
                    variance = sum((r - sum(returns)/len(returns))**2 for r in returns) / len(returns)
                    analysis.volatility = (variance ** 0.5) * (252 ** 0.5)  # Annualized volatility

            # Calculate 24h price change
            if len(prices) >= 2 and analysis.current_price:
                prev_price = prices[-2]
                price_change = analysis.current_price - prev_price
                analysis.price_change_24h = price_change
                analysis.price_change_percent = (price_change / prev_price) * 100

        except Exception as e:
            analysis.errors.append(f"Volume/volatility error: {str(e)}")

    async def _calculate_opportunity_score(self, analysis: SymbolAnalysis):
        """Calculate opportunity score using the ranker"""
        try:
            score_data = {
                'rsi': analysis.rsi,
                'macd_signal': analysis.macd_signal,
                'bollinger_position': analysis.bollinger_position,
                'volume_spike': analysis.volume_spike,
                'volatility': analysis.volatility,
                'price_change_percent': analysis.price_change_percent,
                'current_price': analysis.current_price,
                'volume': analysis.volume
            }

            # Use opportunity ranker
            ranking_result = await self.opportunity_ranker.rank_opportunity(analysis.symbol, score_data)

            analysis.opportunity_score = ranking_result.get('score', 0.0)
            analysis.confidence = ranking_result.get('confidence', 0.0)
            analysis.liquidity_score = ranking_result.get('liquidity_score', 0.0)

            # Determine scan result based on score
            if analysis.opportunity_score >= 0.7:
                analysis.scan_result = ScanResult.BULLISH
            elif analysis.opportunity_score <= 0.3:
                analysis.scan_result = ScanResult.BEARISH
            else:
                analysis.scan_result = ScanResult.NEUTRAL

        except Exception as e:
            analysis.errors.append(f"Opportunity scoring error: {str(e)}")

    async def _generate_trading_signals(self, analysis: SymbolAnalysis):
        """Generate entry/exit signals and position sizing"""
        try:
            # Entry signal logic
            entry_conditions = [
                analysis.opportunity_score > 0.7,
                analysis.confidence > 0.6,
                analysis.rsi and 30 <= analysis.rsi <= 70,  # Not extreme
                analysis.bollinger_position in ['oversold', 'lower_half'],
                analysis.macd_signal == 'bullish'
            ]

            analysis.entry_signal = sum(entry_conditions) >= 3

            # Exit signal logic
            exit_conditions = [
                analysis.opportunity_score < 0.3,
                analysis.rsi and (analysis.rsi > 80 or analysis.rsi < 20),
                analysis.bollinger_position == 'overbought',
                analysis.macd_signal == 'bearish'
            ]

            analysis.exit_signal = sum(exit_conditions) >= 2

            # Position sizing based on confidence and volatility
            base_size = 1000.0  # Base position size in dollars

            if analysis.confidence > 0 and analysis.volatility:
                # Adjust for confidence
                confidence_multiplier = analysis.confidence

                # Adjust for volatility (reduce size for high volatility)
                volatility_multiplier = max(0.5, 1.0 - analysis.volatility)

                analysis.position_size_suggestion = base_size * confidence_multiplier * volatility_multiplier
            else:
                analysis.position_size_suggestion = base_size * 0.5  # Conservative default

        except Exception as e:
            analysis.errors.append(f"Signal generation error: {str(e)}")

    def _is_data_fresh(self, data: Dict, max_age_seconds: int) -> bool:
        """Check if cached data is fresh enough"""
        try:
            if 'cached_at' in data:
                cached_time = datetime.fromisoformat(data['cached_at'])
                age = (datetime.utcnow() - cached_time).total_seconds()
                return age <= max_age_seconds
        except:
            pass
        return False

    async def scan_multiple_symbols(self, symbols: List[str]) -> List[SymbolAnalysis]:
        """Scan multiple symbols concurrently"""
        if not symbols:
            return []

        logger.info(f"Starting concurrent scan of {len(symbols)} symbols...")
        start_time = time.time()

        # Create scanning tasks
        tasks = [self.scan_symbol(symbol) for symbol in symbols]

        # Execute concurrently with semaphore limiting
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        analyses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create error analysis
                error_analysis = SymbolAnalysis(
                    symbol=symbols[i],
                    timestamp=datetime.utcnow(),
                    scan_result=ScanResult.ERROR,
                    errors=[str(result)]
                )
                analyses.append(error_analysis)
            else:
                analyses.append(result)

        # Update statistics
        scan_duration = (time.time() - start_time) * 1000
        successful_scans = sum(1 for a in analyses if a.scan_result != ScanResult.ERROR)
        failed_scans = len(analyses) - successful_scans

        self.scan_statistics['total_scans'] += len(symbols)
        self.scan_statistics['successful_scans'] += successful_scans
        self.scan_statistics['failed_scans'] += failed_scans

        if successful_scans > 0:
            avg_symbol_time = scan_duration / len(symbols)
            self.scan_statistics['avg_scan_time_ms'] = (
                (self.scan_statistics['avg_scan_time_ms'] * (self.scan_statistics['total_scans'] - len(symbols)) +
                 avg_symbol_time * len(symbols)) / self.scan_statistics['total_scans']
            )

        opportunities = sum(1 for a in analyses if a.opportunity_score > 0.7)
        self.scan_statistics['opportunities_found'] += opportunities

        logger.info(f"Scan completed: {len(symbols)} symbols in {scan_duration:.1f}ms, "
                   f"{successful_scans} successful, {opportunities} opportunities")

        return analyses

    async def full_universe_scan(self, asset_classes: List[AssetClass] = None) -> Dict[str, any]:
        """Perform full universe scan with prioritization"""
        logger.info("Starting full universe scan...")

        self.is_scanning = True
        scan_start = time.time()

        try:
            # Get symbols to scan
            if asset_classes:
                symbols_to_scan = []
                for asset_class in asset_classes:
                    symbols_to_scan.extend(self.universe_manager.get_symbols_by_asset_class(asset_class))
            else:
                # Prioritize high-volume symbols for faster scanning
                symbols_to_scan = self.universe_manager.get_high_volume_symbols(limit=5000)

            # Remove duplicates and filter active symbols
            symbols_to_scan = list(set(symbols_to_scan))
            symbols_to_scan = [s for s in symbols_to_scan if self.universe_manager.is_symbol_active(s)]

            logger.info(f"Scanning {len(symbols_to_scan)} symbols from universe...")

            # Scan in batches for memory efficiency
            batch_size = 500
            all_analyses = []

            for i in range(0, len(symbols_to_scan), batch_size):
                batch = symbols_to_scan[i:i + batch_size]
                batch_analyses = await self.scan_multiple_symbols(batch)
                all_analyses.extend(batch_analyses)

                # Brief pause between batches to prevent overwhelming
                await asyncio.sleep(0.1)

            # Sort by opportunity score
            all_analyses.sort(key=lambda x: x.opportunity_score, reverse=True)

            # Generate summary
            scan_duration = (time.time() - scan_start) * 1000
            opportunities = [a for a in all_analyses if a.opportunity_score > 0.7]

            result = {
                'status': 'completed',
                'total_symbols_scanned': len(all_analyses),
                'scan_duration_ms': scan_duration,
                'opportunities_found': len(opportunities),
                'top_opportunities': [
                    {
                        'symbol': a.symbol,
                        'opportunity_score': a.opportunity_score,
                        'scan_result': a.scan_result.value,
                        'confidence': a.confidence,
                        'entry_signal': a.entry_signal,
                        'current_price': a.current_price,
                        'price_change_percent': a.price_change_percent
                    }
                    for a in opportunities[:20]  # Top 20 opportunities
                ],
                'scan_statistics': self.scan_statistics.copy(),
                'timestamp': datetime.utcnow().isoformat()
            }

            self.last_scan_time = datetime.utcnow()

            logger.info(f"Full universe scan completed: {len(opportunities)} opportunities found "
                       f"from {len(all_analyses)} symbols in {scan_duration:.1f}ms")

            return result

        finally:
            self.is_scanning = False

    def get_scan_statistics(self) -> Dict[str, any]:
        """Get current scanning statistics"""
        return {
            'statistics': self.scan_statistics.copy(),
            'is_scanning': self.is_scanning,
            'last_scan_time': self.last_scan_time.isoformat() if self.last_scan_time else None,
            'universe_stats': self.universe_manager.get_universe_stats(),
            'max_concurrent': self.max_concurrent,
            'avg_symbol_scan_time_ms': sum(self.symbol_scan_times.values()) / len(self.symbol_scan_times) if self.symbol_scan_times else 0.0
        }