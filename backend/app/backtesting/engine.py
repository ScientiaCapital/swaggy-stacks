"""
Refactored Backtesting Engine - Lightweight orchestrator

This replaces the original 1144-line BacktestEngine by delegating to specialized services
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from app.backtesting.performance_analyzer import BacktestResult, PerformanceAnalyzer
from app.backtesting.portfolio_manager import PortfolioManager
from app.backtesting.signal_generator import SignalGenerator
from app.core.config import settings

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtest execution"""

    initial_capital: float = 100000
    max_position_size: float = 0.15
    transaction_cost: float = 0.001
    years_lookback: int = 3
    trade_count_per_signal: int = 1
    enable_portfolio_correlation: bool = True
    cache_results: bool = True
    risk_free_rate: float = 0.02


class RefactoredBacktestEngine:
    """
    Refactored Backtesting Engine - Main interface for backtesting

    This engine acts as a lightweight orchestrator, delegating work to:
    - SignalGenerator: Multi-strategy signal generation
    - PortfolioManager: Portfolio and trade management
    - PerformanceAnalyzer: Advanced metrics and analytics
    """

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()

        # Initialize specialized services
        self.signal_generator = SignalGenerator()
        self.portfolio_manager = PortfolioManager(
            initial_capital=self.config.initial_capital,
            transaction_cost=self.config.transaction_cost,
        )
        self.performance_analyzer = PerformanceAnalyzer(
            risk_free_rate=self.config.risk_free_rate
        )

        # Caching
        self.redis_client = None
        self._initialize_caching()

        # Risk management integration
        self.risk_manager = None
        self._initialize_risk_management()

        logger.info("RefactoredBacktestEngine initialized with specialized services")

    def _initialize_caching(self):
        """Initialize Redis caching if available"""
        if self.config.cache_results and REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=getattr(settings, "REDIS_HOST", "localhost"),
                    port=getattr(settings, "REDIS_PORT", 6379),
                    decode_responses=True,
                    socket_timeout=5,
                )
                self.redis_client.ping()
                logger.info("✅ Redis caching enabled")
            except Exception as e:
                logger.warning(f"⚠️ Redis connection failed: {e}")
                self.redis_client = None

    def _initialize_risk_management(self):
        """Initialize risk management system"""
        try:
            from app.risk.position_manager import IntegratedRiskManager

            risk_config = {
                "base_risk": {
                    "max_position_size": self.config.max_position_size
                    * self.config.initial_capital,
                    "max_daily_loss": self.config.initial_capital * 0.02,
                },
                "position_sizing": {
                    "kelly_max_fraction": 0.25,
                    "fixed_fraction_default": 0.02,
                    "min_position_size": self.config.initial_capital * 0.001,
                },
                "stop_loss": {
                    "atr_periods": 14,
                    "atr_multiplier": 2.0,
                    "min_stop_distance": 0.02,
                    "max_stop_distance": 0.10,
                },
                "portfolio_risk": {
                    "var_confidence_level": 0.95,
                    "var_lookback_days": 252,
                    "max_correlation": 0.7,
                    "max_sector_exposure": 0.30,
                },
            }

            self.risk_manager = IntegratedRiskManager(
                user_id="backtest_system", config=risk_config
            )
            logger.info("✅ Integrated Risk Management initialized")
        except Exception as e:
            logger.warning(f"⚠️ Risk Management initialization failed: {e}")

    # Core Backtesting Methods
    async def single_instrument_backtest(
        self,
        symbol: str,
        strategy_name: str = "integrated_alpha",
        custom_signals: List[Dict[str, Any]] = None,
    ) -> BacktestResult:
        """Execute backtest for single instrument using specified strategy"""
        start_time = time.time()
        backtest_id = f"single_{symbol}_{strategy_name}_{int(start_time)}"

        logger.info(
            f"📊 Starting single instrument backtest: {symbol} with {strategy_name}"
        )

        # Check cache first
        cached_result = self._get_cached_result(backtest_id, [symbol])
        if cached_result:
            logger.info(f"📋 Using cached result for {symbol}")
            return cached_result

        # Get historical data
        data = self.get_historical_data([symbol], years=self.config.years_lookback)
        if not data or symbol not in data:
            raise ValueError(f"Failed to fetch data for {symbol}")

        # Reset portfolio state
        self.portfolio_manager.reset_portfolio()

        # Execute backtest
        symbol_data = data[symbol]
        if custom_signals:
            await self._execute_signal_based_backtest(
                symbol, symbol_data, custom_signals
            )
        else:
            await self._execute_strategy_backtest(symbol, symbol_data, strategy_name)

        # Calculate performance metrics
        result = self.performance_analyzer.calculate_comprehensive_metrics(
            backtest_id=backtest_id,
            symbols=[symbol],
            portfolio_manager=self.portfolio_manager,
            market_data=data,
            start_date=symbol_data.index[0],
            end_date=symbol_data.index[-1],
            execution_start_time=start_time,
            config={"strategy": strategy_name, **self.config.__dict__},
        )

        # Add correlation analysis if enabled
        if self.config.enable_portfolio_correlation:
            result.correlation_matrix = (
                self.performance_analyzer.calculate_correlation_matrix([symbol], data)
            )
            result.performance_attribution = (
                self.performance_analyzer.calculate_performance_attribution(
                    [symbol], self.portfolio_manager.get_trade_history()
                )
            )

        # Cache results
        self._cache_result(result)

        execution_time = time.time() - start_time
        logger.info(f"✅ Single instrument backtest completed in {execution_time:.2f}s")

        return result

    async def multi_instrument_backtest(
        self, symbols: List[str], strategy_name: str = "integrated_alpha"
    ) -> BacktestResult:
        """Execute multi-instrument backtest using specified strategy"""
        start_time = time.time()
        backtest_id = f"multi_{strategy_name}_{'_'.join(symbols[:3])}_{int(start_time)}"

        logger.info(
            f"📊 Starting multi-instrument backtest: {len(symbols)} symbols with {strategy_name}"
        )

        # Check cache
        cached_result = self._get_cached_result(backtest_id, symbols)
        if cached_result:
            logger.info("📋 Using cached multi-instrument result")
            return cached_result

        # Get historical data for all symbols
        data = self.get_historical_data(symbols, years=self.config.years_lookback)
        if not data:
            raise ValueError("Failed to fetch historical data")

        # Reset portfolio
        self.portfolio_manager.reset_portfolio()

        # Execute portfolio backtest
        await self._execute_portfolio_backtest(symbols, data, strategy_name)

        # Calculate comprehensive metrics
        earliest_date = min(data[sym].index[0] for sym in symbols if sym in data)
        latest_date = max(data[sym].index[-1] for sym in symbols if sym in data)

        result = self.performance_analyzer.calculate_comprehensive_metrics(
            backtest_id=backtest_id,
            symbols=symbols,
            portfolio_manager=self.portfolio_manager,
            market_data=data,
            start_date=earliest_date,
            end_date=latest_date,
            execution_start_time=start_time,
            config={"strategy": strategy_name, **self.config.__dict__},
        )

        # Add portfolio analysis
        if self.config.enable_portfolio_correlation and len(symbols) > 1:
            result.correlation_matrix = (
                self.performance_analyzer.calculate_correlation_matrix(symbols, data)
            )
            result.performance_attribution = (
                self.performance_analyzer.calculate_performance_attribution(
                    symbols, self.portfolio_manager.get_trade_history()
                )
            )

        # Cache results
        self._cache_result(result)

        execution_time = time.time() - start_time
        logger.info(f"✅ Multi-instrument backtest completed in {execution_time:.2f}s")

        return result

    async def strategy_comparison_backtest(
        self, symbols: List[str], strategies: List[str] = None
    ) -> Dict[str, BacktestResult]:
        """Compare performance of multiple strategies"""
        if strategies is None:
            strategies = [
                "integrated_alpha",
                "markov_traditional",
                "candlestick_patterns",
            ]

        logger.info(
            f"📊 Strategy comparison: {len(strategies)} strategies on {len(symbols)} symbols"
        )

        results = {}

        for strategy in strategies:
            logger.info(f"🎯 Testing {strategy} strategy...")

            try:
                if len(symbols) == 1:
                    result = await self.single_instrument_backtest(symbols[0], strategy)
                else:
                    result = await self.multi_instrument_backtest(symbols, strategy)

                results[strategy] = result
                logger.info(
                    f"✅ {strategy}: {result.total_return:.2%} return, {result.sharpe_ratio:.2f} Sharpe"
                )

            except Exception as e:
                logger.error(f"❌ {strategy} strategy failed: {e}")
                continue

        # Generate comparison analysis
        if len(results) > 1:
            comparison = self.performance_analyzer.compare_strategies(results)
            logger.info(
                f"🏆 Best strategy: {comparison.get('summary', {}).get('best_overall_strategy', 'Unknown')}"
            )

        return results

    # Private execution methods
    async def _execute_strategy_backtest(self, symbol: str, data, strategy_name: str):
        """Execute backtest using signals from specified strategy"""
        for i in range(50, len(data)):  # Start after lookback period
            current_date = data.index[i]

            # Prepare market data for signal generation
            market_data = {
                "prices": data["Close"].iloc[: i + 1].tolist(),
                "highs": data["High"].iloc[: i + 1].tolist(),
                "lows": data["Low"].iloc[: i + 1].tolist(),
                "volumes": (
                    data["Volume"].iloc[: i + 1].tolist()
                    if "Volume" in data.columns
                    else []
                ),
                "current_price": data.loc[current_date, "Close"],
            }

            # Generate signals
            all_signals = await self.signal_generator.generate_integrated_signals(
                symbol, market_data
            )

            # Extract signal for specified strategy
            action, confidence = self._extract_strategy_signal(
                all_signals, strategy_name
            )

            # Execute trade if signal is strong enough
            if action != "HOLD" and confidence > 0.6:
                current_price = data.loc[current_date, "Close"]

                success = self.portfolio_manager.execute_trade(
                    symbol=symbol,
                    action=action,
                    price=current_price,
                    date=current_date,
                    position_multiplier=confidence * self.config.trade_count_per_signal,
                    strategy=strategy_name,
                    confidence=confidence,
                )

                if success:
                    # Record portfolio snapshot
                    self.portfolio_manager.record_portfolio_snapshot(
                        date=current_date,
                        market_data={symbol: data},
                        strategy=strategy_name,
                        additional_data={"signal_confidence": confidence},
                    )

    async def _execute_portfolio_backtest(
        self, symbols: List[str], data: Dict, strategy_name: str
    ):
        """Execute portfolio backtest across multiple symbols"""
        # Create unified timeline
        all_dates = set()
        for symbol_data in data.values():
            all_dates.update(symbol_data.index)

        timeline = sorted(all_dates)[50:]  # Skip first 50 days

        # Execute trades chronologically
        for i, date in enumerate(timeline):
            if i % 100 == 0:  # Progress logging
                logger.info(
                    f"📈 Processing date {i+1}/{len(timeline)}: {date.strftime('%Y-%m-%d')}"
                )

            for symbol in symbols:
                if symbol not in data or date not in data[symbol].index:
                    continue

                symbol_data = data[symbol]
                date_index = symbol_data.index.get_loc(date)

                if date_index < 50:  # Need sufficient lookback
                    continue

                # Prepare market data
                market_data = {
                    "prices": symbol_data["Close"].iloc[: date_index + 1].tolist(),
                    "highs": symbol_data["High"].iloc[: date_index + 1].tolist(),
                    "lows": symbol_data["Low"].iloc[: date_index + 1].tolist(),
                    "volumes": (
                        symbol_data["Volume"].iloc[: date_index + 1].tolist()
                        if "Volume" in symbol_data.columns
                        else []
                    ),
                    "current_price": symbol_data.loc[date, "Close"],
                }

                # Generate signals
                all_signals = await self.signal_generator.generate_integrated_signals(
                    symbol, market_data
                )

                # Execute portfolio trade
                await self._execute_portfolio_trade(
                    symbol, all_signals, data, date, strategy_name
                )

    async def _execute_portfolio_trade(
        self,
        symbol: str,
        all_signals: Dict[str, Any],
        data: Dict,
        date: datetime,
        strategy_name: str,
    ):
        """Execute portfolio trade with correlation adjustment"""
        action, confidence = self._extract_strategy_signal(all_signals, strategy_name)

        if action == "HOLD" or confidence <= 0.5:
            return

        current_price = data[symbol].loc[date, "Close"]

        # Portfolio-aware position sizing
        correlation_adjustment = (
            self.portfolio_manager.calculate_correlation_adjustment(
                symbol, current_price
            )
        )
        adjusted_confidence = confidence * correlation_adjustment

        if adjusted_confidence > 0.5:
            success = self.portfolio_manager.execute_trade(
                symbol=symbol,
                action=action,
                price=current_price,
                date=date,
                position_multiplier=adjusted_confidence
                * self.config.trade_count_per_signal,
                strategy=strategy_name,
                confidence=confidence,
            )

            if success:
                self.portfolio_manager.record_portfolio_snapshot(
                    date=date,
                    market_data=data,
                    strategy=strategy_name,
                    additional_data={
                        "correlation_adjustment": correlation_adjustment,
                        "signal_confidence": confidence,
                        "adjusted_confidence": adjusted_confidence,
                    },
                )

    async def _execute_signal_based_backtest(
        self, symbol: str, data, signals: List[Dict[str, Any]]
    ):
        """Execute backtest using provided trading signals"""
        for signal in signals:
            signal_date = signal["date"]
            if signal_date not in data.index:
                continue

            price = data.loc[signal_date, "Close"]
            action = signal.get("action", "HOLD")
            confidence = signal.get("confidence", 0.5)

            if action in ["BUY", "SELL"]:
                success = self.portfolio_manager.execute_trade(
                    symbol=symbol,
                    action=action,
                    price=price,
                    date=signal_date,
                    position_multiplier=confidence * self.config.trade_count_per_signal,
                    strategy="custom_signals",
                    confidence=confidence,
                )

                if success:
                    self.portfolio_manager.record_portfolio_snapshot(
                        date=signal_date,
                        market_data={symbol: data},
                        strategy="custom_signals",
                    )

    def _extract_strategy_signal(
        self, all_signals: Dict[str, Any], strategy_name: str
    ) -> Tuple[str, float]:
        """Extract action and confidence from strategy signals"""
        if strategy_name == "integrated_alpha" and "integrated_alpha" in all_signals:
            signal_data = all_signals["integrated_alpha"]
            action = self._convert_signal_to_action(signal_data["signal"])
            confidence = signal_data["confidence"]
            return action, confidence

        elif strategy_name in all_signals:
            signal_data = all_signals[strategy_name]
            action = signal_data.get("signal", "HOLD")
            confidence = signal_data.get("confidence", 0.5)
            return action, confidence

        elif (
            strategy_name == "candlestick_patterns"
            and "candlestick_patterns" in all_signals
        ):
            patterns = all_signals["candlestick_patterns"]
            return self._patterns_to_signal(patterns)

        else:
            # Fallback to ensemble signal
            if "ensemble_signal" in all_signals:
                ensemble = all_signals["ensemble_signal"]
                action = ensemble.get("ensemble_direction", "NEUTRAL")
                confidence = ensemble.get("confidence", 0.0)
                return self._convert_signal_to_action(action), confidence

        return "HOLD", 0.0

    def _convert_signal_to_action(self, signal: str) -> str:
        """Convert signal to trading action"""
        signal_upper = str(signal).upper()
        if signal_upper in ["LONG", "BULLISH"]:
            return "BUY"
        elif signal_upper in ["SHORT", "BEARISH"]:
            return "SELL"
        else:
            return "HOLD"

    def _patterns_to_signal(self, patterns: Dict) -> Tuple[str, float]:
        """Convert candlestick patterns to trading signal"""
        if not patterns:
            return "HOLD", 0.0

        bullish_strength = 0.0
        bearish_strength = 0.0

        for pattern_name, pattern_data in patterns.items():
            if isinstance(pattern_data, dict) and pattern_data.get("detected", False):
                strength = pattern_data.get("confidence", 0.0)
                if "bullish" in pattern_name.lower() or pattern_name.lower() in [
                    "hammer",
                    "doji",
                    "morning_star",
                ]:
                    bullish_strength += strength
                elif "bearish" in pattern_name.lower() or pattern_name.lower() in [
                    "hanging_man",
                    "shooting_star",
                    "evening_star",
                ]:
                    bearish_strength += strength

        if bullish_strength > bearish_strength and bullish_strength > 0.6:
            return "BUY", bullish_strength
        elif bearish_strength > bullish_strength and bearish_strength > 0.6:
            return "SELL", bearish_strength
        else:
            return "HOLD", max(bullish_strength, bearish_strength)

    # Caching methods
    def _get_cached_result(
        self, backtest_id: str, symbols: List[str]
    ) -> Optional[BacktestResult]:
        """Retrieve cached backtest result"""
        if not self.redis_client:
            return None

        try:
            cache_key = f"backtest:{backtest_id}"
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                # Would implement proper serialization/deserialization here
                logger.info(f"Cache hit for {backtest_id}")
                return None  # Simplified for now
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")

        return None

    def _cache_result(self, result: BacktestResult):
        """Cache backtest result"""
        if not self.redis_client:
            return

        try:
            cache_key = f"backtest:{result.backtest_id}"
            # Would implement proper serialization here
            # self.redis_client.setex(cache_key, 3600, serialized_result)
            logger.debug(f"Cached result for {result.backtest_id}")
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")

    def get_historical_data(
        self, symbols: List[str], years: int = 3
    ) -> Dict[str, pd.DataFrame]:
        """Get historical market data - placeholder implementation"""
        # This would connect to real data sources
        logger.info(f"Fetching {years} years of data for {len(symbols)} symbols")

        # Return mock data for now
        import numpy as np

        mock_data = {}
        for symbol in symbols:
            dates = pd.date_range(
                start=datetime.now() - timedelta(days=years * 365),
                end=datetime.now(),
                freq="D",
            )

            # Generate realistic price data
            base_price = np.random.uniform(50, 200)
            returns = np.random.normal(0.0005, 0.02, len(dates))
            prices = base_price * np.exp(np.cumsum(returns))

            mock_data[symbol] = pd.DataFrame(
                {
                    "Open": prices * np.random.uniform(0.99, 1.01, len(dates)),
                    "High": prices * np.random.uniform(1.00, 1.03, len(dates)),
                    "Low": prices * np.random.uniform(0.97, 1.00, len(dates)),
                    "Close": prices,
                    "Volume": np.random.randint(100000, 1000000, len(dates)),
                },
                index=dates,
            )

        return mock_data

    # Legacy compatibility methods
    async def single_instrument_backtest_with_strategy(self, *args, **kwargs):
        """Legacy compatibility method"""
        return await self.single_instrument_backtest(*args, **kwargs)

    async def multi_instrument_backtest_with_strategy(self, *args, **kwargs):
        """Legacy compatibility method"""
        return await self.multi_instrument_backtest(*args, **kwargs)
