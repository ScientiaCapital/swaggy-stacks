"""
Unified Backtesting Engine - Enterprise-grade backtesting system
Extends MarkovBacktester with multi-instrument, portfolio-level capabilities
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

# Import existing components - DISABLED: External dependencies not available
# sys.path.append('../../private_ai_modules')
# from backtest_markov import MarkovBacktester
# from integrated_backtesting.pattern_validation_pipeline import PatternValidationPipeline, PatternType

# App imports
from app.core.config import settings
from app.analysis.modern_indicators import ModernIndicators
from app.analysis.indicator_factory import IndicatorFactory, IndicatorType
from app.analysis.llm_predictors import LLMPredictor, get_llm_predictor

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtest execution"""
    initial_capital: float = 100000
    max_position_size: float = 0.15
    transaction_cost: float = 0.001
    years_lookback: int = 3
    trade_count_per_signal: int = 1
    enable_mlx_acceleration: bool = True
    enable_portfolio_correlation: bool = True
    cache_results: bool = True
    risk_free_rate: float = 0.02


@dataclass
class BacktestResult:
    """Comprehensive backtest results"""
    backtest_id: str
    config: BacktestConfig
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    alpha_generated: float
    beta: float
    volatility: float
    var_95: float  # Value at Risk 95%
    calmar_ratio: float
    sortino_ratio: float
    portfolio_value_history: List[Dict[str, Any]]
    trade_history: List[Dict[str, Any]]
    correlation_matrix: Optional[Dict[str, Any]] = None
    performance_attribution: Optional[Dict[str, float]] = None
    execution_time_seconds: float = 0.0
    memory_used_mb: float = 0.0


class BacktestEngine:
    """
    Unified Backtesting Engine

    Standalone backtesting engine with enterprise features:
    - Multi-instrument backtesting
    - Portfolio-level analysis
    - MLX acceleration
    - Redis caching
    - All strategy integration (Markov, Elliott Wave, Fibonacci, Golden Zone, Wyckoff, Alpha Patterns)
    - Advanced performance metrics
    """

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()

        # Initialize backtesting parameters
        self.initial_capital = self.config.initial_capital
        self.max_position_size = self.config.max_position_size
        self.transaction_cost = self.config.transaction_cost

        # Enhanced components
        self.pattern_pipeline = None
        self.redis_client = None
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Strategy integration
        self.integrated_alpha_detector = None
        self.strategy_agents = {}

        # Modern indicator and ML components
        self.indicator_factory = IndicatorFactory()
        self.llm_predictor = None  # Lazy loaded for performance

        # Signal weighting configuration - can be customized per strategy
        self.signal_weights = {
            'integrated_alpha': 0.25,
            'modern_indicators': 0.20,
            'llm_predictions': 0.20,
            'candlestick_patterns': 0.15,
            'markov_traditional': 0.10,
            'strategy_agents': 0.10  # Combined weight for all strategy agents
        }

        # Initialize enhanced features
        self._initialize_enhanced_features()

        logger.info("🚀 BacktestEngine initialized with all strategies and enterprise features")

    def _initialize_enhanced_features(self):
        """Initialize MLX acceleration, Redis caching, and strategy integrations"""
        # MLX Pattern Validation Pipeline
        if self.config.enable_mlx_acceleration and MLX_AVAILABLE:
            try:
                # TODO: Implement PatternValidationPipeline once external dependencies are resolved
                # self.pattern_pipeline = PatternValidationPipeline(
                #     initial_capital=self.config.initial_capital,
                #     max_position_size=self.config.max_position_size,
                #     min_pattern_confidence=0.6
                # )
                logger.info("✅ MLX acceleration placeholder enabled")
            except Exception as e:
                logger.warning(f"⚠️ MLX initialization failed: {e}")

        # Redis caching
        if self.config.cache_results and REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=getattr(settings, 'REDIS_HOST', 'localhost'),
                    port=getattr(settings, 'REDIS_PORT', 6379),
                    decode_responses=True,
                    socket_timeout=5
                )
                self.redis_client.ping()  # Test connection
                logger.info("✅ Redis caching enabled")
            except Exception as e:
                logger.warning(f"⚠️ Redis connection failed: {e}")
                self.redis_client = None

        # Initialize integrated strategy components
        self._initialize_strategy_integrations()

        # Initialize risk management integration
        self._initialize_risk_management()

    def _initialize_strategy_integrations(self):
        """Initialize all available trading strategies"""
        try:
            # Import and initialize IntegratedAlphaDetector
            from app.analysis.integrated_alpha_detector import IntegratedAlphaDetector
            self.integrated_alpha_detector = IntegratedAlphaDetector()
            logger.info("✅ Integrated Alpha Detector initialized (Elliott Wave, Fibonacci, Golden Zone, Wyckoff, Alpha Patterns)")
        except Exception as e:
            logger.warning(f"⚠️ IntegratedAlphaDetector initialization failed: {e}")

        try:
            # Import strategy agents
            from app.rag.agents.strategy_agent import (
                MarkovStrategy, WyckoffStrategy, FibonacciStrategy,
                create_markov_agent, create_wyckoff_agent, create_fibonacci_agent
            )
            
            self.strategy_agents = {
                'markov': create_markov_agent(),
                'wyckoff': create_wyckoff_agent(), 
                'fibonacci': create_fibonacci_agent()
            }
            logger.info("✅ Strategy agents initialized (Markov, Wyckoff, Fibonacci)")
        except Exception as e:
            logger.warning(f"⚠️ Strategy agents initialization failed: {e}")

        try:
            # Import candlestick pattern detection
            from app.rag.tools.pattern_tool import PatternTool
            self.pattern_tool = PatternTool()
            logger.info("✅ Candlestick pattern detection initialized (Pin Bar, Engulfing, Doji, Morning/Evening Star, Harami, Tweezers)")
        except Exception as e:
            logger.warning(f"⚠️ PatternTool initialization failed: {e}")

    def _initialize_risk_management(self):
        """Initialize comprehensive risk management system"""
        try:
            from app.risk.position_manager import IntegratedRiskManager

            # Risk management configuration
            risk_config = {
                "base_risk": {
                    "max_position_size": self.config.max_position_size * self.config.initial_capital,
                    "max_daily_loss": self.config.initial_capital * 0.02,  # 2% max daily loss
                },
                "position_sizing": {
                    "kelly_max_fraction": 0.25,
                    "fixed_fraction_default": 0.02,
                    "min_position_size": self.config.initial_capital * 0.001,  # 0.1% minimum
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
                }
            }

            self.risk_manager = IntegratedRiskManager(
                user_id="backtest_system",
                config=risk_config
            )
            logger.info("✅ Integrated Risk Management initialized (Kelly Criterion, ATR stops, VaR, Correlation limits)")
        except Exception as e:
            logger.warning(f"⚠️ Risk Management initialization failed: {e}")
            self.risk_manager = None

    async def generate_integrated_signals(self, symbol: str, market_data: Dict[str, Any],
                                        real_time_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate trading signals using all available strategies including modern indicators and LLM predictions"""
        signals = {}
        signal_scores = {}  # For weighted combination

        # Convert market data to DataFrame for indicator calculations
        df_data = self._prepare_market_dataframe(market_data)

        # 1. Integrated Alpha Detection (includes Elliott Wave, Fibonacci, Golden Zone, Wyckoff)
        if self.integrated_alpha_detector:
            try:
                alpha_signal = await self.integrated_alpha_detector.detect_alpha_opportunity(
                    symbol, market_data, real_time_data or {}
                )
                if alpha_signal:
                    signals['integrated_alpha'] = {
                        'signal': alpha_signal.direction,
                        'confidence': alpha_signal.alpha_potential,
                        'strength': alpha_signal.signal_strength,
                        'time_horizon': alpha_signal.time_horizon,
                        'components': {
                            'markov_state': alpha_signal.markov_state,
                            'elliott_wave': alpha_signal.elliott_wave_position,
                            'fibonacci_level': alpha_signal.fibonacci_level,
                            'golden_zone': alpha_signal.golden_zone_signal,
                            'wyckoff_phase': alpha_signal.wyckoff_phase,
                            'momentum_score': alpha_signal.momentum_score,
                            'volatility_score': alpha_signal.volatility_score
                        }
                    }
                    signal_scores['integrated_alpha'] = self._extract_signal_score(alpha_signal.direction, alpha_signal.alpha_potential)
            except Exception as e:
                logger.warning(f"IntegratedAlpha signal generation failed for {symbol}: {e}")

        # 2. Modern Technical Indicators (Kaabar's indicators)
        if df_data is not None and len(df_data) >= 20:
            try:
                modern_indicators = self.indicator_factory.calculate_all(
                    df_data,
                    indicator_type=IndicatorType.MODERN,
                    use_cache=True
                )

                # Generate ensemble signal from modern indicators
                modern_signal = self._generate_modern_indicator_signal(modern_indicators, symbol)
                if modern_signal:
                    signals['modern_indicators'] = modern_signal
                    signal_scores['modern_indicators'] = modern_signal.get('signal_strength', 0.0)

            except Exception as e:
                logger.warning(f"Modern indicators calculation failed for {symbol}: {e}")

        # 3. LLM-based Predictions (Chinese LLMs ensemble)
        if df_data is not None and len(df_data) >= 10:
            try:
                # Lazy load LLM predictor for performance
                if not self.llm_predictor:
                    self.llm_predictor = get_llm_predictor()

                if self.llm_predictor:
                    llm_prediction = await self.llm_predictor.ensemble_predict(
                        symbol=symbol,
                        market_data=df_data,
                        horizon_days=5
                    )

                    if llm_prediction:
                        signals['llm_predictions'] = {
                            'ensemble_prediction': llm_prediction.get('ensemble_prediction'),
                            'confidence': llm_prediction.get('confidence', 0.0),
                            'individual_predictions': llm_prediction.get('model_predictions', {}),
                            'reasoning': llm_prediction.get('reasoning', ''),
                            'risk_assessment': llm_prediction.get('risk_assessment', {})
                        }
                        signal_scores['llm_predictions'] = self._extract_llm_signal_score(llm_prediction)

            except Exception as e:
                logger.warning(f"LLM predictions failed for {symbol}: {e}")

        # 4. Individual Strategy Agents (backward compatibility)
        for strategy_name, agent in self.strategy_agents.items():
            try:
                if hasattr(agent, 'generate_signal'):
                    strategy_signal = agent.generate_signal(market_data)
                    signals[strategy_name] = strategy_signal
                    signal_scores[strategy_name] = self._extract_signal_score(
                        strategy_signal.get('direction', 'NEUTRAL'),
                        strategy_signal.get('confidence', 0.0)
                    )
            except Exception as e:
                logger.warning(f"{strategy_name} strategy signal failed for {symbol}: {e}")

        # 5. Candlestick Pattern Detection (backward compatibility)
        if self.pattern_tool:
            try:
                prices = market_data.get('prices', [])
                volumes = market_data.get('volumes', [])
                if len(prices) >= 4:  # Minimum for pattern detection
                    patterns = self.pattern_tool._detect_candlestick_patterns(prices, volumes)
                    signals['candlestick_patterns'] = patterns
                    signal_scores['candlestick_patterns'] = self._extract_pattern_signal_score(patterns)
            except Exception as e:
                logger.warning(f"Candlestick pattern detection failed for {symbol}: {e}")

        # 6. Traditional Markov Chain Analysis (backward compatibility)
        try:
            prices = market_data.get('prices', [])
            if len(prices) >= 20:
                markov_signal = self.calculate_markov_signals(pd.Series(prices))
                signals['markov_traditional'] = markov_signal
                signal_scores['markov_traditional'] = self._extract_signal_score(
                    markov_signal.get('signal', 'NEUTRAL'),
                    markov_signal.get('confidence', 0.0)
                )
        except Exception as e:
            logger.warning(f"Traditional Markov analysis failed for {symbol}: {e}")

        # 7. Generate Weighted Ensemble Signal
        ensemble_signal = self._combine_signals_with_weights(signal_scores, signals)
        signals['ensemble_signal'] = ensemble_signal

        return signals

    def _prepare_market_dataframe(self, market_data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Convert market data dictionary to DataFrame for indicator calculations"""
        try:
            # Extract price data
            prices = market_data.get('prices', [])
            volumes = market_data.get('volumes', [])
            timestamps = market_data.get('timestamps', [])

            if not prices or len(prices) < 10:
                return None

            # Handle different price data formats
            if isinstance(prices[0], dict):
                # OHLCV format
                df_data = pd.DataFrame(prices)
                if 'timestamp' in df_data.columns:
                    df_data['timestamp'] = pd.to_datetime(df_data['timestamp'])
                    df_data.set_index('timestamp', inplace=True)
            elif isinstance(prices[0], (int, float)):
                # Simple price array format
                data_dict = {'close': prices}
                if volumes:
                    data_dict['volume'] = volumes[:len(prices)]
                if timestamps:
                    data_dict['timestamp'] = pd.to_datetime(timestamps[:len(prices)])
                    df_data = pd.DataFrame(data_dict).set_index('timestamp')
                else:
                    df_data = pd.DataFrame(data_dict)

                # Generate OHLC from close prices if not available
                if 'open' not in df_data.columns:
                    df_data['open'] = df_data['close'].shift(1).fillna(df_data['close'])
                if 'high' not in df_data.columns:
                    df_data['high'] = df_data[['open', 'close']].max(axis=1)
                if 'low' not in df_data.columns:
                    df_data['low'] = df_data[['open', 'close']].min(axis=1)
            else:
                logger.warning("Unsupported price data format")
                return None

            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close']
            for col in required_cols:
                if col not in df_data.columns:
                    df_data[col] = df_data['close']

            return df_data

        except Exception as e:
            logger.warning(f"Failed to prepare market DataFrame: {e}")
            return None

    def _extract_signal_score(self, direction: str, confidence: float) -> float:
        """Convert signal direction and confidence to normalized score (-1 to +1)"""
        try:
            direction_map = {
                'BULLISH': 1.0, 'BUY': 1.0, 'LONG': 1.0,
                'BEARISH': -1.0, 'SELL': -1.0, 'SHORT': -1.0,
                'NEUTRAL': 0.0, 'HOLD': 0.0
            }
            direction_score = direction_map.get(str(direction).upper(), 0.0)
            confidence_normalized = max(0.0, min(1.0, confidence))
            return direction_score * confidence_normalized
        except Exception as e:
            logger.warning(f"Failed to extract signal score: {e}")
            return 0.0

    def _generate_modern_indicator_signal(self, indicators: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Generate ensemble signal from modern technical indicators"""
        try:
            signals = []
            signal_strengths = []

            # Process each modern indicator
            for indicator_name, values in indicators.items():
                if isinstance(values, dict) and 'values' in values:
                    latest_values = values['values'][-5:]  # Use last 5 values for trend analysis

                    # Determine signal based on indicator type
                    if 'kama' in indicator_name.lower():
                        # KAMA: Compare current vs previous
                        if len(latest_values) >= 2:
                            trend = 1 if latest_values[-1] > latest_values[-2] else -1
                            signals.append(trend)
                            signal_strengths.append(abs(latest_values[-1] - latest_values[-2]) / latest_values[-2])

                    elif 'envelope' in indicator_name.lower():
                        # K's Envelopes: Position relative to bands
                        if isinstance(latest_values[-1], dict):
                            upper = latest_values[-1].get('upper', 0)
                            lower = latest_values[-1].get('lower', 0)
                            middle = latest_values[-1].get('middle', 0)
                            if upper > lower > 0:
                                # Compare current price position
                                position = (middle - lower) / (upper - lower)
                                signal = 1 if position > 0.7 else (-1 if position < 0.3 else 0)
                                signals.append(signal)
                                signal_strengths.append(abs(position - 0.5) * 2)

                    elif 'rvi' in indicator_name.lower():
                        # RVI: Momentum oscillator
                        if latest_values[-1] > 0.5:
                            signals.append(1)
                            signal_strengths.append(latest_values[-1])
                        elif latest_values[-1] < -0.5:
                            signals.append(-1)
                            signal_strengths.append(abs(latest_values[-1]))
                        else:
                            signals.append(0)
                            signal_strengths.append(0.1)

            # Calculate ensemble signal
            if signals:
                weighted_signal = np.average(signals, weights=signal_strengths) if signal_strengths else np.mean(signals)
                confidence = np.mean(signal_strengths) if signal_strengths else 0.0

                signal_direction = 'BULLISH' if weighted_signal > 0.1 else ('BEARISH' if weighted_signal < -0.1 else 'NEUTRAL')

                return {
                    'signal': signal_direction,
                    'signal_strength': abs(weighted_signal),
                    'confidence': confidence,
                    'indicator_count': len(signals),
                    'individual_signals': dict(zip(indicators.keys(), signals))
                }

            return None

        except Exception as e:
            logger.warning(f"Failed to generate modern indicator signal for {symbol}: {e}")
            return None

    def _extract_llm_signal_score(self, llm_prediction: Dict[str, Any]) -> float:
        """Extract normalized signal score from LLM prediction"""
        try:
            ensemble_pred = llm_prediction.get('ensemble_prediction', 'NEUTRAL')
            confidence = llm_prediction.get('confidence', 0.0)
            return self._extract_signal_score(ensemble_pred, confidence)
        except Exception as e:
            logger.warning(f"Failed to extract LLM signal score: {e}")
            return 0.0

    def _extract_pattern_signal_score(self, patterns: Dict[str, Any]) -> float:
        """Extract signal score from candlestick patterns"""
        try:
            if not patterns:
                return 0.0

            # Weight different pattern types
            bullish_patterns = ['hammer', 'doji', 'engulfing_bullish', 'morning_star']
            bearish_patterns = ['hanging_man', 'shooting_star', 'engulfing_bearish', 'evening_star']

            total_score = 0.0
            pattern_count = 0

            for pattern_name, pattern_data in patterns.items():
                if isinstance(pattern_data, dict) and pattern_data.get('detected', False):
                    confidence = pattern_data.get('confidence', 0.5)
                    if pattern_name.lower() in bullish_patterns:
                        total_score += confidence
                        pattern_count += 1
                    elif pattern_name.lower() in bearish_patterns:
                        total_score -= confidence
                        pattern_count += 1

            return total_score / max(pattern_count, 1)

        except Exception as e:
            logger.warning(f"Failed to extract pattern signal score: {e}")
            return 0.0

    def _combine_signals_with_weights(self, signal_scores: Dict[str, float], all_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Combine all signals using configured weights to generate ensemble signal"""
        try:
            weighted_score = 0.0
            total_weight = 0.0
            contributing_signals = {}

            # Calculate weighted average of individual signal scores
            for signal_name, score in signal_scores.items():
                weight = self.signal_weights.get(signal_name, 0.0)
                if weight > 0 and abs(score) > 0.01:  # Only include meaningful signals
                    weighted_score += score * weight
                    total_weight += weight
                    contributing_signals[signal_name] = {
                        'score': score,
                        'weight': weight,
                        'contribution': score * weight
                    }

            # Handle strategy agents separately (they may have multiple entries)
            strategy_agent_scores = []
            for signal_name, score in signal_scores.items():
                if signal_name not in self.signal_weights and signal_name in all_signals:
                    strategy_agent_scores.append(score)

            if strategy_agent_scores:
                avg_strategy_score = np.mean(strategy_agent_scores)
                weight = self.signal_weights.get('strategy_agents', 0.1)
                weighted_score += avg_strategy_score * weight
                total_weight += weight
                contributing_signals['strategy_agents_combined'] = {
                    'score': avg_strategy_score,
                    'weight': weight,
                    'contribution': avg_strategy_score * weight
                }

            # Normalize final score
            if total_weight > 0:
                final_score = weighted_score / total_weight
            else:
                final_score = 0.0

            # Generate ensemble prediction
            confidence = min(abs(final_score), 1.0)
            if final_score > 0.15:
                ensemble_direction = 'BULLISH'
            elif final_score < -0.15:
                ensemble_direction = 'BEARISH'
            else:
                ensemble_direction = 'NEUTRAL'

            return {
                'ensemble_direction': ensemble_direction,
                'ensemble_score': final_score,
                'confidence': confidence,
                'total_weight_used': total_weight,
                'contributing_signals': contributing_signals,
                'signal_count': len(contributing_signals)
            }

        except Exception as e:
            logger.warning(f"Failed to combine signals with weights: {e}")
            return {
                'ensemble_direction': 'NEUTRAL',
                'ensemble_score': 0.0,
                'confidence': 0.0,
                'total_weight_used': 0.0,
                'contributing_signals': {},
                'signal_count': 0
            }

    async def single_instrument_backtest_with_strategy(self,
                                                     symbol: str,
                                                     strategy_name: str = 'integrated_alpha',
                                                     custom_signals: List[Dict[str, Any]] = None) -> BacktestResult:
        """Execute backtest for single instrument using specified strategy"""
        start_time = time.time()
        backtest_id = f"single_{symbol}_{strategy_name}_{int(start_time)}"

        logger.info(f"📊 Starting single instrument backtest: {symbol} with {strategy_name} strategy")

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
        self._reset_portfolio()

        # Execute backtest with specified strategy
        symbol_data = data[symbol]
        if custom_signals:
            await self._execute_signal_based_backtest(symbol, symbol_data, custom_signals)
        else:
            await self._execute_strategy_backtest_with_signals(symbol, symbol_data, strategy_name)

        # Calculate performance metrics
        result = self._calculate_performance_metrics(
            backtest_id, [symbol], symbol_data.index[0], symbol_data.index[-1], start_time
        )

        # Cache results
        self._cache_result(result)

        execution_time = time.time() - start_time
        logger.info(f"✅ Single instrument backtest with {strategy_name} completed in {execution_time:.2f}s")

        return result

    async def _execute_strategy_backtest_with_signals(self, symbol: str, data: pd.DataFrame, strategy_name: str):
        """Execute backtest using signals from specified strategy"""
        for i in range(50, len(data)):  # Start after sufficient lookback period
            current_date = data.index[i]
            
            # Prepare market data for signal generation
            market_data = {
                'prices': data['Close'].iloc[:i+1].tolist(),
                'highs': data['High'].iloc[:i+1].tolist(),
                'lows': data['Low'].iloc[:i+1].tolist(),
                'volumes': data['Volume'].iloc[:i+1].tolist() if 'Volume' in data.columns else [],
                'current_price': data.loc[current_date, 'Close']
            }
            
            # Generate signals using all strategies
            all_signals = await self.generate_integrated_signals(symbol, market_data)
            
            # Extract signals for specified strategy
            if strategy_name == 'integrated_alpha' and 'integrated_alpha' in all_signals:
                signal_data = all_signals['integrated_alpha']
                action = self._convert_signal_to_action(signal_data['signal'])
                confidence = signal_data['confidence']
                
            elif strategy_name in all_signals:
                signal_data = all_signals[strategy_name]
                action = signal_data.get('signal', 'HOLD')
                confidence = signal_data.get('confidence', 0.5)
                
            elif strategy_name == 'candlestick_patterns' and 'candlestick_patterns' in all_signals:
                patterns = all_signals['candlestick_patterns']
                action, confidence = self._patterns_to_signal(patterns)
                
            else:
                # Fallback to traditional Markov
                if 'markov_traditional' in all_signals:
                    signal_data = all_signals['markov_traditional']
                    action = signal_data['signal']
                    confidence = signal_data['confidence']
                else:
                    continue

            # Execute trade if signal is strong enough
            if action != 'HOLD' and confidence > 0.6:
                current_price = data.loc[current_date, 'Close']

                success = self.execute_trade(
                    symbol=symbol,
                    action=action,
                    price=current_price,
                    date=current_date,
                    position_multiplier=confidence * self.config.trade_count_per_signal
                )

                if success:
                    portfolio_value = self._calculate_current_portfolio_value(data, current_date)
                    self.portfolio_history.append({
                        'date': current_date,
                        'total_value': portfolio_value,
                        'cash': self.cash,
                        'positions': dict(self.positions),
                        'strategy': strategy_name,
                        'signal_confidence': confidence
                    })

    def _convert_signal_to_action(self, signal: str) -> str:
        """Convert IntegratedAlphaDetector signal to trading action"""
        if signal == 'long':
            return 'BUY'
        elif signal == 'short':
            return 'SELL'
        else:
            return 'HOLD'

    def _patterns_to_signal(self, patterns: List[Dict]) -> Tuple[str, float]:
        """Convert candlestick patterns to trading signal"""
        if not patterns:
            return 'HOLD', 0.0
        
        # Aggregate pattern signals
        bullish_strength = 0.0
        bearish_strength = 0.0
        
        for pattern in patterns:
            strength = pattern.get('strength', 0.0)
            if pattern.get('direction') == 'bullish':
                bullish_strength += strength
            elif pattern.get('direction') == 'bearish':
                bearish_strength += strength
        
        # Determine overall signal
        if bullish_strength > bearish_strength and bullish_strength > 0.6:
            return 'BUY', bullish_strength
        elif bearish_strength > bullish_strength and bearish_strength > 0.6:
            return 'SELL', bearish_strength
        else:
            return 'HOLD', max(bullish_strength, bearish_strength)

    async def strategy_comparison_backtest(self, symbols: List[str], 
                                         strategies: List[str] = None) -> Dict[str, BacktestResult]:
        """Compare performance of multiple strategies on the same instruments"""
        if strategies is None:
            strategies = ['integrated_alpha', 'markov_traditional', 'candlestick_patterns']
            
        logger.info(f"📊 Starting strategy comparison backtest: {len(strategies)} strategies on {len(symbols)} symbols")
        
        results = {}
        
        for strategy in strategies:
            logger.info(f"🎯 Testing {strategy} strategy...")
            
            try:
                if len(symbols) == 1:
                    result = await self.single_instrument_backtest_with_strategy(symbols[0], strategy)
                else:
                    result = await self.multi_instrument_backtest_with_strategy(symbols, strategy)
                
                results[strategy] = result
                
                logger.info(f"✅ {strategy}: {result.total_return:.2%} return, {result.sharpe_ratio:.2f} Sharpe")
                
            except Exception as e:
                logger.error(f"❌ {strategy} strategy failed: {e}")
                continue
        
        return results

    async def multi_instrument_backtest_with_strategy(self,
                                                    symbols: List[str],
                                                    strategy_name: str = 'integrated_alpha') -> BacktestResult:
        """Execute multi-instrument backtest using specified strategy"""
        start_time = time.time()
        backtest_id = f"multi_{strategy_name}_{'_'.join(symbols[:3])}_{int(start_time)}"

        logger.info(f"📊 Starting multi-instrument backtest: {len(symbols)} symbols with {strategy_name}")

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
        self._reset_portfolio()

        # Execute with strategy-aware portfolio management
        await self._execute_portfolio_backtest_with_strategy(symbols, data, strategy_name)

        # Calculate comprehensive portfolio metrics
        earliest_date = min(data[sym].index[0] for sym in symbols if sym in data)
        latest_date = max(data[sym].index[-1] for sym in symbols if sym in data)

        result = self._calculate_performance_metrics(
            backtest_id, symbols, earliest_date, latest_date, start_time
        )

        # Add correlation analysis
        if self.config.enable_portfolio_correlation and len(symbols) > 1:
            result.correlation_matrix = self._calculate_correlation_matrix(symbols, data)
            result.performance_attribution = self._calculate_performance_attribution(symbols)

        # Cache results
        self._cache_result(result)

        execution_time = time.time() - start_time
        logger.info(f"✅ Multi-instrument backtest with {strategy_name} completed in {execution_time:.2f}s")

        return result

    async def _execute_portfolio_backtest_with_strategy(self,
                                                      symbols: List[str],
                                                      data: Dict[str, pd.DataFrame],
                                                      strategy_name: str):
        """Execute portfolio backtest with specified strategy"""
        # Create unified timeline
        all_dates = set()
        for symbol_data in data.values():
            all_dates.update(symbol_data.index)

        timeline = sorted(all_dates)[50:]  # Skip first 50 days for lookback

        # Execute trades chronologically across all symbols
        for i, date in enumerate(timeline):
            if i % 100 == 0:  # Progress logging
                logger.info(f"📈 Processing date {i+1}/{len(timeline)}: {date.strftime('%Y-%m-%d')}")
            
            for symbol in symbols:
                if symbol not in data or date not in data[symbol].index:
                    continue

                symbol_data = data[symbol]
                date_index = symbol_data.index.get_loc(date)
                
                if date_index < 50:  # Need sufficient lookback
                    continue

                # Prepare market data
                market_data = {
                    'prices': symbol_data['Close'].iloc[:date_index+1].tolist(),
                    'highs': symbol_data['High'].iloc[:date_index+1].tolist(),
                    'lows': symbol_data['Low'].iloc[:date_index+1].tolist(),
                    'volumes': symbol_data['Volume'].iloc[:date_index+1].tolist() if 'Volume' in symbol_data.columns else [],
                    'current_price': symbol_data.loc[date, 'Close']
                }

                # Generate signals
                all_signals = await self.generate_integrated_signals(symbol, market_data)
                
                # Execute trade based on strategy
                await self._execute_portfolio_trade_with_strategy(
                    symbol, all_signals, symbol_data, date, strategy_name
                )

    async def _execute_portfolio_trade_with_strategy(self,
                                                   symbol: str,
                                                   all_signals: Dict[str, Any],
                                                   data: pd.DataFrame,
                                                   date: datetime,
                                                   strategy_name: str):
        """Execute portfolio trade using specified strategy"""
        # Extract signal based on strategy
        if strategy_name == 'integrated_alpha' and 'integrated_alpha' in all_signals:
            signal_data = all_signals['integrated_alpha']
            action = self._convert_signal_to_action(signal_data['signal'])
            confidence = signal_data['confidence']
            
        elif strategy_name in all_signals:
            signal_data = all_signals[strategy_name]
            action = signal_data.get('signal', 'HOLD')
            confidence = signal_data.get('confidence', 0.5)
            
        elif strategy_name == 'candlestick_patterns' and 'candlestick_patterns' in all_signals:
            patterns = all_signals['candlestick_patterns']
            action, confidence = self._patterns_to_signal(patterns)
            
        else:
            return  # No valid signal

        if action == 'HOLD' or confidence <= 0.5:
            return

        current_price = data.loc[date, 'Close']

        # Portfolio-aware position sizing
        correlation_adjustment = self._calculate_correlation_adjustment(symbol, current_price)
        adjusted_confidence = confidence * correlation_adjustment

        if adjusted_confidence > 0.5:
            success = self.execute_trade(
                symbol=symbol,
                action=action,
                price=current_price,
                date=date,
                position_multiplier=adjusted_confidence * self.config.trade_count_per_signal
            )

            if success:
                portfolio_value = self._calculate_current_portfolio_value(data, date)
                self.portfolio_history.append({
                    'date': date,
                    'total_value': portfolio_value,
                    'cash': self.cash,
                    'positions': dict(self.positions),
                    'strategy': strategy_name,
                    'correlation_adjustment': correlation_adjustment,
                    'signal_confidence': confidence,
                    'adjusted_confidence': adjusted_confidence
                })

    async def single_instrument_backtest(self,
                                       symbol: str,
                                       strategy_signals: List[Dict[str, Any]] = None) -> BacktestResult:
        """Execute backtest for single instrument (backwards compatibility)"""
        return await self.single_instrument_backtest_with_strategy(symbol, 'integrated_alpha', strategy_signals)

    async def multi_instrument_backtest(self,
                                      symbols: List[str],
                                      strategy_signals: Dict[str, List[Dict[str, Any]]] = None) -> BacktestResult:
        """Execute backtest across multiple instruments (backwards compatibility)"""
        if strategy_signals:
            # Convert strategy_signals to custom signal format and use traditional method
            return await super().multi_instrument_backtest(symbols, strategy_signals)
        else:
            return await self.multi_instrument_backtest_with_strategy(symbols, 'integrated_alpha')

    async def multi_trade_per_signal_backtest(self,
                                            symbols: List[str],
                                            trade_count_per_signal: int = 3) -> BacktestResult:
        """Execute multiple trades per signal with position pyramiding"""
        start_time = time.time()
        backtest_id = f"multi_trade_{len(symbols)}_{trade_count_per_signal}_{int(start_time)}"

        logger.info(f"📊 Multi-trade backtest: {trade_count_per_signal} trades per signal")

        # Temporarily modify config
        original_trade_count = self.config.trade_count_per_signal
        self.config.trade_count_per_signal = trade_count_per_signal

        try:
            # Execute with modified position sizing
            result = await self.multi_instrument_backtest_with_strategy(symbols)
            result.backtest_id = backtest_id
            return result
        finally:
            # Restore original config
            self.config.trade_count_per_signal = original_trade_count

    async def _execute_strategy_backtest(self,
                                       symbol: str,
                                       data: pd.DataFrame,
                                       strategy_signals: List[Dict[str, Any]] = None):
        """Execute backtest logic for a single symbol (backwards compatibility)"""
        if strategy_signals:
            # Use provided signals
            await self._execute_signal_based_backtest(symbol, data, strategy_signals)
        else:
            # Use integrated alpha strategy as default
            await self._execute_strategy_backtest_with_signals(symbol, data, 'integrated_alpha')

    async def _execute_signal_based_backtest(self,
                                           symbol: str,
                                           data: pd.DataFrame,
                                           signals: List[Dict[str, Any]]):
        """Execute backtest using provided trading signals"""
        for signal in signals:
            signal_date = pd.to_datetime(signal['date'])
            if signal_date not in data.index:
                continue

            price = data.loc[signal_date, 'Close']
            action = signal.get('action', 'HOLD')
            confidence = signal.get('confidence', 0.5)

            # Scale position size by confidence and trade count
            position_multiplier = confidence * self.config.trade_count_per_signal

            if action in ['BUY', 'SELL']:
                success = self.execute_trade(
                    symbol=symbol,
                    action=action,
                    price=price,
                    date=signal_date,
                    position_multiplier=position_multiplier
                )

                if success:
                    # Record portfolio value
                    portfolio_value = self._calculate_current_portfolio_value(data, signal_date)
                    self.portfolio_history.append({
                        'date': signal_date,
                        'total_value': portfolio_value,
                        'cash': self.cash,
                        'positions': dict(self.positions)
                    })

    async def _execute_markov_based_backtest(self, symbol: str, data: pd.DataFrame):
        """Execute backtest using Markov chain analysis (kept for compatibility)"""
        await self._execute_strategy_backtest_with_signals(symbol, data, 'markov_traditional')

    def enhance_backtest_results_with_risk_metrics(
        self,
        base_results: Dict[str, Any],
        positions: List[Dict[str, Any]],
        price_history: Dict[str, List[float]],
        portfolio_value: float
    ) -> Dict[str, Any]:
        """Enhance backtest results with comprehensive risk management metrics"""
        try:
            if not self.risk_manager:
                logger.warning("Risk manager not initialized, skipping risk metrics enhancement")
                return base_results

            # Calculate daily P&L from base results
            daily_pnl = 0.0
            if 'portfolio_value_history' in base_results and len(base_results['portfolio_value_history']) > 1:
                recent_values = base_results['portfolio_value_history'][-2:]
                daily_pnl = recent_values[-1].get('value', 0) - recent_values[-2].get('value', 0)

            # Generate comprehensive risk assessment
            risk_assessment = self.risk_manager.comprehensive_risk_assessment(
                positions=positions,
                account_value=portfolio_value,
                daily_pnl=daily_pnl,
                price_history=price_history
            )

            # Calculate optimal position sizes for all symbols
            position_sizing_analysis = {}
            for symbol in base_results.get('symbols', []):
                if symbol in price_history and price_history[symbol]:
                    current_price = price_history[symbol][-1]

                    # Get historical performance from base results
                    historical_performance = {
                        "win_rate": base_results.get('win_rate', 0.6),
                        "avg_win": 0.05,  # Default 5% average win
                        "avg_loss": 0.03,  # Default 3% average loss
                    }

                    # Calculate optimal position using Kelly Criterion
                    optimal_position, sizing_details = self.risk_manager.calculate_optimal_position_size(
                        symbol=symbol,
                        entry_price=current_price,
                        account_value=portfolio_value,
                        historical_performance=historical_performance,
                        method="kelly"
                    )

                    position_sizing_analysis[symbol] = {
                        "optimal_position_size": optimal_position,
                        "current_price": current_price,
                        "sizing_method": sizing_details.get("method", "kelly"),
                        "kelly_fraction": sizing_details.get("kelly_fraction", 0),
                        "shares_recommended": sizing_details.get("shares", 0)
                    }

            # Calculate stop loss recommendations
            stop_loss_analysis = {}
            for symbol in base_results.get('symbols', []):
                if symbol in price_history and len(price_history[symbol]) >= 14:
                    current_price = price_history[symbol][-1]

                    # Prepare market data for ATR calculation
                    prices = price_history[symbol]
                    market_data = {
                        "highs": [p * 1.01 for p in prices],  # Mock highs (1% above close)
                        "lows": [p * 0.99 for p in prices],   # Mock lows (1% below close)
                        "closes": prices
                    }

                    # Calculate comprehensive stop loss
                    stop_price, stop_details = self.risk_manager.calculate_comprehensive_stop_loss(
                        symbol=symbol,
                        entry_price=current_price,
                        side="BUY",
                        market_data=market_data,
                        method="atr"
                    )

                    stop_loss_analysis[symbol] = {
                        "recommended_stop_loss": stop_price,
                        "stop_distance_pct": abs(stop_price - current_price) / current_price,
                        "method": stop_details.get("method", "atr"),
                        "atr_value": stop_details.get("atr", 0)
                    }

            # Enhanced results with risk metrics
            enhanced_results = {
                **base_results,
                "risk_management": {
                    "risk_assessment": risk_assessment,
                    "position_sizing_analysis": position_sizing_analysis,
                    "stop_loss_analysis": stop_loss_analysis,
                    "risk_score": risk_assessment.get("risk_score", 50),
                    "var_analysis": risk_assessment.get("var_analysis", {}),
                    "correlation_analysis": risk_assessment.get("correlation_analysis", {}),
                    "sector_analysis": risk_assessment.get("sector_analysis", {}),
                },
                "risk_adjusted_metrics": {
                    "risk_adjusted_return": base_results.get('total_return', 0) / max(risk_assessment.get("risk_score", 50) / 100, 0.1),
                    "var_utilization": risk_assessment.get("var_analysis", {}).get("var_pct_of_portfolio", 0),
                    "portfolio_heat": sum(
                        sizing["kelly_fraction"] for sizing in position_sizing_analysis.values()
                        if "kelly_fraction" in sizing
                    ),
                    "diversification_ratio": 1 - risk_assessment.get("correlation_analysis", {}).get("correlation_compliant", True),
                }
            }

            logger.info(
                "Backtest results enhanced with comprehensive risk metrics",
                risk_score=risk_assessment.get("risk_score", 0),
                position_count=len(position_sizing_analysis),
                var_dollar=risk_assessment.get("var_analysis", {}).get("var_dollar", 0)
            )

            return enhanced_results

        except Exception as e:
            logger.error("Error enhancing backtest results with risk metrics", error=str(e))
            # Return base results if enhancement fails
            return {
                **base_results,
                "risk_management_error": str(e)
            }