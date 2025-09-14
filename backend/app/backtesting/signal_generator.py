"""
Signal Generation Service - Extracted from BacktestEngine

Handles signal generation using multiple strategies and indicators
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.analysis.modern_indicators import ModernIndicators
from app.analysis.indicator_factory import IndicatorFactory, IndicatorType
from app.analysis.llm_predictors import LLMPredictor, get_llm_predictor

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    Unified signal generation system for backtesting

    Coordinates multiple signal sources:
    - Integrated Alpha Detection (Elliott Wave, Fibonacci, etc.)
    - Modern Technical Indicators
    - LLM-based Predictions
    - Strategy Agents
    - Candlestick Patterns
    - Traditional Markov Chain Analysis
    """

    def __init__(self, signal_weights: Dict[str, float] = None):
        self.signal_weights = signal_weights or {
            'integrated_alpha': 0.25,
            'modern_indicators': 0.20,
            'llm_predictions': 0.20,
            'candlestick_patterns': 0.15,
            'markov_traditional': 0.10,
            'strategy_agents': 0.10
        }

        # Initialize components
        self.indicator_factory = IndicatorFactory()
        self.llm_predictor = None  # Lazy loaded

        # Strategy integration components (lazy loaded)
        self.integrated_alpha_detector = None
        self.strategy_agents = {}
        self.pattern_tool = None

        logger.info("SignalGenerator initialized with weighted signal combination")

    def _initialize_strategy_integrations(self):
        """Initialize all available trading strategies"""
        if self.integrated_alpha_detector is not None:
            return  # Already initialized

        try:
            # Import and initialize IntegratedAlphaDetector
            from app.analysis.integrated_alpha_detector import IntegratedAlphaDetector
            self.integrated_alpha_detector = IntegratedAlphaDetector()
            logger.info("✅ Integrated Alpha Detector initialized")
        except Exception as e:
            logger.warning(f"⚠️ IntegratedAlphaDetector initialization failed: {e}")

        try:
            # Import strategy agents
            from app.rag.agents.strategy_agent import (
                create_markov_agent, create_wyckoff_agent, create_fibonacci_agent
            )

            self.strategy_agents = {
                'markov': create_markov_agent(),
                'wyckoff': create_wyckoff_agent(),
                'fibonacci': create_fibonacci_agent()
            }
            logger.info("✅ Strategy agents initialized")
        except Exception as e:
            logger.warning(f"⚠️ Strategy agents initialization failed: {e}")

        try:
            # Import candlestick pattern detection
            from app.rag.tools.pattern_tool import PatternTool
            self.pattern_tool = PatternTool()
            logger.info("✅ Candlestick pattern detection initialized")
        except Exception as e:
            logger.warning(f"⚠️ PatternTool initialization failed: {e}")

    async def generate_integrated_signals(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        real_time_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate trading signals using all available strategies"""
        self._initialize_strategy_integrations()

        signals = {}
        signal_scores = {}

        # Convert market data to DataFrame for indicator calculations
        df_data = self._prepare_market_dataframe(market_data)

        # 1. Integrated Alpha Detection
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
                    signal_scores['integrated_alpha'] = self._extract_signal_score(
                        alpha_signal.direction, alpha_signal.alpha_potential
                    )
            except Exception as e:
                logger.warning(f"IntegratedAlpha signal generation failed for {symbol}: {e}")

        # 2. Modern Technical Indicators
        if df_data is not None and len(df_data) >= 20:
            try:
                modern_indicators = self.indicator_factory.calculate_all(
                    df_data, indicator_type=IndicatorType.MODERN, use_cache=True
                )

                modern_signal = self._generate_modern_indicator_signal(modern_indicators, symbol)
                if modern_signal:
                    signals['modern_indicators'] = modern_signal
                    signal_scores['modern_indicators'] = modern_signal.get('signal_strength', 0.0)

            except Exception as e:
                logger.warning(f"Modern indicators calculation failed for {symbol}: {e}")

        # 3. LLM-based Predictions
        if df_data is not None and len(df_data) >= 10:
            try:
                if not self.llm_predictor:
                    self.llm_predictor = get_llm_predictor()

                if self.llm_predictor:
                    llm_prediction = await self.llm_predictor.ensemble_predict(
                        symbol=symbol, market_data=df_data, horizon_days=5
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

        # 4. Individual Strategy Agents
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

        # 5. Candlestick Pattern Detection
        if self.pattern_tool:
            try:
                prices = market_data.get('prices', [])
                volumes = market_data.get('volumes', [])
                if len(prices) >= 4:
                    patterns = self.pattern_tool._detect_candlestick_patterns(prices, volumes)
                    signals['candlestick_patterns'] = patterns
                    signal_scores['candlestick_patterns'] = self._extract_pattern_signal_score(patterns)
            except Exception as e:
                logger.warning(f"Candlestick pattern detection failed for {symbol}: {e}")

        # 6. Traditional Markov Chain Analysis
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
                    latest_values = values['values'][-5:]

                    # Determine signal based on indicator type
                    if 'kama' in indicator_name.lower():
                        if len(latest_values) >= 2:
                            trend = 1 if latest_values[-1] > latest_values[-2] else -1
                            signals.append(trend)
                            signal_strengths.append(abs(latest_values[-1] - latest_values[-2]) / latest_values[-2])

                    elif 'envelope' in indicator_name.lower():
                        if isinstance(latest_values[-1], dict):
                            upper = latest_values[-1].get('upper', 0)
                            lower = latest_values[-1].get('lower', 0)
                            middle = latest_values[-1].get('middle', 0)
                            if upper > lower > 0:
                                position = (middle - lower) / (upper - lower)
                                signal = 1 if position > 0.7 else (-1 if position < 0.3 else 0)
                                signals.append(signal)
                                signal_strengths.append(abs(position - 0.5) * 2)

                    elif 'rvi' in indicator_name.lower():
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
                if weight > 0 and abs(score) > 0.01:
                    weighted_score += score * weight
                    total_weight += weight
                    contributing_signals[signal_name] = {
                        'score': score,
                        'weight': weight,
                        'contribution': score * weight
                    }

            # Handle strategy agents separately
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

    def calculate_markov_signals(self, price_series: pd.Series) -> Dict[str, Any]:
        """Calculate traditional Markov chain signals"""
        try:
            if len(price_series) < 20:
                return {'signal': 'NEUTRAL', 'confidence': 0.0, 'state': 'insufficient_data'}

            # Calculate returns
            returns = price_series.pct_change().dropna()

            # Simple state classification based on returns
            current_return = returns.iloc[-1]
            recent_volatility = returns.tail(20).std()

            # Classify current state
            if abs(current_return) < recent_volatility * 0.5:
                state = 'neutral'
                confidence = 0.4
                signal = 'NEUTRAL'
            elif current_return > recent_volatility * 0.5:
                state = 'bullish'
                confidence = min(0.8, abs(current_return) / recent_volatility)
                signal = 'BULLISH'
            else:
                state = 'bearish'
                confidence = min(0.8, abs(current_return) / recent_volatility)
                signal = 'BEARISH'

            return {
                'signal': signal,
                'confidence': confidence,
                'state': state,
                'current_return': current_return,
                'volatility': recent_volatility
            }

        except Exception as e:
            logger.warning(f"Markov signal calculation failed: {e}")
            return {'signal': 'NEUTRAL', 'confidence': 0.0, 'state': 'error'}

    def update_signal_weights(self, new_weights: Dict[str, float]):
        """Update signal combination weights"""
        # Validate weights sum to 1.0
        total_weight = sum(new_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Signal weights sum to {total_weight}, not 1.0. Normalizing...")
            new_weights = {k: v/total_weight for k, v in new_weights.items()}

        self.signal_weights.update(new_weights)
        logger.info(f"Updated signal weights: {self.signal_weights}")