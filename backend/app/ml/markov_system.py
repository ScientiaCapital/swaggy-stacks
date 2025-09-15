"""
Markov Trading System
Combines functionality from markov_analyzer.py, markov_agent.py, and legacy enhanced_markov_system.py
Eliminates redundancy while maintaining all features
"""

import warnings
from typing import Dict, Tuple, Any, List, Optional

import numpy as np
import pandas as pd
import structlog
from scipy import stats
from sklearn.preprocessing import StandardScaler

from app.core.exceptions import TradingError

warnings.filterwarnings("ignore")
logger = structlog.get_logger()

# Lazy import to avoid circular dependencies
def _get_regime_detector():
    """Lazy import of MarketRegimeDetector to avoid circular dependencies"""
    try:
        from app.ml.unsupervised.market_regime import MarketRegimeDetector
        return MarketRegimeDetector
    except ImportError:
        logger.warning("MarketRegimeDetector not available, regime detection disabled")
        return None

# ============================================================================
# CORE MARKOV ANALYSIS ENGINE
# ============================================================================


class MarkovCore:
    """
    Core Markov Chain Analysis - Base functionality consolidated from markov_analyzer.py
    """

    def __init__(self, lookback_period: int = 100, n_states: int = 5):
        self.lookback_period = lookback_period
        self.n_states = n_states
        self.transition_matrix = None
        self.state_probabilities = None
        self.state_bounds = None
        self.scaler = StandardScaler()

        logger.info(
            "Markov core initialized",
            lookback_period=lookback_period,
            n_states=n_states,
        )

    def _calculate_returns(self, price_data: pd.DataFrame) -> pd.Series:
        """Calculate returns from price data"""
        if isinstance(price_data, pd.DataFrame):
            close_prices = (
                price_data["close"]
                if "close" in price_data.columns
                else price_data.iloc[:, 0]
            )
        else:
            close_prices = pd.Series(price_data)

        returns = close_prices.pct_change().dropna()
        return returns

    def _discretize_returns(self, returns: pd.Series) -> pd.Series:
        """Discretize returns into states using quantile-based method"""
        quantiles = np.linspace(0, 1, self.n_states + 1)
        state_bounds = returns.quantile(quantiles)
        self.state_bounds = state_bounds

        states = pd.cut(returns, bins=state_bounds, labels=False, include_lowest=True)
        return states

    def _build_transition_matrix(self, states: pd.Series) -> np.ndarray:
        """Build transition probability matrix"""
        n_states = self.n_states
        transition_matrix = np.zeros((n_states, n_states))

        # Count transitions
        for i in range(len(states) - 1):
            current_state = int(states.iloc[i])
            next_state = int(states.iloc[i + 1])
            if not np.isnan(current_state) and not np.isnan(next_state):
                transition_matrix[current_state, next_state] += 1

        # Normalize to probabilities
        row_sums = transition_matrix.sum(axis=1)
        transition_matrix = np.divide(
            transition_matrix,
            row_sums[:, np.newaxis],
            out=np.zeros_like(transition_matrix),
            where=row_sums[:, np.newaxis] != 0,
        )

        return transition_matrix

    def _calculate_state_probabilities(self, states: pd.Series) -> np.ndarray:
        """Calculate current state probabilities"""
        state_counts = states.value_counts().sort_index()
        probabilities = np.zeros(self.n_states)

        for state, count in state_counts.items():
            if not np.isnan(state):
                probabilities[int(state)] = count / len(states)

        return probabilities

    def predict_next_state(
        self, current_state: int, steps: int = 1
    ) -> Tuple[int, float]:
        """Predict future states using transition matrix"""
        if self.transition_matrix is None:
            return 0, 0.0

        try:
            # Start with current state
            state_vector = np.zeros(self.n_states)
            state_vector[current_state] = 1.0

            # Apply transition matrix
            for _ in range(steps):
                state_vector = np.dot(state_vector, self.transition_matrix)

            # Get most likely next state
            predicted_state = np.argmax(state_vector)
            confidence = state_vector[predicted_state]

            return int(predicted_state), float(confidence)
        except Exception as e:
            logger.error("Error predicting next state", error=str(e))
            return 0, 0.0


# ============================================================================
# DATA HANDLING AND UTILITIES
# ============================================================================


class DataHandler:
    """Data handling with validation and cleaning"""

    def __init__(self):
        self.data_cache = {}
        self.validated_data = {}

    def validate_and_clean_data(self, prices, volumes=None):
        """Validate and clean market data, handling missing values and outliers"""
        prices = np.array(prices)

        # Handle NaN values
        if np.any(np.isnan(prices)):
            nan_indices = np.where(np.isnan(prices))[0]
            for idx in nan_indices:
                if idx == 0:
                    next_val_idx = np.where(~np.isnan(prices))[0]
                    if len(next_val_idx) > 0:
                        prices[idx] = prices[next_val_idx[0]]
                elif idx == len(prices) - 1:
                    prices[idx] = prices[idx - 1]
                else:
                    prev_val = prices[idx - 1]
                    next_val_idx = np.where(~np.isnan(prices[idx + 1 :]))[0]
                    if len(next_val_idx) > 0:
                        next_val = prices[idx + 1 + next_val_idx[0]]
                        prices[idx] = (prev_val + next_val) / 2
                    else:
                        prices[idx] = prev_val

        # Handle outliers using Z-score method
        if len(prices) > 10:
            returns = np.diff(np.log(prices))
            z_scores = np.abs(stats.zscore(returns))
            outlier_indices = np.where(z_scores > 3)[0]

            for idx in outlier_indices:
                if idx > 0 and idx < len(returns) - 1:
                    returns[idx] = (returns[idx - 1] + returns[idx + 1]) / 2

            # Reconstruct prices from cleaned returns
            cleaned_prices = [prices[0]]
            for i, r in enumerate(returns):
                cleaned_prices.append(cleaned_prices[-1] * np.exp(r))
            prices = np.array(cleaned_prices)

        # Process volumes if provided
        cleaned_volumes = None
        if volumes is not None:
            volumes = np.array(volumes)
            if np.any(np.isnan(volumes)):
                mask = np.isnan(volumes)
                volumes[mask] = np.interp(
                    np.flatnonzero(mask), np.flatnonzero(~mask), volumes[~mask]
                )
            cleaned_volumes = volumes

        return prices, cleaned_volumes


class PositionSizer:
    """Position sizing based on market uncertainty and confidence"""

    def __init__(self, account_size=100000, max_risk_per_trade=0.02):
        self.account_size = account_size
        self.max_risk_per_trade = max_risk_per_trade
        self.risk_free_rate = 0.02

    def calculate_size(self, uncertainty, current_price, stop_loss_price=None):
        """Calculate position size using modified Kelly criterion"""
        max_risk_amount = self.account_size * self.max_risk_per_trade

        if stop_loss_price and current_price > 0:
            risk_per_share = abs(current_price - stop_loss_price)
            if risk_per_share > 0:
                shares = max_risk_amount / risk_per_share
                return shares

        # Use uncertainty to determine position size
        if uncertainty < 0.2:
            position_pct = 0.08
        elif uncertainty < 0.4:
            position_pct = 0.05
        elif uncertainty < 0.6:
            position_pct = 0.03
        else:
            position_pct = 0.01

        shares = (self.account_size * position_pct) / current_price
        return shares


# ============================================================================
# MARKOV SYSTEM
# ============================================================================


class MarkovSystem(MarkovCore):
    """
    Markov System combining core functionality with advanced features
    Consolidates enhanced_markov_system.py functionality
    """

    def __init__(self, lookback_period: int = 100, n_states: int = 5, **kwargs):
        super().__init__(lookback_period, n_states)

        # Enhanced components
        self.data_handler = DataHandler()
        self.position_sizer = PositionSizer(**kwargs)

        # Market regime detection
        self.regime_detector = None
        self.enable_regime_detection = kwargs.get("enable_regime_detection", True)

        if self.enable_regime_detection:
            RegimeDetectorClass = _get_regime_detector()
            if RegimeDetectorClass:
                self.regime_detector = RegimeDetectorClass(
                    n_regimes=kwargs.get("n_regimes", 5),
                    lookback_period=lookback_period,
                    rolling_window=kwargs.get("regime_rolling_window", 20),
                    transition_threshold=kwargs.get("regime_transition_threshold", 0.7),
                    min_regime_duration=kwargs.get("min_regime_duration", 5)
                )
                logger.info("Market regime detection enabled")

        # Enhanced parameters
        self.confidence_adjustment_factor = kwargs.get(
            "confidence_adjustment_factor", 1.2
        )
        self.state_transition_threshold = kwargs.get("state_transition_threshold", 0.6)
        self.lookback_periods = kwargs.get("lookback_periods", [5, 10, 20, 50])

        # Cache for analysis results
        self.analysis_cache = {}

        logger.info("Markov system initialized")

    def analyze(self, price_data: pd.DataFrame) -> Dict:
        """
        Comprehensive Markov analysis combining all features
        """
        try:
            if len(price_data) < self.lookback_period:
                raise TradingError(
                    f"Insufficient data: need {self.lookback_period}, got {len(price_data)}"
                )

            # Clean data
            if isinstance(price_data, pd.DataFrame):
                prices = (
                    price_data["close"].values
                    if "close" in price_data.columns
                    else price_data.iloc[:, 0].values
                )
            else:
                prices = np.array(price_data)

            cleaned_prices, cleaned_volumes = self.data_handler.validate_and_clean_data(
                prices
            )

            # Convert back to DataFrame format
            clean_df = pd.DataFrame({"close": cleaned_prices})

            # Calculate returns and states
            returns = self._calculate_returns(clean_df)
            states = self._discretize_returns(returns)

            # Build transition matrix
            self.transition_matrix = self._build_transition_matrix(states)

            # Calculate state probabilities
            self.state_probabilities = self._calculate_state_probabilities(states)

            # Generate signals
            signals = self._generate_enhanced_signals(clean_df, returns, states)

            # Calculate confidence metrics
            confidence = self._calculate_confidence()

            # Multi-timeframe analysis
            multi_timeframe = self._multi_timeframe_analysis(cleaned_prices)

            # Market regime detection
            regime_info = self._perform_regime_detection(price_data)

            result = {
                "signal": signals["signal"],
                "confidence": confidence,
                "state_probabilities": self.state_probabilities.tolist(),
                "transition_matrix": self.transition_matrix.tolist(),
                "current_state": (
                    int(states.iloc[-1]) if not pd.isna(states.iloc[-1]) else 0
                ),
                "expected_return": signals["expected_return"],
                "volatility": signals["volatility"],
                "risk_score": signals["risk_score"],
                "multi_timeframe": multi_timeframe,
                "position_sizing": signals.get("position_sizing", {}),
                "market_regime": regime_info,
                "analysis_metadata": {
                    "lookback_period": self.lookback_period,
                    "n_states": self.n_states,
                    "data_points": len(price_data),
                    "analysis_timestamp": pd.Timestamp.now().isoformat(),
                },
            }

            logger.info(
                "Markov analysis completed",
                signal=signals["signal"],
                confidence=confidence,
                current_state=(
                    int(states.iloc[-1]) if not pd.isna(states.iloc[-1]) else 0
                ),
            )

            return result

        except Exception as e:
            logger.error("Error in Markov analysis", error=str(e))
            raise TradingError(f"Markov analysis failed: {str(e)}")

    def _generate_enhanced_signals(
        self, price_data: pd.DataFrame, returns: pd.Series, states: pd.Series
    ) -> Dict:
        """Generate trading signals with position sizing"""
        current_state = int(states.iloc[-1]) if not pd.isna(states.iloc[-1]) else 0
        current_price = price_data["close"].iloc[-1]
        current_return = returns.iloc[-1]

        # Get transition probabilities
        if self.transition_matrix is not None:
            transition_probs = self.transition_matrix[current_state, :]

            # Predict next state
            next_state, next_state_prob = self.predict_next_state(current_state)

            # Calculate expected return
            expected_return = self._calculate_expected_return_from_states(
                transition_probs
            )

            # Calculate volatility and risk
            volatility = returns.rolling(window=min(20, len(returns))).std().iloc[-1]
            risk_score = self._calculate_risk_score(
                current_return, volatility, transition_probs
            )

            # Enhanced signal logic
            confidence = min(1.0, next_state_prob * self.confidence_adjustment_factor)

            if confidence >= self.state_transition_threshold:
                if next_state >= 3:  # Bull states
                    signal = "BUY"
                elif next_state <= 1:  # Bear states
                    signal = "SELL"
                else:
                    signal = "HOLD"
            else:
                signal = "HOLD"

            # Position sizing
            uncertainty = 1 - confidence
            position_size = self.position_sizer.calculate_size(
                uncertainty, current_price
            )
            stop_loss = self.position_sizer.calculate_stop_loss(
                {"consensus": {"action": signal}}, current_price
            )

            position_sizing = {
                "position_size": position_size,
                "stop_loss": stop_loss,
                "uncertainty": uncertainty,
                "max_risk": self.position_sizer.max_risk_per_trade,
            }

        else:
            signal = "HOLD"
            expected_return = 0.0
            volatility = returns.rolling(window=min(20, len(returns))).std().iloc[-1]
            risk_score = 0.5
            position_sizing = {}

        return {
            "signal": signal,
            "expected_return": expected_return,
            "volatility": volatility,
            "risk_score": risk_score,
            "position_sizing": position_sizing,
        }

    def _multi_timeframe_analysis(self, prices: np.ndarray) -> Dict:
        """Perform multi-timeframe Markov analysis"""
        multi_tf_results = {}

        for period in self.lookback_periods:
            if len(prices) >= period:
                period_prices = prices[-period:]
                period_df = pd.DataFrame({"close": period_prices})

                try:
                    # Mini analysis for this timeframe
                    returns = self._calculate_returns(period_df)
                    states = self._discretize_returns(returns)

                    if len(states) > 0:
                        current_state = (
                            int(states.iloc[-1]) if not pd.isna(states.iloc[-1]) else 0
                        )
                        state_prob = (
                            self.state_probabilities[current_state]
                            if self.state_probabilities is not None
                            else 0
                        )

                        multi_tf_results[f"tf_{period}"] = {
                            "state": current_state,
                            "probability": state_prob,
                            "volatility": returns.std() if len(returns) > 0 else 0,
                        }
                except Exception as e:
                    logger.warning(
                        f"Multi-timeframe analysis failed for period {period}",
                        error=str(e),
                    )
                    continue

        return multi_tf_results

    def _calculate_expected_return_from_states(
        self, transition_probs: np.ndarray
    ) -> float:
        """Calculate expected return based on state transition probabilities"""
        if self.state_bounds is None:
            return 0.0

        expected_return = 0.0
        for state, prob in enumerate(transition_probs):
            if prob > 0 and state < len(self.state_bounds) - 1:
                lower_bound = self.state_bounds.iloc[state]
                upper_bound = self.state_bounds.iloc[state + 1]
                state_return = (lower_bound + upper_bound) / 2
                expected_return += prob * state_return

        return expected_return

    def _calculate_risk_score(
        self, current_return: float, volatility: float, transition_probs: np.ndarray
    ) -> float:
        """Calculate risk score based on volatility and state uncertainty"""
        vol_score = min(volatility * 10, 1.0)

        # Entropy of transition probabilities
        entropy = -np.sum(transition_probs * np.log(transition_probs + 1e-10))
        max_entropy = np.log(self.n_states)
        uncertainty_score = entropy / max_entropy if max_entropy > 0 else 0

        risk_score = (vol_score + uncertainty_score) / 2
        return min(risk_score, 1.0)

    def _calculate_confidence(self) -> float:
        """Calculate confidence in the analysis"""
        if self.transition_matrix is None or self.state_probabilities is None:
            return 0.0

        max_state_prob = np.max(self.state_probabilities)

        transition_entropy = -np.sum(
            self.transition_matrix * np.log(self.transition_matrix + 1e-10)
        )
        max_transition_entropy = np.log(self.n_states)
        transition_confidence = 1 - (transition_entropy / max_transition_entropy)

        confidence = (max_state_prob + transition_confidence) / 2
        return min(confidence, 1.0)

    def _perform_regime_detection(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform market regime detection and return regime information.

        Args:
            price_data: Price data for regime analysis

        Returns:
            Dictionary containing regime detection results
        """
        if not self.enable_regime_detection or self.regime_detector is None:
            return {
                "regime_detected": False,
                "regime_type": "unknown",
                "regime_confidence": 0.0,
                "regime_transition_risk": 0.0,
                "regime_detection_enabled": False
            }

        try:
            # Ensure we have sufficient data for regime detection
            if len(price_data) < self.regime_detector.rolling_window:
                logger.warning(
                    "Insufficient data for regime detection",
                    data_points=len(price_data),
                    required=self.regime_detector.rolling_window
                )
                return {
                    "regime_detected": False,
                    "regime_type": "insufficient_data",
                    "regime_confidence": 0.0,
                    "regime_transition_risk": 0.0,
                    "data_points": len(price_data),
                    "required_points": self.regime_detector.rolling_window
                }

            # Check if regime detector is fitted
            if not self.regime_detector.is_fitted:
                logger.info("Fitting regime detector with historical data")
                # Use sufficient historical data for fitting
                fit_data = price_data.tail(max(self.regime_detector.lookback_period, len(price_data)))
                self.regime_detector.fit(fit_data)

            # Get regime prediction
            regime_result = self.regime_detector.predict(price_data)

            # Enhance with MarkovSystem-specific information
            enhanced_result = {
                **regime_result,
                "regime_detection_enabled": True,
                "markov_integration": {
                    "current_markov_state": getattr(self, 'current_state', 0),
                    "markov_confidence": self._calculate_confidence(),
                    "regime_markov_alignment": self._assess_regime_markov_alignment(regime_result)
                }
            }

            logger.debug(
                "Regime detection completed",
                regime_type=enhanced_result.get("current_regime"),
                confidence=enhanced_result.get("regime_confidence"),
                transition_risk=enhanced_result.get("regime_transition_risk")
            )

            return enhanced_result

        except Exception as e:
            logger.error("Regime detection failed", error=str(e))
            return {
                "regime_detected": False,
                "regime_type": "error",
                "regime_confidence": 0.0,
                "regime_transition_risk": 1.0,  # High risk due to error
                "error": str(e),
                "regime_detection_enabled": True
            }

    def _assess_regime_markov_alignment(self, regime_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess alignment between regime detection and Markov state analysis.

        Args:
            regime_result: Result from regime detection

        Returns:
            Dictionary with alignment assessment
        """
        try:
            regime_type = regime_result.get("current_regime", "unknown")
            regime_confidence = regime_result.get("regime_confidence", 0.0)
            markov_confidence = self._calculate_confidence()

            # Simple alignment scoring based on regime type and Markov signals
            alignment_score = 0.5  # Neutral baseline

            # High confidence in both systems suggests good alignment
            if regime_confidence > 0.7 and markov_confidence > 0.7:
                alignment_score = 0.8
            elif regime_confidence > 0.5 and markov_confidence > 0.5:
                alignment_score = 0.6
            elif regime_confidence < 0.3 or markov_confidence < 0.3:
                alignment_score = 0.3

            return {
                "alignment_score": alignment_score,
                "regime_confidence": regime_confidence,
                "markov_confidence": markov_confidence,
                "confidence_differential": abs(regime_confidence - markov_confidence),
                "alignment_quality": (
                    "strong" if alignment_score > 0.7 else
                    "moderate" if alignment_score > 0.5 else
                    "weak"
                )
            }

        except Exception as e:
            logger.warning("Failed to assess regime-Markov alignment", error=str(e))
            return {
                "alignment_score": 0.5,
                "alignment_quality": "unknown",
                "error": str(e)
            }


# ============================================================================
# BACKWARDS COMPATIBILITY
# ============================================================================

# Maintain backwards compatibility with existing imports
MarkovAnalyzer = MarkovCore  # For markov_analyzer.py imports
# Backward compatibility aliases
EnhancedMarkovSystem = MarkovSystem  # For old imports
EnhancedDataHandler = DataHandler  # For old imports
EnhancedMarkovTradingSystem = MarkovSystem  # Legacy alias

# Export main classes
__all__ = [
    "MarkovCore",
    "MarkovSystem",
    "DataHandler",
    "PositionSizer",
    # Backward compatibility aliases
    "EnhancedMarkovSystem",
    "EnhancedDataHandler",
    "MarkovAnalyzer",
    "EnhancedMarkovTradingSystem",
]
