"""
Enhanced Markov Chain Analysis for Trading
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.preprocessing import StandardScaler
import structlog
from app.core.exceptions import TradingError

logger = structlog.get_logger()


class MarkovAnalyzer:
    """Enhanced Markov Chain Analysis for trading signals"""
    
    def __init__(self, lookback_period: int = 100, n_states: int = 5):
        self.lookback_period = lookback_period
        self.n_states = n_states
        self.transition_matrix = None
        self.state_probabilities = None
        self.state_bounds = None
        self.scaler = StandardScaler()
        
        logger.info("Markov analyzer initialized", lookback_period=lookback_period, n_states=n_states)
    
    def analyze(self, price_data: pd.DataFrame) -> Dict:
        """
        Perform Markov chain analysis on price data
        
        Args:
            price_data: DataFrame with OHLCV data
            
        Returns:
            Dict: Analysis results with signals and probabilities
        """
        try:
            if len(price_data) < self.lookback_period:
                raise TradingError(f"Insufficient data: need {self.lookback_period}, got {len(price_data)}")
            
            # Prepare data
            returns = self._calculate_returns(price_data)
            states = self._discretize_returns(returns)
            
            # Build transition matrix
            self.transition_matrix = self._build_transition_matrix(states)
            
            # Calculate state probabilities
            self.state_probabilities = self._calculate_state_probabilities(states)
            
            # Generate signals
            signals = self._generate_signals(price_data, returns, states)
            
            # Calculate confidence metrics
            confidence = self._calculate_confidence()
            
            result = {
                "signal": signals["signal"],
                "confidence": confidence,
                "state_probabilities": self.state_probabilities.tolist(),
                "transition_matrix": self.transition_matrix.tolist(),
                "current_state": states.iloc[-1],
                "expected_return": signals["expected_return"],
                "volatility": signals["volatility"],
                "risk_score": signals["risk_score"],
                "analysis_metadata": {
                    "lookback_period": self.lookback_period,
                    "n_states": self.n_states,
                    "data_points": len(price_data),
                    "analysis_timestamp": pd.Timestamp.now().isoformat()
                }
            }
            
            logger.info(
                "Markov analysis completed",
                signal=signals["signal"],
                confidence=confidence,
                current_state=states.iloc[-1]
            )
            
            return result
            
        except Exception as e:
            logger.error("Error in Markov analysis", error=str(e))
            raise TradingError(f"Markov analysis failed: {str(e)}")
    
    def _calculate_returns(self, price_data: pd.DataFrame) -> pd.Series:
        """Calculate returns from price data"""
        close_prices = price_data['close']
        returns = close_prices.pct_change().dropna()
        return returns
    
    def _discretize_returns(self, returns: pd.Series) -> pd.Series:
        """Discretize returns into states"""
        # Use quantile-based discretization
        quantiles = np.linspace(0, 1, self.n_states + 1)
        state_bounds = returns.quantile(quantiles)
        self.state_bounds = state_bounds
        
        # Assign states
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
            where=row_sums[:, np.newaxis] != 0
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
    
    def _generate_signals(self, price_data: pd.DataFrame, returns: pd.Series, states: pd.Series) -> Dict:
        """Generate trading signals based on Markov analysis"""
        current_state = int(states.iloc[-1])
        current_return = returns.iloc[-1]
        
        # Get transition probabilities from current state
        if self.transition_matrix is not None:
            transition_probs = self.transition_matrix[current_state, :]
            
            # Calculate expected next state
            expected_next_state = np.argmax(transition_probs)
            
            # Calculate expected return based on state transitions
            expected_return = self._calculate_expected_return_from_states(transition_probs)
            
            # Generate signal based on expected return and volatility
            volatility = returns.rolling(window=20).std().iloc[-1]
            risk_score = self._calculate_risk_score(current_return, volatility, transition_probs)
            
            # Signal logic
            if expected_return > 0.002 and risk_score < 0.7:  # 0.2% expected return, low risk
                signal = "BUY"
            elif expected_return < -0.002 and risk_score < 0.7:  # -0.2% expected return, low risk
                signal = "SELL"
            else:
                signal = "HOLD"
        else:
            signal = "HOLD"
            expected_return = 0.0
            volatility = returns.rolling(window=20).std().iloc[-1]
            risk_score = 0.5
        
        return {
            "signal": signal,
            "expected_return": expected_return,
            "volatility": volatility,
            "risk_score": risk_score
        }
    
    def _calculate_expected_return_from_states(self, transition_probs: np.ndarray) -> float:
        """Calculate expected return based on state transition probabilities"""
        if self.state_bounds is None:
            return 0.0
        
        expected_return = 0.0
        for state, prob in enumerate(transition_probs):
            if prob > 0:
                # Use midpoint of state bounds as representative return
                lower_bound = self.state_bounds.iloc[state]
                upper_bound = self.state_bounds.iloc[state + 1]
                state_return = (lower_bound + upper_bound) / 2
                expected_return += prob * state_return
        
        return expected_return
    
    def _calculate_risk_score(self, current_return: float, volatility: float, transition_probs: np.ndarray) -> float:
        """Calculate risk score based on volatility and state uncertainty"""
        # Volatility component (0-1 scale)
        vol_score = min(volatility * 10, 1.0)  # Scale volatility
        
        # Uncertainty component (entropy of transition probabilities)
        entropy = -np.sum(transition_probs * np.log(transition_probs + 1e-10))
        max_entropy = np.log(self.n_states)
        uncertainty_score = entropy / max_entropy if max_entropy > 0 else 0
        
        # Combine components
        risk_score = (vol_score + uncertainty_score) / 2
        
        return min(risk_score, 1.0)
    
    def _calculate_confidence(self) -> float:
        """Calculate confidence in the analysis"""
        if self.transition_matrix is None or self.state_probabilities is None:
            return 0.0
        
        # Confidence based on state probability concentration
        max_state_prob = np.max(self.state_probabilities)
        
        # Confidence based on transition matrix stability
        transition_entropy = -np.sum(
            self.transition_matrix * np.log(self.transition_matrix + 1e-10)
        )
        max_transition_entropy = np.log(self.n_states)
        transition_confidence = 1 - (transition_entropy / max_transition_entropy)
        
        # Combine confidence metrics
        confidence = (max_state_prob + transition_confidence) / 2
        
        return min(confidence, 1.0)
    
    def predict_next_state(self, current_state: int, steps: int = 1) -> Dict:
        """Predict future states using transition matrix"""
        if self.transition_matrix is None:
            return {"error": "Transition matrix not available"}
        
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
            
            return {
                "predicted_state": int(predicted_state),
                "confidence": float(confidence),
                "state_probabilities": state_vector.tolist()
            }
            
        except Exception as e:
            logger.error("Error predicting next state", error=str(e))
            return {"error": f"Prediction failed: {str(e)}"}
    
    def get_state_description(self, state: int) -> str:
        """Get human-readable description of a state"""
        if self.state_bounds is None or state >= len(self.state_bounds) - 1:
            return "Unknown state"
        
        lower_bound = self.state_bounds.iloc[state]
        upper_bound = self.state_bounds.iloc[state + 1]
        
        if state == 0:
            return f"Very Low Returns ({lower_bound:.3f} to {upper_bound:.3f})"
        elif state == 1:
            return f"Low Returns ({lower_bound:.3f} to {upper_bound:.3f})"
        elif state == 2:
            return f"Moderate Returns ({lower_bound:.3f} to {upper_bound:.3f})"
        elif state == 3:
            return f"High Returns ({lower_bound:.3f} to {upper_bound:.3f})"
        else:
            return f"Very High Returns ({lower_bound:.3f} to {upper_bound:.3f})"
