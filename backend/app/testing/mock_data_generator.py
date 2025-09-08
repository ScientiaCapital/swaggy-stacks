"""
Mock Data Generator - Generate realistic market data for agent testing
Supports various market scenarios and patterns for comprehensive agent validation
"""

import asyncio
import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Generator, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import numpy as np

import structlog

logger = structlog.get_logger(__name__)


class MarketRegime(Enum):
    """Market regime types for simulation"""
    TRENDING_BULLISH = "trending_bullish"
    TRENDING_BEARISH = "trending_bearish"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT_BULLISH = "breakout_bullish"
    BREAKOUT_BEARISH = "breakout_bearish"
    MEAN_REVERSION = "mean_reversion"


@dataclass
class MockMarketData:
    """Mock market data point"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    bid: float
    ask: float
    
    # Technical indicators
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None
    sma_20: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    atr: Optional[float] = None
    
    # Market metadata
    regime: Optional[str] = None
    volatility: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API compatibility"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class MockTechnicalIndicators:
    """Mock technical indicators"""
    symbol: str
    timestamp: datetime
    rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    sma_20: float
    sma_50: float
    ema_12: float
    ema_26: float
    atr: float
    adx: float
    stochastic_k: float
    stochastic_d: float
    williams_r: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class MockMarkovAnalysis:
    """Mock Markov chain analysis"""
    symbol: str
    timestamp: datetime
    current_state: str
    state_probabilities: Dict[str, float]
    transition_matrix: Dict[str, Dict[str, float]]
    regime_confidence: float
    volatility_state: str
    momentum_state: str
    predicted_next_state: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


class MockDataGenerator:
    """Generate realistic mock market data for testing"""
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Price simulation parameters
        self.base_prices = {
            "AAPL": 150.0,
            "TSLA": 200.0,
            "MSFT": 300.0,
            "GOOGL": 120.0,
            "AMZN": 100.0,
            "SPY": 400.0,
            "QQQ": 350.0
        }
        
        # Market regime parameters
        self.regime_params = {
            MarketRegime.TRENDING_BULLISH: {
                "drift": 0.0008,  # Daily return drift
                "volatility": 0.02,
                "volume_multiplier": 1.2
            },
            MarketRegime.TRENDING_BEARISH: {
                "drift": -0.0006,
                "volatility": 0.025,
                "volume_multiplier": 1.5
            },
            MarketRegime.SIDEWAYS: {
                "drift": 0.0001,
                "volatility": 0.015,
                "volume_multiplier": 0.8
            },
            MarketRegime.HIGH_VOLATILITY: {
                "drift": 0.0,
                "volatility": 0.04,
                "volume_multiplier": 2.0
            },
            MarketRegime.LOW_VOLATILITY: {
                "drift": 0.0002,
                "volatility": 0.008,
                "volume_multiplier": 0.6
            },
            MarketRegime.BREAKOUT_BULLISH: {
                "drift": 0.0015,
                "volatility": 0.03,
                "volume_multiplier": 2.5
            },
            MarketRegime.BREAKOUT_BEARISH: {
                "drift": -0.0012,
                "volatility": 0.035,
                "volume_multiplier": 2.8
            },
            MarketRegime.MEAN_REVERSION: {
                "drift": 0.0,
                "volatility": 0.02,
                "volume_multiplier": 1.0
            }
        }
        
        # Technical indicator calculation state
        self.price_history: Dict[str, List[float]] = {}
        self.volume_history: Dict[str, List[int]] = {}
        
    def generate_market_scenario(
        self,
        symbol: str,
        regime: MarketRegime,
        duration_minutes: int = 1440,  # 1 day
        interval_minutes: int = 1
    ) -> List[MockMarketData]:
        """Generate market data for a specific scenario"""
        
        logger.info("Generating mock market scenario",
                   symbol=symbol,
                   regime=regime.value,
                   duration_minutes=duration_minutes)
        
        data_points = []
        current_time = datetime.now()
        
        # Get regime parameters
        params = self.regime_params[regime]
        base_price = self.base_prices.get(symbol, 100.0)
        current_price = base_price
        
        # Initialize price history if needed
        if symbol not in self.price_history:
            self.price_history[symbol] = [base_price] * 50  # 50-period history
            self.volume_history[symbol] = [1000000] * 50
        
        # Generate data points
        for i in range(0, duration_minutes, interval_minutes):
            timestamp = current_time + timedelta(minutes=i)
            
            # Generate price movement
            if regime == MarketRegime.MEAN_REVERSION:
                # Mean reversion logic
                deviation_from_base = (current_price - base_price) / base_price
                mean_revert_factor = -deviation_from_base * 0.1
                price_change = (params["drift"] + mean_revert_factor) * current_price + \
                              random.gauss(0, params["volatility"] * current_price)
            else:
                # Standard price evolution
                price_change = params["drift"] * current_price + \
                              random.gauss(0, params["volatility"] * current_price)
            
            current_price = max(current_price + price_change, 0.01)  # Prevent negative prices
            
            # Generate OHLC data
            volatility_factor = params["volatility"] * current_price
            
            # Open price (close of previous period)
            open_price = current_price
            
            # Generate high/low with realistic spread
            high_low_range = abs(random.gauss(0, volatility_factor * 0.5))
            high_price = open_price + random.uniform(0, high_low_range)
            low_price = open_price - random.uniform(0, high_low_range)
            
            # Close price is our current_price
            close_price = current_price
            
            # Ensure OHLC consistency
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Generate volume
            base_volume = 1000000
            volume = int(base_volume * params["volume_multiplier"] * 
                        (1 + random.gauss(0, 0.3)))
            volume = max(volume, 1000)
            
            # Generate bid/ask spread
            spread = current_price * random.uniform(0.0001, 0.001)  # 0.01-0.1% spread
            bid = current_price - spread / 2
            ask = current_price + spread / 2
            
            # Update price history
            self.price_history[symbol].append(close_price)
            self.volume_history[symbol].append(volume)
            
            # Keep history manageable
            if len(self.price_history[symbol]) > 200:
                self.price_history[symbol] = self.price_history[symbol][-200:]
                self.volume_history[symbol] = self.volume_history[symbol][-200:]
            
            # Generate technical indicators
            tech_indicators = self._calculate_technical_indicators(symbol)
            
            # Create data point
            data_point = MockMarketData(
                symbol=symbol,
                timestamp=timestamp,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
                bid=bid,
                ask=ask,
                rsi=tech_indicators.get("rsi"),
                macd=tech_indicators.get("macd"),
                macd_signal=tech_indicators.get("macd_signal"),
                bb_upper=tech_indicators.get("bb_upper"),
                bb_lower=tech_indicators.get("bb_lower"),
                sma_20=tech_indicators.get("sma_20"),
                ema_12=tech_indicators.get("ema_12"),
                ema_26=tech_indicators.get("ema_26"),
                atr=tech_indicators.get("atr"),
                regime=regime.value,
                volatility=params["volatility"]
            )
            
            data_points.append(data_point)
        
        logger.info("Mock market scenario generated",
                   symbol=symbol,
                   regime=regime.value,
                   data_points=len(data_points))
        
        return data_points
    
    def _calculate_technical_indicators(self, symbol: str) -> Dict[str, float]:
        """Calculate technical indicators from price history"""
        if symbol not in self.price_history or len(self.price_history[symbol]) < 50:
            return {}
        
        prices = np.array(self.price_history[symbol][-50:])  # Use last 50 periods
        volumes = np.array(self.volume_history[symbol][-50:])
        
        indicators = {}
        
        try:
            # RSI calculation (simplified)
            if len(prices) >= 14:
                price_changes = np.diff(prices)
                gains = np.where(price_changes > 0, price_changes, 0)
                losses = np.where(price_changes < 0, -price_changes, 0)
                
                avg_gain = np.mean(gains[-14:])
                avg_loss = np.mean(losses[-14:])
                
                if avg_loss != 0:
                    rs = avg_gain / avg_loss
                    indicators["rsi"] = 100 - (100 / (1 + rs))
                else:
                    indicators["rsi"] = 100
            
            # Simple Moving Averages
            if len(prices) >= 20:
                indicators["sma_20"] = np.mean(prices[-20:])
            
            # Exponential Moving Averages
            if len(prices) >= 12:
                indicators["ema_12"] = self._calculate_ema(prices, 12)
            if len(prices) >= 26:
                indicators["ema_26"] = self._calculate_ema(prices, 26)
            
            # MACD
            if "ema_12" in indicators and "ema_26" in indicators:
                macd_line = indicators["ema_12"] - indicators["ema_26"]
                indicators["macd"] = macd_line
                
                # Signal line (9-period EMA of MACD)
                indicators["macd_signal"] = macd_line * 0.9 + random.gauss(0, abs(macd_line) * 0.1)
            
            # Bollinger Bands
            if len(prices) >= 20:
                sma_20 = indicators.get("sma_20", np.mean(prices[-20:]))
                std_20 = np.std(prices[-20:])
                indicators["bb_upper"] = sma_20 + (2 * std_20)
                indicators["bb_lower"] = sma_20 - (2 * std_20)
            
            # ATR (Average True Range)
            if len(prices) >= 14:
                high_low = np.array([max(prices[i-5:i+1]) - min(prices[i-5:i+1]) 
                                   for i in range(5, len(prices))])
                indicators["atr"] = np.mean(high_low[-14:])
        
        except Exception as e:
            logger.warning("Error calculating technical indicators", 
                          symbol=symbol, error=str(e))
        
        return indicators
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        multiplier = 2.0 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def generate_technical_indicators(
        self, 
        symbol: str, 
        market_data: MockMarketData
    ) -> MockTechnicalIndicators:
        """Generate comprehensive technical indicators"""
        
        # Use calculated indicators from market data generation
        basic_indicators = self._calculate_technical_indicators(symbol)
        
        # Generate additional indicators
        price = market_data.close
        
        # ADX (simplified)
        adx = random.uniform(20, 80)
        
        # Stochastic oscillator
        stochastic_k = random.uniform(0, 100)
        stochastic_d = stochastic_k * 0.9 + random.gauss(0, 5)
        stochastic_d = max(0, min(100, stochastic_d))
        
        # Williams %R
        williams_r = random.uniform(-100, 0)
        
        return MockTechnicalIndicators(
            symbol=symbol,
            timestamp=market_data.timestamp,
            rsi=basic_indicators.get("rsi", 50.0),
            macd=basic_indicators.get("macd", 0.0),
            macd_signal=basic_indicators.get("macd_signal", 0.0),
            macd_histogram=basic_indicators.get("macd", 0.0) - basic_indicators.get("macd_signal", 0.0),
            bb_upper=basic_indicators.get("bb_upper", price * 1.02),
            bb_middle=basic_indicators.get("sma_20", price),
            bb_lower=basic_indicators.get("bb_lower", price * 0.98),
            sma_20=basic_indicators.get("sma_20", price),
            sma_50=price * random.uniform(0.95, 1.05),
            ema_12=basic_indicators.get("ema_12", price),
            ema_26=basic_indicators.get("ema_26", price),
            atr=basic_indicators.get("atr", price * 0.02),
            adx=adx,
            stochastic_k=stochastic_k,
            stochastic_d=stochastic_d,
            williams_r=williams_r
        )
    
    def generate_markov_analysis(
        self,
        symbol: str,
        current_data: MockMarketData,
        regime: MarketRegime
    ) -> MockMarkovAnalysis:
        """Generate mock Markov chain analysis"""
        
        # Define possible states
        states = ["bullish_trending", "bearish_trending", "sideways", "volatile", "accumulation"]
        
        # Map regime to most likely state
        regime_to_state = {
            MarketRegime.TRENDING_BULLISH: "bullish_trending",
            MarketRegime.TRENDING_BEARISH: "bearish_trending", 
            MarketRegime.SIDEWAYS: "sideways",
            MarketRegime.HIGH_VOLATILITY: "volatile",
            MarketRegime.LOW_VOLATILITY: "accumulation",
            MarketRegime.BREAKOUT_BULLISH: "bullish_trending",
            MarketRegime.BREAKOUT_BEARISH: "bearish_trending",
            MarketRegime.MEAN_REVERSION: "sideways"
        }
        
        current_state = regime_to_state.get(regime, "sideways")
        
        # Generate state probabilities (current state should be highest)
        state_probs = {}
        for state in states:
            if state == current_state:
                state_probs[state] = random.uniform(0.6, 0.8)
            else:
                state_probs[state] = random.uniform(0.05, 0.2)
        
        # Normalize probabilities
        total_prob = sum(state_probs.values())
        state_probs = {k: v / total_prob for k, v in state_probs.items()}
        
        # Generate transition matrix
        transition_matrix = {}
        for from_state in states:
            transition_matrix[from_state] = {}
            for to_state in states:
                if from_state == to_state:
                    # Higher probability to stay in current state
                    prob = random.uniform(0.4, 0.7)
                else:
                    prob = random.uniform(0.05, 0.3)
                transition_matrix[from_state][to_state] = prob
            
            # Normalize row
            row_sum = sum(transition_matrix[from_state].values())
            for to_state in states:
                transition_matrix[from_state][to_state] /= row_sum
        
        # Predict next state (highest transition probability)
        next_state_probs = transition_matrix[current_state]
        predicted_next_state = max(next_state_probs.items(), key=lambda x: x[1])[0]
        
        # Generate confidence score
        regime_confidence = random.uniform(0.6, 0.9)
        
        # Generate volatility and momentum states
        volatility_states = ["low", "normal", "high", "extreme"]
        momentum_states = ["strong_bullish", "weak_bullish", "neutral", "weak_bearish", "strong_bearish"]
        
        volatility_state = random.choice(volatility_states)
        momentum_state = random.choice(momentum_states)
        
        return MockMarkovAnalysis(
            symbol=symbol,
            timestamp=current_data.timestamp,
            current_state=current_state,
            state_probabilities=state_probs,
            transition_matrix=transition_matrix,
            regime_confidence=regime_confidence,
            volatility_state=volatility_state,
            momentum_state=momentum_state,
            predicted_next_state=predicted_next_state
        )
    
    async def stream_market_data(
        self,
        symbol: str,
        regime: MarketRegime,
        callback: callable,
        interval_seconds: int = 60,
        duration_minutes: int = 480  # 8 hours
    ):
        """Stream market data in real-time simulation"""
        
        logger.info("Starting market data stream",
                   symbol=symbol,
                   regime=regime.value,
                   interval_seconds=interval_seconds)
        
        # Generate scenario data
        scenario_data = self.generate_market_scenario(
            symbol=symbol,
            regime=regime,
            duration_minutes=duration_minutes,
            interval_minutes=1
        )
        
        # Stream data at specified intervals
        for i, data_point in enumerate(scenario_data):
            if i % (interval_seconds // 60) == 0:  # Adjust for interval
                # Generate associated technical indicators
                tech_indicators = self.generate_technical_indicators(symbol, data_point)
                markov_analysis = self.generate_markov_analysis(symbol, data_point, regime)
                
                # Prepare streaming payload
                stream_data = {
                    "market_data": data_point.to_dict(),
                    "technical_indicators": tech_indicators.to_dict(),
                    "markov_analysis": markov_analysis.to_dict(),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Call the callback
                try:
                    await callback(stream_data)
                except Exception as e:
                    logger.error("Stream callback failed", error=str(e))
                
                # Wait for next interval
                await asyncio.sleep(interval_seconds)
        
        logger.info("Market data stream completed", symbol=symbol)
    
    def generate_test_scenario_suite(self) -> Dict[str, List[MockMarketData]]:
        """Generate comprehensive test scenarios for agent validation"""
        
        scenarios = {}
        test_symbol = "TEST"
        
        # Generate data for each market regime
        for regime in MarketRegime:
            scenario_data = self.generate_market_scenario(
                symbol=test_symbol,
                regime=regime,
                duration_minutes=240,  # 4 hours
                interval_minutes=5
            )
            scenarios[f"{test_symbol}_{regime.value}"] = scenario_data
        
        logger.info("Generated test scenario suite", 
                   scenarios=len(scenarios),
                   total_data_points=sum(len(data) for data in scenarios.values()))
        
        return scenarios
    
    def save_scenario_data(self, scenarios: Dict[str, List[MockMarketData]], 
                          file_path: str = "mock_scenarios.json"):
        """Save generated scenarios to file"""
        
        serializable_scenarios = {}
        for scenario_name, data_points in scenarios.items():
            serializable_scenarios[scenario_name] = [
                point.to_dict() for point in data_points
            ]
        
        with open(file_path, 'w') as f:
            json.dump(serializable_scenarios, f, indent=2, default=str)
        
        logger.info("Scenarios saved to file", file_path=file_path)
    
    def load_scenario_data(self, file_path: str = "mock_scenarios.json") -> Dict[str, List[MockMarketData]]:
        """Load scenarios from file"""
        
        try:
            with open(file_path, 'r') as f:
                serializable_scenarios = json.load(f)
            
            scenarios = {}
            for scenario_name, data_points in serializable_scenarios.items():
                mock_data_points = []
                for point_dict in data_points:
                    # Convert timestamp back to datetime
                    point_dict['timestamp'] = datetime.fromisoformat(point_dict['timestamp'])
                    mock_data_points.append(MockMarketData(**point_dict))
                scenarios[scenario_name] = mock_data_points
            
            logger.info("Scenarios loaded from file", 
                       file_path=file_path,
                       scenarios=len(scenarios))
            
            return scenarios
            
        except Exception as e:
            logger.error("Failed to load scenarios", file_path=file_path, error=str(e))
            return {}


# Global mock data generator instance
mock_data_generator = MockDataGenerator(seed=42)  # Fixed seed for reproducible tests