"""
Tests for ConsolidatedStrategyAgent trading algorithms
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from app.rag.agents.strategy_agent import (
    StrategyAgent,
    MarkovStrategy,
    WyckoffStrategy,
    FibonacciStrategy
)
from app.rag.agents.base_agent import TradingSignal


class TestMarkovStrategy:
    """Test suite for Markov trading strategy"""
    
    @pytest.fixture
    def markov_strategy(self):
        """Create Markov strategy instance"""
        return MarkovStrategy()
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        np.random.seed(42)  # For reproducible tests
        
        # Generate realistic price movements
        returns = np.random.normal(0.001, 0.02, 100)  # 0.1% average daily return, 2% volatility
        prices = [100.0]  # Starting price
        
        for return_rate in returns[1:]:
            new_price = prices[-1] * (1 + return_rate)
            prices.append(new_price)
        
        return pd.DataFrame({
            'close': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'volume': np.random.normal(1000000, 200000, 100)
        }, index=dates)
    
    async def test_markov_strategy_initialization(self, markov_strategy):
        """Test Markov strategy initialization"""
        assert markov_strategy.name == "markov"
        assert hasattr(markov_strategy, 'logger')
    
    async def test_markov_state_detection(self, markov_strategy, sample_price_data):
        """Test Markov state detection logic"""
        market_data = {
            'symbol': 'AAPL',
            'current_price': sample_price_data['close'].iloc[-1],
            'historical_data': sample_price_data
        }
        
        signal = await markov_strategy.analyze_market(market_data)
        
        assert isinstance(signal, dict)
        assert 'action' in signal
        assert 'confidence' in signal
        assert 'reasoning' in signal
        assert signal['action'] in ['buy', 'sell', 'hold']
        assert 0 <= signal['confidence'] <= 1
    
    async def test_markov_regime_detection(self, markov_strategy, sample_price_data):
        """Test regime detection in Markov model"""
        # Create trending data
        trending_data = sample_price_data.copy()
        trending_data['close'] = np.linspace(100, 150, len(trending_data))  # Strong uptrend
        
        market_data = {
            'symbol': 'AAPL',
            'current_price': trending_data['close'].iloc[-1],
            'historical_data': trending_data
        }
        
        signal = await markov_strategy.analyze_market(market_data)
        
        # Should detect trending regime and potentially give buy signal
        assert signal['confidence'] > 0.3  # Should have reasonable confidence
        if signal['action'] == 'buy':
            assert 'trend' in signal['reasoning'].lower() or 'momentum' in signal['reasoning'].lower()
    
    async def test_markov_volatility_regime(self, markov_strategy):
        """Test Markov model handles high volatility periods"""
        # Create high volatility data
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        high_vol_returns = np.random.normal(0, 0.05, 50)  # High volatility
        prices = [100.0]
        
        for return_rate in high_vol_returns[1:]:
            prices.append(prices[-1] * (1 + return_rate))
        
        volatile_data = pd.DataFrame({
            'close': prices,
            'high': [p * 1.03 for p in prices],
            'low': [p * 0.97 for p in prices],
            'volume': np.random.normal(2000000, 500000, 50)  # High volume
        }, index=dates)
        
        market_data = {
            'symbol': 'AAPL',
            'current_price': volatile_data['close'].iloc[-1],
            'historical_data': volatile_data
        }
        
        signal = await markov_strategy.analyze_market(market_data)
        
        # In high volatility, should be more cautious
        assert signal is not None
        if signal['action'] != 'hold':
            assert signal['confidence'] < 0.8  # Should be less confident in volatile conditions
    
    async def test_markov_insufficient_data(self, markov_strategy):
        """Test Markov strategy with insufficient data"""
        insufficient_data = pd.DataFrame({
            'close': [100, 101, 99],
            'high': [102, 103, 101],
            'low': [98, 99, 97],
            'volume': [1000000, 1100000, 900000]
        })
        
        market_data = {
            'symbol': 'AAPL',
            'current_price': 99,
            'historical_data': insufficient_data
        }
        
        signal = await markov_strategy.analyze_market(market_data)
        
        # Should handle gracefully and return hold or low confidence signal
        assert signal is not None
        assert signal['action'] in ['buy', 'sell', 'hold']
        assert signal['confidence'] <= 0.5  # Should have low confidence with little data


class TestWyckoffStrategy:
    """Test suite for Wyckoff trading strategy"""
    
    @pytest.fixture
    def wyckoff_strategy(self):
        """Create Wyckoff strategy instance"""
        return WyckoffStrategy()
    
    async def test_wyckoff_accumulation_phase(self, wyckoff_strategy, sample_market_data):
        """Test Wyckoff accumulation phase detection"""
        # Create accumulation pattern - sideways movement with increasing volume
        accumulation_data = sample_market_data['historical_data'].copy()
        
        # Simulate accumulation - price stays in range, volume increases
        price_range = [98, 102]
        accumulation_data['close'] = np.random.uniform(price_range[0], price_range[1], len(accumulation_data))
        accumulation_data['volume'] = np.linspace(500000, 1500000, len(accumulation_data))  # Increasing volume
        
        market_data = {
            'symbol': 'AAPL',
            'current_price': accumulation_data['close'].iloc[-1],
            'historical_data': accumulation_data
        }
        
        signal = await wyckoff_strategy.analyze_market(market_data)
        
        assert signal is not None
        assert signal['action'] in ['buy', 'sell', 'hold']
        
        # In accumulation phase, should lean towards buy or hold
        if 'accumulation' in signal['reasoning'].lower():
            assert signal['action'] in ['buy', 'hold']
    
    async def test_wyckoff_distribution_phase(self, wyckoff_strategy, sample_market_data):
        """Test Wyckoff distribution phase detection"""
        # Create distribution pattern - topping action with high volume
        distribution_data = sample_market_data['historical_data'].copy()
        
        # Simulate distribution - price at highs, high volume, showing weakness
        distribution_data['close'] = np.random.uniform(145, 155, len(distribution_data))
        distribution_data['high'] = distribution_data['close'] * 1.02
        distribution_data['low'] = distribution_data['close'] * 0.97
        distribution_data['volume'] = np.random.normal(2000000, 300000, len(distribution_data))  # High volume
        
        market_data = {
            'symbol': 'AAPL',
            'current_price': distribution_data['close'].iloc[-1],
            'historical_data': distribution_data
        }
        
        signal = await wyckoff_strategy.analyze_market(market_data)
        
        assert signal is not None
        if 'distribution' in signal['reasoning'].lower():
            assert signal['action'] in ['sell', 'hold']
    
    async def test_wyckoff_markup_phase(self, wyckoff_strategy, sample_market_data):
        """Test Wyckoff markup phase detection"""
        # Create markup pattern - strong uptrend with confirming volume
        markup_data = sample_market_data['historical_data'].copy()
        
        # Simulate markup - strong uptrend
        markup_data['close'] = np.linspace(100, 150, len(markup_data))
        markup_data['high'] = markup_data['close'] * 1.02
        markup_data['low'] = markup_data['close'] * 0.98
        markup_data['volume'] = np.random.normal(1200000, 200000, len(markup_data))
        
        market_data = {
            'symbol': 'AAPL',
            'current_price': markup_data['close'].iloc[-1],
            'historical_data': markup_data
        }
        
        signal = await wyckoff_strategy.analyze_market(market_data)
        
        assert signal is not None
        # In markup phase, might give buy signal early or hold signal if late
        if 'markup' in signal['reasoning'].lower() or 'uptrend' in signal['reasoning'].lower():
            assert signal['action'] in ['buy', 'hold']


class TestFibonacciStrategy:
    """Test suite for Fibonacci retracement strategy"""
    
    @pytest.fixture
    def fibonacci_strategy(self):
        """Create Fibonacci strategy instance"""
        return FibonacciStrategy()
    
    async def test_fibonacci_retracement_levels(self, fibonacci_strategy, sample_market_data):
        """Test Fibonacci retracement level calculation"""
        # Create data with clear swing high and low
        fib_data = sample_market_data['historical_data'].copy()
        
        # Create a pattern: low -> high -> retracement
        swing_low = 100
        swing_high = 150
        
        # First part: move from low to high
        uptrend = np.linspace(swing_low, swing_high, 30)
        # Second part: retracement to 38.2% level
        retracement_target = swing_high - (swing_high - swing_low) * 0.382
        retracement = np.linspace(swing_high, retracement_target, 20)
        
        fib_prices = np.concatenate([uptrend, retracement])
        fib_data = fib_data.iloc[:len(fib_prices)].copy()
        fib_data['close'] = fib_prices
        fib_data['high'] = fib_prices * 1.01
        fib_data['low'] = fib_prices * 0.99
        
        market_data = {
            'symbol': 'AAPL',
            'current_price': fib_data['close'].iloc[-1],
            'historical_data': fib_data
        }
        
        signal = await fibonacci_strategy.analyze_market(market_data)
        
        assert signal is not None
        assert signal['action'] in ['buy', 'sell', 'hold']
        
        # At 38.2% retracement, should consider buy opportunity
        if abs(fib_data['close'].iloc[-1] - retracement_target) < 2:  # Near retracement level
            assert signal['confidence'] > 0.3
    
    async def test_fibonacci_extension_levels(self, fibonacci_strategy, sample_market_data):
        """Test Fibonacci extension level calculation"""
        # Create breakout pattern that might reach extension levels
        extension_data = sample_market_data['historical_data'].copy()
        
        # Pattern: low -> high -> higher high (extension)
        swing_low = 100
        swing_high = 140
        extension_target = swing_high + (swing_high - swing_low) * 1.618  # 161.8% extension
        
        prices = np.concatenate([
            np.linspace(swing_low, swing_high, 25),  # Initial move
            np.linspace(swing_high, swing_high - 10, 10),  # Small pullback
            np.linspace(swing_high - 10, extension_target, 15)  # Extension move
        ])
        
        extension_data = extension_data.iloc[:len(prices)].copy()
        extension_data['close'] = prices
        extension_data['high'] = prices * 1.01
        extension_data['low'] = prices * 0.99
        
        market_data = {
            'symbol': 'AAPL',
            'current_price': extension_data['close'].iloc[-1],
            'historical_data': extension_data
        }
        
        signal = await fibonacci_strategy.analyze_market(market_data)
        
        assert signal is not None
        # Near extension levels, might suggest taking profits (sell) or caution (hold)
        if abs(extension_data['close'].iloc[-1] - extension_target) < 5:
            assert signal['action'] in ['sell', 'hold']


class TestStrategyAgent:
    """Test suite for StrategyAgent"""
    
    @pytest.fixture
    async def consolidated_agent(self, mock_market_research_service):
        """Create consolidated strategy agent"""
        agent = StrategyAgent(
            strategies=['markov', 'wyckoff', 'fibonacci'],
            use_market_research=False  # Disable for unit tests
        )
        agent.market_research_service = mock_market_research_service
        return agent
    
    async def test_consolidated_agent_initialization(self, consolidated_agent):
        """Test consolidated agent initialization"""
        assert len(consolidated_agent.strategies) == 3
        assert 'markov' in consolidated_agent.strategies
        assert 'wyckoff' in consolidated_agent.strategies
        assert 'fibonacci' in consolidated_agent.strategies
    
    async def test_multi_strategy_consensus(self, consolidated_agent, sample_market_data, sample_technical_indicators):
        """Test consensus building from multiple strategies"""
        market_data = {
            **sample_market_data,
            'technical_indicators': sample_technical_indicators
        }
        
        signal = await consolidated_agent.analyze_market(market_data)
        
        assert isinstance(signal, TradingSignal)
        assert signal.symbol == 'AAPL'
        assert signal.action in ['buy', 'sell', 'hold']
        assert 0 <= signal.confidence <= 1
        assert len(signal.reasoning) > 0
        assert isinstance(signal.metadata, dict)
    
    async def test_market_research_integration(self, mock_market_research_service):
        """Test market research integration in consolidated agent"""
        agent = StrategyAgent(
            strategies=['markov'],
            use_market_research=True
        )
        agent.market_research_service = mock_market_research_service
        
        market_data = {
            'symbol': 'AAPL',
            'current_price': 150.0,
            'historical_data': pd.DataFrame({
                'close': np.random.normal(150, 5, 50),
                'volume': np.random.normal(1000000, 100000, 50)
            })
        }
        
        signal = await agent.analyze_market(market_data)
        
        assert isinstance(signal, TradingSignal)
        # Market research should be integrated into reasoning
        assert 'market_research' in signal.metadata or 'research' in signal.reasoning.lower()
    
    async def test_consensus_weighted_average(self, consolidated_agent, sample_market_data):
        """Test weighted average consensus method"""
        consolidated_agent.consensus_method = 'weighted_average'
        
        signal = await consolidated_agent.analyze_market(sample_market_data)
        
        assert isinstance(signal, TradingSignal)
        assert signal.confidence > 0  # Should have some confidence from averaging
    
    async def test_error_handling_in_strategies(self, consolidated_agent):
        """Test error handling when individual strategies fail"""
        # Create malformed market data to trigger errors
        malformed_data = {
            'symbol': 'AAPL',
            'current_price': None,  # This should cause issues
            'historical_data': pd.DataFrame()  # Empty dataframe
        }
        
        signal = await consolidated_agent.analyze_market(malformed_data)
        
        # Should handle errors gracefully
        assert isinstance(signal, TradingSignal)
        assert signal.action == 'hold'  # Default safe action
        assert signal.confidence == 0.0  # No confidence due to errors
        assert 'error' in signal.reasoning.lower()
    
    async def test_strategy_plugin_loading(self):
        """Test dynamic loading of strategy plugins"""
        # Test loading subset of strategies
        agent = StrategyAgent(strategies=['markov', 'wyckoff'])
        
        assert len(agent.strategies) == 2
        assert 'markov' in agent.strategies
        assert 'wyckoff' in agent.strategies
        assert 'fibonacci' not in agent.strategies
    
    async def test_signal_aggregation_methods(self, consolidated_agent, sample_market_data):
        """Test different signal aggregation methods"""
        methods = ['weighted_average', 'majority_vote', 'confidence_weighted']
        
        for method in methods:
            if hasattr(consolidated_agent, 'consensus_method'):
                consolidated_agent.consensus_method = method
                
                signal = await consolidated_agent.analyze_market(sample_market_data)
                
                assert isinstance(signal, TradingSignal)
                assert signal.action in ['buy', 'sell', 'hold']
    
    async def test_market_condition_adaptation(self, consolidated_agent):
        """Test agent adaptation to different market conditions"""
        # Test trending market
        trending_data = {
            'symbol': 'AAPL',
            'current_price': 150.0,
            'historical_data': pd.DataFrame({
                'close': np.linspace(100, 150, 50),  # Strong uptrend
                'high': np.linspace(102, 152, 50),
                'low': np.linspace(98, 148, 50),
                'volume': np.random.normal(1000000, 100000, 50)
            })
        }
        
        trending_signal = await consolidated_agent.analyze_market(trending_data)
        
        # Test sideways market
        sideways_data = {
            'symbol': 'AAPL',
            'current_price': 150.0,
            'historical_data': pd.DataFrame({
                'close': np.random.uniform(148, 152, 50),  # Sideways movement
                'high': np.random.uniform(150, 154, 50),
                'low': np.random.uniform(146, 150, 50),
                'volume': np.random.normal(800000, 50000, 50)
            })
        }
        
        sideways_signal = await consolidated_agent.analyze_market(sideways_data)
        
        # Signals should adapt to market conditions
        assert isinstance(trending_signal, TradingSignal)
        assert isinstance(sideways_signal, TradingSignal)
        
        # Trending market might have higher confidence
        # Sideways market might prefer hold
        if sideways_signal.action == 'hold':
            assert sideways_signal.confidence >= 0
    
    async def test_performance_metrics_tracking(self, consolidated_agent, sample_market_data):
        """Test that agent tracks performance metrics"""
        signal = await consolidated_agent.analyze_market(sample_market_data)
        
        # Check if metadata includes performance tracking info
        assert isinstance(signal.metadata, dict)
        
        # Should include strategy information
        if 'strategies_used' in signal.metadata:
            assert len(signal.metadata['strategies_used']) > 0
        
        # Should include timing information
        assert signal.timestamp is not None
    
    async def test_risk_management_integration(self, consolidated_agent, sample_market_data):
        """Test risk management considerations in signals"""
        # Add risk-relevant data
        risky_data = sample_market_data.copy()
        risky_data['volatility'] = 0.8  # High volatility
        
        signal = await consolidated_agent.analyze_market(risky_data)
        
        # In high volatility, should be more conservative
        if signal.action in ['buy', 'sell']:
            # Should either have lower confidence or include risk warning
            risk_mentioned = any(keyword in signal.reasoning.lower() 
                               for keyword in ['risk', 'volatility', 'caution'])
            
            assert signal.confidence < 0.9 or risk_mentioned