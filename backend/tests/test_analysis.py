"""
Analysis engine tests for Swaggy Stacks
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from app.analysis.markov_system import MarkovSystem
from app.analysis.fibonacci_analysis import FibonacciAnalyzer
from app.analysis.wyckoff_analysis import WyckoffAnalyzer


def test_markov_analysis_with_data():
    """Test Markov analysis with sample data"""
    markov = MarkovSystem()
    
    # Create sample price data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    prices = pd.Series(
        data=np.random.randn(100).cumsum() + 100,
        index=dates
    )
    
    # Analyze the data
    result = markov.analyze(prices)
    
    assert result is not None
    assert 'state' in result
    assert 'confidence' in result
    assert result['confidence'] >= 0 and result['confidence'] <= 1


def test_fibonacci_levels_calculation():
    """Test Fibonacci retracement level calculations"""
    fib = FibonacciAnalyzer()
    
    high = 100
    low = 50
    
    levels = fib.calculate_retracement_levels(high, low)
    
    assert levels is not None
    assert '0.236' in levels
    assert '0.382' in levels
    assert '0.5' in levels
    assert '0.618' in levels
    assert '0.786' in levels
    
    # Check level values are correct
    assert abs(levels['0.5'] - 75) < 0.01  # 50% level should be at 75


def test_wyckoff_phase_detection():
    """Test Wyckoff phase detection"""
    wyckoff = WyckoffAnalyzer()
    
    # Create sample volume and price data
    dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
    prices = pd.Series(
        data=np.random.randn(50).cumsum() + 100,
        index=dates
    )
    volume = pd.Series(
        data=np.random.randint(1000000, 5000000, 50),
        index=dates
    )
    
    phase = wyckoff.detect_phase(prices, volume)
    
    assert phase is not None
    assert phase in ['accumulation', 'markup', 'distribution', 'markdown', 'unknown']


def test_analysis_integration():
    """Test that different analysis methods can work together"""
    markov = MarkovSystem()
    fib = FibonacciAnalyzer()
    wyckoff = WyckoffAnalyzer()
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    prices = pd.Series(
        data=np.random.randn(30).cumsum() + 100,
        index=dates
    )
    volume = pd.Series(
        data=np.random.randint(1000000, 5000000, 30),
        index=dates
    )
    
    # Run all analyses
    markov_result = markov.analyze(prices)
    fib_levels = fib.calculate_retracement_levels(prices.max(), prices.min())
    wyckoff_phase = wyckoff.detect_phase(prices, volume)
    
    # All should return valid results
    assert markov_result is not None
    assert fib_levels is not None
    assert wyckoff_phase is not None