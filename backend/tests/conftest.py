"""
Clean test configuration - minimal imports for reliable testing
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from unittest.mock import Mock, AsyncMock
from datetime import datetime, timedelta


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')

    # Generate realistic price data
    base_price = 100
    returns = np.random.normal(0.001, 0.02, 100)
    prices = [base_price]

    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))

    return pd.DataFrame({
        'open': prices,
        'high': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
        'close': prices,
        'volume': np.random.uniform(100000, 1000000, 100)
    }, index=dates)


@pytest.fixture
def mock_database():
    """Mock database session"""
    return Mock()


@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    mock_redis = Mock()
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.ping.return_value = True
    return mock_redis


@pytest.fixture
def sample_market_data():
    """Sample market data for testing"""
    return {
        'symbol': 'AAPL',
        'price': 150.25,
        'volume': 2500000,
        'change': 2.15,
        'change_percent': 1.45,
        'high': 152.00,
        'low': 148.50,
        'open': 149.00
    }


@pytest.fixture
def sample_portfolio():
    """Sample portfolio data"""
    return {
        'total_value': 125000.0,
        'cash': 25000.0,
        'positions': [
            {
                'symbol': 'AAPL',
                'quantity': 100,
                'entry_price': 145.0,
                'current_price': 150.0,
                'pnl': 500.0
            },
            {
                'symbol': 'GOOGL',
                'quantity': 25,
                'entry_price': 2400.0,
                'current_price': 2500.0,
                'pnl': 2500.0
            }
        ]
    }