"""Test fixtures for market data used across RAG tests."""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List
import numpy as np
import pandas as pd


@pytest.fixture
def sample_market_data() -> Dict[str, Any]:
    """Sample market data for testing."""
    return {
        "symbol": "AAPL",
        "current_price": 150.25,
        "volume": 45000000,
        "timestamp": datetime.now().isoformat(),
        "open": 149.50,
        "high": 151.00,
        "low": 148.75,
        "close": 150.25,
        "change": 0.75,
        "change_percent": 0.50,
        "market_cap": 2400000000000,
        "pe_ratio": 25.5,
        "dividend_yield": 0.66
    }


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Sample OHLCV data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='D')
    np.random.seed(42)  # For reproducible tests
    
    base_price = 150.0
    data = []
    
    for i, date in enumerate(dates):
        # Simulate realistic price movement
        change = np.random.normal(0, 2)  # Mean 0, std dev 2
        base_price += change
        
        open_price = base_price
        high_price = base_price + abs(np.random.normal(0, 1))
        low_price = base_price - abs(np.random.normal(0, 1))
        close_price = base_price + np.random.normal(0, 0.5)
        volume = int(np.random.normal(40000000, 10000000))
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': max(volume, 1000000)  # Ensure positive volume
        })
        
        base_price = close_price
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_trading_signals() -> List[Dict[str, Any]]:
    """Sample trading signals for testing."""
    return [
        {
            "signal_id": "sig_001",
            "symbol": "AAPL",
            "signal_type": "BUY",
            "confidence": 0.85,
            "reasoning": "Strong bullish momentum with volume confirmation",
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "strategy": "momentum",
                "indicators": ["RSI", "MACD", "Volume"]
            }
        },
        {
            "signal_id": "sig_002",
            "symbol": "AAPL",
            "signal_type": "SELL",
            "confidence": 0.72,
            "reasoning": "Overbought conditions with divergence",
            "timestamp": (datetime.now() + timedelta(hours=1)).isoformat(),
            "metadata": {
                "strategy": "mean_reversion",
                "indicators": ["RSI", "Bollinger_Bands"]
            }
        }
    ]


@pytest.fixture
def sample_agent_memory() -> List[Dict[str, Any]]:
    """Sample agent memory data for testing."""
    return [
        {
            "memory_id": "mem_001",
            "agent_id": "strategy_agent_1",
            "memory_type": "decision",
            "content": "Bought AAPL at $150.25 based on bullish momentum signals",
            "metadata": {
                "symbol": "AAPL",
                "action": "BUY",
                "price": 150.25,
                "confidence": 0.85,
                "outcome": "profitable"
            },
            "timestamp": datetime.now().isoformat(),
            "importance": 0.8
        },
        {
            "memory_id": "mem_002",
            "agent_id": "strategy_agent_1",
            "memory_type": "pattern",
            "content": "High volume breakout pattern successful in tech stocks",
            "metadata": {
                "pattern_type": "volume_breakout",
                "sector": "technology",
                "success_rate": 0.73,
                "sample_size": 45
            },
            "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
            "importance": 0.9
        }
    ]


@pytest.fixture
def sample_tool_definitions() -> List[Dict[str, Any]]:
    """Sample tool definitions for testing."""
    return [
        {
            "name": "get_market_data",
            "description": "Retrieve current market data for a symbol",
            "category": "market_data",
            "parameters": {
                "symbol": {"type": "string", "required": True},
                "fields": {"type": "array", "items": {"type": "string"}, "required": False}
            },
            "permissions": ["read_market_data"],
            "async_capable": True
        },
        {
            "name": "calculate_rsi",
            "description": "Calculate RSI indicator for given price data",
            "category": "technical_indicators",
            "parameters": {
                "prices": {"type": "array", "items": {"type": "number"}, "required": True},
                "period": {"type": "integer", "default": 14, "required": False}
            },
            "permissions": ["calculate_indicators"],
            "async_capable": False
        }
    ]


@pytest.fixture
def mock_embedding_vectors() -> Dict[str, np.ndarray]:
    """Mock embedding vectors for testing."""
    np.random.seed(42)  # For reproducible tests
    return {
        "bullish_momentum": np.random.rand(384).astype(np.float32),
        "bearish_reversal": np.random.rand(384).astype(np.float32),
        "volume_breakout": np.random.rand(384).astype(np.float32),
        "support_level": np.random.rand(384).astype(np.float32),
        "resistance_level": np.random.rand(384).astype(np.float32)
    }


@pytest.fixture
def sample_context_data() -> Dict[str, Any]:
    """Sample context data for testing context builder."""
    return {
        "current_market": {
            "symbol": "AAPL",
            "price": 150.25,
            "volume": 45000000,
            "trend": "bullish"
        },
        "memories": [
            {
                "content": "Previous successful trade at $148",
                "relevance": 0.85,
                "timestamp": "2024-01-15T10:30:00"
            }
        ],
        "tool_outputs": [
            {
                "tool": "rsi_calculator",
                "result": {"rsi": 65.5, "signal": "neutral"},
                "confidence": 0.8
            }
        ],
        "user_preferences": {
            "risk_tolerance": "medium",
            "strategy_focus": ["momentum", "mean_reversion"]
        }
    }


@pytest.fixture
def sample_workflow_state() -> Dict[str, Any]:
    """Sample LangGraph workflow state for testing."""
    return {
        "session_id": "test_session_001",
        "symbol": "AAPL",
        "current_step": "analysis",
        "market_context": {
            "price": 150.25,
            "volume": 45000000,
            "trend": "bullish",
            "volatility": "low"
        },
        "strategy_signals": [
            {
                "strategy": "momentum",
                "signal": "BUY",
                "confidence": 0.75
            }
        ],
        "risk_metrics": {
            "portfolio_exposure": 0.15,
            "position_size": 100,
            "max_loss": 500
        },
        "decision_confidence": 0.82,
        "workflow_metadata": {
            "started_at": datetime.now().isoformat(),
            "steps_completed": ["market_analysis", "signal_generation"]
        }
    }


class MockAsyncContextManager:
    """Mock async context manager for testing database connections."""
    
    def __init__(self, return_value=None):
        self.return_value = return_value
        self.queries_executed = []
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def execute(self, query: str, *args):
        self.queries_executed.append((query, args))
        return self.return_value
    
    async def fetch(self, query: str, *args):
        self.queries_executed.append((query, args))
        return [] if self.return_value is None else [self.return_value]
    
    async def fetchrow(self, query: str, *args):
        self.queries_executed.append((query, args))
        return self.return_value


@pytest.fixture
def mock_db_connection():
    """Mock database connection for testing."""
    return MockAsyncContextManager()


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing."""
    class MockRedis:
        def __init__(self):
            self.data = {}
            self.operations = []
        
        async def get(self, key: str):
            self.operations.append(("GET", key))
            return self.data.get(key)
        
        async def set(self, key: str, value: str, ex: int = None):
            self.operations.append(("SET", key, value, ex))
            self.data[key] = value
        
        async def delete(self, key: str):
            self.operations.append(("DEL", key))
            if key in self.data:
                del self.data[key]
        
        async def exists(self, key: str) -> bool:
            self.operations.append(("EXISTS", key))
            return key in self.data
    
    return MockRedis()