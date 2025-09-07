"""
Shared test fixtures and configuration for the Swaggy Stacks trading system
"""

import asyncio
import os
import pytest
import pytest_asyncio
from typing import Dict, Any, Generator, AsyncGenerator
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from fastapi.testclient import TestClient
from fastapi import FastAPI
import redis.asyncio as aioredis

# Import application modules
from app.main import app
from app.core.database import get_db
from app.core.models import BaseModel
from app.core.config import get_settings
from app.mcp.orchestrator import MCPOrchestrator
from app.services.github_automation import GitHubAutomationService
from app.services.market_research import MarketResearchService
from app.trading.trading_manager import TradingManager


# Test configuration
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"
TEST_REDIS_URL = "redis://localhost:6379/15"  # Test database


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def settings():
    """Test settings configuration"""
    with patch.dict(os.environ, {
        "DATABASE_URL": TEST_DATABASE_URL,
        "REDIS_URL": TEST_REDIS_URL,
        "TESTING": "true",
        "LOG_LEVEL": "WARNING",
        "ALPACA_API_KEY": "test_key",
        "ALPACA_SECRET_KEY": "test_secret",
        "ALPACA_BASE_URL": "https://paper-api.alpaca.markets",
        "ML_FEATURES_ENABLED": "false",  # Disable ML features for testing
        "EMBEDDING_SERVICE_TYPE": "mock"  # Use mock embedding service
    }):
        yield get_settings()


@pytest_asyncio.fixture(scope="session")
async def async_db_engine():
    """Create async test database engine"""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={
            "check_same_thread": False,
        },
        poolclass=StaticPool,
        echo=False
    )
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(BaseModel.metadata.create_all)
    
    yield engine
    
    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(BaseModel.metadata.drop_all)
    
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(async_db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session"""
    async with AsyncSession(async_db_engine, expire_on_commit=False) as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def redis_client():
    """Create test Redis client"""
    try:
        client = aioredis.from_url(TEST_REDIS_URL, decode_responses=True)
        await client.ping()  # Test connection
        yield client
        
        # Cleanup - flush test database
        await client.flushdb()
        await client.close()
    except Exception:
        # If Redis not available, use mock
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_client.get.return_value = None
        mock_client.set.return_value = True
        mock_client.delete.return_value = 1
        mock_client.flushdb.return_value = True
        yield mock_client


@pytest.fixture
def client(db_session) -> TestClient:
    """Create test client with database dependency override"""
    def get_test_db():
        return db_session
    
    app.dependency_overrides[get_db] = get_test_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    # Cleanup
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def mock_mcp_orchestrator():
    """Mock MCP orchestrator for testing"""
    orchestrator = AsyncMock(spec=MCPOrchestrator)
    orchestrator._initialized = True
    
    # Mock server availability
    orchestrator.is_server_available.return_value = True
    
    # Mock server calls
    async def mock_call_mcp_method(server, method, *args, **kwargs):
        if server == 'github' and method == 'list_commits':
            return [{
                'sha': 'test_commit_sha',
                'commit': {'message': 'test commit'}
            }]
        elif server == 'tavily' and method == 'search':
            return {
                'results': [{'title': 'Test News', 'content': 'Test content'}]
            }
        return {'status': 'success', 'data': 'mock_data'}
    
    orchestrator.call_mcp_method = mock_call_mcp_method
    
    # Mock health status
    orchestrator.get_server_status.return_value = {
        'github': Mock(connected=True, error_count=0),
        'memory': Mock(connected=True, error_count=0),
        'tavily': Mock(connected=True, error_count=0)
    }
    
    return orchestrator


@pytest_asyncio.fixture
async def mock_github_service(mock_mcp_orchestrator):
    """Mock GitHub automation service"""
    service = AsyncMock(spec=GitHubAutomationService)
    service.owner = "test_owner"
    service.repo = "test_repo"
    service.orchestrator = mock_mcp_orchestrator
    
    # Mock methods
    service.create_automated_pr.return_value = {
        'number': 123,
        'title': 'Test PR',
        'state': 'open'
    }
    
    service.manage_deployment_workflow.return_value = {
        'workflow_name': 'test-workflow',
        'status': 'initiated'
    }
    
    service.health_check.return_value = {
        'service': 'github_automation',
        'status': 'healthy',
        'checks': {'orchestrator': 'ok', 'github_mcp': 'ok'}
    }
    
    return service


@pytest_asyncio.fixture
async def mock_market_research_service(mock_mcp_orchestrator):
    """Mock market research service"""
    from app.services.market_research import MarketSentiment, ComplexAnalysisResult, IntegratedAnalysis
    from app.core.models import SentimentLevel, AnalysisComplexity
    
    service = AsyncMock(spec=MarketResearchService)
    service.orchestrator = mock_mcp_orchestrator
    
    # Mock sentiment analysis
    mock_sentiment = MarketSentiment(
        overall_sentiment=SentimentLevel.BULLISH,
        confidence_score=85.0,
        key_factors=["positive earnings", "market momentum"],
        news_sentiment=0.6,
        social_sentiment=0.7
    )
    
    # Mock complex analysis
    mock_complex = ComplexAnalysisResult(
        analysis_type="technical_analysis",
        key_insights=["breakout pattern", "strong support"],
        confidence_level=0.8,
        risk_factors=["market volatility"],
        thought_process=[
            {"stage": "Analysis", "thought": "Strong technical indicators"}
        ]
    )
    
    # Mock integrated analysis
    mock_integrated = IntegratedAnalysis(
        symbol="AAPL",
        timestamp=datetime.utcnow(),
        market_sentiment=mock_sentiment,
        complex_analysis=mock_complex,
        confidence_score=0.82,
        trading_recommendation={"action": "buy", "confidence": 0.82}
    )
    
    service.analyze_market_sentiment.return_value = mock_sentiment
    service.complex_analysis_workflow.return_value = mock_complex
    service.integrated_strategy_analysis.return_value = mock_integrated
    
    return service


@pytest_asyncio.fixture
async def mock_trading_manager():
    """Mock trading manager"""
    manager = AsyncMock(spec=TradingManager)
    manager._initialized = True
    
    # Mock account info
    manager.get_account_info.return_value = {
        'account_value': 100000.0,
        'cash': 50000.0,
        'buying_power': 200000.0
    }
    
    # Mock position info
    manager.get_positions.return_value = [
        {
            'symbol': 'AAPL',
            'qty': '10',
            'side': 'long',
            'market_value': '1500.00',
            'unrealized_pl': '50.00'
        }
    ]
    
    # Mock order submission
    manager.submit_order.return_value = {
        'id': 'test_order_id',
        'status': 'accepted',
        'symbol': 'AAPL',
        'qty': '10',
        'side': 'buy'
    }
    
    return manager


@pytest.fixture
def sample_market_data() -> Dict[str, Any]:
    """Sample market data for testing"""
    return {
        'symbol': 'AAPL',
        'current_price': 150.0,
        'volume': 50000000,
        'high_52w': 180.0,
        'low_52w': 120.0,
        'volatility': 0.25,
        'timestamp': datetime.utcnow(),
        'historical_data': pd.DataFrame({
            'close': np.random.normal(150, 5, 100),
            'volume': np.random.normal(50000000, 10000000, 100),
            'high': np.random.normal(152, 5, 100),
            'low': np.random.normal(148, 5, 100)
        })
    }


@pytest.fixture
def sample_technical_indicators() -> Dict[str, Any]:
    """Sample technical indicators for testing"""
    return {
        'rsi': 65.0,
        'ma20': 148.0,
        'ma50': 145.0,
        'ma200': 140.0,
        'macd': 2.5,
        'macd_signal': 2.2,
        'macd_histogram': 0.3,
        'atr': 3.2,
        'bollinger_upper': 155.0,
        'bollinger_lower': 145.0,
        'stochastic_k': 70.0,
        'stochastic_d': 68.0,
        'williams_r': -25.0
    }


@pytest.fixture
def sample_trading_signal():
    """Sample trading signal for testing"""
    # Create mock trading signal without importing from ML modules
    from dataclasses import dataclass
    from datetime import datetime
    
    @dataclass
    class MockTradingSignal:
        symbol: str
        action: str
        confidence: float
        reasoning: str
        timestamp: datetime
        metadata: dict
    
    return MockTradingSignal(
        symbol='AAPL',
        action='buy',
        confidence=0.75,
        reasoning='Strong technical indicators and positive sentiment',
        timestamp=datetime.utcnow(),
        metadata={
            'strategy': 'consolidated',
            'signals': ['markov', 'wyckoff'],
            'market_conditions': 'trending'
        }
    )


@pytest_asyncio.fixture
async def consolidated_strategy_agent(mock_market_research_service):
    """Create mock consolidated strategy agent for testing"""
    # Create a mock agent that doesn't require ML dependencies
    mock_agent = AsyncMock()
    mock_agent.strategies = ['markov', 'wyckoff']
    mock_agent.use_market_research = False
    mock_agent.market_research_service = mock_market_research_service
    mock_agent.agent_name = "MockConsolidatedAgent"
    
    # Mock analyze method to return a sample signal
    async def mock_analyze(symbol: str, timeframe: str = "1h"):
        return {
            'signal': 'buy',
            'confidence': 0.75,
            'reasoning': f'Mock analysis for {symbol}',
            'metadata': {
                'strategy': 'consolidated',
                'signals': ['markov', 'wyckoff'],
                'timeframe': timeframe
            }
        }
    
    mock_agent.analyze = mock_analyze
    mock_agent.initialize = AsyncMock()
    mock_agent.health_check = AsyncMock(return_value={'status': 'healthy', 'agent': 'mock'})
    
    return mock_agent


@pytest.fixture
def mock_alpaca_api():
    """Mock Alpaca API client"""
    with patch('alpaca_trade_api.REST') as mock_rest:
        mock_client = Mock()
        
        # Mock account
        mock_account = Mock()
        mock_account.portfolio_value = '100000.0'
        mock_account.cash = '50000.0'
        mock_account.buying_power = '200000.0'
        mock_client.get_account.return_value = mock_account
        
        # Mock positions
        mock_position = Mock()
        mock_position.symbol = 'AAPL'
        mock_position.qty = '10'
        mock_position.side = 'long'
        mock_position.market_value = '1500.00'
        mock_position.unrealized_pl = '50.00'
        mock_client.list_positions.return_value = [mock_position]
        
        # Mock orders
        mock_order = Mock()
        mock_order.id = 'test_order_id'
        mock_order.status = 'accepted'
        mock_order.symbol = 'AAPL'
        mock_order.qty = '10'
        mock_order.side = 'buy'
        mock_client.submit_order.return_value = mock_order
        
        # Mock historical data
        mock_bars = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='D'),
            'open': np.random.normal(150, 5, 100),
            'high': np.random.normal(152, 5, 100),
            'low': np.random.normal(148, 5, 100),
            'close': np.random.normal(150, 5, 100),
            'volume': np.random.normal(50000000, 10000000, 100)
        })
        mock_client.get_bars.return_value = mock_bars
        
        mock_rest.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_yfinance():
    """Mock yfinance for market data"""
    with patch('yfinance.Ticker') as mock_ticker_class:
        mock_ticker = Mock()
        
        # Mock historical data
        mock_history = pd.DataFrame({
            'Open': np.random.normal(150, 5, 100),
            'High': np.random.normal(152, 5, 100),
            'Low': np.random.normal(148, 5, 100),
            'Close': np.random.normal(150, 5, 100),
            'Volume': np.random.normal(50000000, 10000000, 100)
        }, index=pd.date_range('2024-01-01', periods=100, freq='D'))
        
        mock_ticker.history.return_value = mock_history
        
        # Mock info
        mock_ticker.info = {
            'symbol': 'AAPL',
            'longName': 'Apple Inc.',
            'marketCap': 3000000000000,
            'beta': 1.2,
            'trailingPE': 25.0,
            'dividendYield': 0.005
        }
        
        mock_ticker_class.return_value = mock_ticker
        yield mock_ticker


# Pytest configuration
def pytest_configure(config):
    """Pytest configuration"""
    # Set asyncio mode
    config.option.asyncio_mode = "auto"


def pytest_collection_modifyitems(config, items):
    """Mark all async tests appropriately"""
    for item in items:
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)


# Test utilities
class AsyncContextManager:
    """Helper for testing async context managers"""
    def __init__(self, return_value=None):
        self.return_value = return_value
    
    async def __aenter__(self):
        return self.return_value
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


# Performance testing utilities
@pytest.fixture
def performance_monitor():
    """Monitor performance metrics during tests"""
    import psutil
    import time
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
            self.process = psutil.Process()
        
        def start(self):
            self.start_time = time.time()
            self.start_memory = self.process.memory_info().rss / 1024 / 1024
        
        def stop(self):
            if self.start_time is None:
                return {}
            
            end_time = time.time()
            end_memory = self.process.memory_info().rss / 1024 / 1024
            
            return {
                'duration': end_time - self.start_time,
                'memory_start': self.start_memory,
                'memory_end': end_memory,
                'memory_diff': end_memory - self.start_memory
            }
    
    return PerformanceMonitor()


# Error simulation utilities
@pytest.fixture
def error_simulator():
    """Utility for simulating various error conditions"""
    class ErrorSimulator:
        @staticmethod
        def network_error():
            from requests.exceptions import ConnectionError
            raise ConnectionError("Simulated network error")
        
        @staticmethod
        def timeout_error():
            from asyncio import TimeoutError
            raise TimeoutError("Simulated timeout")
        
        @staticmethod
        def auth_error():
            from app.core.exceptions import AuthenticationError
            raise AuthenticationError("Simulated auth error")
        
        @staticmethod
        def mcp_error():
            from app.core.exceptions import MCPError
            raise MCPError("Simulated MCP error")
    
    return ErrorSimulator()