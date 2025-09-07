# Testing Patterns and Task Completion Workflow

## Current Testing Infrastructure

### Organized Test Structure
All tests are now properly organized in `backend/tests/`:

#### Integration Tests (`backend/tests/integration/`)
- **`test_trading_simple.py`**: Core Alpaca integration and trading functionality
- **`test_ai_system.py`**: AI and Ollama integration testing
- **`test_markov_mvp.py`**: Markov analysis system validation
- **`test_live_trading_system.py`**: Complete system integration
- **`test_health_endpoints.py`**: Health check endpoint testing
- **`test_github_endpoints.py`**: GitHub integration testing

#### Unit Tests (`backend/tests/unit/`)
- **`test_account_info.py`**: Account information utilities
- **`test_free_data_sources.py`**: Free data source integrations
- **`test_friday_prices.py`**: Price data validation
- **`test_exceptions.py`**: Custom exception handling
- **`test_mcp_orchestrator.py`**: MCP orchestration testing
- **`test_github_automation.py`**: GitHub automation testing

#### Specialized Tests
- **`backend/tests/api/`**: API endpoint testing
- **`backend/tests/mcp/`**: MCP integration testing
- **`backend/tests/trading/`**: Trading system component testing

### Root Level Utilities
- **`backtest_markov.py`**: Backtesting framework for Markov strategies
- **`setup_ollama_models.py`**: Ollama model configuration utility

## Testing Patterns and Standards

### Async Test Patterns
```python
@pytest.mark.asyncio
async def test_trading_manager_initialization():
    """Test TradingManager singleton initialization"""
    manager1 = get_trading_manager()
    manager2 = get_trading_manager()
    assert manager1 is manager2  # Singleton validation

@pytest.mark.asyncio
async def test_strategy_agent_consensus():
    """Test multi-strategy consensus in ConsolidatedStrategyAgent"""
    agent = ConsolidatedStrategyAgent(strategies=['markov', 'fibonacci'])
    await agent.initialize()
    
    signal = await agent.analyze_market(test_market_data)
    assert signal.confidence >= 0.0 and signal.confidence <= 1.0
```

### Configuration Files
- **`backend/tests/conftest.py`**: Shared fixtures and test configuration
- **`backend/tests/pytest.ini`**: Pytest configuration
- **`backend/tests/run_tests.py`**: Custom test runner

## Task Completion Workflow

### When a Development Task is Complete

#### 1. Code Quality Checks (Mandatory)
```bash
cd backend

# Format and lint code
black app/
isort app/
flake8 app/
mypy app/

# Frontend checks (if applicable)
cd ../frontend
npm run lint
npm run type-check
```

#### 2. Run Relevant Tests
```bash
# Run specific tests related to your changes
pytest backend/tests/unit/ -v                    # Unit tests
pytest backend/tests/integration/ -v             # Integration tests
pytest backend/tests/trading/ -v                 # Trading system tests

# Run integration tests at root level
python test_ai_system.py          # If AI/analysis changes  
python backtest_markov.py          # If Markov system changes

# Full test suite (before major commits)
cd backend && pytest tests/ -v --cov=app
```

#### 3. System Health Validation
```bash
# Start services and verify health
docker-compose up -d
curl http://localhost:8000/health
curl http://localhost:3000

# Check logs for errors
docker-compose logs backend | grep ERROR
```

### Current Integration Directories

#### Active Integration Projects
- **`deep-rl/`**: Deep Reinforcement Learning components
  - RL training pipeline and enhanced DQN models
  - Trading dashboard and validation framework
  - Meta orchestrator for multi-agent coordination

- **`mooncake-integration/`**: KV-cache architecture integration
  - Mooncake client and security implementations
  - Enhanced models and seven-model orchestrator
  - Performance monitoring and revenue models

- **`finrl-integration/`**: FinRL framework integration
  - Structured for backtesting, live trading, environments
  - Agent implementations and utilities
  - Meta orchestrator and monitoring capabilities

- **`api-monetization/`**: API billing and monetization system
  - Client SDK and billing service
  - Admin interfaces and webhook handling
  - Deployment configurations

## Performance and Load Testing

### Trading System Performance
- **Latency Testing**: Order execution response times
- **Throughput Testing**: Multiple concurrent trading operations
- **Memory Usage**: Monitor memory consumption during extended operations

### API Performance Testing
- **Endpoint Response Times**: Target < 200ms for standard operations
- **Database Query Performance**: Monitor slow queries
- **Caching Effectiveness**: Redis cache hit rates

## Test Organization Best Practices
- **Isolated Testing**: Each test should be independent
- **Test Data Management**: Use fixtures and factories for consistent data
- **Mock External Services**: Avoid dependencies on external APIs during testing
- **Cleanup**: Ensure tests clean up after themselves
- **Proper Location**: Integration tests test workflows, unit tests test components