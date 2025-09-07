# Testing Patterns and Task Completion Workflow

## Existing Testing Infrastructure

### Root-Level Integration Tests
The project has comprehensive integration tests at the root level:

#### Core Trading System Tests
- **`test_trading_simple.py`**: Core Alpaca integration and trading functionality
  - Alpaca API connection testing
  - Kelly Criterion position sizing validation
  - Markov signal generation and validation
  - Risk management system testing

- **`test_ai_system.py`**: AI and Ollama integration testing
  - Ollama health checks and model availability
  - Market analysis functionality
  - Memory usage monitoring
  - Comprehensive AI system validation

- **`test_markov_mvp.py`**: Markov analysis system validation
  - Trading signal generation testing
  - Market data processing validation
  - Technical indicator calculations

- **`test_live_trading_system.py`**: Complete system integration
  - End-to-end trading workflow testing
  - Quick signal generation validation
  - System health monitoring

### Testing Patterns and Standards

#### Async Test Patterns
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

#### Test Data and Fixtures
```python
# conftest.py patterns for shared fixtures
@pytest.fixture
async def test_trading_manager():
    """Provide isolated TradingManager for testing"""
    # Setup test manager with mock configurations
    yield manager
    # Cleanup

@pytest.fixture
def sample_market_data():
    """Provide consistent test market data"""
    return {
        'symbol': 'AAPL',
        'price': 150.00,
        'volume': 1000000,
        'timestamp': datetime.now()
    }
```

#### Mock Integration Patterns
```python
# Alpaca API mocking for safe testing
@patch('app.trading.alpaca_client.TradingClient')
async def test_order_execution_without_real_trades(mock_client):
    """Test order execution with mocked Alpaca client"""
    mock_client.submit_order.return_value = MockOrderResponse()
    
    result = await trading_manager.execute_order(test_order)
    assert result.status == 'submitted'
```

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
pytest tests/test_your_module.py -v

# Run integration tests
python test_trading_simple.py      # If trading changes
python test_ai_system.py          # If AI/analysis changes  
python test_markov_mvp.py          # If Markov system changes

# Full test suite (before major commits)
pytest tests/ -v --cov=app
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

#### 4. Documentation Updates (If Required)
- Update docstrings for new/modified functions
- Update type hints and method signatures
- Update memory files if architectural changes made
- Update CLAUDE.md if new commands or patterns introduced

#### 5. Git Workflow
```bash
# Standard commit workflow
git add .
git commit -m "feat: description of changes"

# For significant changes
git push
gh pr create --title "Feature: Description" --body "Detailed description"
```

## Testing Strategy by Component

### Trading System Testing
- **Unit Tests**: Individual strategy algorithms, risk calculations
- **Integration Tests**: Alpaca API integration, database operations
- **End-to-End**: Complete trading workflows from signal to execution

### API Endpoint Testing  
- **Request/Response Validation**: Pydantic model validation
- **Authentication**: User authentication and authorization
- **Error Handling**: Proper HTTP status codes and error responses

### Database Testing
- **Model Validation**: SQLAlchemy model constraints and relationships
- **Migration Testing**: Database schema changes
- **Data Integrity**: Referential integrity and constraints

### Frontend Testing
- **Component Testing**: Individual React component functionality
- **Integration Testing**: API integration and data flow
- **E2E Testing**: Complete user workflows

## Continuous Integration Alignment

### GitHub Actions Integration
The project uses `.github/workflows/ci.yml` with:
- **Backend Tests**: pytest with coverage reporting
- **Frontend Tests**: Jest/React Testing Library
- **Code Quality**: Linting and type checking
- **Security Scanning**: Vulnerability assessment
- **Docker Building**: Container build validation

### Local Development Alignment
Match CI/CD requirements locally:
```bash
# Run the same checks as CI
pytest tests/ -v --cov=app --cov-report=xml
npm test -- --coverage --watchAll=false
black --check app/
isort --check-only app/
flake8 app/
mypy app/
npm run lint && npm run type-check
```

## Performance and Load Testing

### Trading System Performance
- **Latency Testing**: Order execution response times
- **Throughput Testing**: Multiple concurrent trading operations
- **Memory Usage**: Monitor memory consumption during extended operations

### API Performance Testing
- **Endpoint Response Times**: Target < 200ms for standard operations
- **Database Query Performance**: Monitor slow queries
- **Caching Effectiveness**: Redis cache hit rates

### Integration Testing Best Practices
- **Isolated Testing**: Each test should be independent
- **Test Data Management**: Use fixtures and factories for consistent data
- **Mock External Services**: Avoid dependencies on external APIs during testing
- **Cleanup**: Ensure tests clean up after themselves