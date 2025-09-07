# Code Organization and Design Patterns

## Project Directory Structure

### Backend Application (`backend/app/`)
```
app/
├── core/                          # Core system components
│   ├── logging.py                 # Centralized structured logging
│   ├── common_imports.py          # Shared imports and utilities
│   ├── config.py                  # Configuration management
│   ├── database.py                # Database connection and session management
│   └── exceptions.py              # Custom exception definitions
├── api/                           # API layer
│   └── v1/                        # API version 1
│       ├── endpoints/             # API endpoint implementations  
│       ├── models.py              # Shared Pydantic models
│       ├── dependencies.py        # Shared API dependencies
│       └── api.py                 # API router configuration
├── analysis/                      # Trading analysis components
│   ├── consolidated_markov_system.py  # Consolidated Markov analysis
│   └── technical_indicators.py   # Technical analysis tools
├── trading/                       # Trading system components
│   ├── trading_manager.py         # Singleton trading manager
│   ├── trading_utils.py           # Trading utility functions
│   ├── alpaca_client.py           # Alpaca API integration
│   ├── risk_manager.py            # Risk management system
│   └── order_manager.py           # Order execution management
├── rag/                           # RAG and AI components
│   ├── agents/                    # AI trading agents
│   │   ├── consolidated_strategy_agent.py  # Plugin-based strategy system
│   │   ├── base_agent.py          # Base agent functionality
│   │   └── strategies/            # Individual strategy implementations
│   └── services/                  # AI services
└── models/                        # Database models
    ├── user.py                    # User management models
    ├── trade.py                   # Trading models
    └── market_data.py             # Market data models
```

## Design Patterns Implementation

### Singleton Pattern
**Used for system-wide resource management:**

#### TradingManager Singleton
```python
class TradingManager:
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TradingManager, cls).__new__(cls)
        return cls._instance
```
- **Purpose**: Centralize trading operations, prevent multiple client initializations
- **Benefits**: Consistent state management, resource efficiency
- **Usage**: `get_trading_manager()` factory function for access

### Plugin Pattern
**Implemented in consolidated strategy agent system:**

#### Strategy Plugin Architecture
```python
class StrategyPlugin(ABC):
    @abstractmethod
    async def analyze(self, data: Dict) -> TradingSignal:
        pass

class ConsolidatedStrategyAgent:
    AVAILABLE_STRATEGIES = {
        'markov': MarkovStrategy,
        'wyckoff': WyckoffStrategy,
        'fibonacci': FibonacciStrategy,
        'elliott_wave': ElliottWaveStrategy
    }
```
- **Benefits**: Easy strategy addition, modular testing, performance isolation
- **Consensus**: Weighted average, majority vote, confidence-based selection

### Dependency Injection Pattern
**FastAPI-based dependency system:**

#### Common Dependencies
```python
async def get_trading_manager(user: User = Depends(get_current_user)) -> TradingManager:
    return get_trading_manager()

async def get_strategy_agent(strategies: List[str] = Query(None)) -> ConsolidatedStrategyAgent:
    return ConsolidatedStrategyAgent(strategies=strategies or ['markov'])
```

### Factory Pattern
**For creating different types of objects:**

#### Agent Factory
```python
async def create_trading_agent(agent_type: str, **kwargs) -> BaseTradingAgent:
    # Uses ConsolidatedStrategyAgent with specified strategies
    return ConsolidatedStrategyAgent(strategies=[agent_type], **kwargs)
```

## Code Organization Principles

### Separation of Concerns
- **API Layer**: Request/response handling, validation, documentation
- **Business Logic**: Trading algorithms, risk management, portfolio operations  
- **Data Layer**: Database models, market data processing, persistence
- **Integration Layer**: External API clients, background tasks

### Centralization Strategy (Post Phase 1)
- **Logging**: Single logging configuration for all modules
- **Imports**: Common imports module reducing redundancy
- **Models**: Shared Pydantic models eliminating duplication  
- **Dependencies**: Reusable FastAPI dependencies
- **Configuration**: Centralized settings management

### Modularity
- **Plugin Architecture**: Easy addition of new trading strategies
- **Microservice Boundaries**: Clear service responsibilities
- **Interface Abstractions**: Clean contracts between components
- **Backward Compatibility**: Maintained through wrapper functions and aliases

### Error Handling Strategy
- **Custom Exceptions**: `TradingError`, `RiskManagementError`, `MarketDataError`
- **Structured Logging**: Consistent error logging with context
- **Graceful Degradation**: Fallback strategies for service failures
- **HTTP Status Mapping**: Proper HTTP status codes for API errors