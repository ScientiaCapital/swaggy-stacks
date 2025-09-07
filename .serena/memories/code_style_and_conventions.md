# Code Style and Conventions

## Python Code Style (Backend)

### Formatting Standards
- **Code Formatter**: Black with default settings (88 character line length)
- **Import Sorting**: isort with profile compatible with Black
- **Line Length**: 88 characters (Black default)
- **Indentation**: 4 spaces (no tabs)
- **String Quotes**: Double quotes preferred by Black formatter

### Naming Conventions
- **Variables/Functions**: snake_case (e.g., `trading_manager`, `get_current_user`)
- **Classes**: PascalCase (e.g., `TradingManager`, `ConsolidatedStrategyAgent`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `API_V1_STR`, `MAX_POSITION_SIZE`)
- **Private Members**: Leading underscore (e.g., `_instance`, `_lock`)
- **Module Names**: snake_case (e.g., `trading_manager.py`, `consolidated_markov_system.py`)

### Type Hints (Mandatory)
```python
# Function signatures with type hints
async def get_trading_manager(user: User = Depends(get_current_user)) -> TradingManager:
    return trading_manager_instance

# Variable annotations  
trading_signals: List[TradingSignal] = []
config: Optional[Dict[str, Any]] = None

# Class attributes
class TradingManager:
    _instance: Optional['TradingManager'] = None
    _lock: asyncio.Lock = asyncio.Lock()
```

### Documentation Standards
```python
class TradingManager:
    """
    Singleton trading manager for centralized operations.
    
    Manages trading sessions, order execution, and portfolio state.
    Integrates with Alpaca API for paper trading execution.
    """
    
    async def execute_trade(self, order: OrderRequest) -> OrderResponse:
        """
        Execute a trading order with risk validation.
        
        Args:
            order: Order request with symbol, quantity, side, and type
            
        Returns:
            OrderResponse with execution details and status
            
        Raises:
            TradingError: If order validation fails
            RiskManagementError: If risk limits exceeded
        """
```

### Error Handling Patterns
```python
# Custom exception hierarchy
class TradingSystemException(Exception):
    """Base exception for trading system errors"""
    
class TradingError(TradingSystemException):
    """Trading operation errors"""
    
class RiskManagementError(TradingSystemException):
    """Risk management validation errors"""

# Structured error handling with logging
try:
    result = await trading_operation()
except TradingError as e:
    logger.error("Trading operation failed", error=str(e), context=context)
    raise HTTPException(status_code=400, detail=str(e))
```

### Async/Await Patterns
```python
# Consistent async patterns
async def async_function() -> ReturnType:
    async with session_manager() as session:
        result = await database_operation(session)
        return result

# Context managers for resource management
@asynccontextmanager
async def trading_session(config: Dict[str, Any]):
    session = await create_session(config)
    try:
        yield session
    finally:
        await session.close()
```

## TypeScript Code Style (Frontend)

### Configuration
- **TypeScript**: Strict mode enabled
- **ESLint**: Airbnb configuration with TypeScript extensions
- **Prettier**: Integrated with ESLint for consistent formatting
- **Import Organization**: Automatic sorting and grouping

### Naming Conventions
```typescript
// Interfaces and Types - PascalCase
interface TradingSignal {
  symbol: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
}

// Components - PascalCase
const TradingDashboard: React.FC<TradingDashboardProps> = ({ data }) => {
  return <div>{/* component JSX */}</div>;
};

// Variables and functions - camelCase
const tradingData = useTradingData();
const handleOrderSubmit = async (order: OrderRequest) => {
  // implementation
};

// Constants - UPPER_SNAKE_CASE
const API_ENDPOINTS = {
  TRADING: '/api/v1/trading',
  MARKET_DATA: '/api/v1/market-data'
} as const;
```

### Component Patterns
```typescript
// Functional components with TypeScript
interface ComponentProps {
  data: TradingData;
  onUpdate: (data: TradingData) => void;
}

const Component: React.FC<ComponentProps> = ({ data, onUpdate }) => {
  const [state, setState] = useState<ComponentState>({ loading: false });
  
  useEffect(() => {
    // effect implementation
  }, [dependencies]);
  
  return (
    <div className="component-container">
      {/* JSX implementation */}
    </div>
  );
};
```

## Database and API Conventions

### SQLAlchemy Models
```python
class Trade(Base):
    __tablename__ = "trades"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"))
    symbol: Mapped[str] = mapped_column(String(10), nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
```

### Pydantic Models (API)
```python
class OrderRequest(BaseModel):
    symbol: str = Field(..., description="Trading symbol")
    quantity: float = Field(..., gt=0, description="Order quantity")
    side: OrderSide = Field(..., description="Order side")
    
    @validator('symbol')
    def symbol_must_be_uppercase(cls, v):
        return v.strip().upper()

class Config:
    use_enum_values = True
    validate_assignment = True
```

### API Endpoint Patterns
```python
@router.post("/orders", response_model=OrderResponse)
async def create_order(
    order: OrderRequest,
    trading_manager: TradingManager = Depends(get_trading_manager),
    current_user: User = Depends(get_current_user)
) -> OrderResponse:
    """Create a new trading order with risk validation."""
    try:
        result = await trading_manager.execute_order(order, current_user)
        return result
    except TradingError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

## Logging and Monitoring

### Structured Logging
```python
# Using centralized logger from app.core.logging
from app.core.logging import get_logger, get_trading_logger

logger = get_logger()  # Standard structured logger
trading_logger = get_trading_logger()  # Trading-specific logger

# Structured log entries
logger.info("Operation completed", 
           operation="order_execution", 
           symbol=order.symbol, 
           duration=execution_time)

# Trading-specific events
trading_logger.trade_event("order_submitted", 
                          symbol=order.symbol, 
                          quantity=order.quantity)
```

### Error Logging Standards
```python
# Comprehensive error context
logger.error("Trading operation failed",
            error=str(e),
            error_type=type(e).__name__,
            user_id=user.id,
            operation_context=context,
            stack_trace=traceback.format_exc())
```

## Testing Conventions

### Test File Organization
```
tests/
├── unit/              # Unit tests for individual components
├── integration/       # API integration tests  
├── trading/          # Trading algorithm tests
└── conftest.py       # Shared test fixtures
```

### Test Naming and Structure
```python
# Test function naming
def test_trading_manager_singleton_pattern():
    """Test that TradingManager implements singleton correctly"""
    
def test_markov_system_state_transitions():
    """Test Markov system state transition calculations"""

# Async test patterns
@pytest.mark.asyncio
async def test_order_execution_with_risk_validation():
    """Test order execution includes proper risk validation"""
    # Arrange
    order = OrderRequest(symbol="AAPL", quantity=100, side="BUY")
    
    # Act  
    result = await trading_manager.execute_order(order)
    
    # Assert
    assert result.status == "submitted"
    assert result.symbol == "AAPL"
```