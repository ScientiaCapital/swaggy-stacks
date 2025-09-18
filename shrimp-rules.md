# SwaggyStacks AI Development Rules

**Purpose**: AI Agent operational guidelines for SwaggyStacks trading system
**Target**: Coding agents working on options trading, risk management, and MCP integration

## Critical Safety Rules

### Trading System Constraints
- **ALWAYS** maintain paper trading mode - NEVER enable live trading
- **ALWAYS** validate market hours before executing any trading logic
- **ALWAYS** enforce position size limits defined in config
- **ALWAYS** check daily loss limits before opening positions
- **ALWAYS** validate Greeks calculations for options positions

### Risk Management Requirements
- **MUST** use Black-Scholes calculator for all options pricing
- **MUST** calculate position-level Greeks for risk assessment
- **MUST** enforce portfolio exposure limits
- **MUST** validate stop-loss and profit targets
- **NEVER** bypass risk checks in trading logic

## Architecture Patterns

### Strategy Implementation Pattern
```python
# REQUIRED pattern for all trading strategies
class NewStrategyConfig(StrategyConfig):
    # Configuration parameters

class NewStrategyPosition(BasePosition):
    # Position tracking

class NewStrategy(BaseStrategy):
    # Strategy implementation with risk management
```

### Singleton Pattern Usage
- **TradingManager**: Use existing singleton instance
- **RiskManager**: Access through singleton pattern
- **MarketDataService**: Use singleton for data access
- **NEVER** create multiple instances of manager classes

### Factory Pattern Requirements
- **ALWAYS** register new strategies in OptionsStrategyFactory
- **ALWAYS** add strategy types to StrategyType enum
- **ALWAYS** define market regime mappings

## File Coordination Requirements

### Strategy Addition Workflow
1. Create strategy file in `app/strategies/options/`
2. **MUST** update `app/strategies/options/__init__.py` exports
3. **MUST** register in `OptionsStrategyFactory._strategy_registry`
4. **MUST** add to `StrategyType` enum
5. **MUST** update regime mappings if applicable

### Testing File Updates
- Add strategy tests in `tests/strategies/options/`
- Update `tests/strategies/options/test_factory.py`
- **MUST** include backtesting validation

### Monitoring Integration
- Add strategy metrics to `app/monitoring/metrics.py`
- Update Grafana dashboards if needed
- **MUST** include performance tracking

## Integration Standards

### Black-Scholes Calculator Usage
```python
# REQUIRED pattern for options pricing
from .black_scholes import BlackScholesCalculator, GreeksData

calculator = BlackScholesCalculator()
greeks = calculator.calculate_greeks(
    spot_price=spot,
    strike_price=strike,
    time_to_expiry=tte,
    risk_free_rate=rate,
    volatility=vol,
    option_type=option_type
)
```

### Prometheus Metrics Integration
- **ALWAYS** use existing metric names from `app/monitoring/metrics.py`
- **NEVER** create duplicate metrics
- **MUST** include strategy-specific labels

### MCP System Integration
- Use TaskMaster-AI for strategic planning
- Use Shrimp for tactical breakdown
- Use Serena for code navigation
- **ALWAYS** update Memory with architectural decisions

## Testing Requirements

### Mandatory Test Coverage
- **Unit tests**: All strategy logic, Greeks calculations
- **Integration tests**: API endpoints, database operations
- **Backtesting**: Historical performance validation
- **Performance tests**: Real-time processing capabilities

### Test Data Requirements
- **MUST** use mock Alpaca API for testing
- **MUST** include edge case scenarios
- **MUST** test risk limit enforcement
- **NEVER** use real API keys in tests

## Code Standards

### Import Requirements
- **ALWAYS** check existing imports before adding new dependencies
- **MUST** use relative imports within strategy modules
- **NEVER** add new third-party dependencies without verification
- **ALWAYS** use existing utility functions

### Type Hints and Validation
- **MUST** use Pydantic v2 for all configuration classes
- **MUST** include type hints for all function parameters
- **MUST** use Decimal for financial calculations
- **NEVER** use float for monetary values

### Logging Standards
```python
# REQUIRED logging pattern
import structlog
logger = structlog.get_logger(__name__)

# Use structured logging with context
logger.info("Strategy signal generated",
           strategy=self.config.strategy_name,
           symbol=symbol,
           signal_type=signal_type)
```

## Decision Trees

### When Adding New Strategy
1. Is it an options strategy? → Use options pattern
2. Does it require new Greeks? → Extend Black-Scholes
3. Is it directional? → Add to regime mappings
4. Requires new risk controls? → Update RiskManager

### Error Handling Priority
1. **Critical**: Market data failures → Halt trading
2. **High**: Risk limit breaches → Close positions
3. **Medium**: Strategy errors → Log and continue
4. **Low**: Metric failures → Log only

### Testing Priority
1. **Critical**: Risk management functions
2. **High**: Strategy entry/exit logic
3. **Medium**: Performance calculations
4. **Low**: Logging and metrics

## Prohibited Actions

### Never Perform These Actions
- **NEVER** enable live trading mode
- **NEVER** bypass position size limits
- **NEVER** ignore market hours validation
- **NEVER** create new dependencies without checking existing code
- **NEVER** modify core risk management logic without thorough testing
- **NEVER** hardcode API keys or secrets
- **NEVER** commit sensitive configuration data

### Code Quality Prohibitions
- **NEVER** use `print()` statements (use structured logging)
- **NEVER** catch exceptions without logging
- **NEVER** use global variables for state management
- **NEVER** create circular imports
- **NEVER** modify singleton instances directly

### Architecture Violations
- **NEVER** bypass factory pattern for strategy creation
- **NEVER** create multiple instances of manager classes
- **NEVER** modify base classes without understanding impact
- **NEVER** implement trading logic outside strategy classes

## AI Decision-Making Standards

### Ambiguous Situation Handling
1. **Code conflicts**: Always prefer existing patterns
2. **Testing uncertainty**: Include more tests rather than fewer
3. **Performance concerns**: Measure before optimizing
4. **Integration questions**: Check existing implementations first

### Priority Framework
1. **Safety first**: Risk management over performance
2. **Consistency**: Follow existing patterns over innovation
3. **Testing**: Validation before implementation
4. **Documentation**: Update related files when making changes

### When to Seek Guidance
- Modifying core risk management logic
- Adding new third-party dependencies
- Changing database schema
- Implementing new trading venues beyond Alpaca

## Workflow Standards

### Development Sequence
1. Analyze existing code patterns
2. Design following established architecture
3. Implement with comprehensive testing
4. Integrate with existing systems
5. Validate through backtesting
6. Update documentation and exports

### Quality Assurance Checklist
- [ ] Follows established architecture patterns
- [ ] Includes comprehensive test coverage
- [ ] Updates all related files (__init__.py, factory, etc.)
- [ ] Maintains risk management standards
- [ ] Uses structured logging
- [ ] Validates with backtesting framework

This document provides specific guidance for AI agents working on SwaggyStacks to maintain consistency, safety, and quality standards.