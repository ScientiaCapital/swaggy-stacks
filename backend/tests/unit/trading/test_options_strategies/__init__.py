"""
Unit tests for options trading strategies

This module contains comprehensive unit tests for all four options strategies:
- Zero-DTE (Zero Days to Expiration) strategy
- Wheel strategy (Cash-Secured Puts + Covered Calls)
- Iron Condor strategy (Four-leg neutral strategy)
- Gamma Scalping strategy (Delta-neutral volatility trading)

Each strategy test suite covers:
- Configuration validation
- Signal analysis and generation
- Order execution and position management
- Risk management and exit conditions
- Edge cases and error handling
- Mock data scenarios and realistic market conditions

Test Coverage Target: 95%+ for all strategy implementations
"""