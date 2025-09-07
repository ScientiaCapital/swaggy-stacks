"""
Swaggy Stacks Backend Test Suite

Comprehensive test infrastructure for the algorithmic trading system.
Tests cover:
- Unit tests for core modules and components
- Integration tests for API endpoints and workflows
- Trading algorithm validation and performance tests
- MCP server integration and communication tests

Test Structure:
- tests/unit/ - Unit tests for individual components
- tests/integration/ - API endpoint and service integration tests
- tests/trading/ - Trading algorithm and strategy tests
- tests/mcp/ - MCP server integration tests

Usage:
    pytest tests/                           # Run all tests
    pytest tests/unit/                      # Run unit tests only
    pytest tests/integration/               # Run integration tests only
    pytest tests/trading/                   # Run trading algorithm tests
    pytest tests/mcp/                       # Run MCP integration tests
    
    pytest --cov=app                        # Run tests with coverage
    pytest -v --tb=short                    # Verbose output with short traceback
    pytest -k "test_github"                 # Run tests matching pattern
    pytest --maxfail=5                      # Stop after 5 failures
"""