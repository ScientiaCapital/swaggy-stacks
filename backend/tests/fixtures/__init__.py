"""
Comprehensive Pytest Fixtures for Options Trading Tests

Reusable fixtures supporting the entire SwaggyStacks testing ecosystem:

- Market data scenarios (bull, bear, sideways, crisis)
- Options chains with realistic pricing and Greeks
- Portfolio states and risk assessments
- Agent communication messages
- WebSocket integration fixtures
- Strategy configurations
- Mock API responses

These fixtures enable comprehensive testing from unit tests
to full integration scenarios with agent coordination.
"""

# Re-export commonly used fixtures for convenience
from .options_test_fixtures import *

__all__ = [
    # Market data fixtures
    'bull_market_data',
    'bear_market_data',
    'sideways_market_data',
    'high_volatility_data',
    'crisis_market_data',
    'earnings_event_data',
    'recent_market_data',

    # Options chain fixtures
    'standard_option_chain',
    'high_vol_option_chain',
    'earnings_option_chain',
    'crisis_option_chain',
    'edge_case_options',

    # Single option fixtures
    'atm_call_option',
    'otm_put_option',
    'itm_call_option',
    'zero_dte_option',

    # Portfolio fixtures
    'empty_portfolio',
    'moderate_portfolio',
    'high_risk_portfolio',

    # Agent communication fixtures
    'agent_coordination_message',
    'risk_manager_response',
    'execution_manager_message',

    # WebSocket fixtures
    'websocket_connection_message',
    'websocket_market_data_message',
    'websocket_trade_execution_message',
    'websocket_alert_message',

    # Strategy configuration fixtures
    'zero_dte_strategy_config',
    'wheel_strategy_config',
    'iron_condor_strategy_config',
    'gamma_scalping_config',

    # Integration test fixtures
    'full_trading_scenario',
    'agent_communication_scenario',

    # Mock client fixtures
    'mock_alpaca_client',
    'mock_websocket_manager',
    'mock_ai_agent'
]