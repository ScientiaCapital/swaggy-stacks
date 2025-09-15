"""
Comprehensive Pytest Fixtures for Options Trading Tests

Provides reusable, realistic test fixtures for all options trading components:
- Market data scenarios
- Options chains with proper Greeks
- Portfolio states and positions
- Agent coordination data
- WebSocket message fixtures

These fixtures support the entire testing ecosystem and enable
end-to-end testing of agent communication and trading execution.
"""

import pytest
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, Mock
import json

from tests.mocks.mock_alpaca_options import get_mock_option_chain, get_mock_single_option
from tests.mocks.mock_market_scenarios import get_market_scenario, get_recent_market_data


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Market Data Fixtures
@pytest.fixture
def bull_market_data():
    """Bull market scenario data"""
    return get_market_scenario("bull_market", symbol="AAPL", duration_days=60)


@pytest.fixture
def bear_market_data():
    """Bear market scenario data"""
    return get_market_scenario("bear_market", symbol="SPY", duration_days=45)


@pytest.fixture
def sideways_market_data():
    """Sideways/neutral market scenario data"""
    return get_market_scenario("sideways", symbol="QQQ", duration_days=90)


@pytest.fixture
def high_volatility_data():
    """High volatility market scenario data"""
    return get_market_scenario("high_volatility", symbol="TSLA", duration_days=30)


@pytest.fixture
def crisis_market_data():
    """Market crisis scenario data"""
    return get_market_scenario("crisis", symbol="SPY", duration_days=20)


@pytest.fixture
def earnings_event_data():
    """Earnings event scenario data"""
    return get_market_scenario("earnings", symbol="AAPL", earnings_surprise="positive")


@pytest.fixture
def recent_market_data():
    """Recent 30-day market data for general testing"""
    return get_recent_market_data(symbol="AAPL", days=30)


# Options Chain Fixtures
@pytest.fixture
def standard_option_chain():
    """Standard options chain for AAPL"""
    return get_mock_option_chain(
        scenario="normal",
        symbol="AAPL",
        underlying_price=150.0,
        expiry_days=[7, 14, 30, 60]
    )


@pytest.fixture
def high_vol_option_chain():
    """High volatility options chain"""
    return get_mock_option_chain(
        scenario="high_vol",
        symbol="TSLA",
        underlying_price=200.0,
        base_volatility=0.55
    )


@pytest.fixture
def earnings_option_chain():
    """Options chain around earnings event"""
    return get_mock_option_chain(
        scenario="earnings",
        symbol="AAPL",
        underlying_price=150.0
    )


@pytest.fixture
def crisis_option_chain():
    """Options chain during crisis conditions"""
    return get_mock_option_chain(
        scenario="crisis",
        symbol="SPY",
        underlying_price=420.0
    )


@pytest.fixture
def edge_case_options():
    """Edge case options for robust testing"""
    return get_mock_option_chain(scenario="edge_cases")


# Single Option Fixtures
@pytest.fixture
def atm_call_option():
    """At-the-money call option"""
    return get_mock_single_option(
        symbol="AAPL",
        strike=150.0,
        option_type="call",
        underlying_price=150.0,
        days_to_expiry=30
    )


@pytest.fixture
def otm_put_option():
    """Out-of-the-money put option"""
    return get_mock_single_option(
        symbol="AAPL",
        strike=140.0,
        option_type="put",
        underlying_price=150.0,
        days_to_expiry=30
    )


@pytest.fixture
def itm_call_option():
    """In-the-money call option"""
    return get_mock_single_option(
        symbol="AAPL",
        strike=140.0,
        option_type="call",
        underlying_price=150.0,
        days_to_expiry=30
    )


@pytest.fixture
def zero_dte_option():
    """Zero days to expiration option"""
    return get_mock_single_option(
        symbol="SPY",
        strike=450.0,
        option_type="call",
        underlying_price=450.0,
        days_to_expiry=0
    )


# Portfolio State Fixtures
@pytest.fixture
def empty_portfolio():
    """Empty portfolio state"""
    return {
        'total_delta': 0.0,
        'total_gamma': 0.0,
        'total_theta': 0.0,
        'total_vega': 0.0,
        'total_rho': 0.0,
        'portfolio_value': 100000.0,
        'buying_power': 50000.0,
        'positions': [],
        'cash': 100000.0
    }


@pytest.fixture
def moderate_portfolio():
    """Portfolio with moderate options exposure"""
    return {
        'total_delta': 25.5,
        'total_gamma': 15.2,
        'total_theta': -8.7,
        'total_vega': 120.3,
        'total_rho': 5.1,
        'portfolio_value': 125000.0,
        'buying_power': 75000.0,
        'positions': [
            {
                'symbol': 'AAPL250221C00150000',
                'quantity': 10,
                'avg_cost': 5.25,
                'current_value': 5500.0,
                'delta': 0.65,
                'gamma': 0.025,
                'theta': -0.05,
                'vega': 0.18
            },
            {
                'symbol': 'SPY250221P00440000',
                'quantity': -5,
                'avg_cost': 3.80,
                'current_value': -1800.0,
                'delta': -0.35,
                'gamma': 0.020,
                'theta': 0.03,
                'vega': 0.12
            }
        ],
        'cash': 98000.0
    }


@pytest.fixture
def high_risk_portfolio():
    """Portfolio near risk limits"""
    return {
        'total_delta': 85.2,
        'total_gamma': 45.8,
        'total_theta': -25.3,
        'total_vega': 280.7,
        'total_rho': 18.9,
        'portfolio_value': 150000.0,
        'buying_power': 30000.0,
        'positions': [
            # Multiple positions near limits
        ],
        'cash': 45000.0,
        'risk_utilization': {
            'delta': 0.85,
            'gamma': 0.92,
            'vega': 0.88
        }
    }


# Agent Communication Fixtures
@pytest.fixture
def agent_coordination_message():
    """Sample agent coordination message"""
    return {
        'message_id': 'coord_001',
        'sender': 'strategy_selector',
        'recipient': 'risk_manager',
        'message_type': 'strategy_recommendation',
        'timestamp': datetime.now(timezone.utc),
        'data': {
            'symbol': 'AAPL',
            'strategy': 'wheel',
            'confidence': 0.85,
            'allocation': 0.15,
            'reasoning': [
                'Bullish market conditions',
                'Low volatility environment',
                'Portfolio has delta capacity'
            ]
        }
    }


@pytest.fixture
def risk_manager_response():
    """Risk manager response to strategy recommendation"""
    return {
        'message_id': 'risk_001',
        'sender': 'risk_manager',
        'recipient': 'strategy_selector',
        'message_type': 'risk_assessment',
        'timestamp': datetime.now(timezone.utc),
        'data': {
            'approved': True,
            'max_position_size': 1000,
            'risk_adjustments': {
                'reduce_allocation': False,
                'suggested_allocation': 0.15
            },
            'warnings': [],
            'limits_status': {
                'delta_utilization': 0.35,
                'gamma_utilization': 0.28,
                'vega_utilization': 0.42
            }
        }
    }


@pytest.fixture
def execution_manager_message():
    """Execution manager order status message"""
    return {
        'message_id': 'exec_001',
        'sender': 'execution_manager',
        'recipient': 'strategy_selector',
        'message_type': 'order_status',
        'timestamp': datetime.now(timezone.utc),
        'data': {
            'order_id': 'order_12345',
            'status': 'filled',
            'symbol': 'AAPL250221P00145000',
            'quantity': 10,
            'filled_price': 4.25,
            'commission': 10.50,
            'strategy': 'wheel',
            'leg': 'csp_entry'
        }
    }


# WebSocket Message Fixtures
@pytest.fixture
def websocket_connection_message():
    """WebSocket connection establishment message"""
    return {
        'type': 'connection_established',
        'client_id': 'test_client_001',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'session_id': 'session_12345'
    }


@pytest.fixture
def websocket_market_data_message():
    """WebSocket market data update message"""
    return {
        'type': 'market_data_update',
        'symbol': 'AAPL',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'data': {
            'price': 150.25,
            'change': 1.25,
            'change_percent': 0.84,
            'volume': 2500000,
            'bid': 150.23,
            'ask': 150.27
        }
    }


@pytest.fixture
def websocket_trade_execution_message():
    """WebSocket trade execution notification"""
    return {
        'type': 'trade_executed',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'data': {
            'order_id': 'order_67890',
            'symbol': 'AAPL250221C00150000',
            'side': 'buy',
            'quantity': 5,
            'price': 5.75,
            'strategy': 'gamma_scalping',
            'execution_time': datetime.now(timezone.utc).isoformat()
        }
    }


@pytest.fixture
def websocket_alert_message():
    """WebSocket alert/notification message"""
    return {
        'type': 'alert',
        'severity': 'warning',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'data': {
            'message': 'Portfolio delta utilization approaching 80% limit',
            'current_utilization': 0.78,
            'limit': 0.85,
            'recommended_action': 'Consider reducing delta exposure'
        }
    }


# Strategy-Specific Fixtures
@pytest.fixture
def zero_dte_strategy_config():
    """Zero-DTE strategy configuration"""
    return {
        'strategy_name': 'zero_dte',
        'max_dte': 0,
        'min_delta': 0.15,
        'max_delta': 0.35,
        'profit_target': 0.50,
        'stop_loss': -2.0,
        'min_premium': 0.05,
        'max_positions': 5,
        'symbols': ['SPY', 'QQQ', 'IWM']
    }


@pytest.fixture
def wheel_strategy_config():
    """Wheel strategy configuration"""
    return {
        'strategy_name': 'wheel',
        'csp_dte_range': [30, 45],
        'cc_dte_range': [21, 35],
        'csp_delta_range': [0.15, 0.30],
        'cc_delta_range': [0.15, 0.30],
        'assignment_handling': 'accept',
        'max_positions': 3,
        'symbols': ['AAPL', 'MSFT', 'GOOGL']
    }


@pytest.fixture
def iron_condor_strategy_config():
    """Iron Condor strategy configuration"""
    return {
        'strategy_name': 'iron_condor',
        'dte_range': [30, 45],
        'wing_width': 10.0,
        'credit_target': 1.0,
        'profit_target': 0.50,
        'loss_limit': -2.0,
        'max_positions': 2,
        'symbols': ['SPY', 'QQQ']
    }


@pytest.fixture
def gamma_scalping_config():
    """Gamma Scalping strategy configuration"""
    return {
        'strategy_name': 'gamma_scalping',
        'dte_range': [21, 60],
        'delta_target': 0.0,
        'delta_tolerance': 0.10,
        'rebalance_threshold': 0.05,
        'min_gamma': 0.01,
        'max_positions': 1,
        'symbols': ['AAPL']
    }


# Mock API Response Fixtures
@pytest.fixture
def mock_alpaca_account():
    """Mock Alpaca account response"""
    return {
        'id': 'test_account_123',
        'account_number': '12345678',
        'status': 'ACTIVE',
        'currency': 'USD',
        'buying_power': '50000.00',
        'portfolio_value': '75000.00',
        'equity': '75000.00',
        'last_equity': '74500.00',
        'multiplier': '4',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'trading_blocked': False,
        'transfers_blocked': False,
        'account_blocked': False
    }


@pytest.fixture
def mock_order_response():
    """Mock Alpaca order response"""
    return {
        'id': 'order_12345',
        'client_order_id': 'client_order_001',
        'symbol': 'AAPL250221C00150000',
        'qty': '10',
        'side': 'buy',
        'order_type': 'limit',
        'time_in_force': 'day',
        'limit_price': '5.25',
        'status': 'new',
        'submitted_at': datetime.now(timezone.utc).isoformat(),
        'filled_qty': '0',
        'filled_avg_price': None,
        'asset_class': 'option'
    }


@pytest.fixture
def mock_position_response():
    """Mock Alpaca position response"""
    return {
        'symbol': 'AAPL250221C00150000',
        'qty': '10',
        'avg_entry_price': '5.15',
        'market_value': '5250.00',
        'cost_basis': '5150.00',
        'unrealized_pl': '100.00',
        'unrealized_plpc': '0.0194',
        'current_price': '5.25',
        'lastday_price': '5.10',
        'change_today': '0.15',
        'side': 'long',
        'asset_class': 'option'
    }


# Integration Test Fixtures
@pytest.fixture
def full_trading_scenario():
    """Complete trading scenario for integration testing"""
    return {
        'market_data': get_recent_market_data(symbol="AAPL", days=30),
        'option_chain': get_mock_option_chain(
            scenario="normal",
            symbol="AAPL",
            underlying_price=150.0
        ),
        'portfolio_state': {
            'total_delta': 15.0,
            'total_gamma': 8.5,
            'total_theta': -5.2,
            'total_vega': 85.0,
            'buying_power': 75000.0
        },
        'strategy_config': {
            'strategy': 'wheel',
            'risk_tolerance': 'moderate',
            'max_allocation': 0.20
        },
        'expected_trades': [
            {
                'action': 'sell_csp',
                'symbol': 'AAPL',
                'strike': 145.0,
                'expiry_days': 30,
                'quantity': 5
            }
        ]
    }


@pytest.fixture
def agent_communication_scenario():
    """Complete agent communication scenario"""
    return {
        'initial_market_analysis': {
            'symbol': 'AAPL',
            'trend': 'bullish',
            'volatility': 'low',
            'confidence': 0.85
        },
        'strategy_recommendation': {
            'strategy': 'wheel',
            'allocation': 0.15,
            'reasoning': ['Bullish trend', 'Low volatility', 'Portfolio capacity']
        },
        'risk_assessment': {
            'approved': True,
            'max_size': 1000,
            'warnings': []
        },
        'execution_plan': {
            'orders': [
                {
                    'symbol': 'AAPL250221P00145000',
                    'action': 'sell',
                    'quantity': 5,
                    'order_type': 'limit',
                    'limit_price': 3.25
                }
            ]
        }
    }


# Performance Test Fixtures
@pytest.fixture
def large_option_chain():
    """Large options chain for performance testing"""
    return get_mock_option_chain(
        scenario="normal",
        symbol="SPY",
        underlying_price=450.0,
        expiry_days=[1, 2, 3, 7, 14, 21, 30, 45, 60, 90, 120],
        strike_range=0.3  # 30% range for many strikes
    )


@pytest.fixture
def high_frequency_data():
    """High frequency market data for performance testing"""
    return get_recent_market_data(symbol="SPY", days=252)  # Full year


# Mock Client Fixtures
@pytest.fixture
def mock_alpaca_client():
    """Mock Alpaca API client"""
    client = AsyncMock()
    client.get_account = AsyncMock()
    client.get_positions = AsyncMock()
    client.place_order = AsyncMock()
    client.cancel_order = AsyncMock()
    client.get_order = AsyncMock()
    client.get_option_chain = AsyncMock()
    return client


@pytest.fixture
def mock_websocket_manager():
    """Mock WebSocket manager"""
    manager = AsyncMock()
    manager.connect = AsyncMock()
    manager.disconnect = AsyncMock()
    manager.send_message = AsyncMock()
    manager.broadcast = AsyncMock()
    return manager


@pytest.fixture
def mock_ai_agent():
    """Mock AI agent for testing"""
    agent = AsyncMock()
    agent.analyze_market = AsyncMock()
    agent.recommend_strategy = AsyncMock()
    agent.assess_risk = AsyncMock()
    agent.generate_reasoning = AsyncMock()
    return agent


# Utility Fixtures
@pytest.fixture
def current_timestamp():
    """Current timestamp for testing"""
    return datetime.now(timezone.utc)


@pytest.fixture
def test_symbols():
    """List of symbols for testing"""
    return ["AAPL", "SPY", "QQQ", "TSLA", "MSFT", "GOOGL", "IWM"]


@pytest.fixture
def options_symbols():
    """List of options symbols for testing"""
    return [
        "AAPL250221C00150000",
        "AAPL250221P00145000",
        "SPY250221C00450000",
        "SPY250221P00440000",
        "QQQ250221C00380000"
    ]


# Clean up fixture
@pytest.fixture(autouse=True)
def cleanup_test_data():
    """Automatically clean up test data after each test"""
    yield
    # Any cleanup code would go here
    pass