"""
Mock Infrastructure for SwaggyStacks Testing

Comprehensive mock data generators for realistic testing scenarios:

- MockAlpacaOptionsGenerator: Realistic options chains with Black-Scholes pricing
- MockMarketScenariosGenerator: Various market conditions and regimes
- Edge cases and crisis scenarios for robust testing

All mocks are designed to be deterministic for reproducible testing
while maintaining realistic market characteristics.
"""

from .mock_alpaca_options import (
    MockAlpacaOptionsGenerator,
    mock_options_generator,
    get_mock_option_chain,
    get_mock_single_option
)

from .mock_market_scenarios import (
    MockMarketScenariosGenerator,
    mock_market_generator,
    get_market_scenario,
    get_recent_market_data
)

__all__ = [
    'MockAlpacaOptionsGenerator',
    'MockMarketScenariosGenerator',
    'mock_options_generator',
    'mock_market_generator',
    'get_mock_option_chain',
    'get_mock_single_option',
    'get_market_scenario',
    'get_recent_market_data'
]