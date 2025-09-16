"""
Trading system tests for Swaggy Stacks
"""

import pytest
from app.trading.trading_manager import TradingManager
from app.trading.risk_manager import RiskManager
from app.ml.markov_system import MarkovSystem


def test_trading_manager_singleton():
    """Test that TradingManager follows singleton pattern"""
    tm1 = TradingManager()
    tm2 = TradingManager()
    assert tm1 is tm2
    assert id(tm1) == id(tm2)


def test_trading_manager_initialization():
    """Test TradingManager initializes with correct defaults"""
    tm = TradingManager()
    assert tm is not None
    assert hasattr(tm, 'config')
    assert hasattr(tm, 'strategies')
    assert hasattr(tm, 'logger')


def test_risk_manager_initialization():
    """Test RiskManager initializes correctly"""
    rm = RiskManager()
    assert rm is not None
    assert hasattr(rm, 'max_position_size')
    assert hasattr(rm, 'max_daily_loss')
    assert hasattr(rm, 'max_portfolio_exposure')


def test_risk_calculations():
    """Test basic risk calculations"""
    rm = RiskManager()
    
    # Test position sizing
    position_size = rm.calculate_position_size(
        account_value=100000,
        risk_per_trade=0.02,
        stop_loss_percent=0.05
    )
    assert position_size > 0
    assert position_size <= 100000 * 0.02 / 0.05


def test_markov_system_initialization():
    """Test Markov System initialization"""
    markov = MarkovSystem()
    assert markov is not None
    assert hasattr(markov, 'states')
    assert hasattr(markov, 'transition_matrix')
    assert hasattr(markov, 'current_state')


def test_markov_state_transitions():
    """Test Markov state transition logic"""
    markov = MarkovSystem()
    
    # Test that states are defined
    assert len(markov.states) > 0
    assert 'BULL' in markov.states
    assert 'BEAR' in markov.states
    assert 'NEUTRAL' in markov.states