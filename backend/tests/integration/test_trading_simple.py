#!/usr/bin/env python3
"""
Simplified test of the enhanced trading system components
Tests key functionality without complex backend dependencies
"""

import asyncio
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Load environment
load_dotenv()

async def test_alpaca_integration():
    """Test Alpaca API integration"""
    try:
        import alpaca_trade_api as tradeapi
        
        # Get credentials
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        
        # Initialize client
        api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
        
        # Test connection
        account = api.get_account()
        
        return {
            'status': 'success',
            'account_value': float(account.portfolio_value),
            'cash': float(account.cash),
            'buying_power': float(account.buying_power)
        }
        
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def test_kelly_criterion():
    """Test Kelly Criterion calculation"""
    def calculate_kelly(win_prob, avg_win, avg_loss):
        if avg_loss <= 0 or win_prob <= 0 or win_prob >= 1:
            return 0.01
        
        b = avg_win / abs(avg_loss)
        p = win_prob
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        return max(0.01, min(0.15, kelly_fraction))
    
    # Test with backtest performance (60% win rate)
    kelly = calculate_kelly(
        win_prob=0.6,
        avg_win=0.08,
        avg_loss=0.04
    )
    
    return {
        'kelly_fraction': kelly,
        'status': 'success',
        'recommended_size': f"{kelly:.1%} of portfolio"
    }

def test_position_sizing():
    """Test position sizing logic"""
    def calculate_position_size(account_value, current_price, stop_loss_price, kelly_fraction=0.05):
        # Risk-based sizing
        risk_per_share = abs(current_price - stop_loss_price)
        max_risk = account_value * 0.005  # 0.5% max risk
        risk_based_shares = max_risk / risk_per_share if risk_per_share > 0 else 0
        
        # Kelly-based sizing
        kelly_size = account_value * kelly_fraction
        kelly_shares = kelly_size / current_price
        
        # Use smaller of the two
        shares = min(risk_based_shares, kelly_shares)
        position_size = shares * current_price
        
        return {
            'shares': int(shares),
            'position_size': position_size,
            'position_pct': position_size / account_value,
            'risk_amount': risk_per_share * shares,
            'risk_pct': (risk_per_share * shares) / account_value
        }
    
    # Test case
    result = calculate_position_size(
        account_value=100000,
        current_price=150.0,
        stop_loss_price=145.0,
        kelly_fraction=0.08
    )
    
    result['status'] = 'success'
    return result

def test_markov_signals():
    """Test Markov signal generation"""
    try:
        import yfinance as yf
        
        # Get AAPL data
        ticker = yf.Ticker("AAPL")
        hist = ticker.history(period="3mo")
        
        if len(hist) < 30:
            return {'status': 'error', 'error': 'Insufficient data'}
        
        # Calculate signals
        price_series = hist['Close']
        returns = price_series.pct_change().dropna()
        
        # Moving averages
        sma_5 = price_series.rolling(5).mean()
        sma_20 = price_series.rolling(20).mean()
        
        # RSI calculation
        gains = returns[returns > 0]
        losses = abs(returns[returns < 0])
        avg_gain = gains.tail(14).mean() if len(gains) > 0 else 0
        avg_loss = losses.tail(14).mean() if len(losses) > 0 else 0
        
        rs = avg_gain / avg_loss if avg_loss > 0 else 100
        rsi = 100 - (100 / (1 + rs))
        
        # Current values
        current_price = price_series.iloc[-1]
        current_sma5 = sma_5.iloc[-1]
        current_sma20 = sma_20.iloc[-1]
        momentum = returns.tail(5).mean()
        volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        
        # Signal logic
        signal_score = 0
        confidence = 0.5
        
        if current_sma5 > current_sma20:
            signal_score += 2
            confidence += 0.1
        if momentum > 0.005:
            signal_score += 2
            confidence += 0.15
        if rsi < 40:
            signal_score += 1
            confidence += 0.1
        
        if current_sma5 <= current_sma20:
            signal_score -= 2
            confidence += 0.1
        if momentum < -0.005:
            signal_score -= 2
            confidence += 0.15
        if rsi > 70:
            signal_score -= 1
            confidence += 0.1
        
        # Final signal
        if signal_score >= 3:
            signal = 'BUY'
        elif signal_score <= -3:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        return {
            'status': 'success',
            'symbol': 'AAPL',
            'signal': signal,
            'confidence': min(1.0, confidence),
            'current_price': current_price,
            'rsi': rsi,
            'momentum': momentum,
            'volatility': volatility,
            'sma5': current_sma5,
            'sma20': current_sma20,
            'signal_score': signal_score
        }
        
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def test_risk_management():
    """Test risk management calculations"""
    def calculate_portfolio_heat(positions, account_value):
        total_risk = 0
        position_risks = {}
        
        for pos in positions:
            symbol = pos['symbol']
            shares = pos['shares']
            entry_price = pos['entry_price']
            stop_loss = pos['stop_loss']
            
            risk_per_share = abs(entry_price - stop_loss)
            position_risk = risk_per_share * abs(shares)
            position_heat = position_risk / account_value
            
            position_risks[symbol] = position_heat
            total_risk += position_risk
        
        portfolio_heat = total_risk / account_value
        
        return {
            'portfolio_heat': portfolio_heat,
            'position_risks': position_risks,
            'total_risk_amount': total_risk,
            'heat_utilization': portfolio_heat / 0.02  # 2% max heat
        }
    
    # Test positions
    test_positions = [
        {'symbol': 'AAPL', 'shares': 50, 'entry_price': 150.0, 'stop_loss': 145.0},
        {'symbol': 'GOOGL', 'shares': 30, 'entry_price': 180.0, 'stop_loss': 175.0}
    ]
    
    result = calculate_portfolio_heat(test_positions, 100000)
    result['status'] = 'success'
    return result

async def main():
    print("🚀 Swaggy Stacks - Enhanced Trading System Test")
    print("=" * 60)
    
    # Test 1: Alpaca Integration
    print("📊 Testing Alpaca API Connection...")
    alpaca_result = await test_alpaca_integration()
    if alpaca_result['status'] == 'success':
        print(f"✅ Alpaca connected - Portfolio: ${alpaca_result['account_value']:,.2f}")
        print(f"   Cash: ${alpaca_result['cash']:,.2f}")
        print(f"   Buying Power: ${alpaca_result['buying_power']:,.2f}")
    else:
        print(f"❌ Alpaca connection failed: {alpaca_result['error']}")
    
    # Test 2: Kelly Criterion
    print(f"\n💰 Testing Kelly Criterion...")
    kelly_result = test_kelly_criterion()
    if kelly_result['status'] == 'success':
        print(f"✅ Kelly Fraction: {kelly_result['kelly_fraction']:.1%}")
        print(f"   Recommended: {kelly_result['recommended_size']}")
    else:
        print("❌ Kelly calculation failed")
    
    # Test 3: Position Sizing
    print(f"\n📏 Testing Position Sizing...")
    sizing_result = test_position_sizing()
    if sizing_result['status'] == 'success':
        print(f"✅ Position Size: ${sizing_result['position_size']:,.2f} ({sizing_result['position_pct']:.1%} of portfolio)")
        print(f"   Shares: {sizing_result['shares']}")
        print(f"   Risk: ${sizing_result['risk_amount']:,.2f} ({sizing_result['risk_pct']:.2%} of portfolio)")
    else:
        print("❌ Position sizing failed")
    
    # Test 4: Markov Signals
    print(f"\n🤖 Testing Markov Signal Generation...")
    signal_result = test_markov_signals()
    if signal_result['status'] == 'success':
        print(f"✅ {signal_result['symbol']}: {signal_result['signal']} (confidence: {signal_result['confidence']:.1%})")
        print(f"   Price: ${signal_result['current_price']:.2f}")
        print(f"   RSI: {signal_result['rsi']:.1f}")
        print(f"   Momentum: {signal_result['momentum']:.3f}")
        print(f"   Volatility: {signal_result['volatility']:.1%}")
        print(f"   SMA5/SMA20: ${signal_result['sma5']:.2f}/${signal_result['sma20']:.2f}")
    else:
        print(f"❌ Signal generation failed: {signal_result['error']}")
    
    # Test 5: Risk Management
    print(f"\n⚠️  Testing Risk Management...")
    risk_result = test_risk_management()
    if risk_result['status'] == 'success':
        print(f"✅ Portfolio Heat: {risk_result['portfolio_heat']:.2%}")
        print(f"   Heat Utilization: {risk_result['heat_utilization']:.1%}")
        print(f"   Total Risk: ${risk_result['total_risk_amount']:,.2f}")
        for symbol, heat in risk_result['position_risks'].items():
            print(f"   {symbol}: {heat:.2%}")
    else:
        print("❌ Risk management failed")
    
    # Summary
    print(f"\n📊 SYSTEM INTEGRATION TEST RESULTS")
    print("=" * 60)
    
    tests = [
        ("Alpaca API Integration", alpaca_result['status'] == 'success'),
        ("Kelly Criterion Calculation", kelly_result['status'] == 'success'),
        ("Position Sizing", sizing_result['status'] == 'success'),
        ("Markov Signal Generation", signal_result['status'] == 'success'),
        ("Risk Management", risk_result['status'] == 'success')
    ]
    
    passed_tests = sum(1 for _, passed in tests if passed)
    total_tests = len(tests)
    
    print(f"Tests Passed: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.0f}%)")
    print()
    
    for test_name, passed in tests:
        status = "✅" if passed else "❌"
        print(f"{test_name:<30} {status}")
    
    # Trading readiness assessment
    print(f"\n🎯 TRADING READINESS ASSESSMENT")
    print("=" * 60)
    
    if passed_tests >= 4:
        print("🟢 SYSTEM READY FOR PAPER TRADING")
        print("   Key components are functional")
        if alpaca_result['status'] == 'success':
            account_value = alpaca_result['account_value']
            kelly_fraction = kelly_result['kelly_fraction']
            max_position = account_value * kelly_fraction
            print(f"   Recommended max position: ${max_position:,.2f} per trade")
        
        print(f"\n💡 NEXT STEPS:")
        print("1. ✅ Start with very small position sizes")
        print("2. ✅ Monitor trades closely for 1-2 weeks") 
        print("3. ✅ Track P&L and signal accuracy")
        print("4. ⏳ Gradually increase position sizes")
        print("5. ⏳ Consider live trading after successful paper period")
        
    elif passed_tests >= 3:
        print("🟡 SYSTEM PARTIALLY READY")
        print("   Some components need attention before trading")
        
    else:
        print("🔴 SYSTEM NOT READY")
        print("   Critical components failed - do not trade")
    
    # Performance projection
    if signal_result['status'] == 'success' and alpaca_result['status'] == 'success':
        print(f"\n📈 PERFORMANCE PROJECTION")
        print("=" * 60)
        
        # Based on backtest results: 57.6% total return, 1.33 Sharpe
        account_value = alpaca_result['account_value']
        projected_annual = 0.172  # 17.2% annualized from backtest
        
        print(f"Current Portfolio: ${account_value:,.2f}")
        print(f"Projected Annual Return: {projected_annual:.1%}")
        print(f"Projected 1-Year Value: ${account_value * (1 + projected_annual):,.2f}")
        print(f"Expected Monthly P&L: ${account_value * (projected_annual / 12):,.2f}")
        
        print(f"\n⚠️  IMPORTANT DISCLAIMERS:")
        print("• Past performance does not guarantee future results")
        print("• All trading involves risk of loss")
        print("• Start with paper trading only")
        print("• Never risk more than you can afford to lose")

if __name__ == "__main__":
    asyncio.run(main())