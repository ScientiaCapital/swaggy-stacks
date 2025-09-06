#!/usr/bin/env python3
"""
Markov Trading System Backtest - Test performance over several years
Bootstrap MVP approach with real performance metrics
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from collections import deque

class MarkovBacktester:
    def __init__(self, initial_capital=100000, max_position_size=0.15, transaction_cost=0.001):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_position_size = max_position_size  # 15% max per position
        self.transaction_cost = transaction_cost  # 0.1% transaction cost
        self.positions = {}  # {symbol: {'shares': int, 'entry_price': float}}
        self.cash = initial_capital
        self.portfolio_history = []
        self.trade_history = []
        self.daily_returns = []
        
    def get_historical_data(self, symbols, years=3):
        """Get historical data for backtesting"""
        print(f"📊 Fetching {years} years of data for {len(symbols)} symbols...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365 + 30)  # Extra buffer
        
        try:
            # Download all data at once
            data = yf.download(symbols, start=start_date, end=end_date, group_by='ticker')
            
            if len(symbols) == 1:
                # Single symbol handling
                data = {symbols[0]: data}
            
            print(f"✅ Downloaded data from {start_date.date()} to {end_date.date()}")
            return data
            
        except Exception as e:
            print(f"❌ Error downloading data: {e}")
            return None
    
    def calculate_markov_signals(self, price_series, lookback=30):
        """Enhanced Markov-like analysis with more data"""
        if len(price_series) < lookback:
            return {'signal': 'HOLD', 'confidence': 0.0, 'strength': 0}
        
        # Calculate returns and volatility
        returns = price_series.pct_change().dropna()
        recent_returns = returns.tail(10)
        
        # Momentum indicators
        sma_short = price_series.rolling(5).mean()
        sma_long = price_series.rolling(20).mean()
        current_price = price_series.iloc[-1]
        
        # Volatility
        volatility = returns.rolling(20).std().iloc[-1]
        
        # RSI-like calculation
        gains = returns[returns > 0]
        losses = abs(returns[returns < 0])
        avg_gain = gains.tail(14).mean() if len(gains) > 0 else 0
        avg_loss = losses.tail(14).mean() if len(losses) > 0 else 0
        
        rs = avg_gain / avg_loss if avg_loss > 0 else 100
        rsi = 100 - (100 / (1 + rs))
        
        # Price position in recent range
        recent_high = price_series.tail(20).max()
        recent_low = price_series.tail(20).min()
        price_position = (current_price - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5
        
        # Trend analysis
        trend_up = sma_short.iloc[-1] > sma_long.iloc[-1] if not pd.isna(sma_short.iloc[-1]) and not pd.isna(sma_long.iloc[-1]) else False
        momentum = recent_returns.mean()
        
        # Signal calculation
        signal_score = 0
        confidence = 0.5
        
        # Bullish conditions
        if trend_up:
            signal_score += 2
            confidence += 0.1
        if momentum > 0.005:  # 0.5% positive momentum
            signal_score += 2
            confidence += 0.15
        if rsi < 40:  # Oversold
            signal_score += 1
            confidence += 0.1
        if price_position < 0.3:  # Near support
            signal_score += 1
            confidence += 0.05
        
        # Bearish conditions
        if not trend_up:
            signal_score -= 2
            confidence += 0.1
        if momentum < -0.005:  # -0.5% negative momentum
            signal_score -= 2
            confidence += 0.15
        if rsi > 70:  # Overbought
            signal_score -= 1
            confidence += 0.1
        if price_position > 0.8:  # Near resistance
            signal_score -= 1
            confidence += 0.05
        
        # Volatility adjustment
        if volatility > 0.03:  # High volatility
            confidence *= 0.8
        
        # Final signal
        if signal_score >= 3:
            signal = 'BUY'
        elif signal_score <= -3:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        confidence = min(1.0, confidence)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'strength': abs(signal_score),
            'metrics': {
                'rsi': rsi,
                'momentum': momentum,
                'volatility': volatility,
                'trend_up': trend_up,
                'price_position': price_position
            }
        }
    
    def execute_trade(self, symbol, action, price, date, shares=None):
        """Execute a trade and record it"""
        if action == 'BUY':
            # Calculate position size
            position_value = min(
                self.capital * self.max_position_size,  # Max 15% of portfolio
                self.cash * 0.95  # Leave 5% cash buffer
            )
            
            if shares is None:
                shares = int(position_value / price)
            
            cost = shares * price * (1 + self.transaction_cost)
            
            if cost <= self.cash and shares > 0:
                self.cash -= cost
                
                if symbol in self.positions:
                    # Add to existing position
                    old_shares = self.positions[symbol]['shares']
                    old_value = old_shares * self.positions[symbol]['entry_price']
                    new_value = shares * price
                    
                    self.positions[symbol]['shares'] += shares
                    self.positions[symbol]['entry_price'] = (old_value + new_value) / (old_shares + shares)
                else:
                    # New position
                    self.positions[symbol] = {'shares': shares, 'entry_price': price}
                
                self.trade_history.append({
                    'date': date,
                    'symbol': symbol,
                    'action': 'BUY',
                    'shares': shares,
                    'price': price,
                    'value': cost,
                    'cash_after': self.cash
                })
                
                return True
        
        elif action == 'SELL' and symbol in self.positions:
            shares = self.positions[symbol]['shares']
            proceeds = shares * price * (1 - self.transaction_cost)
            
            self.cash += proceeds
            entry_price = self.positions[symbol]['entry_price']
            pnl = (price - entry_price) * shares
            
            del self.positions[symbol]
            
            self.trade_history.append({
                'date': date,
                'symbol': symbol,
                'action': 'SELL',
                'shares': shares,
                'price': price,
                'value': proceeds,
                'pnl': pnl,
                'cash_after': self.cash
            })
            
            return True
        
        return False
    
    def calculate_portfolio_value(self, current_prices):
        """Calculate total portfolio value"""
        portfolio_value = self.cash
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                portfolio_value += position['shares'] * current_prices[symbol]
        
        return portfolio_value
    
    def run_backtest(self, data, symbols, rebalance_days=5):
        """Run the full backtest"""
        print(f"🎯 Running backtest with ${self.initial_capital:,} starting capital")
        
        # Get aligned dates across all symbols
        all_dates = None
        for symbol in symbols:
            if symbol in data and not data[symbol].empty:
                symbol_dates = data[symbol].index
                if all_dates is None:
                    all_dates = symbol_dates
                else:
                    all_dates = all_dates.intersection(symbol_dates)
        
        if all_dates is None or len(all_dates) < 100:
            print("❌ Insufficient data for backtest")
            return
        
        print(f"📅 Backtesting {len(all_dates)} trading days")
        
        day_count = 0
        
        for date in all_dates[50:]:  # Skip first 50 days for indicators
            day_count += 1
            current_prices = {}
            
            # Get current prices for all symbols
            for symbol in symbols:
                if symbol in data:
                    try:
                        current_prices[symbol] = data[symbol]['Close'].loc[date]
                    except:
                        continue
            
            # Calculate portfolio value
            portfolio_value = self.calculate_portfolio_value(current_prices)
            daily_return = (portfolio_value / (self.portfolio_history[-1] if self.portfolio_history else self.initial_capital)) - 1
            
            self.portfolio_history.append(portfolio_value)
            self.daily_returns.append(daily_return)
            
            # Generate trading signals every few days
            if day_count % rebalance_days == 0:
                for symbol in symbols:
                    if symbol not in current_prices:
                        continue
                        
                    # Get price history up to this date
                    try:
                        price_history = data[symbol]['Close'].loc[:date]
                        signal_data = self.calculate_markov_signals(price_history)
                        
                        # Execute trades based on signals
                        if signal_data['confidence'] > 0.7:  # High confidence threshold
                            if signal_data['signal'] == 'BUY' and symbol not in self.positions:
                                self.execute_trade(symbol, 'BUY', current_prices[symbol], date)
                            elif signal_data['signal'] == 'SELL' and symbol in self.positions:
                                self.execute_trade(symbol, 'SELL', current_prices[symbol], date)
                                
                    except Exception as e:
                        continue
            
            # Progress update
            if day_count % 100 == 0:
                pct_complete = (day_count / len(all_dates[50:])) * 100
                print(f"📈 Progress: {pct_complete:.1f}% | Portfolio: ${portfolio_value:,.2f}")
        
        # Final portfolio value
        final_value = self.calculate_portfolio_value(current_prices)
        self.portfolio_history.append(final_value)
        
        print(f"✅ Backtest complete!")
        return self.analyze_performance()
    
    def analyze_performance(self):
        """Analyze backtest performance"""
        if len(self.portfolio_history) < 2:
            return {}
        
        final_value = self.portfolio_history[-1]
        total_return = (final_value / self.initial_capital) - 1
        
        # Calculate metrics
        returns = np.array(self.daily_returns)
        returns_pct = returns * 100
        
        # Annualized return
        trading_days = len(returns)
        years = trading_days / 252
        annualized_return = ((final_value / self.initial_capital) ** (1/years)) - 1 if years > 0 else 0
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(252)
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        portfolio_series = np.array(self.portfolio_history)
        peaks = np.maximum.accumulate(portfolio_series)
        drawdowns = (peaks - portfolio_series) / peaks
        max_drawdown = np.max(drawdowns)
        
        # Win rate
        profitable_trades = [t for t in self.trade_history if t['action'] == 'SELL' and t.get('pnl', 0) > 0]
        total_sell_trades = [t for t in self.trade_history if t['action'] == 'SELL']
        win_rate = len(profitable_trades) / len(total_sell_trades) if total_sell_trades else 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(self.trade_history),
            'profitable_trades': len(profitable_trades),
            'trading_days': trading_days
        }

def main():
    print("🚀 Swaggy Stacks - Multi-Year Markov Backtest")
    print("=" * 60)
    
    # Test symbols
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    
    # Initialize backtester
    backtester = MarkovBacktester(initial_capital=100000)
    
    # Get historical data
    data = backtester.get_historical_data(symbols, years=3)
    
    if data is None:
        print("❌ Failed to get historical data")
        return
    
    # Run backtest
    performance = backtester.run_backtest(data, symbols)
    
    if not performance:
        print("❌ Backtest failed")
        return
    
    # Display results
    print(f"\n📊 BACKTEST RESULTS")
    print("=" * 60)
    print(f"Initial Capital:      ${performance['initial_capital']:>12,.2f}")
    print(f"Final Value:          ${performance['final_value']:>12,.2f}")
    print(f"Total Return:         {performance['total_return']:>12.1%}")
    print(f"Annualized Return:    {performance['annualized_return']:>12.1%}")
    print(f"Volatility:           {performance['volatility']:>12.1%}")
    print(f"Sharpe Ratio:         {performance['sharpe_ratio']:>12.2f}")
    print(f"Max Drawdown:         {performance['max_drawdown']:>12.1%}")
    print(f"Win Rate:             {performance['win_rate']:>12.1%}")
    print(f"Total Trades:         {performance['total_trades']:>12}")
    print(f"Profitable Trades:    {performance['profitable_trades']:>12}")
    print(f"Trading Days:         {performance['trading_days']:>12}")
    
    # Benchmark comparison (buy and hold)
    print(f"\n📈 VS BUY & HOLD COMPARISON")
    print("=" * 60)
    
    # Calculate buy and hold return for comparison
    if 'AAPL' in data and not data['AAPL'].empty:
        start_price = data['AAPL']['Close'].iloc[50]  # Same starting point as backtest
        end_price = data['AAPL']['Close'].iloc[-1]
        buy_hold_return = (end_price / start_price) - 1
        
        print(f"AAPL Buy & Hold:      {buy_hold_return:>12.1%}")
        print(f"Strategy vs B&H:      {performance['total_return'] - buy_hold_return:>12.1%}")
    
    # Recent trades
    if backtester.trade_history:
        print(f"\n💼 RECENT TRADES (Last 10)")
        print("=" * 60)
        recent_trades = backtester.trade_history[-10:]
        print(f"{'Date':<12} {'Symbol':<6} {'Action':<4} {'Shares':<6} {'Price':<8} {'P&L':<10}")
        print("-" * 60)
        for trade in recent_trades:
            date_str = trade['date'].strftime('%Y-%m-%d')
            pnl_str = f"${trade.get('pnl', 0):.2f}" if 'pnl' in trade else ""
            print(f"{date_str:<12} {trade['symbol']:<6} {trade['action']:<4} {trade['shares']:<6} ${trade['price']:<7.2f} {pnl_str:<10}")
    
    print(f"\n🎯 STRATEGY ASSESSMENT")
    print("=" * 60)
    if performance['total_return'] > 0.1:  # 10%+ total return
        print("✅ STRONG PERFORMANCE - Strategy shows positive results")
    elif performance['total_return'] > 0:
        print("⚠️  MODEST PERFORMANCE - Strategy is profitable but could improve")
    else:
        print("❌ POOR PERFORMANCE - Strategy needs optimization")
    
    if performance['sharpe_ratio'] > 1.0:
        print("✅ GOOD RISK ADJUSTMENT - Strong risk-adjusted returns")
    elif performance['sharpe_ratio'] > 0.5:
        print("⚠️  MODERATE RISK ADJUSTMENT - Acceptable risk-adjusted returns")
    else:
        print("❌ POOR RISK ADJUSTMENT - High risk relative to returns")
    
    print(f"\n🔧 NEXT STEPS:")
    print("1. Optimize signal thresholds and confidence levels")
    print("2. Add more sophisticated risk management")
    print("3. Consider sector diversification")
    print("4. Implement stop-losses and take-profits")
    print("5. Test on different market conditions")

if __name__ == "__main__":
    main()