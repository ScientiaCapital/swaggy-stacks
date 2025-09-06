"""
Enhanced Comprehensive Markov Trading System
Incorporates all advanced features from the provided implementation
"""

import numpy as np
import pandas as pd
from collections import defaultdict, deque
from datetime import datetime, timedelta
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

class EnhancedDataHandler:
    """
    Enhanced data handling with validation, cleaning, and multiple timeframe support
    """
    
    def __init__(self):
        self.data_cache = {}
        self.validated_data = {}
        
    def validate_and_clean_data(self, prices, volumes=None):
        """
        Validate and clean market data, handling missing values and outliers
        """
        # Convert to numpy arrays for processing
        prices = np.array(prices)
        
        # Handle NaN values
        if np.any(np.isnan(prices)):
            # Linear interpolation for missing values
            nan_indices = np.where(np.isnan(prices))[0]
            for idx in nan_indices:
                if idx == 0:
                    # First value missing, use next available
                    next_val_idx = np.where(~np.isnan(prices))[0]
                    if len(next_val_idx) > 0:
                        prices[idx] = prices[next_val_idx[0]]
                    else:
                        prices[idx] = 0
                elif idx == len(prices) - 1:
                    # Last value missing, use previous
                    prices[idx] = prices[idx-1]
                else:
                    # Interpolate between previous and next
                    prev_val = prices[idx-1]
                    next_val_idx = np.where(~np.isnan(prices[idx+1:]))[0]
                    if len(next_val_idx) > 0:
                        next_val = prices[idx+1+next_val_idx[0]]
                        prices[idx] = (prev_val + next_val) / 2
                    else:
                        prices[idx] = prev_val
        
        # Handle outliers using Z-score method
        if len(prices) > 10:
            returns = np.diff(np.log(prices))
            z_scores = np.abs(stats.zscore(returns))
            outlier_indices = np.where(z_scores > 3)[0]
            
            for idx in outlier_indices:
                if idx > 0 and idx < len(returns) - 1:
                    # Replace outlier with average of neighbors
                    returns[idx] = (returns[idx-1] + returns[idx+1]) / 2
            
            # Reconstruct prices from cleaned returns
            cleaned_prices = [prices[0]]
            for i, r in enumerate(returns):
                cleaned_prices.append(cleaned_prices[-1] * np.exp(r))
            prices = np.array(cleaned_prices)
        
        # Process volumes if provided
        cleaned_volumes = None
        if volumes is not None:
            volumes = np.array(volumes)
            # Simple forward fill for volume data
            if np.any(np.isnan(volumes)):
                mask = np.isnan(volumes)
                volumes[mask] = np.interp(
                    np.flatnonzero(mask), 
                    np.flatnonzero(~mask), 
                    volumes[~mask]
                )
            cleaned_volumes = volumes
        
        return prices, cleaned_volumes
    
    def resample_data(self, prices, volumes, original_freq, target_freq):
        """
        Resample data to different timeframes (e.g., from 1-min to 5-min)
        """
        # Create datetime index
        if len(prices) > 0:
            start_time = datetime.now() - timedelta(minutes=len(prices)*int(original_freq))
            time_index = [start_time + timedelta(minutes=i*int(original_freq)) for i in range(len(prices))]
            
            # Create DataFrame
            df = pd.DataFrame({
                'price': prices,
                'volume': volumes if volumes is not None else np.zeros(len(prices))
            }, index=time_index)
            
            # Resample
            resampled = df.resample(f'{target_freq}T').agg({
                'price': 'ohlc',
                'volume': 'sum'
            })
            
            # Flatten columns
            resampled.columns = ['_'.join(col).strip() for col in resampled.columns.values]
            
            return (
                resampled['price_close'].values,
                resampled['volume_sum'].values,
                resampled['price_open'].values,
                resampled['price_high'].values,
                resampled['price_low'].values
            )
        
        return prices, volumes, None, None, None

class PositionSizer:
    """
    Position sizing based on market uncertainty and confidence
    """
    
    def __init__(self, account_size=100000, max_risk_per_trade=0.02):
        self.account_size = account_size
        self.max_risk_per_trade = max_risk_per_trade
        self.risk_free_rate = 0.02  # Annual risk-free rate
    
    def calculate_size(self, uncertainty, current_price, stop_loss_price=None):
        """
        Calculate position size using modified Kelly criterion
        """
        # Base position size on account size and risk per trade
        max_risk_amount = self.account_size * self.max_risk_per_trade
        
        # If stop loss is provided, use it to determine position size
        if stop_loss_price and current_price > 0:
            risk_per_share = abs(current_price - stop_loss_price)
            if risk_per_share > 0:
                shares = max_risk_amount / risk_per_share
                return shares
        
        # Otherwise, use uncertainty to determine position size
        if uncertainty < 0.2:  # High confidence
            position_pct = 0.08
        elif uncertainty < 0.4:  # Medium-high confidence
            position_pct = 0.05
        elif uncertainty < 0.6:  # Medium confidence
            position_pct = 0.03
        else:  # Low confidence
            position_pct = 0.01
        
        shares = (self.account_size * position_pct) / current_price
        return shares
    
    def calculate_stop_loss(self, analysis, current_price, atr=None, multiplier=2):
        """
        Calculate stop loss based on volatility and analysis
        """
        if atr is not None and atr > 0:
            # Use ATR for stop loss
            if analysis['consensus']['action'] == 'BUY':
                return current_price - (multiplier * atr)
            else:
                return current_price + (multiplier * atr)
        
        # Fallback: percentage-based stop loss
        stop_loss_pct = 0.02  # 2% stop loss
        
        if analysis['consensus']['action'] == 'BUY':
            return current_price * (1 - stop_loss_pct)
        else:
            return current_price * (1 + stop_loss_pct)
    
    def calculate_sharpe_ratio(self, returns, periods=252):
        """
        Calculate Sharpe ratio for performance evaluation
        """
        if len(returns) < 2:
            return 0
        
        excess_returns = returns - (self.risk_free_rate / periods)
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods)
        return sharpe

class RegimeDetector:
    """
    Detect different market regimes using statistical methods
    """
    
    def __init__(self):
        self.history = deque(maxlen=100)
    
    def calculate_hurst_exponent(self, prices):
        """
        Calculate the Hurst exponent to detect trending vs mean-reverting behavior
        """
        if len(prices) < 50:
            return 0.5  # Neutral value
        
        # Convert prices to log returns
        returns = np.diff(np.log(prices))
        
        # Calculate the rescaled range
        max_lag = min(20, len(returns) // 2)
        lags = range(2, max_lag)
        
        # Calculate the R/S statistic for each lag
        rs_values = []
        for lag in lags:
            # Split the data into chunks of size lag
            chunks = [returns[i:i+lag] for i in range(0, len(returns), lag)]
            if len(chunks[-1]) < lag // 2:  # Discard the last chunk if it's too small
                chunks = chunks[:-1]
            
            chunk_rs = []
            for chunk in chunks:
                mean_chunk = np.mean(chunk)
                deviation = chunk - mean_chunk
                cumulative_deviation = np.cumsum(deviation)
                range_val = np.max(cumulative_deviation) - np.min(cumulative_deviation)
                std_val = np.std(chunk)
                
                if std_val > 0:
                    chunk_rs.append(range_val / std_val)
            
            if chunk_rs:
                rs_values.append(np.mean(chunk_rs))
        
        if len(rs_values) < 2:
            return 0.5
        
        # Fit a line to the log of the R/S statistics
        x = np.log(lags[:len(rs_values)])
        y = np.log(rs_values)
        
        # Handle any infinite values
        finite_mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(finite_mask) < 2:
            return 0.5
        
        x, y = x[finite_mask], y[finite_mask]
        
        # Linear regression
        slope, intercept = np.polyfit(x, y, 1)
        return slope
    
    def detect_regime(self, prices, volumes=None):
        """
        Identify the current market regime
        """
        if len(prices) < 50:
            return "INSUFFICIENT_DATA"
        
        # Calculate metrics
        returns = np.diff(np.log(prices))
        volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
        
        # Calculate Hurst exponent
        hurst = self.calculate_hurst_exponent(prices)
        
        # Determine regime based on metrics
        if hurst > 0.6:
            trend = "TRENDING"
        elif hurst < 0.4:
            trend = "MEAN_REVERTING"
        else:
            trend = "TRANSITIONAL"
        
        # Volatility classification
        if volatility > 0.25:
            vol_class = "HIGH_VOLATILITY"
        elif volatility > 0.15:
            vol_class = "MEDIUM_VOLATILITY"
        else:
            vol_class = "LOW_VOLATILITY"
        
        # Volume analysis (if available)
        volume_class = ""
        if volumes is not None and len(volumes) > 10:
            volume_ratio = np.mean(volumes[-5:]) / np.mean(volumes[-20:])
            if volume_ratio > 1.5:
                volume_class = "_HIGH_VOLUME"
            elif volume_ratio < 0.7:
                volume_class = "_LOW_VOLUME"
        
        return f"{trend}_{vol_class}{volume_class}"

class EnhancedRiskManager:
    """
    Comprehensive risk management system
    """
    
    def __init__(self, max_drawdown=0.2, correlation_threshold=0.7):
        self.max_drawdown = max_drawdown
        self.correlation_threshold = correlation_threshold
        self.portfolio = {}
        self.drawdown_history = deque(maxlen=100)
    
    def assess_trade_risk(self, analysis, prices, current_price):
        """
        Assess risk for a potential trade
        """
        risk_score = 0
        warnings = []
        
        # Check RSI for overbought/oversold conditions
        rsi = self.calculate_rsi(prices)
        if rsi > 70:
            risk_score += 0.3
            warnings.append("OVERBOUGHT")
        elif rsi < 30:
            risk_score += 0.2
            warnings.append("OVERSOLD")
        
        # Check for volatility expansion
        recent_volatility = np.std(np.diff(np.log(prices[-10:]))) * np.sqrt(252)
        historical_volatility = np.std(np.diff(np.log(prices[-50:]))) * np.sqrt(252)
        
        if recent_volatility > historical_volatility * 1.5:
            risk_score += 0.4
            warnings.append("HIGH_VOLATILITY_EXPANSION")
        
        # Check if price is at extreme compared to recent range
        price_position = (current_price - np.min(prices[-20:])) / (np.max(prices[-20:]) - np.min(prices[-20:]))
        if price_position > 0.8:
            risk_score += 0.2
            warnings.append("NEAR_RESISTANCE")
        elif price_position < 0.2:
            risk_score += 0.1
            warnings.append("NEAR_SUPPORT")
        
        # Check market regime from analysis
        if 'market_regime' in analysis:
            if "HIGH_VOLATILITY" in analysis['market_regime']:
                risk_score += 0.2
                warnings.append("HIGH_VOL_REGIME")
        
        return {
            'risk_score': min(1.0, risk_score),
            'warnings': warnings,
            'rsi': rsi,
            'volatility_ratio': recent_volatility / historical_volatility if historical_volatility > 0 else 1
        }
    
    def calculate_rsi(self, prices, period=14):
        """
        Calculate Relative Strength Index
        """
        if len(prices) < period + 1:
            return 50
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def check_portfolio_correlation(self, new_symbol, historical_correlations):
        """
        Check if adding a new position would increase portfolio correlation too much
        """
        if not self.portfolio:
            return True  # No positions yet
            
        avg_correlation = np.mean([
            historical_correlations.get((symbol, new_symbol), 0)
            for symbol in self.portfolio.keys()
        ])
        
        return avg_correlation < self.correlation_threshold
    
    def update_portfolio(self, symbol, shares, price):
        """
        Update portfolio with new position
        """
        if shares == 0:
            if symbol in self.portfolio:
                del self.portfolio[symbol]
        else:
            self.portfolio[symbol] = {
                'shares': shares,
                'entry_price': price,
                'value': shares * price
            }
    
    def calculate_drawdown(self, equity_curve):
        """
        Calculate maximum drawdown
        """
        if len(equity_curve) < 2:
            return 0
        
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        return np.max(drawdown) if len(drawdown) > 0 else 0

class Backtester:
    """
    Backtesting framework for system evaluation
    """
    
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.results = {}
    
    def run_backtest(self, system, historical_data, symbols=None, transaction_cost=0.001):
        """
        Run backtest on historical data
        """
        if symbols is None:
            symbols = list(historical_data.keys())
        
        equity_curve = [self.initial_capital]
        positions = {}
        trades = []
        
        # Prepare data for each symbol
        symbol_data = {}
        for symbol in symbols:
            symbol_data[symbol] = {
                'prices': historical_data[symbol]['close'],
                'volumes': historical_data[symbol]['volume'],
                'timestamps': historical_data[symbol]['timestamp']
            }
        
        # Iterate through historical data
        for i in range(50, len(symbol_data[symbols[0]]['prices'])):
            current_equity = equity_curve[-1]
            current_date = symbol_data[symbols[0]]['timestamps'][i]
            
            # Update system with latest data
            for symbol in symbols:
                prices = symbol_data[symbol]['prices'][:i+1]
                volumes = symbol_data[symbol]['volumes'][:i+1]
                
                # Analyze current market state
                analysis = system.analyze_market(prices, volumes)
                
                # Get current price
                current_price = prices[-1]
                
                # Make trading decision
                action = analysis['consensus']['action']
                confidence = analysis['consensus']['confidence']
                
                # Execute trades
                if symbol in positions:
                    # Check if we should exit position
                    if (action == 'SELL' and positions[symbol]['shares'] > 0) or \
                       (action == 'BUY' and positions[symbol]['shares'] < 0):
                        # Exit position
                        exit_price = current_price * (1 - transaction_cost) if positions[symbol]['shares'] > 0 else current_price * (1 + transaction_cost)
                        pnl = (exit_price - positions[symbol]['entry_price']) * positions[symbol]['shares']
                        current_equity += pnl
                        
                        # Record trade
                        trades.append({
                            'symbol': symbol,
                            'entry_date': positions[symbol]['entry_date'],
                            'exit_date': current_date,
                            'entry_price': positions[symbol]['entry_price'],
                            'exit_price': exit_price,
                            'shares': positions[symbol]['shares'],
                            'pnl': pnl,
                            'type': 'LONG' if positions[symbol]['shares'] > 0 else 'SHORT'
                        })
                        
                        del positions[symbol]
                
                # Enter new position
                if action in ['BUY', 'SELL'] and symbol not in positions:
                    # Calculate position size
                    uncertainty = 1 - confidence
                    position_size = system.position_sizer.calculate_size(
                        uncertainty, current_price
                    )
                    
                    if action == 'BUY':
                        shares = position_size
                    else:  # SELL (short)
                        shares = -position_size
                    
                    # Account for transaction costs
                    entry_price = current_price * (1 + transaction_cost) if shares > 0 else current_price * (1 - transaction_cost)
                    
                    # Update positions
                    positions[symbol] = {
                        'shares': shares,
                        'entry_price': entry_price,
                        'entry_date': current_date
                    }
            
            # Update equity curve
            # Calculate current portfolio value
            portfolio_value = 0
            for symbol, position in positions.items():
                current_symbol_price = symbol_data[symbol]['prices'][i]
                portfolio_value += position['shares'] * current_symbol_price
            
            equity_curve.append(current_equity + portfolio_value)
        
        # Calculate performance metrics
        returns = np.diff(equity_curve) / equity_curve[:-1]
        total_return = (equity_curve[-1] - self.initial_capital) / self.initial_capital
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        max_drawdown = self.calculate_max_drawdown(equity_curve)
        
        self.results = {
            'equity_curve': equity_curve,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trades': trades
        }
        
        return self.results
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        """
        Calculate annualized Sharpe ratio
        """
        if len(returns) == 0:
            return 0
        
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return sharpe
    
    def calculate_max_drawdown(self, equity_curve):
        """
        Calculate maximum drawdown
        """
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        return np.max(drawdown) if len(drawdown) > 0 else 0

# Note: This file contains the enhanced components. The full system would also include
# the original Fibonacci, Elliott Wave, Wyckoff, and Markov Chain classes from the provided code.
# For brevity, I'm including the enhanced components that extend the original system.
