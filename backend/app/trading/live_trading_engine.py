"""
Live Trading Engine - Integrates Markov analysis with Alpaca for live trading
"""

import asyncio
import yfinance as yf
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import structlog

from app.trading.alpaca_client import AlpacaClient
from app.trading.risk_manager import RiskManager
from app.trading.order_manager import OrderManager
from app.trading.position_optimizer import PositionOptimizer
from app.ai.trading_agents import AIAgentCoordinator
from app.core.config import settings
from app.core.exceptions import TradingError

logger = structlog.get_logger()


class LiveTradingEngine:
    """
    Main live trading engine that coordinates all components
    """
    
    def __init__(self, user_id: int, symbols: List[str], enable_ai: bool = True):
        self.user_id = user_id
        self.symbols = symbols
        self.is_running = False
        self.paper_trading = True  # Always start with paper trading
        self.enable_ai = enable_ai
        
        # Initialize components
        self.alpaca_client = AlpacaClient(paper=True)
        self.risk_manager = RiskManager(user_id=user_id)
        self.order_manager = OrderManager(self.alpaca_client, self.risk_manager)
        self.position_optimizer = PositionOptimizer()
        
        # Initialize AI agents if enabled
        if self.enable_ai:
            try:
                self.ai_coordinator = AIAgentCoordinator()
                logger.info("AI agents initialized successfully")
            except Exception as e:
                logger.error("Failed to initialize AI agents", error=str(e))
                self.ai_coordinator = None
                self.enable_ai = False
        else:
            self.ai_coordinator = None
        
        # Trading parameters
        self.signal_check_interval = 300  # 5 minutes
        self.market_hours = {
            'start': '09:30',
            'end': '16:00',
            'timezone': 'US/Eastern'
        }
        
        # Market data cache
        self.market_data_cache = {}
        self.last_signals = {}
        
        logger.info("Live trading engine initialized", user_id=user_id, symbols=symbols)
    
    async def start_trading(self):
        """Start the live trading engine"""
        try:
            self.is_running = True
            logger.info("Starting live trading engine")
            
            # Verify account connection
            account = await self.alpaca_client.get_account()
            logger.info("Connected to Alpaca account", account_value=account['portfolio_value'])
            
            # Start main trading loop
            while self.is_running:
                try:
                    if self._is_market_hours():
                        await self._trading_cycle()
                        await self.order_manager.monitor_positions()
                    else:
                        logger.info("Outside market hours, waiting...")
                    
                    # Wait before next cycle
                    await asyncio.sleep(self.signal_check_interval)
                    
                except Exception as e:
                    logger.error("Error in trading cycle", error=str(e))
                    await asyncio.sleep(60)  # Wait 1 minute before retrying
            
        except Exception as e:
            logger.error("Fatal error in trading engine", error=str(e))
            self.is_running = False
        finally:
            logger.info("Live trading engine stopped")
    
    def stop_trading(self):
        """Stop the live trading engine"""
        self.is_running = False
        logger.info("Stopping live trading engine")
    
    async def _trading_cycle(self):
        """Main trading cycle - analyze signals and execute trades"""
        try:
            logger.info("Starting trading cycle")
            
            # Get current account state
            account = await self.alpaca_client.get_account()
            positions = await self.alpaca_client.get_positions()
            
            account_value = float(account['portfolio_value'])
            cash = float(account['cash'])
            
            # Get market data for all symbols
            market_data = await self._get_market_data()
            if not market_data:
                logger.warning("No market data available, skipping cycle")
                return
            
            # Analyze each symbol
            for symbol in self.symbols:
                try:
                    await self._analyze_and_trade_symbol(
                        symbol, market_data.get(symbol), account_value, cash, positions
                    )
                except Exception as e:
                    logger.error("Error analyzing symbol", symbol=symbol, error=str(e))
            
            # Clean up old orders
            await self.order_manager.cleanup_expired_orders()
            
            logger.info("Trading cycle completed")
            
        except Exception as e:
            logger.error("Error in trading cycle", error=str(e))
    
    async def _analyze_and_trade_symbol(
        self,
        symbol: str,
        price_data: Optional[pd.DataFrame],
        account_value: float,
        cash: float,
        current_positions: List[Dict]
    ):
        """Analyze a symbol and execute trades using AI-enhanced analysis"""
        try:
            if price_data is None or len(price_data) < 50:
                logger.warning("Insufficient data for analysis", symbol=symbol)
                return
            
            current_price = price_data['Close'].iloc[-1]
            
            # Check if we already have a position
            existing_position = next(
                (pos for pos in current_positions if pos['symbol'] == symbol), None
            )
            
            # Calculate technical indicators
            technical_indicators = self._calculate_technical_indicators(price_data)
            atr = await self.alpaca_client.calculate_atr(symbol)
            volatility = price_data['Close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
            
            # Prepare market data for AI analysis
            market_data = {
                'current_price': current_price,
                'volume': price_data['Volume'].iloc[-1] if 'Volume' in price_data.columns else 0,
                'high_52w': price_data['High'].rolling(252).max().iloc[-1] if len(price_data) >= 252 else price_data['High'].max(),
                'low_52w': price_data['Low'].rolling(252).min().iloc[-1] if len(price_data) >= 252 else price_data['Low'].min(),
                'volatility': volatility
            }
            
            # Run traditional Markov analysis
            markov_analysis = self._calculate_markov_signals(price_data, symbol)
            
            # Use AI-enhanced analysis if available
            if self.enable_ai and self.ai_coordinator:
                try:
                    ai_analysis = await self.ai_coordinator.comprehensive_analysis(
                        symbol=symbol,
                        market_data=market_data,
                        technical_indicators=technical_indicators,
                        account_info={'equity': account_value, 'cash': cash},
                        current_positions=[{'symbol': p['symbol'], 'qty': p['qty'], 'market_value': p.get('market_value', 0)} for p in current_positions],
                        markov_analysis=markov_analysis
                    )
                    
                    # Use AI recommendation
                    final_signal = ai_analysis.get('final_recommendation', 'HOLD')
                    ai_confidence = ai_analysis.get('strategy_signal', {}).get('confidence', 0.0)
                    
                    # Require high confidence for AI signals
                    if ai_confidence < 0.6:
                        logger.info("Low AI confidence, using fallback", symbol=symbol, ai_confidence=ai_confidence)
                        final_signal = markov_analysis['signal']
                        confidence = markov_analysis['confidence']
                    else:
                        confidence = ai_confidence
                        
                    logger.info("AI analysis completed", symbol=symbol, 
                              ai_signal=final_signal, ai_confidence=ai_confidence,
                              markov_signal=markov_analysis['signal'], markov_confidence=markov_analysis['confidence'])
                    
                except Exception as e:
                    logger.error("AI analysis failed, using Markov fallback", symbol=symbol, error=str(e))
                    final_signal = markov_analysis['signal']
                    confidence = markov_analysis['confidence']
            else:
                # Fallback to traditional Markov analysis
                final_signal = markov_analysis['signal']
                confidence = markov_analysis['confidence']
            
            # Apply confidence threshold
            if confidence < 0.6:  # Lowered threshold slightly for AI-enhanced system
                logger.info("Low confidence signal, skipping", symbol=symbol, confidence=confidence)
                return
            
            # Execute trades based on final signal
            if final_signal == 'BUY' and not existing_position:
                await self._execute_buy_signal(
                    symbol, current_price, confidence, account_value, cash, volatility, atr
                )
            elif final_signal == 'SELL' and existing_position:
                await self._execute_sell_signal(
                    symbol, current_price, existing_position, confidence
                )
            
            # Store enhanced signal data for tracking
            self.last_signals[symbol] = {
                'signal': final_signal,
                'confidence': confidence,
                'price': current_price,
                'markov_signal': markov_analysis['signal'],
                'markov_confidence': markov_analysis['confidence'],
                'ai_enhanced': self.enable_ai and self.ai_coordinator is not None,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error("Error analyzing symbol", symbol=symbol, error=str(e))
    
    async def _execute_buy_signal(
        self,
        symbol: str,
        current_price: float,
        confidence: float,
        account_value: float,
        cash: float,
        volatility: float,
        atr: Optional[float]
    ):
        """Execute a buy signal with proper risk management"""
        try:
            # Calculate stop-loss price
            if atr:
                stop_loss_price = current_price - (atr * 2.0)  # 2x ATR stop
            else:
                stop_loss_price = current_price * 0.95  # 5% stop loss
            
            # Calculate take-profit price
            take_profit_price = current_price * 1.15  # 15% take profit
            
            # Calculate optimal position size
            position_size = self.risk_manager.calculate_position_size(
                symbol=symbol,
                price=current_price,
                account_value=account_value,
                volatility=volatility,
                confidence=confidence,
                stop_loss_price=stop_loss_price,
                use_optimizer=True
            )
            
            if position_size < 100:  # Minimum $100 position
                logger.info("Position size too small, skipping", symbol=symbol, size=position_size)
                return
            
            # Check if we have enough cash
            if position_size > cash * 0.95:
                logger.warning("Insufficient cash for position", symbol=symbol, needed=position_size, available=cash)
                return
            
            shares = int(position_size / current_price)
            
            # Validate order with risk manager
            is_valid, reason = self.risk_manager.validate_order(
                symbol=symbol,
                quantity=shares,
                price=current_price,
                side='BUY',
                current_positions=await self.alpaca_client.get_positions(),
                account_value=account_value,
                daily_pnl=0.0  # Simplified - would track daily P&L
            )
            
            if not is_valid:
                logger.warning("Order validation failed", symbol=symbol, reason=reason)
                return
            
            # Execute bracket order (entry + stop loss + take profit)
            bracket_result = await self.order_manager.create_bracket_order(
                symbol=symbol,
                quantity=shares,
                side='BUY',
                entry_price=None,  # Market order
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                atr=atr
            )
            
            logger.info(
                "BUY signal executed",
                symbol=symbol,
                shares=shares,
                price=current_price,
                position_size=position_size,
                stop_loss=stop_loss_price,
                take_profit=take_profit_price,
                confidence=confidence,
                order_id=bracket_result.get('main_order_id')
            )
            
        except Exception as e:
            logger.error("Error executing buy signal", symbol=symbol, error=str(e))
    
    async def _execute_sell_signal(
        self,
        symbol: str,
        current_price: float,
        position: Dict,
        confidence: float
    ):
        """Execute a sell signal for an existing position"""
        try:
            shares = abs(float(position['qty']))
            
            # Execute market sell order
            sell_order = await self.alpaca_client.execute_order(
                symbol=symbol,
                qty=shares,
                side='sell',
                order_type='market'
            )
            
            # Cancel any existing stop orders for this position
            await self.order_manager.cancel_stop_orders(symbol)
            
            # Calculate P&L
            entry_price = float(position.get('cost_basis', 0)) / shares if shares > 0 else 0
            pnl = (current_price - entry_price) * shares
            
            logger.info(
                "SELL signal executed",
                symbol=symbol,
                shares=shares,
                price=current_price,
                entry_price=entry_price,
                pnl=pnl,
                confidence=confidence,
                order_id=sell_order['id']
            )
            
            # Update position optimizer performance
            result = 'win' if pnl > 0 else 'loss'
            self.position_optimizer.update_performance_history(
                symbol, entry_price, current_price, shares, result
            )
            
        except Exception as e:
            logger.error("Error executing sell signal", symbol=symbol, error=str(e))
    
    def _calculate_technical_indicators(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicators for AI analysis"""
        try:
            price_series = price_data['Close']
            
            # RSI calculation
            gains = price_series.diff().clip(lower=0)
            losses = (-1 * price_series.diff()).clip(lower=0)
            avg_gains = gains.rolling(14).mean()
            avg_losses = losses.rolling(14).mean()
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            # Moving averages
            ma20 = price_series.rolling(20).mean().iloc[-1]
            ma50 = price_series.rolling(50).mean().iloc[-1]
            
            # MACD
            exp12 = price_series.ewm(span=12).mean()
            exp26 = price_series.ewm(span=26).mean()
            macd = (exp12 - exp26).iloc[-1]
            
            # ATR (simplified)
            high_low = price_data['High'] - price_data['Low']
            high_close = (price_data['High'] - price_data['Close'].shift()).abs()
            low_close = (price_data['Low'] - price_data['Close'].shift()).abs()
            ranges = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = ranges.rolling(14).mean().iloc[-1]
            
            # Bollinger Bands
            sma20 = price_series.rolling(20).mean()
            std20 = price_series.rolling(20).std()
            bb_upper = (sma20 + 2 * std20).iloc[-1]
            bb_lower = (sma20 - 2 * std20).iloc[-1]
            
            return {
                'rsi': rsi if not pd.isna(rsi) else 50.0,
                'ma20': ma20 if not pd.isna(ma20) else price_series.iloc[-1],
                'ma50': ma50 if not pd.isna(ma50) else price_series.iloc[-1],
                'macd': macd if not pd.isna(macd) else 0.0,
                'atr': atr if not pd.isna(atr) else 0.02 * price_series.iloc[-1],
                'bollinger_bands': {
                    'upper': bb_upper if not pd.isna(bb_upper) else price_series.iloc[-1] * 1.02,
                    'lower': bb_lower if not pd.isna(bb_lower) else price_series.iloc[-1] * 0.98
                }
            }
            
        except Exception as e:
            logger.error("Error calculating technical indicators", error=str(e))
            current_price = price_data['Close'].iloc[-1]
            return {
                'rsi': 50.0,
                'ma20': current_price,
                'ma50': current_price,
                'macd': 0.0,
                'atr': 0.02 * current_price,
                'bollinger_bands': {
                    'upper': current_price * 1.02,
                    'lower': current_price * 0.98
                }
            }
    
    def _calculate_markov_signals(self, price_data: pd.DataFrame, symbol: str) -> Dict:
        """Calculate Markov trading signals from price data"""
        try:
            # Use the enhanced Markov analysis from backtest
            price_series = price_data['Close']
            
            if len(price_series) < 30:
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
                    'price_position': price_position,
                    'signal_score': signal_score
                }
            }
            
        except Exception as e:
            logger.error("Error calculating Markov signals", symbol=symbol, error=str(e))
            return {'signal': 'HOLD', 'confidence': 0.0, 'strength': 0}
    
    async def _get_market_data(self) -> Dict[str, pd.DataFrame]:
        """Get current market data for all symbols"""
        try:
            market_data = {}
            
            for symbol in self.symbols:
                try:
                    # Get recent data from yfinance (free)
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="3mo", interval="1d")  # 3 months of daily data
                    
                    if not hist.empty:
                        market_data[symbol] = hist
                        logger.debug("Market data retrieved", symbol=symbol, days=len(hist))
                    else:
                        logger.warning("No market data for symbol", symbol=symbol)
                        
                except Exception as e:
                    logger.error("Error getting market data", symbol=symbol, error=str(e))
            
            return market_data
            
        except Exception as e:
            logger.error("Error getting market data", error=str(e))
            return {}
    
    def _is_market_hours(self) -> bool:
        """Check if market is currently open"""
        try:
            from datetime import datetime
            import pytz
            
            # Get current time in market timezone
            market_tz = pytz.timezone(self.market_hours['timezone'])
            now = datetime.now(market_tz)
            
            # Check if it's a weekday
            if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return False
            
            # Check if it's within market hours
            market_start = now.replace(
                hour=int(self.market_hours['start'].split(':')[0]),
                minute=int(self.market_hours['start'].split(':')[1]),
                second=0,
                microsecond=0
            )
            
            market_end = now.replace(
                hour=int(self.market_hours['end'].split(':')[0]),
                minute=int(self.market_hours['end'].split(':')[1]),
                second=0,
                microsecond=0
            )
            
            return market_start <= now <= market_end
            
        except Exception as e:
            logger.error("Error checking market hours", error=str(e))
            return False  # Fail safe
    
    async def get_trading_status(self) -> Dict[str, Any]:
        """Get current status of the trading engine"""
        try:
            account = await self.alpaca_client.get_account()
            positions = await self.alpaca_client.get_positions()
            
            # Get order manager status
            order_status = self.order_manager.get_monitoring_status()
            
            # Get position optimizer summary
            optimizer_summary = self.position_optimizer.get_optimization_summary()
            
            return {
                'is_running': self.is_running,
                'paper_trading': self.paper_trading,
                'market_hours': self._is_market_hours(),
                'symbols': self.symbols,
                'account': {
                    'portfolio_value': account['portfolio_value'],
                    'cash': account['cash'],
                    'buying_power': account['buying_power']
                },
                'positions': len(positions),
                'active_positions': [pos['symbol'] for pos in positions],
                'order_management': order_status,
                'position_optimizer': optimizer_summary,
                'last_signals': self.last_signals,
                'last_update': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error("Error getting trading status", error=str(e))
            return {'error': str(e)}
    
    async def emergency_stop(self):
        """Emergency stop - close all positions and cancel all orders"""
        try:
            logger.warning("EMERGENCY STOP INITIATED")
            
            # Stop the trading loop
            self.is_running = False
            
            # Get all open orders and cancel them
            orders = await self.alpaca_client.get_orders(status='open')
            for order in orders:
                try:
                    await self.alpaca_client.cancel_order(order['id'])
                    logger.info("Cancelled order", order_id=order['id'], symbol=order['symbol'])
                except Exception as e:
                    logger.error("Failed to cancel order", order_id=order['id'], error=str(e))
            
            # Get all positions and close them
            positions = await self.alpaca_client.get_positions()
            for position in positions:
                try:
                    symbol = position['symbol']
                    qty = abs(float(position['qty']))
                    side = 'sell' if float(position['qty']) > 0 else 'buy'
                    
                    await self.alpaca_client.execute_order(
                        symbol=symbol,
                        qty=qty,
                        side=side,
                        order_type='market'
                    )
                    
                    logger.info("Emergency position closed", symbol=symbol, qty=qty, side=side)
                    
                except Exception as e:
                    logger.error("Failed to close position", symbol=position['symbol'], error=str(e))
            
            logger.warning("EMERGENCY STOP COMPLETED")
            
        except Exception as e:
            logger.error("Error during emergency stop", error=str(e))