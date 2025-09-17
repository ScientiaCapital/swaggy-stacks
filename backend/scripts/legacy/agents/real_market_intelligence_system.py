#!/usr/bin/env python3
"""
🧠 REAL MARKET INTELLIGENCE SYSTEM
Gathers REAL market data, creates data-driven trading ideas, and backtests on historical data
No mock data, no fake communication - everything based on actual market analysis
"""

import asyncio
import time
import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
from sqlalchemy.orm import sessionmaker

# Add the backend directory to Python path
sys.path.append('/Users/tmkipper/repos/swaggy-stacks/backend')

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.live import CryptoDataStream
from alpaca.data.requests import CryptoBarsRequest, CryptoLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest, StopLossRequest, TakeProfitRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from app.core.config import settings
from app.core.database import engine
from app.models.trade import Trade
from app.models.market_data import MarketData
# Portfolio model not available - will track portfolio data internally
from app.indicators.technical_indicators import TechnicalIndicators
from app.ml.markov_system import MarkovChainAnalyzer

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MarketSignal:
    """Real market signal based on actual data analysis"""
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    reasoning: str
    technical_analysis: Dict[str, Any]
    timestamp: datetime

@dataclass
class BacktestResult:
    """Results from backtesting on real historical data"""
    symbol: str
    strategy_name: str
    total_return: float
    win_rate: float
    max_drawdown: float
    sharpe_ratio: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    timestamp: datetime

class RealMarketDataCollector:
    """Collects REAL market data from Alpaca"""

    def __init__(self):
        self.crypto_client = CryptoHistoricalDataClient()
        self.stream = CryptoDataStream(
            api_key=settings.ALPACA_API_KEY,
            secret_key=settings.ALPACA_SECRET_KEY
        )
        self.symbols = [
            'BTC/USD', 'ETH/USD', 'SOL/USD', 'DOGE/USD', 'ADA/USD',
            'LTC/USD', 'DOT/USD', 'LINK/USD', 'UNI/USD', 'MATIC/USD'
        ]

    async def get_real_time_data(self, symbol: str) -> Dict[str, Any]:
        """Get real-time market data for a symbol"""
        try:
            # Get latest quote
            quote_request = CryptoLatestQuoteRequest(symbol_or_symbols=[symbol])
            quotes = self.crypto_client.get_crypto_latest_quote(quote_request)

            if symbol in quotes:
                quote = quotes[symbol]
                return {
                    'symbol': symbol,
                    'bid_price': float(quote.bid_price),
                    'ask_price': float(quote.ask_price),
                    'bid_size': float(quote.bid_size),
                    'ask_size': float(quote.ask_size),
                    'timestamp': quote.timestamp,
                    'spread': float(quote.ask_price - quote.bid_price)
                }
        except Exception as e:
            logger.error(f"❌ Error getting real-time data for {symbol}: {e}")
            return None

    async def get_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get real historical market data"""
        try:
            start_date = datetime.now() - timedelta(days=days)
            end_date = datetime.now()

            request_params = CryptoBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Hour,
                start=start_date,
                end=end_date
            )

            bars = self.crypto_client.get_crypto_bars(request_params)

            if symbol in bars:
                df = bars[symbol].df
                df.reset_index(inplace=True)
                return df
            else:
                logger.warning(f"⚠️ No historical data for {symbol}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"❌ Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()

class RealTechnicalAnalyzer:
    """Performs REAL technical analysis on actual market data"""

    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.markov_analyzer = MarkovChainAnalyzer()

    def analyze_symbol(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive technical analysis on real data"""
        if df.empty:
            return {}

        try:
            # Calculate real technical indicators
            analysis = {}

            # RSI Analysis
            rsi = self.indicators.calculate_rsi(df['close'])
            current_rsi = rsi.iloc[-1] if not rsi.empty else 50
            analysis['rsi'] = {
                'value': current_rsi,
                'signal': 'oversold' if current_rsi < 30 else 'overbought' if current_rsi > 70 else 'neutral'
            }

            # MACD Analysis
            macd_data = self.indicators.calculate_macd(df['close'])
            if not macd_data.empty:
                current_macd = macd_data['macd'].iloc[-1]
                current_signal = macd_data['signal'].iloc[-1]
                analysis['macd'] = {
                    'macd': current_macd,
                    'signal': current_signal,
                    'histogram': macd_data['histogram'].iloc[-1],
                    'trend': 'bullish' if current_macd > current_signal else 'bearish'
                }

            # Bollinger Bands
            bb_data = self.indicators.calculate_bollinger_bands(df['close'])
            if not bb_data.empty:
                current_price = df['close'].iloc[-1]
                upper_band = bb_data['upper'].iloc[-1]
                lower_band = bb_data['lower'].iloc[-1]
                middle_band = bb_data['middle'].iloc[-1]

                analysis['bollinger_bands'] = {
                    'current_price': current_price,
                    'upper_band': upper_band,
                    'lower_band': lower_band,
                    'middle_band': middle_band,
                    'position': 'upper' if current_price > upper_band else 'lower' if current_price < lower_band else 'middle'
                }

            # Volume analysis
            analysis['volume'] = {
                'current': float(df['volume'].iloc[-1]),
                'average_20': float(df['volume'].rolling(20).mean().iloc[-1]),
                'trend': 'high' if df['volume'].iloc[-1] > df['volume'].rolling(20).mean().iloc[-1] else 'low'
            }

            # Price action
            analysis['price_action'] = {
                'current': float(df['close'].iloc[-1]),
                'change_24h': float((df['close'].iloc[-1] - df['close'].iloc[-24]) / df['close'].iloc[-24] * 100),
                'high_24h': float(df['high'].iloc[-24:].max()),
                'low_24h': float(df['low'].iloc[-24:].min())
            }

            return analysis

        except Exception as e:
            logger.error(f"❌ Technical analysis error for {symbol}: {e}")
            return {}

class RealBacktestEngine:
    """Backtests strategies on REAL historical data"""

    def __init__(self):
        self.data_collector = RealMarketDataCollector()
        self.analyzer = RealTechnicalAnalyzer()

    async def backtest_strategy(self, symbol: str, strategy_name: str, days: int = 90) -> BacktestResult:
        """Backtest a strategy on real historical data"""
        logger.info(f"📊 Backtesting {strategy_name} on {symbol} with {days} days of REAL data")

        # Get real historical data
        df = await self.data_collector.get_historical_data(symbol, days)

        if df.empty:
            logger.error(f"❌ No data for backtesting {symbol}")
            return None

        # Simple RSI mean reversion strategy for demonstration
        trades = []
        position = None
        entry_price = 0

        # Calculate indicators for the entire period
        rsi = self.analyzer.indicators.calculate_rsi(df['close'])

        for i in range(len(df)):
            if i < 14:  # Not enough data for RSI
                continue

            current_price = df['close'].iloc[i]
            current_rsi = rsi.iloc[i]
            timestamp = df['timestamp'].iloc[i]

            # Entry signals
            if position is None:
                if current_rsi < 30:  # Oversold - Buy signal
                    position = 'long'
                    entry_price = current_price
                    trades.append({
                        'type': 'entry',
                        'side': 'buy',
                        'price': current_price,
                        'timestamp': timestamp,
                        'rsi': current_rsi
                    })
                elif current_rsi > 70:  # Overbought - Sell signal
                    position = 'short'
                    entry_price = current_price
                    trades.append({
                        'type': 'entry',
                        'side': 'sell',
                        'price': current_price,
                        'timestamp': timestamp,
                        'rsi': current_rsi
                    })

            # Exit signals
            elif position == 'long' and current_rsi > 50:  # Exit long
                pnl = (current_price - entry_price) / entry_price * 100
                trades.append({
                    'type': 'exit',
                    'side': 'sell',
                    'price': current_price,
                    'timestamp': timestamp,
                    'rsi': current_rsi,
                    'pnl': pnl
                })
                position = None

            elif position == 'short' and current_rsi < 50:  # Exit short
                pnl = (entry_price - current_price) / entry_price * 100
                trades.append({
                    'type': 'exit',
                    'side': 'buy',
                    'price': current_price,
                    'timestamp': timestamp,
                    'rsi': current_rsi,
                    'pnl': pnl
                })
                position = None

        # Calculate performance metrics from real trades
        exit_trades = [t for t in trades if t['type'] == 'exit']

        if not exit_trades:
            return BacktestResult(
                symbol=symbol,
                strategy_name=strategy_name,
                total_return=0.0,
                win_rate=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                avg_win=0.0,
                avg_loss=0.0,
                timestamp=datetime.now()
            )

        pnls = [t['pnl'] for t in exit_trades]
        total_return = sum(pnls)
        winning_trades = len([p for p in pnls if p > 0])
        losing_trades = len([p for p in pnls if p <= 0])
        win_rate = winning_trades / len(pnls) * 100 if pnls else 0

        avg_win = np.mean([p for p in pnls if p > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([p for p in pnls if p <= 0]) if losing_trades > 0 else 0

        # Calculate max drawdown
        cumulative_returns = np.cumsum(pnls)
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = rolling_max - cumulative_returns
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

        # Simple Sharpe ratio approximation
        sharpe_ratio = np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0

        result = BacktestResult(
            symbol=symbol,
            strategy_name=strategy_name,
            total_return=total_return,
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            total_trades=len(pnls),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            timestamp=datetime.now()
        )

        logger.info(f"✅ Backtest complete: {symbol} - Return: {total_return:.2f}%, Win Rate: {win_rate:.1f}%")
        return result

class RealSignalGenerator:
    """Generates trading signals based on REAL data analysis"""

    def __init__(self):
        self.data_collector = RealMarketDataCollector()
        self.analyzer = RealTechnicalAnalyzer()
        self.backtest_engine = RealBacktestEngine()

    async def generate_signal(self, symbol: str) -> Optional[MarketSignal]:
        """Generate a trading signal based on real market analysis"""
        logger.info(f"🔍 Analyzing {symbol} for REAL trading opportunities...")

        # Get real historical data
        df = await self.data_collector.get_historical_data(symbol, 30)
        if df.empty:
            logger.warning(f"⚠️ No data available for {symbol}")
            return None

        # Get real-time market data
        real_time_data = await self.data_collector.get_real_time_data(symbol)
        if not real_time_data:
            logger.warning(f"⚠️ No real-time data for {symbol}")
            return None

        # Perform technical analysis on real data
        technical_analysis = self.analyzer.analyze_symbol(symbol, df)
        if not technical_analysis:
            logger.warning(f"⚠️ Technical analysis failed for {symbol}")
            return None

        # Backtest our strategy before generating signal
        backtest_result = await self.backtest_engine.backtest_strategy(symbol, "RSI_MeanReversion", 90)
        if not backtest_result or backtest_result.win_rate < 55:  # Only trade strategies with >55% win rate
            logger.info(f"⚠️ Strategy not profitable enough for {symbol}: Win Rate {backtest_result.win_rate:.1f}%")
            return None

        # Generate signal based on real analysis
        current_price = real_time_data['ask_price']
        rsi = technical_analysis.get('rsi', {})
        macd = technical_analysis.get('macd', {})
        bb = technical_analysis.get('bollinger_bands', {})

        signal_type = 'HOLD'
        confidence = 0.0
        reasoning = "Neutral market conditions"

        # Real signal logic based on multiple indicators
        buy_signals = 0
        sell_signals = 0

        # RSI signals
        if rsi.get('signal') == 'oversold':
            buy_signals += 2
        elif rsi.get('signal') == 'overbought':
            sell_signals += 2

        # MACD signals
        if macd.get('trend') == 'bullish':
            buy_signals += 1
        elif macd.get('trend') == 'bearish':
            sell_signals += 1

        # Bollinger Band signals
        if bb.get('position') == 'lower':
            buy_signals += 1
        elif bb.get('position') == 'upper':
            sell_signals += 1

        # Volume confirmation
        volume_trend = technical_analysis.get('volume', {}).get('trend', 'low')
        if volume_trend == 'high':
            if buy_signals > sell_signals:
                buy_signals += 1
            elif sell_signals > buy_signals:
                sell_signals += 1

        # Determine final signal
        if buy_signals >= 3 and buy_signals > sell_signals:
            signal_type = 'BUY'
            confidence = min(0.9, 0.5 + (buy_signals - sell_signals) * 0.1)
            reasoning = f"Strong buy signal: RSI={rsi.get('value', 0):.1f}, MACD trend={macd.get('trend', 'unknown')}, Volume={volume_trend}"
        elif sell_signals >= 3 and sell_signals > buy_signals:
            signal_type = 'SELL'
            confidence = min(0.9, 0.5 + (sell_signals - buy_signals) * 0.1)
            reasoning = f"Strong sell signal: RSI={rsi.get('value', 0):.1f}, MACD trend={macd.get('trend', 'unknown')}, Volume={volume_trend}"
        else:
            confidence = 0.3
            reasoning = f"Mixed signals: Buy={buy_signals}, Sell={sell_signals}, awaiting clearer direction"

        # Calculate stop loss and take profit based on volatility
        atr = df['high'].rolling(14).max() - df['low'].rolling(14).min()
        avg_atr = atr.mean()

        if signal_type == 'BUY':
            stop_loss = current_price - (avg_atr * 1.5)
            take_profit = current_price + (avg_atr * 2.0)
        elif signal_type == 'SELL':
            stop_loss = current_price + (avg_atr * 1.5)
            take_profit = current_price - (avg_atr * 2.0)
        else:
            stop_loss = current_price
            take_profit = current_price

        signal = MarketSignal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning,
            technical_analysis=technical_analysis,
            timestamp=datetime.now()
        )

        logger.info(f"📊 Signal generated for {symbol}: {signal_type} (confidence: {confidence:.2f})")
        logger.info(f"💡 Reasoning: {reasoning}")

        return signal

class RealMarketIntelligenceSystem:
    """Main system that coordinates real market intelligence"""

    def __init__(self):
        self.signal_generator = RealSignalGenerator()
        self.trading_client = TradingClient(
            api_key=settings.ALPACA_API_KEY,
            secret_key=settings.ALPACA_SECRET_KEY,
            paper=True
        )

        # Database session
        Session = sessionmaker(bind=engine)
        self.db_session = Session()

        self.active_symbols = [
            'BTC/USD', 'ETH/USD', 'SOL/USD', 'DOGE/USD', 'ADA/USD'
        ]

        self.signals_generated = []
        self.trades_executed = []

    async def analyze_market_opportunities(self):
        """Analyze all symbols for real trading opportunities"""
        logger.info("🔍 ANALYZING REAL MARKET OPPORTUNITIES")
        logger.info("=" * 60)

        for symbol in self.active_symbols:
            try:
                signal = await self.signal_generator.generate_signal(symbol)
                if signal:
                    self.signals_generated.append(signal)

                    # Store signal in database
                    await self.store_signal_in_database(signal)

                    # Execute trade if confidence is high enough
                    if signal.confidence >= 0.7 and signal.signal_type in ['BUY', 'SELL']:
                        await self.execute_real_trade(signal)

                # Don't overwhelm the API
                await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"❌ Error analyzing {symbol}: {e}")

    async def store_signal_in_database(self, signal: MarketSignal):
        """Store the real signal analysis in database for ML learning"""
        try:
            # Create market data entry
            market_data = MarketData(
                symbol=signal.symbol,
                price=signal.entry_price,
                volume=signal.technical_analysis.get('volume', {}).get('current', 0),
                timestamp=signal.timestamp,
                metadata={
                    'signal_type': signal.signal_type,
                    'confidence': signal.confidence,
                    'reasoning': signal.reasoning,
                    'technical_analysis': signal.technical_analysis
                }
            )

            self.db_session.add(market_data)
            self.db_session.commit()

            logger.info(f"💾 Stored signal data for {signal.symbol} in database")

        except Exception as e:
            logger.error(f"❌ Database storage error: {e}")
            self.db_session.rollback()

    async def execute_real_trade(self, signal: MarketSignal):
        """Execute a real trade based on the signal"""
        try:
            logger.info(f"🚀 EXECUTING REAL TRADE based on analysis:")
            logger.info(f"   Symbol: {signal.symbol}")
            logger.info(f"   Signal: {signal.signal_type}")
            logger.info(f"   Confidence: {signal.confidence:.2f}")
            logger.info(f"   Reasoning: {signal.reasoning}")

            # Calculate position size (risk 1% of portfolio)
            account = self.trading_client.get_account()
            portfolio_value = float(account.portfolio_value)
            risk_amount = portfolio_value * 0.01  # 1% risk

            position_size = risk_amount / abs(signal.entry_price - signal.stop_loss)
            notional = position_size * signal.entry_price

            # Create limit order with stop loss and take profit
            side = OrderSide.BUY if signal.signal_type == 'BUY' else OrderSide.SELL

            order_request = LimitOrderRequest(
                symbol=signal.symbol,
                notional=notional,
                side=side,
                limit_price=signal.entry_price,
                time_in_force=TimeInForce.GTC,
                order_class="bracket",
                stop_loss=StopLossRequest(stop_price=signal.stop_loss),
                take_profit=TakeProfitRequest(limit_price=signal.take_profit)
            )

            order = self.trading_client.submit_order(order_request)

            logger.info(f"✅ REAL TRADE EXECUTED:")
            logger.info(f"   Order ID: {order.id}")
            logger.info(f"   Status: {order.status}")
            logger.info(f"   Entry: ${signal.entry_price:.4f}")
            logger.info(f"   Stop Loss: ${signal.stop_loss:.4f}")
            logger.info(f"   Take Profit: ${signal.take_profit:.4f}")

            # Store trade in database
            trade = Trade(
                symbol=signal.symbol,
                side=signal.signal_type.lower(),
                quantity=position_size,
                notional=notional,
                price=signal.entry_price,
                status='submitted',
                strategy_name='RealMarketIntelligence',
                alpaca_order_id=str(order.id),
                metadata={
                    'signal_confidence': signal.confidence,
                    'signal_reasoning': signal.reasoning,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'technical_analysis': signal.technical_analysis
                }
            )

            self.db_session.add(trade)
            self.db_session.commit()

            self.trades_executed.append({
                'signal': signal,
                'order': order,
                'timestamp': datetime.now()
            })

        except Exception as e:
            logger.error(f"❌ Trade execution error: {e}")
            self.db_session.rollback()

    async def run_continuous_analysis(self, duration_minutes: int = 60):
        """Run continuous real market analysis"""
        logger.info("🧠 STARTING REAL MARKET INTELLIGENCE SYSTEM")
        logger.info("=" * 60)
        logger.info("📊 Gathering REAL market data")
        logger.info("🔍 Analyzing REAL technical indicators")
        logger.info("📈 Backtesting on REAL historical data")
        logger.info("💡 Generating data-driven trading ideas")
        logger.info("🎯 Executing validated strategies only")
        logger.info("=" * 60)

        start_time = time.time()
        analysis_count = 0

        while (time.time() - start_time) < (duration_minutes * 60):
            try:
                analysis_count += 1
                logger.info(f"\n🔄 ANALYSIS CYCLE #{analysis_count}")

                # Analyze all markets for real opportunities
                await self.analyze_market_opportunities()

                # Report findings
                logger.info(f"\n📊 CYCLE SUMMARY:")
                logger.info(f"   Signals Generated: {len(self.signals_generated)}")
                logger.info(f"   Trades Executed: {len(self.trades_executed)}")

                if self.signals_generated:
                    latest_signal = self.signals_generated[-1]
                    logger.info(f"   Latest Signal: {latest_signal.symbol} {latest_signal.signal_type} (confidence: {latest_signal.confidence:.2f})")

                # Wait before next analysis cycle
                logger.info("⏳ Waiting 5 minutes before next analysis cycle...")
                await asyncio.sleep(300)  # 5 minutes between cycles

            except Exception as e:
                logger.error(f"❌ Analysis cycle error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error

        logger.info(f"\n🎉 ANALYSIS SESSION COMPLETE!")
        logger.info(f"📊 Total Analysis Cycles: {analysis_count}")
        logger.info(f"📈 Total Signals Generated: {len(self.signals_generated)}")
        logger.info(f"💰 Total Trades Executed: {len(self.trades_executed)}")

async def main():
    """Main function to run the real market intelligence system"""
    intelligence_system = RealMarketIntelligenceSystem()

    # Run continuous analysis for 2 hours
    await intelligence_system.run_continuous_analysis(duration_minutes=120)

if __name__ == "__main__":
    asyncio.run(main())