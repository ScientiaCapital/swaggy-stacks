#!/usr/bin/env python3
"""
🚀 SOPHISTICATED BITCOIN TRADER - Limit Orders with Stop-Losses
Advanced trading system using limit orders, stop-losses, and technical analysis
Integrates with our risk management and indicator systems
"""

import asyncio
import time
import sys
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import random
import numpy as np

# Add the backend directory to Python path
sys.path.append('/Users/tmkipper/repos/swaggy-stacks/backend')

# Modern Alpaca SDK imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest, StopLossRequest, TakeProfitRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Database imports
from sqlalchemy.orm import sessionmaker
from app.core.database import engine
from app.models.trade import Trade

# Risk management
from app.trading.risk_manager import RiskManager

# Technical indicators
from app.indicators.technical_indicators import TechnicalIndicators

# Config
from app.core.config import settings

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedCryptoAgent:
    """Advanced Crypto Analysis Agent with Technical Indicators"""

    def __init__(self, name="AdvancedCryptoAgent"):
        self.name = name
        self.technical_indicators = TechnicalIndicators()

    def analyze_bitcoin_advanced(self, current_price: float, price_history: List[float] = None) -> Dict:
        """Advanced Bitcoin analysis with technical indicators"""

        # Generate price history if not provided (for demo)
        if not price_history:
            base_price = current_price
            price_history = [
                base_price + random.uniform(-1000, 1000) for _ in range(50)
            ]

        # Calculate technical indicators
        prices_array = np.array(price_history)

        # RSI Analysis
        rsi = self.technical_indicators.calculate_rsi(prices_array)
        current_rsi = rsi[-1] if len(rsi) > 0 else 50

        # MACD Analysis
        macd_line, signal_line, histogram = self.technical_indicators.calculate_macd(prices_array)
        macd_signal = "BULLISH" if macd_line[-1] > signal_line[-1] else "BEARISH"

        # Bollinger Bands
        upper_band, middle_band, lower_band = self.technical_indicators.calculate_bollinger_bands(prices_array)

        # Support and Resistance
        support = min(price_history[-10:])  # Recent support
        resistance = max(price_history[-10:])  # Recent resistance

        # Generate sophisticated signal
        signal_strength = 0
        reasons = []

        # RSI Analysis
        if current_rsi < 30:
            signal_strength += 0.3
            reasons.append("RSI oversold (bullish)")
        elif current_rsi > 70:
            signal_strength -= 0.3
            reasons.append("RSI overbought (bearish)")

        # MACD Analysis
        if macd_signal == "BULLISH":
            signal_strength += 0.2
            reasons.append("MACD bullish crossover")
        else:
            signal_strength -= 0.2
            reasons.append("MACD bearish signal")

        # Bollinger Band Analysis
        if current_price < lower_band[-1]:
            signal_strength += 0.25
            reasons.append("Price below lower Bollinger Band")
        elif current_price > upper_band[-1]:
            signal_strength -= 0.25
            reasons.append("Price above upper Bollinger Band")

        # Support/Resistance Analysis
        if current_price <= support * 1.02:  # Near support
            signal_strength += 0.15
            reasons.append("Price near support level")
        elif current_price >= resistance * 0.98:  # Near resistance
            signal_strength -= 0.15
            reasons.append("Price near resistance level")

        # Determine final signal
        if signal_strength >= 0.4:
            final_signal = "STRONG_BUY"
        elif signal_strength >= 0.1:
            final_signal = "BUY"
        elif signal_strength <= -0.4:
            final_signal = "STRONG_SELL"
        elif signal_strength <= -0.1:
            final_signal = "SELL"
        else:
            final_signal = "HOLD"

        confidence = min(0.95, 0.5 + abs(signal_strength))

        # Calculate entry prices and levels
        if final_signal in ["STRONG_BUY", "BUY"]:
            # For buy orders, set limit slightly below current price
            entry_price = current_price * 0.999  # 0.1% below market
            stop_loss = support * 0.98  # Stop at 2% below support
            take_profit = resistance * 1.02  # Take profit 2% above resistance
        elif final_signal in ["STRONG_SELL", "SELL"]:
            # For sell orders, set limit slightly above current price
            entry_price = current_price * 1.001  # 0.1% above market
            stop_loss = resistance * 1.02  # Stop at 2% above resistance
            take_profit = support * 0.98  # Take profit 2% below support
        else:
            entry_price = current_price
            stop_loss = current_price * 0.95
            take_profit = current_price * 1.05

        analysis = {
            'signal': final_signal,
            'confidence': round(confidence, 3),
            'signal_strength': round(signal_strength, 3),
            'reasoning': "; ".join(reasons),
            'technical_analysis': {
                'rsi': round(current_rsi, 2),
                'macd_signal': macd_signal,
                'support': round(support, 2),
                'resistance': round(resistance, 2),
                'current_price': round(current_price, 2),
                'bollinger_position': 'middle'  # Simplified
            },
            'trade_levels': {
                'entry_price': round(entry_price, 2),
                'stop_loss': round(stop_loss, 2),
                'take_profit': round(take_profit, 2),
                'risk_reward_ratio': round(abs(take_profit - entry_price) / abs(entry_price - stop_loss), 2) if entry_price != stop_loss else 1.0
            }
        }

        logger.info(f"🔬 {self.name}: {final_signal} @ {confidence:.1%} confidence (RR: {analysis['trade_levels']['risk_reward_ratio']:.1f}:1)")
        return analysis

class SophisticatedBitcoinTrader:
    """Sophisticated Bitcoin Trading System with Limit Orders and Stop-Losses"""

    def __init__(self):
        self.trading_client = None
        self.db_session = None
        self.risk_manager = None
        self.crypto_agent = AdvancedCryptoAgent()
        self.trade_count = 0
        self.setup_system()

    def setup_system(self):
        """Initialize all trading system components"""
        logger.info("🚀 INITIALIZING SOPHISTICATED BITCOIN TRADING SYSTEM")

        # Initialize Alpaca trading client
        try:
            self.trading_client = TradingClient(
                api_key=settings.ALPACA_API_KEY,
                secret_key=settings.ALPACA_SECRET_KEY,
                paper=True
            )
            logger.info("✅ Alpaca trading client connected")
        except Exception as e:
            logger.error(f"❌ Alpaca connection failed: {e}")
            return False

        # Initialize database session
        try:
            Session = sessionmaker(bind=engine)
            self.db_session = Session()
            logger.info("✅ Database connection established")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")

        # Initialize risk manager
        try:
            self.risk_manager = RiskManager(user_id=1)  # Default user
            logger.info("✅ Risk manager initialized")
        except Exception as e:
            logger.error(f"❌ Risk manager initialization failed: {e}")

        return True

    def get_market_data(self) -> Dict:
        """Get comprehensive market data"""
        try:
            account = self.trading_client.get_account()

            # Simulate Bitcoin price with realistic movement
            base_price = 115000
            current_price = base_price + random.uniform(-2000, 2000)

            # Generate price history for technical analysis
            price_history = []
            price = base_price
            for i in range(50):
                price += random.uniform(-500, 500)
                price_history.append(max(110000, min(120000, price)))  # Keep in realistic range

            market_data = {
                'timestamp': datetime.now().isoformat(),
                'btc_price': current_price,
                'price_history': price_history,
                'portfolio_value': float(account.portfolio_value),
                'buying_power': float(account.buying_power),
                'daily_trading_power': float(account.daytrading_buying_power),
                'volatility': random.uniform(0.15, 0.35),  # Crypto volatility
                'volume_24h': random.uniform(1000000, 5000000)  # Daily volume
            }

            return market_data

        except Exception as e:
            logger.error(f"❌ Error getting market data: {e}")
            return {}

    def execute_sophisticated_trade(self, analysis: Dict, market_data: Dict) -> Dict:
        """Execute sophisticated trade with limit orders and stop-losses"""

        signal = analysis['signal']
        if signal == 'HOLD':
            logger.info("🛑 Analysis suggests HOLD - no trade executed")
            return {'status': 'hold', 'reason': 'Technical analysis suggests holding'}

        try:
            # Get trade levels from analysis
            trade_levels = analysis['trade_levels']
            entry_price = trade_levels['entry_price']
            stop_loss_price = trade_levels['stop_loss']
            take_profit_price = trade_levels['take_profit']

            # Determine position side
            side = OrderSide.BUY if signal in ['STRONG_BUY', 'BUY'] else OrderSide.SELL

            # Calculate position size using risk manager
            account_value = market_data['portfolio_value']

            if self.risk_manager:
                position_size_dollars = self.risk_manager.calculate_position_size(
                    symbol="BTC/USD",
                    price=entry_price,
                    account_value=account_value,
                    volatility=market_data.get('volatility'),
                    confidence=analysis['confidence'],
                    stop_loss_price=stop_loss_price,
                    use_optimizer=True
                )
            else:
                # Fallback position sizing (2% of account)
                position_size_dollars = account_value * 0.02

            # Validate order with risk manager
            if self.risk_manager:
                is_valid, validation_reason = self.risk_manager.validate_order(
                    symbol="BTC/USD",
                    quantity=position_size_dollars / entry_price,  # Approximate shares
                    price=entry_price,
                    side=side.value,
                    current_positions=[],  # Simplified for demo
                    account_value=account_value,
                    daily_pnl=0  # Simplified for demo
                )

                if not is_valid:
                    logger.warning(f"🛑 Risk manager rejected order: {validation_reason}")
                    return {'status': 'rejected', 'reason': validation_reason}

            logger.info(f"🚀 EXECUTING SOPHISTICATED {side.value} ORDER")
            logger.info(f"   Entry Price: ${entry_price:,.2f}")
            logger.info(f"   Stop Loss: ${stop_loss_price:,.2f}")
            logger.info(f"   Take Profit: ${take_profit_price:,.2f}")
            logger.info(f"   Position Size: ${position_size_dollars:,.2f}")
            logger.info(f"   Risk/Reward: {trade_levels['risk_reward_ratio']:.1f}:1")

            # Create limit order with stop-loss and take-profit
            order_request = LimitOrderRequest(
                symbol="BTC/USD",
                notional=position_size_dollars,
                side=side,
                limit_price=entry_price,
                time_in_force=TimeInForce.GTC,
                order_class="bracket",  # Bracket order with stop-loss and take-profit
                stop_loss=StopLossRequest(stop_price=stop_loss_price),
                take_profit=TakeProfitRequest(limit_price=take_profit_price)
            )

            # Submit the sophisticated order
            order = self.trading_client.submit_order(order_data=order_request)

            trade_result = {
                'status': 'executed',
                'order_id': order.id,
                'symbol': order.symbol,
                'side': str(order.side),
                'order_type': 'limit_bracket',
                'notional': float(order.notional),
                'limit_price': float(order.limit_price),
                'stop_loss_price': stop_loss_price,
                'take_profit_price': take_profit_price,
                'order_status': str(order.status),
                'submitted_at': order.submitted_at.isoformat() if order.submitted_at else None,
                'analysis': analysis,
                'risk_metrics': {
                    'position_size_dollars': position_size_dollars,
                    'risk_per_share': abs(entry_price - stop_loss_price),
                    'reward_per_share': abs(take_profit_price - entry_price),
                    'risk_reward_ratio': trade_levels['risk_reward_ratio']
                }
            }

            logger.info(f"✅ Sophisticated order executed - Order ID: {order.id}")
            logger.info(f"📊 This is a complete bracket order with stop-loss and take-profit")

            # Save to database
            self.save_sophisticated_trade(trade_result)

            self.trade_count += 1
            return trade_result

        except Exception as e:
            logger.error(f"❌ Sophisticated trade execution failed: {e}")
            return {'status': 'failed', 'error': str(e)}

    def save_sophisticated_trade(self, trade_result: Dict):
        """Save sophisticated trade to database"""

        if not self.db_session:
            logger.warning("⚠️ No database session - skipping save")
            return

        try:
            # Create trade record with sophisticated data
            trade = Trade(
                symbol=trade_result['symbol'],
                side=trade_result['side'],
                quantity=0,  # Will be filled when order completes
                notional=trade_result['notional'],
                status=trade_result['order_status'],
                order_id=trade_result['order_id'],
                strategy_name="Sophisticated_Bitcoin_Trader",
                trade_type="crypto_bracket",
                metadata=json.dumps({
                    'analysis': trade_result['analysis'],
                    'risk_metrics': trade_result['risk_metrics'],
                    'order_type': trade_result['order_type'],
                    'limit_price': trade_result['limit_price'],
                    'stop_loss_price': trade_result['stop_loss_price'],
                    'take_profit_price': trade_result['take_profit_price']
                })
            )

            self.db_session.add(trade)
            self.db_session.commit()

            logger.info(f"💾 Sophisticated trade saved to database - ID: {trade.id}")

        except Exception as e:
            logger.error(f"❌ Database save failed: {e}")
            if self.db_session:
                self.db_session.rollback()

    def run_sophisticated_trading_session(self, duration_minutes=8, cycle_interval=90):
        """Run sophisticated Bitcoin trading session"""

        logger.info("🎯 STARTING SOPHISTICATED BITCOIN TRADING SESSION")
        logger.info("=" * 80)
        logger.info("📊 Advanced technical analysis with limit orders")
        logger.info("🛡️ Risk management with position sizing")
        logger.info("🎚️ Bracket orders with stop-loss and take-profit")
        logger.info("💾 Full database integration for learning")
        logger.info("=" * 80)

        start_time = time.time()

        while (time.time() - start_time) < (duration_minutes * 60):
            try:
                elapsed = int(time.time() - start_time)
                logger.info(f"\n📊 Sophisticated Analysis Cycle {self.trade_count + 1} - Elapsed: {elapsed}s")

                # Get comprehensive market data
                market_data = self.get_market_data()
                if not market_data:
                    logger.warning("⚠️ No market data - skipping cycle")
                    time.sleep(30)
                    continue

                current_price = market_data['btc_price']
                logger.info(f"💰 BTC Price: ${current_price:,.2f}")
                logger.info(f"💸 Buying Power: ${market_data['buying_power']:,.2f}")
                logger.info(f"📈 Volatility: {market_data['volatility']:.1%}")

                # Get sophisticated analysis
                analysis = self.crypto_agent.analyze_bitcoin_advanced(
                    current_price,
                    market_data['price_history']
                )

                # Display analysis results
                logger.info(f"🔬 Technical Analysis: {analysis['signal']} ({analysis['confidence']:.1%})")
                logger.info(f"📝 Reasoning: {analysis['reasoning']}")
                logger.info(f"📊 RSI: {analysis['technical_analysis']['rsi']}")
                logger.info(f"📈 MACD: {analysis['technical_analysis']['macd_signal']}")
                logger.info(f"🎯 Entry: ${analysis['trade_levels']['entry_price']:,.2f}")
                logger.info(f"🛑 Stop Loss: ${analysis['trade_levels']['stop_loss']:,.2f}")
                logger.info(f"🎉 Take Profit: ${analysis['trade_levels']['take_profit']:,.2f}")

                # Execute sophisticated trade
                trade_result = self.execute_sophisticated_trade(analysis, market_data)

                if trade_result['status'] == 'executed':
                    logger.info(f"🎉 SOPHISTICATED TRADE #{self.trade_count} EXECUTED!")
                    logger.info("🔔 CHECK YOUR ALPACA ACCOUNT FOR BRACKET ORDER!")
                    logger.info("📊 Order includes automatic stop-loss and take-profit")
                    logger.info("💾 Trade analysis saved to database for ML learning")
                elif trade_result['status'] == 'rejected':
                    logger.info(f"🛑 Trade rejected by risk management: {trade_result['reason']}")
                elif trade_result['status'] == 'hold':
                    logger.info(f"⏸️ Technical analysis suggests holding position")

                logger.info(f"⏳ Waiting {cycle_interval} seconds before next analysis...")
                time.sleep(cycle_interval)

            except Exception as e:
                logger.error(f"❌ Trading cycle error: {e}")
                time.sleep(30)

        logger.info(f"\n🎉 SOPHISTICATED TRADING SESSION COMPLETE!")
        logger.info(f"📊 Total sophisticated trades: {self.trade_count}")
        logger.info(f"🎚️ All trades used bracket orders with stop-loss/take-profit")
        logger.info(f"💾 Technical analysis data saved for machine learning")
        logger.info("📱 Check your Alpaca dashboard for sophisticated orders!")

        # Generate risk summary
        if self.risk_manager:
            try:
                account = self.trading_client.get_account()
                risk_summary = self.risk_manager.get_risk_summary(
                    positions=[],
                    account_value=float(account.portfolio_value),
                    daily_pnl=0
                )
                logger.info(f"🛡️ Risk Summary: {json.dumps(risk_summary, indent=2)}")
            except Exception as e:
                logger.error(f"❌ Risk summary error: {e}")

        # Close database session
        if self.db_session:
            self.db_session.close()

def main():
    """Main function"""
    print("🎯 SOPHISTICATED BITCOIN TRADER")
    print("📊 Advanced technical analysis with indicators")
    print("🎚️ Limit orders with stop-loss and take-profit")
    print("🛡️ Integrated risk management")
    print("💾 Database storage for machine learning")
    print("🔒 Paper trading mode")
    print("📱 Sophisticated orders in your Alpaca account")
    print("\nStarting in 3 seconds...")

    time.sleep(3)

    trader = SophisticatedBitcoinTrader()
    trader.run_sophisticated_trading_session(duration_minutes=6, cycle_interval=75)

if __name__ == "__main__":
    main()