#!/usr/bin/env python3
"""
💰 WORKING REAL TRADING SYSTEM
Actually executes trades and stores them in database - no mock data
Uses the modern bitcoin trader that we know works
"""

import asyncio
import time
import sys
import os
from datetime import datetime
from sqlalchemy.orm import sessionmaker

# Add the backend directory to Python path
sys.path.append('/Users/tmkipper/repos/swaggy-stacks/backend')

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from app.core.config import settings
from app.core.database import engine
from app.models.trade import Trade

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WorkingRealTradingSystem:
    """System that actually works and executes real trades"""

    def __init__(self):
        self.trading_client = TradingClient(
            api_key=settings.ALPACA_API_KEY,
            secret_key=settings.ALPACA_SECRET_KEY,
            paper=True
        )

        # Database session
        Session = sessionmaker(bind=engine)
        self.db_session = Session()

        self.symbol = "BTC/USD"
        self.trades_executed = []

    def get_account_info(self):
        """Get real account information"""
        try:
            account = self.trading_client.get_account()
            return {
                'portfolio_value': float(account.portfolio_value),
                'buying_power': float(account.buying_power),
                'status': account.status
            }
        except Exception as e:
            logger.error(f"❌ Account error: {e}")
            return None

    def execute_real_trade(self, side: str, notional: float, reason: str):
        """Execute a REAL trade and store in database"""
        try:
            logger.info(f"🚀 EXECUTING REAL TRADE:")
            logger.info(f"   Symbol: {self.symbol}")
            logger.info(f"   Side: {side}")
            logger.info(f"   Notional: ${notional}")
            logger.info(f"   Reason: {reason}")

            # Submit the order
            market_order_data = MarketOrderRequest(
                symbol=self.symbol,
                notional=notional,
                side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
                time_in_force=TimeInForce.GTC
            )

            order = self.trading_client.submit_order(market_order_data)

            logger.info(f"✅ REAL ORDER SUBMITTED:")
            logger.info(f"   Order ID: {order.id}")
            logger.info(f"   Status: {order.status}")
            logger.info(f"   Symbol: {order.symbol}")
            logger.info(f"   Side: {order.side}")
            logger.info(f"   Notional: ${order.notional}")

            # Store in database immediately
            trade = Trade(
                symbol=self.symbol,
                side=side,
                quantity=None,  # Will be filled when order completes
                notional=notional,
                price=None,    # Will be filled when order completes
                status='submitted',
                strategy_name='WorkingRealSystem',
                alpaca_order_id=str(order.id),
                metadata={
                    'reason': reason,
                    'order_type': 'market',
                    'time_in_force': 'GTC',
                    'real_execution': True,
                    'system_type': 'authentic_trading'
                }
            )

            self.db_session.add(trade)
            self.db_session.commit()

            logger.info(f"💾 TRADE STORED IN DATABASE:")
            logger.info(f"   Trade ID: {trade.id}")
            logger.info(f"   Stored at: {trade.created_at}")

            self.trades_executed.append({
                'trade': trade,
                'order': order,
                'timestamp': datetime.now()
            })

            return order

        except Exception as e:
            logger.error(f"❌ Trade execution error: {e}")
            self.db_session.rollback()
            return None

    def check_recent_trades_in_db(self):
        """Check what trades are actually in the database"""
        try:
            logger.info("📊 CHECKING DATABASE FOR REAL TRADES:")
            trades = self.db_session.query(Trade).order_by(Trade.created_at.desc()).limit(10).all()

            if trades:
                logger.info(f"   Found {len(trades)} trades in database:")
                for trade in trades:
                    logger.info(f"   {trade.created_at}: {trade.symbol} {trade.side} ${trade.notional} - {trade.status}")
            else:
                logger.info("   No trades found in database yet")

            return trades

        except Exception as e:
            logger.error(f"❌ Database error: {e}")
            return []

    def run_real_trading_session(self, duration_minutes=10):
        """Run a real trading session that actually works"""
        logger.info("💰 STARTING WORKING REAL TRADING SYSTEM")
        logger.info("=" * 60)
        logger.info("🚀 This system ACTUALLY executes trades")
        logger.info("💾 Stores ALL trades in the database")
        logger.info("📊 No mock data - everything is real")
        logger.info("✅ Uses proven modern_bitcoin_trader approach")
        logger.info("=" * 60)

        # Get account info
        account = self.get_account_info()
        if account:
            logger.info(f"💰 Portfolio Value: ${account['portfolio_value']:,.2f}")
            logger.info(f"💸 Buying Power: ${account['buying_power']:,.2f}")
        else:
            logger.error("❌ Could not get account info - aborting")
            return

        # Check existing trades in database
        self.check_recent_trades_in_db()

        start_time = time.time()
        trade_count = 0

        logger.info(f"\n🎯 Trading session will run for {duration_minutes} minutes")
        logger.info("🔥 Making REAL trades every 2 minutes")

        while (time.time() - start_time) < (duration_minutes * 60):
            try:
                trade_count += 1

                # Determine trade side (alternate between buy and sell)
                side = 'buy' if trade_count % 2 == 1 else 'sell'
                notional = 50.0  # $50 trades for testing

                reason = f"Real trading cycle #{trade_count} - authentic execution test"

                # Execute real trade
                order = self.execute_real_trade(side, notional, reason)

                if order:
                    logger.info(f"✅ Trade #{trade_count} executed successfully")

                    # Wait 30 seconds then check database again
                    time.sleep(30)
                    logger.info(f"\n🔍 Checking database after trade #{trade_count}:")
                    self.check_recent_trades_in_db()

                else:
                    logger.error(f"❌ Trade #{trade_count} failed")

                # Wait 2 minutes before next trade
                logger.info(f"⏳ Waiting 2 minutes before next trade...")
                time.sleep(120)

            except Exception as e:
                logger.error(f"❌ Trading cycle error: {e}")
                time.sleep(60)

        logger.info(f"\n🎉 REAL TRADING SESSION COMPLETE!")
        logger.info(f"📊 Total trades attempted: {trade_count}")
        logger.info(f"💰 Total successful trades: {len(self.trades_executed)}")

        # Final database check
        logger.info(f"\n📊 FINAL DATABASE STATUS:")
        final_trades = self.check_recent_trades_in_db()

        # Show final account status
        final_account = self.get_account_info()
        if final_account:
            logger.info(f"📊 Final Portfolio Value: ${final_account['portfolio_value']:,.2f}")

def main():
    """Main function"""
    print("💰 WORKING REAL TRADING SYSTEM")
    print("✅ Actually executes trades")
    print("💾 Stores in database")
    print("📊 No mock data")
    print("\nStarting in 3 seconds...")
    time.sleep(3)

    trader = WorkingRealTradingSystem()
    trader.run_real_trading_session(duration_minutes=10)  # 10 minute session

if __name__ == "__main__":
    main()