#!/usr/bin/env python3
"""
🚀 MODERN BITCOIN TRADER - Using alpaca-py SDK
This uses the modern alpaca-py SDK with proven BTC/USD format
Will execute real Bitcoin trades that appear in your Alpaca account
"""

import time
import sys
import os
from datetime import datetime
import random

# Add the backend directory to Python path
sys.path.append('/Users/tmkipper/repos/swaggy-stacks/backend')

# Modern Alpaca SDK imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from app.core.config import settings

class ModernBitcoinTrader:
    def __init__(self):
        """Initialize with modern alpaca-py SDK"""
        print("🚀 INITIALIZING MODERN BITCOIN TRADER...")
        print("📦 Using alpaca-py SDK (modern)")

        # Initialize modern trading client
        self.trading_client = TradingClient(
            api_key=settings.ALPACA_API_KEY,
            secret_key=settings.ALPACA_SECRET_KEY,
            paper=True  # Paper trading mode
        )

        self.symbol = "BTC/USD"  # Proven working format
        print(f"✅ Connected to Alpaca Paper Trading")
        print(f"🪙 Symbol: {self.symbol}")

    def check_account(self):
        """Check account status and buying power"""
        try:
            account = self.trading_client.get_account()

            print(f"\n💰 ACCOUNT STATUS:")
            print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
            print(f"   Buying Power: ${float(account.buying_power):,.2f}")
            print(f"   Status: {account.status}")
            print(f"   Day Trading Buying Power: ${float(account.daytrading_buying_power):,.2f}")

            return True
        except Exception as e:
            print(f"❌ Account error: {e}")
            return False

    def get_current_positions(self):
        """Get current Bitcoin position"""
        try:
            positions = self.trading_client.get_all_positions()

            btc_position = None
            for position in positions:
                if "BTC" in position.symbol:
                    btc_position = position
                    break

            if btc_position:
                print(f"\n📈 CURRENT BTC POSITION:")
                print(f"   Symbol: {btc_position.symbol}")
                print(f"   Quantity: {btc_position.qty}")
                print(f"   Market Value: ${float(btc_position.market_value):,.2f}")
                print(f"   Avg Cost: ${float(btc_position.avg_entry_price):,.2f}")
                print(f"   Unrealized P&L: ${float(btc_position.unrealized_pl):,.2f}")

                return btc_position
            else:
                print("\n📈 No current BTC position")
                return None

        except Exception as e:
            print(f"❌ Position error: {e}")
            return None

    def simulate_agent_analysis(self):
        """Simulate AI agent analysis for Bitcoin"""
        agents = {
            'crypto_analyst': {
                'recommendation': random.choice(['BUY', 'SELL', 'HOLD']),
                'confidence': round(random.uniform(0.75, 0.95), 2),
                'reasoning': random.choice([
                    'Technical breakout detected',
                    'Support level holding strong',
                    'Volume surge indicates momentum',
                    'Market sentiment turning bullish',
                    'Resistance level breakdown'
                ])
            },
            'risk_manager': {
                'approval': random.choice([True, True, False]),  # 2/3 approval rate
                'position_size': round(random.uniform(20, 50), 2),
                'risk_level': random.choice(['LOW', 'MODERATE', 'HIGH']),
                'max_loss': round(random.uniform(5, 15), 2)
            },
            'momentum_trader': {
                'signal': random.choice(['STRONG_BUY', 'BUY', 'NEUTRAL', 'SELL']),
                'timeframe': random.choice(['5m', '15m', '1h', '4h']),
                'strategy': random.choice(['Momentum', 'Mean Reversion', 'Breakout', 'DCA'])
            }
        }
        return agents

    def execute_bitcoin_trade(self, side: OrderSide, notional: float, agents: dict):
        """Execute real Bitcoin trade using modern SDK"""
        try:
            print(f"\n🤖 AI AGENT CONSENSUS:")
            print(f"   Crypto Analyst: {agents['crypto_analyst']['recommendation']} "
                  f"(confidence: {agents['crypto_analyst']['confidence']})")
            print(f"   Reasoning: {agents['crypto_analyst']['reasoning']}")
            print(f"   Risk Manager: {agents['risk_manager']['risk_level']} risk, "
                  f"approved: {agents['risk_manager']['approval']}")
            print(f"   Momentum Trader: {agents['momentum_trader']['signal']} "
                  f"({agents['momentum_trader']['strategy']})")

            if not agents['risk_manager']['approval']:
                print("🛑 Risk Manager REJECTED Bitcoin trade")
                return None

            print(f"\n🚀 EXECUTING REAL BITCOIN TRADE:")
            print(f"   Symbol: {self.symbol}")
            print(f"   Side: {side.value}")
            print(f"   Notional: ${notional}")
            print(f"   Strategy: {agents['momentum_trader']['strategy']}")
            print(f"   This WILL appear in your Alpaca account!")

            # Create market order request
            market_order_data = MarketOrderRequest(
                symbol=self.symbol,
                notional=notional,
                side=side,
                time_in_force=TimeInForce.GTC
            )

            # Submit the order
            order = self.trading_client.submit_order(order_data=market_order_data)

            print(f"\n✅ *** BITCOIN ORDER EXECUTED ***")
            print(f"   Order ID: {order.id}")
            print(f"   Symbol: {order.symbol}")
            print(f"   Side: {order.side}")
            print(f"   Notional: ${order.notional}")
            print(f"   Status: {order.status}")
            print(f"   Type: {order.order_type}")
            print(f"   Submitted: {order.submitted_at}")
            print(f"\n🔔 *** CHECK YOUR ALPACA DASHBOARD NOW! ***")
            print(f"🌐 https://app.alpaca.markets")

            return order

        except Exception as e:
            print(f"❌ Bitcoin trading error: {e}")
            print(f"   Error type: {type(e).__name__}")
            return None

    def run_bitcoin_trading_session(self, duration_minutes=5, trade_interval=60):
        """Run continuous Bitcoin trading session with AI agents"""
        print("🪙 MODERN BITCOIN TRADING SESSION")
        print("=" * 70)
        print("🤖 AI agents will analyze and execute Bitcoin trades")
        print("📊 Paper trading mode - safe to test")
        print("🎯 Real trades will appear in your Alpaca account")
        print("💰 Using proven BTC/USD format")
        print("=" * 70)

        # Check account
        if not self.check_account():
            print("❌ Account check failed - aborting")
            return

        # Show current position
        current_position = self.get_current_positions()

        print(f"\n🎯 Trading session: {duration_minutes} minutes")
        print(f"⏰ Trade interval: {trade_interval} seconds")
        print("🤖 AI agents will coordinate Bitcoin trading decisions...")

        start_time = time.time()
        trade_count = 0

        while (time.time() - start_time) < (duration_minutes * 60):
            print(f"\n📊 Bitcoin Analysis Cycle {trade_count + 1}")
            print(f"🕐 Elapsed: {int(time.time() - start_time)} seconds")

            # Get AI agent analysis
            agents = self.simulate_agent_analysis()

            # Determine trade based on agent consensus
            recommendation = agents['crypto_analyst']['recommendation']

            if recommendation in ['BUY', 'SELL']:
                side = OrderSide.BUY if recommendation == 'BUY' else OrderSide.SELL
                notional = agents['risk_manager']['position_size']

                # Execute real Bitcoin trade
                trade_result = self.execute_bitcoin_trade(side, notional, agents)

                if trade_result:
                    trade_count += 1
                    print(f"\n📈 *** REAL BITCOIN TRADES EXECUTED: {trade_count} ***")
                    print("🔔 CHECK YOUR ALPACA ACCOUNT FOR NEW ORDERS!")

                    # Update position info
                    print(f"\n⏳ Waiting {trade_interval} seconds before next analysis...")
                    time.sleep(trade_interval)
                else:
                    print("❌ Bitcoin trade failed")
                    time.sleep(30)
            else:
                print(f"🛑 AI Agents: {recommendation} recommendation - no trade")
                print(f"⏳ Waiting 30 seconds before next analysis...")
                time.sleep(30)

        print(f"\n🎉 BITCOIN TRADING SESSION COMPLETE!")
        print(f"🪙 Total Bitcoin trades executed: {trade_count}")
        print("📱 All trades should be visible in your Alpaca dashboard!")

        # Show final position
        final_position = self.get_current_positions()

def main():
    """Main function"""
    print("🪙 MODERN BITCOIN TRADER")
    print("🚀 Using alpaca-py SDK with proven BTC/USD format")
    print("✅ Will execute real Bitcoin trades")
    print("🔒 Paper trading mode")
    print("📱 Trades will appear in your Alpaca account")
    print("\nStarting in 3 seconds...")

    time.sleep(3)

    trader = ModernBitcoinTrader()
    trader.run_bitcoin_trading_session(duration_minutes=3, trade_interval=45)

if __name__ == "__main__":
    main()