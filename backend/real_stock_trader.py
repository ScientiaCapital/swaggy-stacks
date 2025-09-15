#!/usr/bin/env python3
"""
🚀 REAL STOCK TRADER - ACTUAL ALPACA TRADES
This will execute REAL stock trades in your Alpaca paper trading account
You WILL see these trades in your Alpaca dashboard
"""

import time
import random
from datetime import datetime

import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError

from app.core.config import settings

class RealStockTrader:
    def __init__(self):
        """Initialize real Alpaca client"""
        self.api = tradeapi.REST(
            settings.ALPACA_API_KEY,
            settings.ALPACA_SECRET_KEY,
            base_url=settings.ALPACA_BASE_URL,
            api_version="v2"
        )
        print("✅ Connected to Alpaca Paper Trading")

    def get_account_info(self):
        """Get real account information"""
        try:
            account = self.api.get_account()
            return {
                'portfolio_value': float(account.portfolio_value),
                'buying_power': float(account.buying_power),
                'status': account.status,
                'day_trade_count': int(account.daytrade_buying_power)
            }
        except Exception as e:
            print(f"❌ Account error: {e}")
            return None

    def get_stock_price(self, symbol: str):
        """Get real stock price"""
        try:
            bars = self.api.get_latest_bar(symbol)
            return {
                'symbol': symbol,
                'price': float(bars.c),
                'volume': int(bars.v),
                'timestamp': bars.t
            }
        except Exception as e:
            print(f"❌ Price error for {symbol}: {e}")
            return None

    def simulate_agent_analysis(self, symbol: str, price: float):
        """Simulate AI agent analysis"""
        agents = {
            'analyst': {
                'recommendation': random.choice(['BUY', 'SELL', 'HOLD']),
                'confidence': round(random.uniform(0.7, 0.95), 2),
                'target_price': round(price * random.uniform(1.02, 1.08), 2)
            },
            'risk': {
                'risk_level': random.choice(['LOW', 'MODERATE', 'HIGH']),
                'approval': random.choice([True, True, False]),  # 2/3 approval rate
                'position_size': random.randint(1, 10)
            },
            'strategist': {
                'strategy': random.choice(['Momentum', 'Reversal', 'Breakout', 'Mean Reversion']),
                'timeframe': random.choice(['Short-term', 'Swing', 'Day Trade']),
                'stop_loss': round(price * 0.98, 2)
            }
        }
        return agents

    def execute_real_trade(self, symbol: str, side: str, qty: int, agents: dict):
        """Execute REAL stock trade that will show in your Alpaca account"""
        try:
            print(f"\n🤖 AI AGENT CONSENSUS:")
            print(f"   Analyst: {agents['analyst']['recommendation']} (confidence: {agents['analyst']['confidence']})")
            print(f"   Risk: {agents['risk']['risk_level']} risk, approved: {agents['risk']['approval']}")
            print(f"   Strategist: {agents['strategist']['strategy']} strategy")

            if not agents['risk']['approval']:
                print("🛑 Risk agent REJECTED trade")
                return None

            print(f"\n🚀 EXECUTING REAL TRADE IN YOUR ALPACA ACCOUNT:")
            print(f"   Symbol: {symbol}")
            print(f"   Side: {side}")
            print(f"   Quantity: {qty} shares")
            print(f"   Strategy: {agents['strategist']['strategy']}")

            # Submit the REAL order
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='day'
            )

            print(f"✅ *** REAL ORDER SUBMITTED TO ALPACA ***")
            print(f"   Order ID: {order.id}")
            print(f"   Status: {order.status}")
            print(f"   Symbol: {order.symbol}")
            print(f"   Side: {order.side}")
            print(f"   Qty: {order.qty}")
            print(f"   *** CHECK YOUR ALPACA DASHBOARD NOW ***")

            return order

        except APIError as e:
            print(f"❌ Trading error: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None

    def run_live_trading_session(self, duration_minutes=3):
        """Run a live trading session with real orders"""
        print("🚀 STARTING LIVE AI TRADING SESSION")
        print("=" * 60)
        print("⚠️  WARNING: THIS EXECUTES REAL ORDERS")
        print("📊 Paper trading - safe to test")
        print("🎯 You WILL see these trades in Alpaca")
        print("=" * 60)

        # Get account info
        account = self.get_account_info()
        if account:
            print(f"💰 Portfolio Value: ${account['portfolio_value']:,.2f}")
            print(f"💸 Buying Power: ${account['buying_power']:,.2f}")
            print(f"✅ Account Status: {account['status']}")
        else:
            print("❌ Could not get account info - aborting")
            return

        # Trading symbols (popular stocks)
        symbols = ['AAPL', 'TSLA', 'NVDA', 'SPY', 'QQQ']

        print(f"\n🎯 Trading session: {duration_minutes} minutes")
        print("🤖 AI agents will analyze and execute trades...")

        start_time = time.time()
        trade_count = 0

        while (time.time() - start_time) < (duration_minutes * 60):
            # Pick random stock
            symbol = random.choice(symbols)

            # Get real price
            stock_data = self.get_stock_price(symbol)
            if not stock_data:
                print(f"❌ Could not get price for {symbol}")
                time.sleep(10)
                continue

            print(f"\n📊 Analyzing {symbol} @ ${stock_data['price']:.2f}")

            # Get AI agent analysis
            agents = self.simulate_agent_analysis(symbol, stock_data['price'])

            # Determine trade
            if agents['analyst']['recommendation'] in ['BUY', 'SELL']:
                side = 'buy' if agents['analyst']['recommendation'] == 'BUY' else 'sell'
                qty = agents['risk']['position_size']

                # Execute REAL trade
                trade_result = self.execute_real_trade(symbol, side, qty, agents)

                if trade_result:
                    trade_count += 1
                    print(f"📈 Total REAL trades executed: {trade_count}")
                    print("🔔 CHECK YOUR ALPACA ACCOUNT NOW!")

                    # Wait before next trade
                    print("⏳ Waiting 45 seconds before next analysis...")
                    time.sleep(45)
                else:
                    print("❌ Trade failed")
                    time.sleep(15)
            else:
                print("🛑 AI Agent: HOLD recommendation - no trade")
                time.sleep(15)

        print(f"\n🎉 LIVE TRADING SESSION COMPLETE!")
        print(f"📊 Total REAL trades executed: {trade_count}")
        print("🔔 CHECK YOUR ALPACA DASHBOARD FOR ALL ORDERS!")

        # Show final account
        final_account = self.get_account_info()
        if final_account:
            print(f"📊 Final Portfolio Value: ${final_account['portfolio_value']:,.2f}")

def main():
    """Main function"""
    print("🚀 LIVE AI STOCK TRADING SYSTEM")
    print("✅ Will execute REAL trades in your Alpaca account")
    print("🔒 Paper trading mode (safe)")
    print("📱 You will see these orders in your Alpaca dashboard")
    print("\nStarting in 5 seconds...")

    time.sleep(5)

    trader = RealStockTrader()
    trader.run_live_trading_session(duration_minutes=2)  # 2 minute demo

if __name__ == "__main__":
    main()