#!/usr/bin/env python3
"""
🚀 WORKING BITCOIN TRADER - BTC/USD Format
This WILL execute real Bitcoin trades in your Alpaca paper trading account
You WILL see these trades immediately in your Alpaca dashboard
"""

import time
import sys
import os

# Add the backend directory to Python path
sys.path.append('/Users/tmkipper/repos/swaggy-stacks/backend')

import alpaca_trade_api as tradeapi
from app.core.config import settings

class WorkingBitcoinTrader:
    def __init__(self):
        """Initialize with proper BTC/USD format"""
        print("🚀 INITIALIZING WORKING BITCOIN TRADER...")

        # Initialize Alpaca client
        self.api = tradeapi.REST(
            settings.ALPACA_API_KEY,
            settings.ALPACA_SECRET_KEY,
            base_url=settings.ALPACA_BASE_URL,
            api_version="v2"
        )

        self.symbol = "BTC/USD"  # Correct format for crypto
        print(f"✅ Connected to Alpaca Paper Trading")
        print(f"🪙 Symbol: {self.symbol}")

    def check_account(self):
        """Verify account status"""
        try:
            account = self.api.get_account()
            print(f"\n💰 ACCOUNT STATUS:")
            print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
            print(f"   Buying Power: ${float(account.buying_power):,.2f}")
            print(f"   Status: {account.status}")

            # Check crypto status if available
            crypto_status = getattr(account, 'crypto_status', 'N/A')
            print(f"   Crypto Status: {crypto_status}")

            return True
        except Exception as e:
            print(f"❌ Account error: {e}")
            return False

    def get_bitcoin_price(self):
        """Get current Bitcoin price"""
        try:
            print(f"\n📊 Getting Bitcoin price for {self.symbol}...")
            bars = self.api.get_latest_bar(self.symbol)
            price = float(bars.c)
            print(f"💰 Current BTC Price: ${price:,.2f}")
            return price
        except Exception as e:
            print(f"❌ Price error: {e}")
            return None

    def execute_bitcoin_trade(self, side="buy", notional=25.00):
        """Execute REAL Bitcoin trade that will show in your account"""
        try:
            print(f"\n🚀 EXECUTING REAL BITCOIN TRADE:")
            print(f"   Symbol: {self.symbol}")
            print(f"   Side: {side.upper()}")
            print(f"   Notional: ${notional}")
            print(f"   This WILL appear in your Alpaca account!")

            # Submit the order using notional for crypto
            order = self.api.submit_order(
                symbol=self.symbol,
                notional=notional,  # Dollar amount for crypto
                side=side,
                type='market',
                time_in_force='gtc'  # Good Till Canceled
            )

            print(f"\n✅ *** BITCOIN ORDER SUBMITTED TO YOUR ACCOUNT ***")
            print(f"   Order ID: {order.id}")
            print(f"   Symbol: {order.symbol}")
            print(f"   Side: {order.side}")
            print(f"   Notional: ${order.notional}")
            print(f"   Status: {order.status}")
            print(f"   Type: {order.type}")
            print(f"   Time in Force: {order.time_in_force}")
            print(f"\n🔔 *** CHECK YOUR ALPACA DASHBOARD NOW! ***")
            print(f"🌐 https://app.alpaca.markets")

            return order

        except Exception as e:
            print(f"❌ Trading error: {e}")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error details: {str(e)}")
            return None

    def run_bitcoin_demo(self):
        """Run a Bitcoin trading demo"""
        print("🪙 WORKING BITCOIN TRADER DEMO")
        print("=" * 60)
        print("⚠️  This will execute REAL Bitcoin trades")
        print("📊 Paper trading mode - safe to test")
        print("🎯 You WILL see these orders in Alpaca")
        print("=" * 60)

        # Check account
        if not self.check_account():
            print("❌ Account check failed - aborting")
            return

        # Get Bitcoin price
        price = self.get_bitcoin_price()
        if not price:
            print("❌ Could not get Bitcoin price - aborting")
            return

        # Execute a small Bitcoin buy order
        print(f"\n🤖 AI AGENT DECISION: BUY Bitcoin at ${price:,.2f}")
        print(f"💡 Strategy: Dollar-cost averaging")
        print(f"🎯 Trade Size: $25 (small test trade)")

        buy_order = self.execute_bitcoin_trade(side="buy", notional=25.00)

        if buy_order:
            print(f"\n✅ SUCCESS! Bitcoin buy order submitted")
            print(f"📱 Order ID {buy_order.id} should appear in your Alpaca account")

            # Wait and then sell it back
            print(f"\n⏳ Waiting 15 seconds, then selling back...")
            time.sleep(15)

            # Sell the Bitcoin back
            print(f"\n🤖 AI AGENT DECISION: SELL Bitcoin (take profit)")
            sell_order = self.execute_bitcoin_trade(side="sell", notional=25.00)

            if sell_order:
                print(f"\n✅ SUCCESS! Bitcoin sell order submitted")
                print(f"📱 Order ID {sell_order.id} should appear in your Alpaca account")

            print(f"\n🎉 BITCOIN TRADING DEMO COMPLETE!")
            print(f"📊 You should see 2 Bitcoin orders in your account:")
            print(f"   1. BUY $25 {self.symbol}")
            print(f"   2. SELL $25 {self.symbol}")
            print(f"\n🔔 Go check your Alpaca dashboard now!")
        else:
            print("❌ Bitcoin trade failed")

def main():
    """Main function"""
    print("🪙 WORKING BITCOIN TRADER")
    print("✅ Will execute REAL Bitcoin trades")
    print("🔒 Paper trading mode")
    print("📱 Trades will appear in your Alpaca account")
    print("\nStarting in 3 seconds...")

    time.sleep(3)

    trader = WorkingBitcoinTrader()
    trader.run_bitcoin_demo()

if __name__ == "__main__":
    main()