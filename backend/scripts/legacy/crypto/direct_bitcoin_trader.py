#!/usr/bin/env python3
"""
🚀 DIRECT BITCOIN TRADER - No Price Check Required
This bypasses price checking and directly executes Bitcoin trades
"""

import time
import sys
import os

# Add the backend directory to Python path
sys.path.append('/Users/tmkipper/repos/swaggy-stacks/backend')

import alpaca_trade_api as tradeapi
from app.core.config import settings

class DirectBitcoinTrader:
    def __init__(self):
        """Initialize with direct trading approach"""
        print("🚀 INITIALIZING DIRECT BITCOIN TRADER...")

        # Initialize Alpaca client
        self.api = tradeapi.REST(
            settings.ALPACA_API_KEY,
            settings.ALPACA_SECRET_KEY,
            base_url=settings.ALPACA_BASE_URL,
            api_version="v2"
        )
        print(f"✅ Connected to Alpaca Paper Trading")

    def check_crypto_assets(self):
        """Check available crypto assets"""
        try:
            print("\n🔍 Checking available crypto assets...")
            assets = self.api.list_assets(asset_class='crypto', status='active')

            print(f"📊 Found {len(assets)} crypto assets:")
            available_symbols = []

            for asset in assets[:10]:  # Show first 10
                if hasattr(asset, 'tradable') and asset.tradable:
                    available_symbols.append(asset.symbol)
                    print(f"   ✅ {asset.symbol} - Tradable: {asset.tradable}")

            return available_symbols
        except Exception as e:
            print(f"❌ Error checking crypto assets: {e}")
            return []

    def execute_direct_bitcoin_trade(self, symbol="BTC/USD", side="buy", notional=10.00):
        """Execute direct Bitcoin trade without price check"""
        try:
            print(f"\n🚀 EXECUTING DIRECT BITCOIN TRADE:")
            print(f"   Symbol: {symbol}")
            print(f"   Side: {side.upper()}")
            print(f"   Notional: ${notional}")
            print(f"   Bypassing price check - going direct to trade")

            # Submit the order directly
            order = self.api.submit_order(
                symbol=symbol,
                notional=notional,
                side=side,
                type='market',
                time_in_force='gtc'
            )

            print(f"\n✅ *** DIRECT BITCOIN ORDER EXECUTED ***")
            print(f"   Order ID: {order.id}")
            print(f"   Symbol: {order.symbol}")
            print(f"   Side: {order.side}")
            print(f"   Notional: ${order.notional}")
            print(f"   Status: {order.status}")
            print(f"\n🔔 *** CHECK YOUR ALPACA ACCOUNT NOW! ***")
            print(f"🌐 https://app.alpaca.markets")

            return order

        except Exception as e:
            print(f"❌ Trading error: {e}")
            print(f"   Error details: {str(e)}")

            # Try alternate symbol formats
            if "/" in symbol:
                alt_symbol = symbol.replace("/", "")
                print(f"\n🔄 Trying alternate format: {alt_symbol}")
                return self.execute_direct_bitcoin_trade(alt_symbol, side, notional)
            elif "/" not in symbol:
                alt_symbol = symbol[:3] + "/" + symbol[3:]
                print(f"\n🔄 Trying alternate format: {alt_symbol}")
                return self.execute_direct_bitcoin_trade(alt_symbol, side, notional)

            return None

    def run_direct_trading(self):
        """Run direct Bitcoin trading"""
        print("🪙 DIRECT BITCOIN TRADER")
        print("=" * 60)
        print("⚠️  Direct trade execution - no price checks")
        print("📊 Paper trading mode")
        print("🎯 Will try multiple Bitcoin symbol formats")
        print("=" * 60)

        # Check account
        try:
            account = self.api.get_account()
            print(f"\n💰 Account Status: {account.status}")
            print(f"💰 Portfolio Value: ${float(account.portfolio_value):,.2f}")
            crypto_status = getattr(account, 'crypto_status', 'N/A')
            print(f"🪙 Crypto Status: {crypto_status}")
        except Exception as e:
            print(f"❌ Account error: {e}")

        # Check available crypto assets
        available_cryptos = self.check_crypto_assets()

        # Try different Bitcoin symbols
        bitcoin_symbols = ["BTC/USD", "BTCUSD", "BTC-USD"]

        if available_cryptos:
            # Use symbols from the available list
            bitcoin_symbols = [s for s in available_cryptos if "BTC" in s]
            print(f"\n🎯 Found Bitcoin symbols: {bitcoin_symbols}")

        if not bitcoin_symbols:
            bitcoin_symbols = ["BTC/USD", "BTCUSD"]  # Fallback

        for symbol in bitcoin_symbols:
            print(f"\n🎯 Attempting trade with symbol: {symbol}")

            result = self.execute_direct_bitcoin_trade(symbol, "buy", 10.00)

            if result:
                print(f"\n✅ SUCCESS! Bitcoin trade executed with {symbol}")
                print(f"📱 Order ID: {result.id}")

                # Wait and sell back
                print(f"\n⏳ Waiting 10 seconds, then selling back...")
                time.sleep(10)

                sell_result = self.execute_direct_bitcoin_trade(symbol, "sell", 10.00)
                if sell_result:
                    print(f"\n✅ SUCCESS! Sell order executed")
                    print(f"📱 Sell Order ID: {sell_result.id}")

                print(f"\n🎉 BITCOIN TRADING COMPLETE!")
                print(f"📊 Check your Alpaca account for 2 orders:")
                print(f"   1. BUY $10 {symbol}")
                print(f"   2. SELL $10 {symbol}")
                return
            else:
                print(f"❌ Failed with {symbol}, trying next...")

        print(f"\n❌ All Bitcoin symbol formats failed")
        print(f"📊 Available crypto symbols were: {available_cryptos[:5]}")

def main():
    """Main function"""
    print("🪙 DIRECT BITCOIN TRADER")
    print("🚀 Will execute Bitcoin trades directly")
    print("📱 No price checks - direct execution")
    print("\nStarting in 2 seconds...")

    time.sleep(2)

    trader = DirectBitcoinTrader()
    trader.run_direct_trading()

if __name__ == "__main__":
    main()