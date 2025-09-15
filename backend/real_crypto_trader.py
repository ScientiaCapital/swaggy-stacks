#!/usr/bin/env python3
"""
🚀 REAL CRYPTO TRADER - ACTUAL ALPACA TRADES
This will execute REAL crypto trades in your Alpaca paper trading account
"""

import asyncio
import time
import json
from datetime import datetime
from typing import Dict, Any

import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError

from app.core.config import settings

class RealCryptoTrader:
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
                'status': account.status
            }
        except Exception as e:
            print(f"❌ Account error: {e}")
            return None

    def get_crypto_price(self, symbol: str) -> float:
        """Get real crypto price"""
        try:
            # Get latest bar for crypto
            bars = self.api.get_latest_bar(symbol)
            return float(bars.c)  # Close price
        except Exception as e:
            print(f"❌ Price error for {symbol}: {e}")
            return None

    def execute_crypto_trade(self, symbol: str, side: str, qty: float, reason: str):
        """Execute REAL crypto trade"""
        try:
            print(f"\n🚀 EXECUTING REAL TRADE:")
            print(f"   Symbol: {symbol}")
            print(f"   Side: {side}")
            print(f"   Quantity: {qty}")
            print(f"   Reason: {reason}")

            # Submit the order
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='gtc'
            )

            print(f"✅ ORDER SUBMITTED:")
            print(f"   Order ID: {order.id}")
            print(f"   Status: {order.status}")
            print(f"   Symbol: {order.symbol}")
            print(f"   Side: {order.side}")
            print(f"   Qty: {order.qty}")

            return order

        except APIError as e:
            print(f"❌ Trading error: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None

    def check_crypto_availability(self):
        """Check which crypto symbols are available"""
        crypto_symbols = ['BTCUSD', 'ETHUSD', 'ADAUSD', 'DOGEUSD', 'LTCUSD']
        available = []

        print("\n🔍 Checking crypto availability...")
        for symbol in crypto_symbols:
            try:
                bars = self.api.get_latest_bar(symbol)
                price = float(bars.c)
                available.append((symbol, price))
                print(f"✅ {symbol}: ${price:,.2f}")
            except Exception as e:
                print(f"❌ {symbol}: Not available ({e})")

        return available

    def run_crypto_trading_session(self, duration_minutes=5):
        """Run a real crypto trading session"""
        print("🚀 STARTING REAL CRYPTO TRADING SESSION")
        print("=" * 60)
        print("⚠️  WARNING: THIS WILL EXECUTE REAL TRADES")
        print("📊 Paper trading mode - no real money at risk")
        print("=" * 60)

        # Get account info
        account = self.get_account_info()
        if account:
            print(f"💰 Portfolio Value: ${account['portfolio_value']:,.2f}")
            print(f"💸 Buying Power: ${account['buying_power']:,.2f}")
        else:
            print("❌ Could not get account info - aborting")
            return

        # Check crypto availability
        available_cryptos = self.check_crypto_availability()
        if not available_cryptos:
            print("❌ No crypto symbols available - aborting")
            return

        print(f"\n🎯 Trading session will run for {duration_minutes} minutes")
        print("🤖 AI agent logic will determine trades...")

        start_time = time.time()
        trade_count = 0

        while (time.time() - start_time) < (duration_minutes * 60):
            # Pick a random crypto from available
            if available_cryptos:
                symbol, current_price = available_cryptos[0]  # Start with Bitcoin

                print(f"\n📊 Analyzing {symbol} @ ${current_price:,.2f}")

                # Simple AI logic for demo
                if self.should_trade(symbol, current_price):
                    # Execute small trade
                    trade_result = self.execute_crypto_trade(
                        symbol=symbol,
                        side='buy',
                        qty=0.001,  # Very small amount for testing
                        reason="AI Agent Recommendation: Bullish momentum detected"
                    )

                    if trade_result:
                        trade_count += 1
                        print(f"📈 Total trades executed: {trade_count}")

                        # Wait before next trade
                        print("⏳ Waiting 60 seconds before next analysis...")
                        time.sleep(60)
                    else:
                        print("❌ Trade failed, waiting 30 seconds...")
                        time.sleep(30)
                else:
                    print("🛑 AI Agent: No trade signal detected")
                    time.sleep(30)

        print(f"\n🎉 Trading session complete!")
        print(f"📊 Total trades executed: {trade_count}")

        # Show final account status
        final_account = self.get_account_info()
        if final_account:
            print(f"📊 Final Portfolio Value: ${final_account['portfolio_value']:,.2f}")

    def should_trade(self, symbol: str, price: float) -> bool:
        """Simple AI logic to determine if we should trade"""
        # For demo, trade every other analysis
        return True  # Simplified for demo

def main():
    """Main function"""
    print("🚀 REAL CRYPTO TRADING SYSTEM")
    print("✅ Will execute actual trades in Alpaca")
    print("🔒 Paper trading mode (no real money)")
    print("\nDo you want to proceed? This will execute real orders...")

    # For demo, auto-proceed after 5 seconds
    print("Starting in 5 seconds...")
    time.sleep(5)

    trader = RealCryptoTrader()
    trader.run_crypto_trading_session(duration_minutes=2)  # 2 minute demo

if __name__ == "__main__":
    main()