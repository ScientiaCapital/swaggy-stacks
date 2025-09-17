#!/usr/bin/env python3
"""Execute one Bitcoin trade immediately"""

import sys
sys.path.append('/Users/tmkipper/repos/swaggy-stacks/backend')

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from app.core.config import settings

print("🚀 INSTANT BITCOIN TRADE")
print("=" * 50)

try:
    # Initialize client
    trading_client = TradingClient(
        api_key=settings.ALPACA_API_KEY,
        secret_key=settings.ALPACA_SECRET_KEY,
        paper=True
    )
    print("✅ Connected to Alpaca")

    # Show account
    account = trading_client.get_account()
    print(f"💰 Portfolio: ${float(account.portfolio_value):,.2f}")
    print(f"💸 Buying Power: ${float(account.buying_power):,.2f}")

    # Execute Bitcoin buy order
    print(f"\n🚀 EXECUTING BITCOIN BUY ORDER...")

    market_order = MarketOrderRequest(
        symbol="BTC/USD",
        notional=25.00,  # $25 order
        side=OrderSide.BUY,
        time_in_force=TimeInForce.GTC
    )

    order = trading_client.submit_order(order_data=market_order)

    print(f"\n✅ *** BITCOIN ORDER EXECUTED ***")
    print(f"   Order ID: {order.id}")
    print(f"   Symbol: {order.symbol}")
    print(f"   Side: {order.side}")
    print(f"   Notional: ${order.notional}")
    print(f"   Status: {order.status}")
    print(f"\n🔔 CHECK YOUR ALPACA ACCOUNT!")
    print(f"🌐 https://app.alpaca.markets")

except Exception as e:
    print(f"❌ Error: {e}")
    print(f"   Type: {type(e).__name__}")