#!/usr/bin/env python3
"""Quick test of modern Bitcoin trader"""

import sys
sys.path.append('/Users/tmkipper/repos/swaggy-stacks/backend')

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from app.core.config import settings

print("🧪 TESTING MODERN ALPACA CONNECTION...")

try:
    # Initialize client
    trading_client = TradingClient(
        api_key=settings.ALPACA_API_KEY,
        secret_key=settings.ALPACA_SECRET_KEY,
        paper=True
    )
    print("✅ Trading client initialized")

    # Get account
    account = trading_client.get_account()
    print(f"✅ Account connected: ${float(account.portfolio_value):,.2f}")

    # Get positions
    positions = trading_client.get_all_positions()
    print(f"✅ Found {len(positions)} positions")

    for pos in positions:
        if "BTC" in pos.symbol:
            print(f"🪙 BTC Position: {pos.qty} {pos.symbol} = ${float(pos.market_value):,.2f}")

    print("✅ Modern trader connection working!")

except Exception as e:
    print(f"❌ Error: {e}")
    print(f"   Type: {type(e).__name__}")