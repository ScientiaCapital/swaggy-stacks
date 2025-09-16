#!/usr/bin/env python3
"""
Minimal test script to check if basic trading components work
"""

import os
import sys

# Add backend to path
sys.path.insert(0, 'backend')

print("Testing Swaggy Stacks Trading System")
print("=" * 50)

# Test 1: Check if we can import core modules
print("\n1. Testing core imports...")
try:
    from app.core.config import settings
    print("✓ Config loaded")
except ImportError as e:
    print(f"✗ Config import failed: {e}")
    sys.exit(1)

# Test 2: Check Alpaca credentials
print("\n2. Checking Alpaca API credentials...")
if settings.ALPACA_API_KEY and settings.ALPACA_SECRET_KEY:
    print(f"✓ Alpaca API key configured (starts with {settings.ALPACA_API_KEY[:4]}...)")
    print(f"✓ Using base URL: {settings.ALPACA_BASE_URL}")
else:
    print("✗ Alpaca credentials not set in environment")
    print("  Please set ALPACA_API_KEY and ALPACA_SECRET_KEY")

# Test 3: Try to import and initialize Alpaca client
print("\n3. Testing Alpaca client...")
try:
    from app.trading.alpaca_client import AlpacaClient
    client = AlpacaClient()
    print("✓ Alpaca client initialized")

    # Try to get account info
    account = client.get_account()
    if account:
        print(f"✓ Connected to Alpaca account")
        print(f"  - Account status: {account.get('status', 'Unknown')}")
        print(f"  - Buying power: ${float(account.get('buying_power', 0)):,.2f}")
        print(f"  - Portfolio value: ${float(account.get('portfolio_value', 0)):,.2f}")
except Exception as e:
    print(f"✗ Alpaca client error: {e}")

# Test 4: Check if we can get market data
print("\n4. Testing market data...")
try:
    symbols = ["AAPL", "MSFT", "GOOGL"]
    for symbol in symbols:
        try:
            latest_trade = client.get_latest_trade(symbol)
            if latest_trade:
                print(f"✓ {symbol}: ${latest_trade.get('price', 0):.2f}")
        except:
            print(f"✗ Could not get data for {symbol}")
except Exception as e:
    print(f"✗ Market data error: {e}")

# Test 5: Check database connection
print("\n5. Testing database connection...")
try:
    from app.core.database import engine
    with engine.connect() as conn:
        result = conn.execute("SELECT 1")
        print("✓ Database connection successful")
except Exception as e:
    print(f"✗ Database error: {e}")

print("\n" + "=" * 50)
print("Basic tests completed!")
print("\nIf all tests passed, the core trading system is functional.")
print("You can now work on fixing the more complex import issues.")