#!/usr/bin/env python3
"""
Test script to verify Alpaca API connection
"""

import os
import sys
from dotenv import load_dotenv

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Load environment variables
load_dotenv()

def test_alpaca_connection():
    """Test Alpaca API connection"""
    try:
        import alpaca_trade_api as tradeapi
        
        # Get API credentials from environment
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        
        if not api_key or not secret_key:
            print("❌ Error: ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env file")
            return False
        
        print(f"🔑 API Key: {api_key[:10]}...")
        print(f"🌐 Base URL: {base_url}")
        
        # Initialize Alpaca client
        api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
        
        # Test connection by getting account info
        print("\n📊 Testing Alpaca API connection...")
        account = api.get_account()
        
        print("✅ Successfully connected to Alpaca API!")
        print(f"📈 Account Status: {account.status}")
        print(f"💰 Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"💵 Cash: ${float(account.cash):,.2f}")
        print(f"📊 Buying Power: ${float(account.buying_power):,.2f}")
        
        # Test market data access
        print("\n📡 Testing market data access...")
        bars = api.get_bars('AAPL', '1Day', limit=1)
        if bars:
            latest_bar = bars[0]
            print(f"✅ Market data access successful!")
            print(f"🍎 AAPL Latest Price: ${latest_bar.c}")
        else:
            print("⚠️  No market data available")
        
        return True
        
    except Exception as e:
        print(f"❌ Error connecting to Alpaca API: {str(e)}")
        print("\n🔧 Troubleshooting:")
        print("1. Check your API credentials in .env file")
        print("2. Ensure you're using paper trading credentials")
        print("3. Verify your Alpaca account is active")
        return False

def test_environment_setup():
    """Test environment configuration"""
    print("🔧 Testing environment configuration...")
    
    required_vars = [
        'ALPACA_API_KEY',
        'ALPACA_SECRET_KEY',
        'ALPACA_BASE_URL'
    ]
    
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
        else:
            print(f"✅ {var}: {'*' * 10}...{value[-4:] if len(value) > 4 else '****'}")
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file")
        return False
    
    print("✅ All required environment variables are set")
    return True

if __name__ == "__main__":
    print("🚀 Swaggy Stacks - Alpaca API Test")
    print("=" * 50)
    
    # Test environment setup
    if not test_environment_setup():
        sys.exit(1)
    
    print()
    
    # Test Alpaca connection
    if test_alpaca_connection():
        print("\n🎉 All tests passed! Your Alpaca integration is ready.")
    else:
        print("\n💥 Tests failed. Please check your configuration.")
        sys.exit(1)
