#!/usr/bin/env python3
"""
Test script to get stock prices from Friday using Alpaca API
"""

import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_friday_stock_data():
    """Test getting stock data from last Friday"""
    try:
        import alpaca_trade_api as tradeapi
        
        # Get API credentials from environment
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        
        if not api_key:
            print("❌ Error: ALPACA_API_KEY must be set in .env file")
            print("📝 Please edit .env file and add your Alpaca paper trading credentials")
            return False
            
        if not secret_key or secret_key == 'your-alpaca-secret-key-here':
            print("⚠️  Warning: ALPACA_SECRET_KEY not set, trying with empty secret...")
            secret_key = ""
        
        print(f"🔑 Using API Key: {api_key[:10]}...")
        print(f"🌐 Base URL: {base_url}")
        
        # Initialize Alpaca client
        api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
        
        # Test connection first
        print("\n📊 Testing Alpaca API connection...")
        account = api.get_account()
        print("✅ Successfully connected to Alpaca API!")
        print(f"📈 Account Status: {account.status}")
        print(f"💰 Portfolio Value: ${float(account.portfolio_value):,.2f}")
        
        # Calculate last Friday's date
        today = datetime.now().date()
        days_since_friday = (today.weekday() - 4) % 7  # Friday is weekday 4
        if days_since_friday == 0 and datetime.now().hour < 16:  # If today is Friday before market close
            days_since_friday = 7
        last_friday = today - timedelta(days=days_since_friday)
        
        print(f"\n📅 Getting stock data for {last_friday.strftime('%A, %B %d, %Y')}")
        
        # Test symbols
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        
        print("\n🔍 Fetching stock prices...")
        print("-" * 60)
        
        for symbol in symbols:
            try:
                # Get daily bars for the symbol
                bars = api.get_bars(
                    symbol, 
                    '1Day', 
                    start=last_friday.isoformat(),
                    end=(last_friday + timedelta(days=1)).isoformat(),
                    limit=1
                )
                
                if bars and len(bars) > 0:
                    bar = bars[0]
                    print(f"📈 {symbol:6} | Open: ${bar.o:8.2f} | High: ${bar.h:8.2f} | Low: ${bar.l:8.2f} | Close: ${bar.c:8.2f} | Vol: {bar.v:,}")
                else:
                    print(f"⚠️  {symbol:6} | No data available for {last_friday}")
                    
            except Exception as e:
                print(f"❌ {symbol:6} | Error: {str(e)}")
        
        print("-" * 60)
        
        # Test getting recent historical data (last 5 days)
        print(f"\n📊 Getting 5-day historical data for AAPL...")
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=10)  # Get more days to ensure we have 5 trading days
            
            bars = api.get_bars(
                'AAPL',
                '1Day',
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                limit=5
            )
            
            if bars:
                print(f"✅ Retrieved {len(bars)} days of data:")
                for i, bar in enumerate(bars):
                    bar_date = bar.t.date()
                    print(f"  Day {i+1}: {bar_date} - Close: ${bar.c:.2f}, Volume: {bar.v:,}")
            else:
                print("⚠️  No historical data available")
                
        except Exception as e:
            print(f"❌ Error getting historical data: {str(e)}")
        
        # Test real-time quote (if market is open)
        print(f"\n💹 Getting latest quote for AAPL...")
        try:
            quote = api.get_latest_quote('AAPL')
            if quote:
                print(f"✅ Latest Quote: Bid: ${quote.bid_price:.2f} ({quote.bid_size}), Ask: ${quote.ask_price:.2f} ({quote.ask_size})")
                print(f"📍 Quote Time: {quote.timestamp}")
            else:
                print("⚠️  No real-time quote available")
        except Exception as e:
            print(f"❌ Error getting quote: {str(e)}")
        
        return True
        
    except ImportError:
        print("❌ Error: alpaca-trade-api not installed")
        print("💡 Run: pip install alpaca-trade-api")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\n🔧 Troubleshooting:")
        print("1. Check your API credentials in .env file")
        print("2. Ensure you're using paper trading credentials")
        print("3. Verify your Alpaca account is active")
        print("4. Check your internet connection")
        return False

def check_credentials():
    """Check if credentials are properly set"""
    api_key = os.getenv('ALPACA_API_KEY')
    
    if not api_key or api_key == 'your-alpaca-api-key-here':
        print("⚠️  ALPACA_API_KEY is not set or using default value")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 Swaggy Stacks - Friday Stock Price Test")
    print("=" * 50)
    
    if not check_credentials():
        print("\n📝 To get your Alpaca API credentials:")
        print("1. Sign up at https://alpaca.markets/")
        print("2. Go to Paper Trading section")
        print("3. Generate API Key and Secret")
        print("4. Add them to your .env file")
        sys.exit(1)
    
    print()
    
    if test_friday_stock_data():
        print("\n🎉 Test completed successfully!")
        print("Your Alpaca integration is working and can fetch stock data.")
    else:
        print("\n💥 Test failed. Please check your configuration.")
        sys.exit(1)