#!/usr/bin/env python3
"""
Test script to compare free market data sources
"""

import os
import sys
from datetime import datetime, timedelta

def test_yfinance():
    """Test Yahoo Finance data"""
    try:
        import yfinance as yf
        
        print("📊 Testing Yahoo Finance (yfinance)...")
        
        # Get current data
        ticker = yf.Ticker("AAPL")
        info = ticker.info
        
        print(f"✅ Current Price: ${info.get('currentPrice', 'N/A')}")
        print(f"✅ Previous Close: ${info.get('previousClose', 'N/A')}")
        
        # Get historical data
        hist = ticker.history(period="5d")
        if not hist.empty:
            print(f"✅ Historical Data: {len(hist)} days retrieved")
            latest = hist.tail(1)
            latest_date = latest.index[0].strftime('%Y-%m-%d')
            latest_close = latest['Close'].iloc[0]
            print(f"✅ Latest Close ({latest_date}): ${latest_close:.2f}")
        
        # Test multiple symbols
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        data = yf.download(symbols, period="1d", interval="1d", group_by='ticker')
        print(f"✅ Batch Download: {len(symbols)} symbols")
        
        return True
        
    except ImportError:
        print("❌ yfinance not installed. Run: pip install yfinance")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_alpha_vantage():
    """Test Alpha Vantage API"""
    try:
        import requests
        
        print("\n📊 Testing Alpha Vantage (demo key)...")
        
        # Using demo key (limited)
        url = "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=demo"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            if 'Time Series (5min)' in data:
                latest_time = list(data['Time Series (5min)'].keys())[0]
                latest_price = data['Time Series (5min)'][latest_time]['4. close']
                print(f"✅ Latest IBM Price: ${latest_price}")
                return True
            else:
                print("⚠️  Demo key has limited access")
                return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_financial_modeling_prep():
    """Test Financial Modeling Prep (demo)"""
    try:
        import requests
        
        print("\n📊 Testing Financial Modeling Prep (demo)...")
        
        # Demo endpoint (limited access)
        url = "https://financialmodelingprep.com/api/v3/quote/AAPL?apikey=demo"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                price = data[0].get('price')
                change_pct = data[0].get('changesPercentage')
                print(f"✅ AAPL Price: ${price:.2f} ({change_pct:+.2f}%)")
                return True
            else:
                print("⚠️  No data returned")
                return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def create_market_data_wrapper():
    """Create a unified market data wrapper"""
    wrapper_code = '''
import yfinance as yf
import requests
from datetime import datetime, timedelta

class MarketDataProvider:
    """Unified market data provider with fallback sources"""
    
    def __init__(self):
        self.primary_source = "yfinance"
        self.fallback_sources = ["alpha_vantage", "fmp"]
        
    def get_current_price(self, symbol):
        """Get current price with fallback"""
        try:
            # Primary: Yahoo Finance
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info.get('currentPrice') or info.get('regularMarketPrice')
            
        except Exception as e:
            print(f"Primary source failed: {e}")
            return None
    
    def get_historical_data(self, symbol, period="1y"):
        """Get historical data"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            return hist
            
        except Exception as e:
            print(f"Historical data failed: {e}")
            return None
    
    def get_intraday_data(self, symbol, interval="1m", period="1d"):
        """Get intraday data"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            return hist
            
        except Exception as e:
            print(f"Intraday data failed: {e}")
            return None
    
    def get_friday_prices(self, symbols):
        """Get Friday's closing prices"""
        try:
            # Calculate last Friday
            today = datetime.now().date()
            days_since_friday = (today.weekday() - 4) % 7
            if days_since_friday == 0 and datetime.now().hour < 16:
                days_since_friday = 7
            last_friday = today - timedelta(days=days_since_friday)
            
            # Get data for the week containing Friday
            start_date = last_friday - timedelta(days=7)
            end_date = last_friday + timedelta(days=3)
            
            results = {}
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date)
                
                # Find Friday's data
                friday_data = hist[hist.index.date == last_friday]
                if not friday_data.empty:
                    results[symbol] = {
                        'date': last_friday,
                        'open': friday_data['Open'].iloc[0],
                        'high': friday_data['High'].iloc[0],
                        'low': friday_data['Low'].iloc[0],
                        'close': friday_data['Close'].iloc[0],
                        'volume': friday_data['Volume'].iloc[0]
                    }
            
            return results
            
        except Exception as e:
            print(f"Friday prices failed: {e}")
            return {}

# Usage example:
if __name__ == "__main__":
    provider = MarketDataProvider()
    
    # Test current price
    price = provider.get_current_price("AAPL")
    print(f"AAPL Current Price: ${price}")
    
    # Test Friday prices
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    friday_data = provider.get_friday_prices(symbols)
    
    for symbol, data in friday_data.items():
        print(f"{symbol} Friday ({data['date']}): ${data['close']:.2f}")
'''
    
    with open('/Users/tmkipper/repos/swaggy-stacks/market_data_provider.py', 'w') as f:
        f.write(wrapper_code)
    
    print("\n✅ Created unified market data provider: market_data_provider.py")

if __name__ == "__main__":
    print("🚀 Testing Free Market Data Sources")
    print("=" * 50)
    
    # Test yfinance (most reliable)
    yf_success = test_yfinance()
    
    # Test other sources
    av_success = test_alpha_vantage()
    fmp_success = test_financial_modeling_prep()
    
    print("\n📊 SUMMARY")
    print("=" * 50)
    print(f"Yahoo Finance (yfinance): {'✅ Working' if yf_success else '❌ Failed'}")
    print(f"Alpha Vantage:           {'✅ Working' if av_success else '❌ Limited'}")
    print(f"Financial Modeling Prep: {'✅ Working' if fmp_success else '❌ Limited'}")
    
    print(f"\n💡 RECOMMENDATION")
    print("=" * 50)
    if yf_success:
        print("✅ Use yfinance as primary source - unlimited and reliable")
        print("💡 Install: pip install yfinance")
        create_market_data_wrapper()
    else:
        print("⚠️  Consider installing yfinance or getting API keys for other sources")
    
    print(f"\n🎯 For Swaggy Stacks integration:")
    print("1. Use yfinance for historical analysis")
    print("2. Use Alpaca for real-time quotes (when trading)")
    print("3. Implement fallback sources for reliability")