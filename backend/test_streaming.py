#!/usr/bin/env python3
"""
Test script for Alpaca streaming infrastructure
"""

import asyncio
import sys
import os

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from app.trading.alpaca_stream_manager import AlpacaStreamManager, get_stream_manager
from app.core.config import settings

async def test_stream_callbacks():
    """Test streaming with custom callbacks"""

    async def trade_callback(trade):
        print(f"📈 TRADE: {trade.symbol} @ ${trade.price} (size: {trade.size})")

    async def quote_callback(quote):
        print(f"💰 QUOTE: {quote.symbol} bid: ${quote.bid_price} ask: ${quote.ask_price}")

    async def bar_callback(bar):
        print(f"📊 BAR: {bar.symbol} OHLCV: ${bar.open}/${bar.high}/${bar.low}/${bar.close} vol:{bar.volume}")

    # Test symbols
    test_symbols = ["AAPL", "MSFT", "TSLA"]

    try:
        print("🚀 Starting Alpaca Stream Test...")
        print(f"📊 Data Feed: {settings.ALPACA_DATA_FEED}")
        print(f"📝 Paper Trading: {settings.TRADING_PAPER_MODE}")

        # Get stream manager
        manager = await get_stream_manager()

        # Connect to stream
        print("\n🔌 Connecting to Alpaca stream...")
        await manager.connect()

        # Subscribe to data with callbacks
        print(f"\n📡 Subscribing to data for: {test_symbols}")
        await manager.subscribe_trades(test_symbols, trade_callback)
        await manager.subscribe_quotes(test_symbols, quote_callback)
        await manager.subscribe_bars(test_symbols, bar_callback)

        # Check connection health
        health = await manager.get_connection_health()
        print(f"\n💚 Connection Health: {health}")

        # Stream for 30 seconds
        print("\n⏰ Streaming for 30 seconds...")
        await asyncio.sleep(30)

        # Get buffered data
        print("\n📊 Getting buffered data...")
        for symbol in test_symbols:
            data = await manager.get_buffered_data(symbol, limit=5)
            print(f"\n{symbol} Recent Data:")
            print(f"  Trades: {len(data['trades'])}")
            print(f"  Quotes: {len(data['quotes'])}")
            print(f"  Bars: {len(data['bars'])}")

            if data['trades']:
                latest_trade = data['trades'][-1]
                print(f"  Latest Trade: ${latest_trade['price']} at {latest_trade['timestamp']}")

        print("\n✅ Test completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Disconnect
        if 'manager' in locals():
            await manager.disconnect()
            print("\n🔌 Disconnected from stream")


async def test_crypto_streaming():
    """Test crypto streaming functionality"""

    async def crypto_callback(data):
        if hasattr(data, 'symbol'):
            print(f"🪙 CRYPTO: {data.symbol} @ ${data.price if hasattr(data, 'price') else 'N/A'}")

    from app.trading.crypto_client import CryptoClient

    try:
        print("\n🪙 Testing Crypto Streaming...")

        # Create crypto client
        crypto_client = CryptoClient(paper=True)

        # Test crypto symbols
        crypto_symbols = ["BTC/USD", "ETH/USD", "DOGE/USD"]

        # Start crypto streaming
        result = await crypto_client.stream_crypto_data(
            symbols=crypto_symbols,
            data_types=["trades", "quotes"],
            callback=crypto_callback
        )

        print(f"✅ Crypto streaming result: {result}")

    except Exception as e:
        print(f"❌ Crypto streaming test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🔥 Alpaca Streaming Infrastructure Test")
    print("=" * 50)

    # Run tests
    asyncio.run(test_stream_callbacks())
    asyncio.run(test_crypto_streaming())