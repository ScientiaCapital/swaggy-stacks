#!/usr/bin/env python3
"""
🔍 CRYPTO SYMBOL EXPLORER - Discover Available Crypto Trading Pairs
Research all available crypto symbols for our trading system
"""

import sys
sys.path.append('/Users/tmkipper/repos/swaggy-stacks/backend')

from alpaca.trading.client import TradingClient
from app.core.config import settings

def explore_crypto_symbols():
    """Explore all available crypto symbols"""

    print("🔍 CRYPTO SYMBOL EXPLORER")
    print("=" * 60)

    try:
        # Initialize client
        trading_client = TradingClient(
            api_key=settings.ALPACA_API_KEY,
            secret_key=settings.ALPACA_SECRET_KEY,
            paper=True
        )
        print("✅ Connected to Alpaca")

        # Get all crypto assets using legacy API
        print("\n📊 Fetching all available crypto assets...")
        import alpaca_trade_api as tradeapi

        legacy_api = tradeapi.REST(
            settings.ALPACA_API_KEY,
            settings.ALPACA_SECRET_KEY,
            base_url=settings.ALPACA_BASE_URL,
            api_version="v2"
        )

        assets = legacy_api.list_assets(asset_class='crypto', status='active')

        print(f"📈 Found {len(assets)} total crypto assets")

        # Filter tradable assets
        tradable_assets = [asset for asset in assets if asset.tradable]
        print(f"✅ {len(tradable_assets)} tradable crypto pairs")

        print("\n🪙 TRADABLE CRYPTO PAIRS:")
        print("-" * 60)

        # Categorize by quote currency
        usd_pairs = []
        usdt_pairs = []
        usdc_pairs = []
        btc_pairs = []
        eth_pairs = []
        other_pairs = []

        for asset in tradable_assets:
            symbol = asset.symbol
            if symbol.endswith('/USD'):
                usd_pairs.append(symbol)
            elif symbol.endswith('/USDT'):
                usdt_pairs.append(symbol)
            elif symbol.endswith('/USDC'):
                usdc_pairs.append(symbol)
            elif symbol.endswith('/BTC'):
                btc_pairs.append(symbol)
            elif symbol.endswith('/ETH'):
                eth_pairs.append(symbol)
            else:
                other_pairs.append(symbol)

        # Display by category
        if usd_pairs:
            print(f"\n💵 USD PAIRS ({len(usd_pairs)}):")
            for i, symbol in enumerate(sorted(usd_pairs), 1):
                print(f"  {i:2d}. {symbol}")

        if usdt_pairs:
            print(f"\n🏦 USDT PAIRS ({len(usdt_pairs)}):")
            for i, symbol in enumerate(sorted(usdt_pairs), 1):
                print(f"  {i:2d}. {symbol}")

        if usdc_pairs:
            print(f"\n💰 USDC PAIRS ({len(usdc_pairs)}):")
            for i, symbol in enumerate(sorted(usdc_pairs), 1):
                print(f"  {i:2d}. {symbol}")

        if btc_pairs:
            print(f"\n₿ BTC PAIRS ({len(btc_pairs)}):")
            for i, symbol in enumerate(sorted(btc_pairs), 1):
                print(f"  {i:2d}. {symbol}")

        if eth_pairs:
            print(f"\n🔷 ETH PAIRS ({len(eth_pairs)}):")
            for i, symbol in enumerate(sorted(eth_pairs), 1):
                print(f"  {i:2d}. {symbol}")

        if other_pairs:
            print(f"\n🔄 OTHER PAIRS ({len(other_pairs)}):")
            for i, symbol in enumerate(sorted(other_pairs), 1):
                print(f"  {i:2d}. {symbol}")

        # Recommend top symbols for trading
        top_crypto_symbols = [
            "BTC/USD", "ETH/USD", "ADA/USD", "DOGE/USD", "SOL/USD",
            "AVAX/USD", "MATIC/USD", "DOT/USD", "LTC/USD", "LINK/USD"
        ]

        available_top_symbols = [symbol for symbol in top_crypto_symbols if symbol in [asset.symbol for asset in tradable_assets]]

        print(f"\n🎯 RECOMMENDED SYMBOLS FOR TRADING ({len(available_top_symbols)}):")
        print("-" * 60)
        for i, symbol in enumerate(available_top_symbols, 1):
            print(f"  {i:2d}. {symbol} ✅")

        # Check for any missing recommended symbols
        missing_symbols = [symbol for symbol in top_crypto_symbols if symbol not in available_top_symbols]
        if missing_symbols:
            print(f"\n⚠️ MISSING RECOMMENDED SYMBOLS ({len(missing_symbols)}):")
            for symbol in missing_symbols:
                print(f"     ❌ {symbol}")

        print(f"\n📊 SUMMARY:")
        print(f"   Total Assets: {len(assets)}")
        print(f"   Tradable: {len(tradable_assets)}")
        print(f"   USD Pairs: {len(usd_pairs)}")
        print(f"   USDT Pairs: {len(usdt_pairs)}")
        print(f"   USDC Pairs: {len(usdc_pairs)}")
        print(f"   BTC Pairs: {len(btc_pairs)}")
        print(f"   ETH Pairs: {len(eth_pairs)}")
        print(f"   Other Pairs: {len(other_pairs)}")
        print(f"   Recommended Available: {len(available_top_symbols)}")

        print(f"\n✅ Crypto symbol exploration complete!")
        return available_top_symbols

    except Exception as e:
        print(f"❌ Error exploring crypto symbols: {e}")
        return []

if __name__ == "__main__":
    symbols = explore_crypto_symbols()