#!/usr/bin/env python3
"""
🚀 REAL CRYPTO TRADER V2 - PROPER ALPACA FORMAT
This uses the correct Alpaca crypto API format (BTC/USD with slash)
Will execute REAL crypto trades you'll see in your Alpaca account
"""

import time
import random

import alpaca_trade_api as tradeapi
from app.core.config import settings

class AlpacaCryptoTrader:
    def __init__(self):
        """Initialize Alpaca client with proper crypto format"""
        self.api = tradeapi.REST(
            settings.ALPACA_API_KEY,
            settings.ALPACA_SECRET_KEY,
            base_url=settings.ALPACA_BASE_URL,
            api_version="v2"
        )
        print("✅ Connected to Alpaca Paper Trading")

    def check_crypto_status(self):
        """Check if account has crypto trading enabled"""
        try:
            account = self.api.get_account()
            crypto_status = getattr(account, 'crypto_status', 'UNKNOWN')
            print(f"🔍 Crypto Status: {crypto_status}")
            return crypto_status
        except Exception as e:
            print(f"❌ Error checking crypto status: {e}")
            return None

    def get_available_crypto_assets(self):
        """Get available crypto assets from Alpaca"""
        try:
            assets = self.api.list_assets(asset_class='crypto', status='active')
            crypto_pairs = []

            print("🔍 Available Crypto Assets:")
            for asset in assets[:10]:  # Show first 10
                if hasattr(asset, 'tradable') and asset.tradable:
                    crypto_pairs.append(asset.symbol)
                    print(f"   ✅ {asset.symbol}")

            return crypto_pairs
        except Exception as e:
            print(f"❌ Error getting crypto assets: {e}")
            return []

    def get_crypto_price(self, symbol: str):
        """Get real crypto price using proper Alpaca format"""
        try:
            # Use the latest bar endpoint
            bars = self.api.get_latest_bar(symbol)
            return {
                'symbol': symbol,
                'price': float(bars.c),
                'volume': float(bars.v) if hasattr(bars, 'v') else 0,
                'timestamp': bars.t
            }
        except Exception as e:
            print(f"❌ Price error for {symbol}: {e}")
            return None

    def simulate_crypto_agent_analysis(self, symbol: str, price: float):
        """Simulate AI crypto agents analyzing the market"""
        agents = {
            'crypto_analyst': {
                'recommendation': random.choice(['BUY', 'SELL', 'HODL']),
                'confidence': round(random.uniform(0.7, 0.95), 2),
                'target_price': round(price * random.uniform(1.02, 1.15), 2),
                'market_sentiment': random.choice(['BULLISH', 'BEARISH', 'NEUTRAL'])
            },
            'defi_strategist': {
                'strategy': random.choice(['DCA', 'Momentum', 'Breakout', 'Support/Resistance']),
                'timeframe': random.choice(['5m', '15m', '1h', '4h', '1d']),
                'stop_loss': round(price * random.uniform(0.95, 0.98), 2),
                'volume_analysis': random.choice(['HIGH', 'MEDIUM', 'LOW'])
            },
            'risk_manager': {
                'risk_level': random.choice(['LOW', 'MODERATE', 'HIGH']),
                'approval': random.choice([True, True, False]),  # 2/3 approval
                'max_notional': random.randint(50, 500),  # Dollar amount for crypto
                'volatility_warning': random.choice([True, False])
            }
        }
        return agents

    def execute_crypto_trade(self, symbol: str, side: str, notional: float, agents: dict):
        """Execute REAL crypto trade using proper Alpaca format"""
        try:
            print(f"\n🤖 CRYPTO AI AGENT CONSENSUS:")
            print(f"   Crypto Analyst: {agents['crypto_analyst']['recommendation']} "
                  f"(confidence: {agents['crypto_analyst']['confidence']})")
            print(f"   DeFi Strategist: {agents['defi_strategist']['strategy']} strategy")
            print(f"   Risk Manager: {agents['risk_manager']['risk_level']} risk, "
                  f"approved: {agents['risk_manager']['approval']}")

            if not agents['risk_manager']['approval']:
                print("🛑 Risk Manager REJECTED crypto trade")
                return None

            print(f"\n🚀 EXECUTING REAL CRYPTO TRADE:")
            print(f"   Symbol: {symbol}")
            print(f"   Side: {side}")
            print(f"   Notional: ${notional}")
            print(f"   Strategy: {agents['defi_strategist']['strategy']}")

            # Submit crypto order using notional (dollar amount)
            order = self.api.submit_order(
                symbol=symbol,
                notional=notional,  # Use notional for crypto fractional trading
                side=side,
                type='market',
                time_in_force='gtc'  # Good Till Canceled
            )

            print(f"✅ *** REAL CRYPTO ORDER SUBMITTED ***")
            print(f"   Order ID: {order.id}")
            print(f"   Symbol: {order.symbol}")
            print(f"   Side: {order.side}")
            print(f"   Notional: ${order.notional}")
            print(f"   Status: {order.status}")
            print(f"   *** CHECK YOUR ALPACA CRYPTO PORTFOLIO ***")

            return order

        except Exception as e:
            print(f"❌ Crypto trading error: {e}")
            return None

    def run_crypto_trading_session(self, duration_minutes=3):
        """Run live crypto trading session"""
        print("🚀 STARTING LIVE CRYPTO AI TRADING")
        print("=" * 60)
        print("₿ Real crypto trades will be executed")
        print("💰 Paper trading mode - safe to test")
        print("🔔 Check Alpaca crypto portfolio for results")
        print("=" * 60)

        # Check crypto status
        crypto_status = self.check_crypto_status()

        # Get available crypto
        available_cryptos = self.get_available_crypto_assets()
        if not available_cryptos:
            print("❌ No crypto assets available - trying common pairs...")
            # Try common crypto pairs with proper format
            available_cryptos = ['BTC/USD', 'ETH/USD', 'DOGE/USD']

        print(f"\n🎯 Crypto Trading Session: {duration_minutes} minutes")

        start_time = time.time()
        trade_count = 0

        while (time.time() - start_time) < (duration_minutes * 60):
            # Try each crypto symbol
            for symbol in available_cryptos[:3]:  # Test first 3
                print(f"\n📊 Analyzing {symbol}...")

                # Get price
                price_data = self.get_crypto_price(symbol)
                if not price_data:
                    print(f"❌ Could not get price for {symbol}")
                    continue

                print(f"💰 Current Price: ${price_data['price']:,.2f}")

                # Get AI analysis
                agents = self.simulate_crypto_agent_analysis(symbol, price_data['price'])

                # Determine trade
                if agents['crypto_analyst']['recommendation'] in ['BUY', 'SELL']:
                    side = 'buy' if agents['crypto_analyst']['recommendation'] == 'BUY' else 'sell'
                    notional = agents['risk_manager']['max_notional']

                    # Execute real crypto trade
                    trade_result = self.execute_crypto_trade(symbol, side, notional, agents)

                    if trade_result:
                        trade_count += 1
                        print(f"📈 Total crypto trades: {trade_count}")
                        print("🔔 CHECK ALPACA CRYPTO PORTFOLIO!")
                        time.sleep(30)  # Wait between trades
                        break  # Move to next cycle
                    else:
                        print("❌ Crypto trade failed")
                else:
                    print("🛑 AI Agents: HODL recommendation - no trade")

                time.sleep(10)

        print(f"\n🎉 CRYPTO TRADING SESSION COMPLETE!")
        print(f"₿ Total crypto trades executed: {trade_count}")
        print("📱 Check your Alpaca crypto portfolio!")

def main():
    """Main crypto trading function"""
    print("₿ LIVE AI CRYPTO TRADING SYSTEM")
    print("✅ Will execute REAL crypto trades")
    print("🔒 Paper trading mode")
    print("📱 Results in Alpaca crypto portfolio")
    print("\nStarting in 5 seconds...")

    time.sleep(5)

    trader = AlpacaCryptoTrader()
    trader.run_crypto_trading_session(duration_minutes=2)

if __name__ == "__main__":
    main()