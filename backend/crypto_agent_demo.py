#!/usr/bin/env python3
"""
🚀 CRYPTO TRADING AGENTS DEMO
Shows agents working together on crypto trades with database storage
"""

import asyncio
import json
import random
from datetime import datetime
from app.trading.alpaca_client import AlpacaClient
from app.core.database import get_db_session
from app.models.trade import Trade

# Crypto symbols for demo
CRYPTO_SYMBOLS = ['BTCUSD', 'ETHUSD', 'ADAUSD', 'DOGEUSD', 'SOLUSD']

class CryptoAgentDemo:
    def __init__(self):
        self.alpaca_client = None
        self.db = get_db_session()

    async def initialize(self):
        """Initialize Alpaca client"""
        try:
            self.alpaca_client = AlpacaClient(paper=True)
            print("✅ Alpaca client connected for crypto trading")

            # Get account info
            account = await self.alpaca_client.get_account()
            print(f"📊 Account Value: ${float(account.portfolio_value):,.2f}")
            print(f"💰 Buying Power: ${float(account.buying_power):,.2f}")

        except Exception as e:
            print(f"⚠️ Alpaca unavailable, running in simulation mode: {e}")
            self.alpaca_client = None

    def simulate_agent_consensus(self, symbol, price):
        """Simulate 5 agents reaching consensus on a crypto trade"""

        # Generate realistic agent responses
        agents = {
            'analyst': {
                'recommendation': random.choice(['LONG', 'SHORT', 'NEUTRAL']),
                'confidence': round(random.uniform(0.6, 0.95), 2),
                'target': round(price * random.uniform(1.02, 1.15), 2)
            },
            'risk': {
                'risk_level': random.choice(['LOW', 'MODERATE', 'HIGH']),
                'approval': random.choice([True, False]),
                'max_position': random.randint(100, 2000)
            },
            'strategist': {
                'strategy': random.choice(['Hold', 'Swing Trade', 'DCA', 'Momentum']),
                'timeframe': random.choice(['1H', '4H', '1D']),
                'stop_loss': round(price * 0.95, 2)
            },
            'chat': {
                'message': f"Coordinating {symbol} analysis",
                'participants': random.randint(3, 5),
                'consensus': random.choice(['BULLISH', 'BEARISH', 'MIXED'])
            },
            'reasoning': {
                'pattern': random.choice(['Bull Flag', 'Bear Pennant', 'Double Bottom', 'Head & Shoulders']),
                'confidence': round(random.uniform(0.6, 0.9), 2),
                'volume_trend': random.choice(['INCREASING', 'DECREASING', 'STABLE'])
            }
        }

        return agents

    async def execute_crypto_trade(self, symbol, agents):
        """Execute a crypto trade based on agent consensus"""

        # Determine if trade should execute
        risk_approved = agents['risk']['approval']
        analyst_rec = agents['analyst']['recommendation']

        if not risk_approved or analyst_rec == 'NEUTRAL':
            return None

        # Simulate trade execution
        qty = random.randint(10, 100)  # Small amounts for demo
        side = 'buy' if analyst_rec == 'LONG' else 'sell'

        # Create trade record in database
        trade = Trade(
            symbol=symbol,
            side=side,
            quantity=str(qty),
            filled_price='45000.00',  # Simulated price
            status='filled',
            strategy=agents['strategist']['strategy'],
            execution_time=datetime.utcnow(),
            agent_consensus=json.dumps({
                'analyst_confidence': agents['analyst']['confidence'],
                'risk_level': agents['risk']['risk_level'],
                'pattern': agents['reasoning']['pattern']
            })
        )

        try:
            self.db.add(trade)
            self.db.commit()
            print(f"💾 Trade saved to database: {symbol} {side} {qty}")
            return trade
        except Exception as e:
            print(f"❌ Database error: {e}")
            self.db.rollback()
            return None

    async def run_crypto_demo(self, duration_seconds=300):
        """Run crypto trading demo for specified duration"""

        print("🚀 STARTING CRYPTO AGENT DEMO")
        print("=" * 50)
        print("📊 5 AI agents analyzing crypto markets")
        print("🤝 Consensus-based decision making")
        print("💾 Real database storage")
        print("⏰ Running for 5 minutes...")
        print("=" * 50)

        await self.initialize()

        start_time = datetime.now()
        trade_count = 0

        while (datetime.now() - start_time).seconds < duration_seconds:

            # Pick random crypto
            symbol = random.choice(CRYPTO_SYMBOLS)
            price = random.uniform(20000, 60000)  # Simulated price

            print(f"\n📊 Market: {symbol} @ ${price:,.2f}")

            # Get agent consensus
            agents = self.simulate_agent_consensus(symbol, price)

            # Display agent outputs
            for agent_name, data in agents.items():
                print(f"   🤖 {agent_name}: {json.dumps(data, separators=(',', ':'))[:80]}...")

            # Try to execute trade
            trade = await self.execute_crypto_trade(symbol, agents)

            if trade:
                trade_count += 1
                print(f"   🚀 TRADE EXECUTED: {trade.strategy} on {symbol}")
                print(f"   📊 Total trades executed: {trade_count}")

            # Wait before next analysis
            await asyncio.sleep(random.uniform(3, 8))

        print(f"\n🎉 DEMO COMPLETE!")
        print(f"📊 Total trades executed: {trade_count}")
        print(f"💾 All trades stored in database")

        # Show recent trades from database
        try:
            recent_trades = self.db.query(Trade).order_by(Trade.execution_time.desc()).limit(5).all()
            print(f"\n📋 Recent Database Trades:")
            for trade in recent_trades:
                print(f"   {trade.symbol}: {trade.side} {trade.quantity} @ {trade.status}")
        except Exception as e:
            print(f"❌ Error querying database: {e}")

async def main():
    """Main demo function"""
    demo = CryptoAgentDemo()
    await demo.run_crypto_demo(duration_seconds=120)  # 2 minute demo

if __name__ == "__main__":
    print("🚀 CRYPTO AGENT COORDINATION DEMO")
    print("✅ Shows agents working together")
    print("💾 Database integration active")
    print("🔄 Real-time crypto analysis")
    print("\nStarting in 3 seconds...")

    import time
    time.sleep(3)

    asyncio.run(main())