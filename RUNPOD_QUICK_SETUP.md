# Quick RunPod Setup for Trading Agents

## Your Pod Information
- **Pod ID**: dbbn61wiaes6ru
- **Status**: RUNNING
- **Cost**: $0.22/hour

## Access Your Pod

Go to RunPod dashboard and click on your pod to access the web terminal, or use:
```bash
./runpodctl exec dbbn61wiaes6ru /bin/bash
```

## Copy & Paste These Commands in RunPod Terminal

### Step 1: Install Dependencies
```bash
apt-get update && apt-get install -y git curl wget vim
pip install alpaca-trade-api pandas numpy yfinance websockets aiohttp
```

### Step 2: Create Trading Agent Script
```bash
cd /workspace
cat > trading_agents.py << 'EOF'
import os
import asyncio
import logging
from datetime import datetime
import alpaca_trade_api as tradeapi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingAgent:
    def __init__(self):
        # Using your Alpaca paper trading credentials
        self.api = tradeapi.REST(
            'PKKR3YKNLL4OGDSNVWE1',
            '2JkMtMLIcWlRhNcuSCOBPfun9FTofY0c3tVejZ5n',
            'https://paper-api.alpaca.markets',
            api_version='v2'
        )
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

    async def run(self):
        logger.info(f"🚀 Trading Agents Started at {datetime.now()}")

        # Check account
        try:
            account = self.api.get_account()
            logger.info(f"✅ Account Status: {account.status}")
            logger.info(f"💰 Buying Power: ${account.buying_power}")
        except Exception as e:
            logger.error(f"❌ Account error: {e}")
            return

        # Monitor loop
        while True:
            try:
                logger.info(f"\n📊 Market Check - {datetime.now().strftime('%H:%M:%S')}")

                for symbol in self.symbols:
                    try:
                        quote = self.api.get_latest_quote(symbol)
                        trade = self.api.get_latest_trade(symbol)
                        logger.info(f"{symbol}: ${trade.price:.2f} (Bid: ${quote.bid_price:.2f}, Ask: ${quote.ask_price:.2f})")
                    except Exception as e:
                        logger.error(f"Error with {symbol}: {e}")

                # Check positions
                positions = self.api.list_positions()
                if positions:
                    logger.info("\n📈 Current Positions:")
                    for pos in positions:
                        logger.info(f"  {pos.symbol}: {pos.qty} shares @ ${pos.avg_entry_price}")

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Loop error: {e}")
                await asyncio.sleep(10)

if __name__ == "__main__":
    agent = TradingAgent()
    asyncio.run(agent.run())
EOF
```

### Step 3: Run the Trading Agents
```bash
# Run in foreground to see output
python3 trading_agents.py

# Or run in background
nohup python3 trading_agents.py > agents.log 2>&1 &

# Check logs
tail -f agents.log
```

## Monitoring Your Agents

### Check if running
```bash
ps aux | grep trading_agents
```

### View logs
```bash
tail -f agents.log
```

### Stop agents
```bash
pkill -f trading_agents.py
```

## Access from Outside

Since port 8000 is exposed, if you run a web API:
- Access at: `https://dbbn61wiaes6ru-8000.proxy.runpod.net`

## Stop the Pod (to save money)

From your local machine:
```bash
./runpodctl stop pod dbbn61wiaes6ru
```

## Delete the Pod (when done)
```bash
./runpodctl delete pod dbbn61wiaes6ru
```

## Notes
- The agents are using your Alpaca PAPER trading account (not real money)
- Markets are open Monday-Friday, 9:30 AM - 4:00 PM Eastern Time
- The pod costs $0.22/hour while running - remember to stop it!