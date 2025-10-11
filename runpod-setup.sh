#!/bin/bash

# RunPod Setup Script for Trading Agents
set -e

echo "=== Setting up Swaggy Stacks Trading System on RunPod ==="

# Step 1: Install system dependencies
echo "Installing system dependencies..."
apt-get update
apt-get install -y git curl wget vim build-essential libpq-dev

# Step 2: Create working directory
echo "Creating working directory..."
mkdir -p /workspace
cd /workspace

# Step 3: Clone the repository (using HTTPS)
echo "Cloning repository..."
git clone https://github.com/yourusername/swaggy-stacks.git || echo "Note: Update with your repo URL or use file transfer"
cd swaggy-stacks || cd /workspace

# Step 4: Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install fastapi uvicorn sqlalchemy psycopg2-binary alembic pydantic python-dotenv
pip install alpaca-trade-api pandas numpy yfinance websockets
pip install redis celery httpx aiohttp structlog
pip install prometheus-client

# Step 5: Create minimal trading agents script if not exists
if [ ! -f live_trading_agents.py ]; then
cat > live_trading_agents.py << 'EOF'
import os
import asyncio
import logging
from datetime import datetime
import alpaca_trade_api as tradeapi
from alpaca_trade_api.stream import Stream

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingAgentCoordinator:
    def __init__(self):
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        self.base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

        if not self.api_key or not self.secret_key:
            raise ValueError("Missing Alpaca API credentials")

        self.api = tradeapi.REST(
            self.api_key,
            self.secret_key,
            self.base_url,
            api_version='v2'
        )

        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

    async def start(self):
        logger.info(f"Starting Trading Agent Coordinator at {datetime.now()}")
        logger.info(f"Monitoring symbols: {self.symbols}")
        logger.info(f"Using Alpaca base URL: {self.base_url}")

        # Verify account
        try:
            account = self.api.get_account()
            logger.info(f"Account Status: {account.status}")
            logger.info(f"Buying Power: ${account.buying_power}")
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")

        # Simple monitoring loop
        while True:
            try:
                for symbol in self.symbols:
                    try:
                        quote = self.api.get_latest_quote(symbol)
                        logger.info(f"{symbol}: Bid=${quote.bid_price:.2f} Ask=${quote.ask_price:.2f}")
                    except Exception as e:
                        logger.error(f"Error getting quote for {symbol}: {e}")

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(10)

async def main():
    coordinator = TradingAgentCoordinator()
    await coordinator.start()

if __name__ == "__main__":
    asyncio.run(main())
EOF
fi

# Step 6: Create environment file
echo "Creating environment configuration..."
cat > .env << 'EOF'
# Alpaca API Configuration (Paper Trading)
# IMPORTANT: Get your API keys from https://alpaca.markets/
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_DATA_URL=https://data.alpaca.markets

# System Configuration
ENVIRONMENT=production
DEBUG=false
SECRET_KEY=your-secret-key-here
EOF

echo ""
echo "⚠️  IMPORTANT: Edit .env and add your actual API keys before running!"
echo "   Get Alpaca API keys from: https://alpaca.markets/"
echo ""

# Step 7: Create startup script
echo "Creating startup script..."
cat > start_agents.sh << 'EOF'
#!/bin/bash
export $(cat .env | grep -v '^#' | xargs)
echo "Starting Trading Agents..."
python3 live_trading_agents.py
EOF
chmod +x start_agents.sh

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start the trading agents:"
echo "  ./start_agents.sh"
echo ""
echo "Or run in background:"
echo "  nohup ./start_agents.sh > agents.log 2>&1 &"
echo ""
echo "To check logs:"
echo "  tail -f agents.log"