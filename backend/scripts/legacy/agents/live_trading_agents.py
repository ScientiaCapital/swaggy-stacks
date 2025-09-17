#!/usr/bin/env python3
"""
Live Trading Agents with Real-Time Streaming
Implements WebSocket streaming with proper fallback mechanisms
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add backend to path
sys.path.insert(0, 'backend')

import websockets
from alpaca.data import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.trading import TradingClient

from app.core.config import settings
from app.trading.alpaca_client import AlpacaClient
from app.trading.trading_manager import get_trading_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingAgentCoordinator:
    """Coordinates multiple AI trading agents with real-time data streaming"""

    def __init__(self):
        self.trading_client = None
        self.data_stream = None
        self.trading_manager = None
        self.active_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        self.agents = {}
        self.websocket_connected = False
        self.fallback_mode = False

    async def initialize(self):
        """Initialize trading connections and agents"""
        try:
            # Initialize Alpaca clients
            self.trading_client = TradingClient(
                api_key=settings.ALPACA_API_KEY,
                secret_key=settings.ALPACA_SECRET_KEY,
                paper=True  # Using paper trading
            )

            # Initialize data stream with WebSocket
            self.data_stream = StockDataStream(
                api_key=settings.ALPACA_API_KEY,
                secret_key=settings.ALPACA_SECRET_KEY
            )

            # Get trading manager
            self.trading_manager = get_trading_manager()

            # Initialize AI agents
            await self._initialize_agents()

            logger.info("✅ Trading Agent Coordinator initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize coordinator: {e}")
            return False

    async def _initialize_agents(self):
        """Initialize individual trading agents"""
        agent_configs = [
            {"name": "MarkovAgent", "strategy": "markov", "symbols": self.active_symbols},
            {"name": "MomentumAgent", "strategy": "momentum", "symbols": self.active_symbols},
            {"name": "PatternAgent", "strategy": "pattern", "symbols": self.active_symbols},
        ]

        for config in agent_configs:
            self.agents[config["name"]] = {
                "strategy": config["strategy"],
                "symbols": config["symbols"],
                "active": True,
                "positions": {},
                "performance": {"trades": 0, "wins": 0, "losses": 0}
            }
            logger.info(f"Initialized {config['name']} for {len(config['symbols'])} symbols")

    async def start_real_time_streaming(self):
        """Start real-time data streaming with WebSocket fallback"""
        try:
            # Try WebSocket streaming first
            logger.info("🔄 Attempting WebSocket connection for real-time streaming...")

            # Subscribe to trades for our symbols
            for symbol in self.active_symbols:
                self.data_stream.subscribe_trades(self._handle_trade_update, symbol)
                self.data_stream.subscribe_quotes(self._handle_quote_update, symbol)

            # Start the WebSocket connection
            try:
                await self._connect_websocket()
                self.websocket_connected = True
                logger.info("✅ WebSocket streaming connected successfully")

            except Exception as ws_error:
                logger.warning(f"⚠️ WebSocket connection failed: {ws_error}")
                logger.info("🔄 Falling back to polling mode...")
                self.fallback_mode = True
                await self._start_polling_fallback()

        except Exception as e:
            logger.error(f"Failed to start streaming: {e}")
            raise

    async def _connect_websocket(self):
        """Connect to Alpaca WebSocket stream"""
        try:
            # Use websockets.sync for synchronous context management
            import websockets.sync

            logger.info("Connecting to Alpaca WebSocket stream...")
            await self.data_stream.run()

        except ImportError:
            # Fall back to async websockets if sync not available
            logger.info("Using async websockets...")
            await self.data_stream.run()

    async def _start_polling_fallback(self):
        """Fallback to polling mode if WebSocket fails"""
        logger.info("Starting polling fallback mode...")

        while True:
            try:
                for symbol in self.active_symbols:
                    # Get latest price data
                    client = AlpacaClient()
                    latest_trade = client.get_latest_trade(symbol)

                    if latest_trade:
                        # Simulate trade update
                        await self._process_market_data(symbol, latest_trade)

                # Poll every 1 second in fallback mode
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Polling error: {e}")
                await asyncio.sleep(5)  # Wait longer on error

    async def _handle_trade_update(self, trade):
        """Handle incoming trade updates"""
        try:
            symbol = trade.symbol
            price = trade.price
            size = trade.size
            timestamp = trade.timestamp

            logger.info(f"📊 Trade Update: {symbol} @ ${price:.2f} x {size} shares")

            # Process through agents
            await self._process_market_data(symbol, {
                "price": price,
                "volume": size,
                "timestamp": timestamp
            })

        except Exception as e:
            logger.error(f"Error handling trade update: {e}")

    async def _handle_quote_update(self, quote):
        """Handle incoming quote updates"""
        try:
            symbol = quote.symbol
            bid = quote.bid_price
            ask = quote.ask_price
            spread = ask - bid

            logger.debug(f"Quote: {symbol} Bid: ${bid:.2f} Ask: ${ask:.2f} Spread: ${spread:.4f}")

        except Exception as e:
            logger.error(f"Error handling quote update: {e}")

    async def _process_market_data(self, symbol: str, data: Dict):
        """Process market data through all active agents"""
        for agent_name, agent in self.agents.items():
            if agent["active"] and symbol in agent["symbols"]:
                try:
                    # Simulate agent decision making
                    decision = await self._agent_decision(agent_name, symbol, data)

                    if decision:
                        logger.info(f"🤖 {agent_name} signal for {symbol}: {decision}")
                        # Here you would execute the trade through trading manager

                except Exception as e:
                    logger.error(f"Agent {agent_name} error: {e}")

    async def _agent_decision(self, agent_name: str, symbol: str, data: Dict) -> Optional[str]:
        """Simulate agent decision making"""
        # This would integrate with actual AI strategy
        # For now, return a simple simulation
        import random

        if random.random() > 0.95:  # 5% chance of signal
            return random.choice(["BUY", "SELL", "HOLD"])
        return None

    async def monitor_performance(self):
        """Monitor and log agent performance"""
        while True:
            try:
                logger.info("\n📈 === Agent Performance Report ===")
                for agent_name, agent in self.agents.items():
                    perf = agent["performance"]
                    win_rate = (perf["wins"] / perf["trades"] * 100) if perf["trades"] > 0 else 0
                    logger.info(f"{agent_name}: Trades: {perf['trades']}, Win Rate: {win_rate:.1f}%")

                logger.info(f"WebSocket Status: {'Connected' if self.websocket_connected else 'Disconnected'}")
                logger.info(f"Mode: {'Fallback Polling' if self.fallback_mode else 'Real-time Streaming'}")

                await asyncio.sleep(30)  # Report every 30 seconds

            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)

    async def run(self):
        """Main run loop"""
        try:
            # Initialize system
            if not await self.initialize():
                logger.error("Failed to initialize system")
                return

            # Start real-time streaming
            streaming_task = asyncio.create_task(self.start_real_time_streaming())

            # Start performance monitoring
            monitor_task = asyncio.create_task(self.monitor_performance())

            # Keep running
            logger.info("🚀 Trading agents are now live with real-time streaming!")
            logger.info(f"Monitoring symbols: {', '.join(self.active_symbols)}")

            # Wait for tasks
            await asyncio.gather(streaming_task, monitor_task)

        except KeyboardInterrupt:
            logger.info("Shutting down trading agents...")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.data_stream:
                await self.data_stream.close()
            logger.info("Resources cleaned up")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


async def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("🎯 Swaggy Stacks Live Trading Agents")
    logger.info("=" * 60)

    coordinator = TradingAgentCoordinator()
    await coordinator.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Failed to run trading agents: {e}")
        sys.exit(1)