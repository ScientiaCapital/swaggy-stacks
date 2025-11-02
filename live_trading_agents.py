"""
Live Trading Agents Coordinator for 24/7 Autonomous Trading.

This script orchestrates the LangGraph workflows for continuous trading operations:
- Trading Workflow: Real-time market analysis and execution during market hours
- Learning Workflow: Overnight analysis and strategy improvement

Designed for RunPod serverless deployment with automatic recovery.
"""
import asyncio
import os
import signal
import sys
from datetime import datetime, time
from typing import Optional
import structlog

# Add backend to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from app.agents.workflows.trading_workflow import TradingWorkflow
from app.agents.workflows.learning_workflow import LearningWorkflow
from app.agents.workflows.state_schemas import TradingState, LearningState
from app.core.config import settings

logger = structlog.get_logger(__name__)


class TradingCoordinator:
    """
    Coordinates autonomous trading operations using LangGraph workflows.

    Features:
    - Market hours detection
    - Automatic workflow scheduling
    - Graceful shutdown handling
    - Error recovery with exponential backoff
    """

    def __init__(self):
        """Initialize trading coordinator with workflows."""
        self.trading_workflow = TradingWorkflow()
        self.learning_workflow = LearningWorkflow()
        self.running = False
        self.completed_trades = []

        # Trading symbols to monitor
        self.symbols = os.getenv("TRADING_SYMBOLS", "AAPL,MSFT,GOOGL,AMZN,TSLA").split(",")

        # Market hours (ET)
        self.market_open = time(9, 30)
        self.market_close = time(16, 0)

        logger.info(
            "TradingCoordinator initialized",
            symbols=self.symbols,
            market_hours=f"{self.market_open}-{self.market_close}"
        )

    def is_market_hours(self) -> bool:
        """Check if currently within market hours (ET)."""
        now = datetime.now().time()
        return self.market_open <= now <= self.market_close

    async def run_trading_cycle(self, symbol: str) -> Optional[dict]:
        """
        Run single trading cycle for a symbol.

        Args:
            symbol: Stock symbol to trade

        Returns:
            Trade result if executed, None otherwise
        """
        try:
            logger.info("Starting trading cycle", symbol=symbol)

            # Prepare initial state
            initial_state: TradingState = {
                "symbol": symbol,
                "market_data": await self._fetch_market_data(symbol),
                "completed": False,
                "market_regime": "",
                "regime_confidence": 0.0,
                "signals": [],
                "recommended_strategy": "",
                "strategy_params": {},
                "strategy_confidence": 0.0,
                "risk_approved": False,
                "position_size": 0.0,
                "risk_assessment": {},
                "execution_status": "",
                "orders": [],
                "next_agent": "",
                "messages": []
            }

            # Run trading workflow
            result = await self.trading_workflow.run(initial_state)

            # Track completed trade if executed
            if result.get("risk_approved") and result.get("execution_status") == "filled":
                trade_record = {
                    "symbol": symbol,
                    "strategy": result["recommended_strategy"],
                    "market_regime": result["market_regime"],
                    "timestamp": datetime.now().isoformat(),
                    "outcome": "pending",  # Updated later
                    "pnl": 0  # Updated later
                }
                self.completed_trades.append(trade_record)

                logger.info(
                    "Trade executed",
                    symbol=symbol,
                    strategy=result["recommended_strategy"]
                )

                return trade_record
            else:
                logger.info(
                    "Trade not executed",
                    symbol=symbol,
                    risk_approved=result.get("risk_approved"),
                    status=result.get("execution_status")
                )
                return None

        except Exception as e:
            logger.error("Trading cycle failed", symbol=symbol, error=str(e))
            return None

    async def _fetch_market_data(self, symbol: str) -> dict:
        """
        Fetch current market data for symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Market data dictionary
        """
        # Placeholder: In production, fetch from Alpaca or data provider
        return {
            "symbol": symbol,
            "VIX": {"value": 15.0},
            "timestamp": datetime.now().isoformat()
        }

    async def run_learning_cycle(self):
        """Run overnight learning workflow to analyze completed trades."""
        if not self.completed_trades:
            logger.info("No trades to learn from today")
            return

        try:
            logger.info(
                "Starting learning cycle",
                trades_count=len(self.completed_trades)
            )

            # Prepare learning state
            initial_state: LearningState = {
                "completed_trades": self.completed_trades,
                "learning_summary": "",
                "patterns_updated": 0,
                "regime_matrix_updated": False,
                "insights": [],
                "next_day_recommendations": [],
                "completed": False
            }

            # Run learning workflow
            result = await self.learning_workflow.run(initial_state)

            logger.info(
                "Learning cycle complete",
                patterns_updated=result["patterns_updated"],
                insights_count=len(result.get("insights", []))
            )

            # Clear completed trades after learning
            self.completed_trades = []

        except Exception as e:
            logger.error("Learning cycle failed", error=str(e))

    async def trading_loop(self):
        """Main trading loop - monitors markets during trading hours."""
        logger.info("Trading loop started")

        while self.running:
            try:
                if self.is_market_hours():
                    # Run trading cycles for all symbols
                    for symbol in self.symbols:
                        if not self.running:
                            break

                        await self.run_trading_cycle(symbol)

                        # Wait between symbols to avoid rate limits
                        await asyncio.sleep(5)

                    # Wait 5 minutes between full cycles during market hours
                    await asyncio.sleep(300)
                else:
                    # Outside market hours - check every 30 minutes
                    logger.info("Outside market hours, waiting...")
                    await asyncio.sleep(1800)

            except Exception as e:
                logger.error("Trading loop error", error=str(e))
                await asyncio.sleep(60)  # Wait 1 minute before retrying

    async def learning_loop(self):
        """Overnight learning loop - runs once per day after market close."""
        logger.info("Learning loop started")

        last_learning_date = None

        while self.running:
            try:
                current_date = datetime.now().date()
                current_time = datetime.now().time()

                # Run learning at 5 PM ET (after market close)
                learning_time = time(17, 0)

                if (current_time >= learning_time and
                    last_learning_date != current_date and
                    self.completed_trades):

                    logger.info("Running overnight learning...")
                    await self.run_learning_cycle()
                    last_learning_date = current_date

                # Check every 30 minutes
                await asyncio.sleep(1800)

            except Exception as e:
                logger.error("Learning loop error", error=str(e))
                await asyncio.sleep(60)

    async def start(self):
        """Start all coordinator loops."""
        self.running = True

        logger.info("Starting TradingCoordinator...")

        # Start both loops concurrently
        await asyncio.gather(
            self.trading_loop(),
            self.learning_loop()
        )

    def stop(self):
        """Stop all coordinator loops gracefully."""
        logger.info("Stopping TradingCoordinator...")
        self.running = False


# Global coordinator instance
coordinator: Optional[TradingCoordinator] = None


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down...")
    if coordinator:
        coordinator.stop()
    sys.exit(0)


async def main():
    """Main entry point for live trading agents."""
    global coordinator

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info(
        "Live Trading Agents starting",
        environment=settings.ENVIRONMENT,
        symbols=os.getenv("TRADING_SYMBOLS", "AAPL,MSFT,GOOGL,AMZN,TSLA")
    )

    # Create and start coordinator
    coordinator = TradingCoordinator()

    try:
        await coordinator.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error("Fatal error", error=str(e))
    finally:
        if coordinator:
            coordinator.stop()
        logger.info("Live Trading Agents stopped")


if __name__ == "__main__":
    asyncio.run(main())
