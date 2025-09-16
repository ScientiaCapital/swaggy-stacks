#!/usr/bin/env python3
"""
🚀 SwaggyStacks Production Trading System
Main entry point for live trading with real-time agent coordination
"""

import asyncio
import signal
import sys
import os
from datetime import datetime
from typing import Optional
import structlog
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent))

from app.core.config import settings
from app.core.database import init_db
from app.trading.trading_manager import TradingManager
from app.trading.alpaca_stream_manager import AlpacaStreamManager
from app.ai.coordination_hub import AgentCoordinationHub
from app.events.trigger_engine import TriggerEngine
from app.monitoring.health_checks import HealthCheckService
from app.monitoring.metrics import PrometheusMetrics

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class ProductionTradingSystem:
    """Main production trading system orchestrator"""

    def __init__(self):
        self.trading_manager: Optional[TradingManager] = None
        self.stream_manager: Optional[AlpacaStreamManager] = None
        self.coordination_hub: Optional[AgentCoordinationHub] = None
        self.trigger_engine: Optional[TriggerEngine] = None
        self.health_service: Optional[HealthCheckService] = None
        self.metrics: Optional[PrometheusMetrics] = None
        self.is_running = False
        self.shutdown_event = asyncio.Event()

    async def initialize(self):
        """Initialize all production components"""
        try:
            logger.info("🚀 Initializing SwaggyStacks Production Trading System")

            # Initialize database
            logger.info("📊 Initializing database...")
            await init_db()

            # Initialize metrics
            logger.info("📈 Setting up Prometheus metrics...")
            self.metrics = PrometheusMetrics()
            await self.metrics.initialize()

            # Initialize trading manager
            logger.info("💼 Initializing Trading Manager...")
            self.trading_manager = TradingManager()
            await self.trading_manager.initialize()

            # Verify account and positions
            account = await self.trading_manager.get_account()
            logger.info(f"💰 Account Status: ${account.equity:,.2f} | Buying Power: ${account.buying_power:,.2f}")

            # Initialize stream manager
            logger.info("📡 Setting up Alpaca streaming...")
            self.stream_manager = AlpacaStreamManager(
                api_key=settings.ALPACA_API_KEY,
                secret_key=settings.ALPACA_SECRET_KEY,
                paper=settings.TRADING_PAPER_MODE,
                data_feed=settings.ALPACA_DATA_FEED
            )
            await self.stream_manager.initialize()

            # Initialize agent coordination hub
            logger.info("🤖 Starting Agent Coordination Hub...")
            self.coordination_hub = AgentCoordinationHub()
            await self.coordination_hub.initialize(
                trading_manager=self.trading_manager,
                stream_manager=self.stream_manager
            )

            # Initialize trigger engine
            logger.info("⚡ Activating Trigger Engine...")
            self.trigger_engine = TriggerEngine()
            await self.trigger_engine.initialize()

            # Initialize health checks
            logger.info("🏥 Setting up health monitoring...")
            self.health_service = HealthCheckService()
            await self.health_service.start()

            # Connect components
            await self._connect_components()

            logger.info("✅ All systems initialized successfully!")

        except Exception as e:
            logger.error(f"❌ Failed to initialize: {e}", exc_info=True)
            raise

    async def _connect_components(self):
        """Connect all components together"""
        # Subscribe to market data
        symbols = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "META", "GOOGL"]

        logger.info(f"📊 Subscribing to symbols: {symbols}")
        await self.stream_manager.connect()

        # Subscribe with agent callbacks
        await self.stream_manager.subscribe_all_data(
            symbols=symbols,
            callbacks={
                'trade': self.coordination_hub.handle_market_data,
                'quote': self.coordination_hub.handle_market_data,
                'bar': self.coordination_hub.handle_market_data
            }
        )

        # Connect trigger engine to coordination hub
        self.trigger_engine.register_callback(self.coordination_hub.handle_trigger_event)

        # Start trigger monitoring
        await self.trigger_engine.start_monitoring()

    async def pre_market_validation(self) -> bool:
        """Run pre-market validation checks"""
        logger.info("🔍 Running pre-market validation...")

        checks = {
            "database": False,
            "alpaca_api": False,
            "streaming": False,
            "agents": False,
            "risk_limits": False
        }

        try:
            # Check database
            from app.core.database import SessionLocal
            db = SessionLocal()
            db.execute("SELECT 1")
            db.close()
            checks["database"] = True
            logger.info("✅ Database connection OK")

            # Check Alpaca API
            account = await self.trading_manager.get_account()
            if account:
                checks["alpaca_api"] = True
                logger.info(f"✅ Alpaca API OK - Account: ${account.equity:,.2f}")

            # Check streaming
            health = await self.stream_manager.get_connection_health()
            if health['healthy']:
                checks["streaming"] = True
                logger.info("✅ Streaming connection OK")

            # Check agents
            agent_status = await self.coordination_hub.get_all_agent_status()
            active_agents = sum(1 for a in agent_status.values() if a.get('status') == 'ACTIVE')
            if active_agents > 0:
                checks["agents"] = True
                logger.info(f"✅ {active_agents} agents active")

            # Check risk limits
            risk_check = await self.trading_manager.validate_risk_limits()
            if risk_check:
                checks["risk_limits"] = True
                logger.info("✅ Risk limits configured")

            # Summary
            passed = sum(checks.values())
            total = len(checks)

            if passed == total:
                logger.info(f"🎉 Pre-market validation PASSED ({passed}/{total})")
                return True
            else:
                logger.warning(f"⚠️ Pre-market validation PARTIAL ({passed}/{total})")
                for check, status in checks.items():
                    if not status:
                        logger.error(f"❌ Failed: {check}")
                return False

        except Exception as e:
            logger.error(f"❌ Pre-market validation failed: {e}")
            return False

    async def run(self):
        """Main production run loop"""
        self.is_running = True

        logger.info("🏁 Starting production trading system...")

        # Run pre-market validation
        if not await self.pre_market_validation():
            logger.error("❌ Pre-market validation failed. Aborting startup.")
            return

        # Start agent heartbeats
        asyncio.create_task(self.coordination_hub.start_heartbeat())

        # Main loop
        logger.info("💹 Trading system is now LIVE!")
        logger.info("Press Ctrl+C to shutdown gracefully")

        try:
            while self.is_running:
                # Check system health
                health = await self.health_service.get_system_health()

                if health['status'] != 'healthy':
                    logger.warning(f"⚠️ System health degraded: {health}")

                # Update metrics
                if self.metrics:
                    await self.metrics.update_system_metrics(health)

                # Wait or check for shutdown
                try:
                    await asyncio.wait_for(self.shutdown_event.wait(), timeout=30)
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    continue  # Continue running

        except KeyboardInterrupt:
            logger.info("⚠️ Shutdown signal received")
        except Exception as e:
            logger.error(f"❌ Critical error in main loop: {e}", exc_info=True)
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Initiating graceful shutdown...")

        self.is_running = False
        self.shutdown_event.set()

        # Stop components in order
        if self.trigger_engine:
            await self.trigger_engine.stop()

        if self.coordination_hub:
            await self.coordination_hub.shutdown()

        if self.stream_manager:
            await self.stream_manager.disconnect()

        if self.trading_manager:
            await self.trading_manager.shutdown()

        if self.health_service:
            await self.health_service.stop()

        logger.info("✅ Shutdown complete")


def handle_signals():
    """Setup signal handlers for graceful shutdown"""
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, lambda s, f: asyncio.create_task(shutdown_handler()))


async def shutdown_handler():
    """Handle shutdown signals"""
    logger.info("Shutdown signal received")
    # Signal the main system to shutdown
    if 'system' in globals():
        await system.shutdown()


async def main():
    """Main entry point"""
    global system

    print("""
╔══════════════════════════════════════════════════════╗
║     🚀 SwaggyStacks Production Trading System 🚀     ║
║                                                      ║
║     AI-Powered Algorithmic Trading Platform         ║
║     Version 1.0.0 | Production Mode                 ║
╚══════════════════════════════════════════════════════╝
    """)

    # Verify environment
    if not settings.ALPACA_API_KEY:
        logger.error("❌ ALPACA_API_KEY not configured!")
        sys.exit(1)

    mode = "PAPER" if settings.TRADING_PAPER_MODE else "LIVE"
    logger.info(f"🎯 Trading Mode: {mode}")

    # Create and run system
    system = ProductionTradingSystem()

    try:
        await system.initialize()
        await system.run()
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        await system.shutdown()


if __name__ == "__main__":
    # Setup signal handlers
    handle_signals()

    # Run the system
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"❌ Unhandled exception: {e}", exc_info=True)
        sys.exit(1)