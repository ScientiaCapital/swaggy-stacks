"""
Trading Manager Singleton
Consolidates all trading operations and client management
Eliminates duplicate AlpacaClient initialization across modules
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Optional

import structlog

from app.ml.markov_system import MarkovSystem
from app.core.config import settings
from app.core.exceptions import RiskManagementError, TradingError
from app.risk.position_manager import IntegratedRiskManager
from app.trading.alpaca_client import AlpacaClient
from app.trading.order_manager import OrderManager
from app.trading.risk_manager import RiskManager

logger = structlog.get_logger()
# Import metrics for monitoring
from app.monitoring.metrics import PrometheusMetrics


class TradingManager:
    """
    Singleton Trading Manager
    Central hub for all trading operations to eliminate redundancy
    """

    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TradingManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Core components
        self._alpaca_client = None
        self._risk_manager = None
        self._integrated_risk_manager = None
        self._order_manager = None
        self._markov_system = None

        # Configuration
        self.paper_trading = getattr(settings, "PAPER_TRADING", True)
        self.max_positions = getattr(settings, "MAX_POSITIONS", 10)
        self.account_size = getattr(settings, "ACCOUNT_SIZE", 100000)

        # State tracking
        self.active_positions = {}
        self.pending_orders = {}
        self.performance_metrics = {}
        self.last_account_update = None

        self._initialized = True
        logger.info("TradingManager singleton initialized")

    async def initialize(self, user_config: Optional[Dict] = None):
        """Initialize all trading components"""
        async with self._lock:
            try:
                # Initialize Alpaca client
                api_key = (
                    user_config.get("api_key")
                    if user_config
                    else settings.ALPACA_API_KEY
                )
                secret_key = (
                    user_config.get("secret_key")
                    if user_config
                    else settings.ALPACA_SECRET_KEY
                )

                self._alpaca_client = AlpacaClient(
                    api_key=api_key, secret_key=secret_key, paper=self.paper_trading
                )

                # Initialize risk manager
                user_id = user_config.get("user_id") if user_config else "system"
                account_info = await self._alpaca_client.get_account()

                self._risk_manager = RiskManager(
                    user_id=user_id,
                    account_balance=float(
                        account_info.get("equity", self.account_size)
                    ),
                    max_daily_loss=getattr(settings, "MAX_DAILY_LOSS", 500),
                    max_position_size=getattr(settings, "MAX_POSITION_SIZE", 10000),
                )

                # Initialize integrated risk manager
                risk_config = {
                    "base_risk": {
                        "max_position_size": getattr(
                            settings, "MAX_POSITION_SIZE", 10000
                        ),
                        "max_daily_loss": getattr(settings, "MAX_DAILY_LOSS", 500),
                    },
                    "position_sizing": {
                        "kelly_max_fraction": 0.25,
                        "fixed_fraction_default": 0.02,
                    },
                    "stop_loss": {
                        "atr_periods": 14,
                        "atr_multiplier": 2.0,
                    },
                    "portfolio_risk": {
                        "var_confidence_level": 0.95,
                        "max_correlation": 0.7,
                        "max_sector_exposure": 0.30,
                    },
                }

                self._integrated_risk_manager = IntegratedRiskManager(
                    user_id=user_id, config=risk_config
                )

                # Initialize order manager
                self._order_manager = OrderManager(
                    alpaca_client=self._alpaca_client, risk_manager=self._risk_manager
                )

                # Initialize Markov system
                self._markov_system = MarkovSystem()

                # Update state
                await self._update_positions()
                await self._update_performance_metrics()

                logger.info(
                    "TradingManager fully initialized",
                    paper_trading=self.paper_trading,
                    max_positions=self.max_positions,
                )

            except Exception as e:
                logger.error("Failed to initialize TradingManager", error=str(e))
                raise TradingError(f"TradingManager initialization failed: {str(e)}")

    @property
    def alpaca_client(self) -> AlpacaClient:
        """Get Alpaca client instance"""
        if not self._alpaca_client:
            raise TradingError(
                "TradingManager not initialized. Call initialize() first."
            )
        return self._alpaca_client

    @property
    def risk_manager(self) -> RiskManager:
        """Get risk manager instance"""
        if not self._risk_manager:
            raise TradingError(
                "TradingManager not initialized. Call initialize() first."
            )
        return self._risk_manager

    @property
    def order_manager(self) -> OrderManager:
        """Get order manager instance"""
        if not self._order_manager:
            raise TradingError(
                "TradingManager not initialized. Call initialize() first."
            )
        return self._order_manager

    @property
    def markov_system(self) -> MarkovSystem:
        """Get Markov analysis system"""
        if not self._markov_system:
            raise TradingError(
                "TradingManager not initialized. Call initialize() first."
            )
        return self._markov_system

    @property
    def integrated_risk_manager(self) -> IntegratedRiskManager:
        """Get integrated risk management system"""
        if not self._integrated_risk_manager:
            raise TradingError(
                "TradingManager not initialized. Call initialize() first."
            )
        return self._integrated_risk_manager

    async def execute_trade(
        self,
        symbol: str,
        action: str,
        quantity: float,
        order_type: str = "market",
        analysis_data: Optional[Dict] = None,
    ) -> Dict:
        """
        Execute a trade with full risk management and analysis integration
        """
        start_time = datetime.now()
        metrics = PrometheusMetrics()

        try:
            # Validate inputs
            if action not in ["BUY", "SELL"]:
                raise TradingError(f"Invalid action: {action}")

            # Get current market data
            current_price = await self.get_current_price(symbol)

            # Risk check timing
            risk_check_start = datetime.now()
            risk_check = await self._risk_manager.check_position_risk(
                symbol=symbol,
                quantity=quantity,
                price=current_price,
                action=action,
                current_positions=self.active_positions,
            )
            risk_check_latency = (datetime.now() - risk_check_start).total_seconds()

            # Record risk check latency
            metrics.record_risk_check_latency(
                risk_check_latency, symbol, action.lower()
            )

            if not risk_check["approved"]:
                # Record failed trade execution
                execution_time = (datetime.now() - start_time).total_seconds()
                metrics.record_trade_execution_metrics(
                    status="rejected",
                    latency=execution_time,
                    symbol=symbol,
                    order_type=order_type,
                    broker="alpaca",
                )
                raise RiskManagementError(f"Trade rejected: {risk_check['reason']}")

            # Apply risk-adjusted quantity
            adjusted_quantity = risk_check.get("adjusted_quantity", quantity)

            # Execute order
            order_result = await self._order_manager.place_order(
                symbol=symbol,
                quantity=adjusted_quantity,
                side=action.lower(),
                order_type=order_type,
            )

            # Record successful execution metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            metrics.record_trade_execution_metrics(
                status="success",
                latency=execution_time,
                symbol=symbol,
                order_type=order_type,
                broker="alpaca",
            )

            # Update tracking
            self.pending_orders[order_result["order_id"]] = {
                "symbol": symbol,
                "quantity": adjusted_quantity,
                "side": action,
                "timestamp": datetime.now(),
                "analysis_data": analysis_data or {},
            }

            # Log trade
            logger.info(
                "Trade executed",
                symbol=symbol,
                action=action,
                quantity=adjusted_quantity,
                order_id=order_result["order_id"],
            )

            return {
                "success": True,
                "order_id": order_result["order_id"],
                "symbol": symbol,
                "quantity": adjusted_quantity,
                "action": action,
                "estimated_price": current_price,
                "risk_metrics": risk_check,
            }

        except Exception as e:
            # Record failed execution
            execution_time = (datetime.now() - start_time).total_seconds()
            metrics.record_trade_execution_metrics(
                status="error",
                latency=execution_time,
                symbol=symbol,
                order_type=order_type,
                broker="alpaca",
            )

            logger.error(
                "Trade execution failed", symbol=symbol, action=action, error=str(e)
            )
            return {
                "success": False,
                "error": str(e),
                "symbol": symbol,
                "action": action,
            }

    async def get_market_analysis(self, symbol: str, timeframe: str = "1D") -> Dict:
        """
        Get comprehensive market analysis for a symbol
        """
        datetime.now()
        metrics = PrometheusMetrics()

        try:
            # Get market data with latency tracking
            market_data_start = datetime.now()
            market_data = await self._alpaca_client.get_market_data(
                symbol=symbol, timeframe=timeframe, limit=100
            )
            market_data_latency = (datetime.now() - market_data_start).total_seconds()
            metrics.record_market_data_latency(market_data_latency, symbol, "alpaca")

            # Perform Markov analysis with timing
            analysis_start = datetime.now()
            analysis = await asyncio.to_thread(self._markov_system.analyze, market_data)
            analysis_latency = (datetime.now() - analysis_start).total_seconds()
            metrics.record_strategy_analysis_latency(analysis_latency, "markov", symbol)

            # Add current position info
            current_position = self.active_positions.get(symbol, {})
            analysis["current_position"] = current_position

            # Add risk assessment
            current_price = market_data["close"].iloc[-1] if len(market_data) > 0 else 0
            risk_assessment = await self._assess_symbol_risk(symbol, current_price)
            analysis["risk_assessment"] = risk_assessment

            return analysis

        except Exception as e:
            logger.error("Market analysis failed", symbol=symbol, error=str(e))
            raise TradingError(f"Market analysis failed for {symbol}: {str(e)}")

    async def get_portfolio_status(self) -> Dict:
        """Get comprehensive portfolio status"""
        try:
            await self._update_positions()
            await self._update_performance_metrics()

            account_info = await self._alpaca_client.get_account()
            metrics = PrometheusMetrics()

            # Calculate portfolio metrics
            total_equity = float(account_info.get("equity", 0))
            float(account_info.get("total_pl", 0))
            daily_pnl = float(account_info.get("day_pl", 0))

            # Calculate portfolio risk metrics
            total_exposure = sum(
                pos.get("market_value", 0) for pos in self.active_positions.values()
            )

            exposure_ratio = (
                total_exposure / self.account_size if self.account_size > 0 else 0
            )
            position_concentration = max(
                (
                    pos.get("market_value", 0) / self.account_size
                    for pos in self.active_positions.values()
                ),
                default=0,
            )

            # Record portfolio risk metrics
            metrics.update_portfolio_risk_metrics(
                total_value=total_equity,
                var_95=abs(daily_pnl) * 1.65,  # Simple VaR approximation
                beta=1.0,  # Would calculate against benchmark
                concentration_risk=position_concentration,
                sector_exposures={},  # Would calculate from position sectors
            )

            portfolio_data = {
                "account": {
                    "equity": total_equity,
                    "cash": float(account_info.get("cash", 0)),
                    "buying_power": float(account_info.get("buying_power", 0)),
                    "day_trade_count": int(account_info.get("day_trade_count", 0)),
                },
                "positions": self.active_positions,
                "pending_orders": len(self.pending_orders),
                "performance": self.performance_metrics,
                "risk_status": await self._get_portfolio_risk_status(),
                "last_updated": datetime.now().isoformat(),
            }

            return portfolio_data

        except Exception as e:
            logger.error("Failed to get portfolio status", error=str(e))
            raise TradingError(f"Portfolio status retrieval failed: {str(e)}")

    async def close_position(
        self, symbol: str, quantity: Optional[float] = None
    ) -> Dict:
        """Close a position (partial or full)"""
        try:
            current_position = self.active_positions.get(symbol)
            if not current_position:
                raise TradingError(f"No active position found for {symbol}")

            # Determine quantity to close
            position_qty = float(current_position.get("quantity", 0))
            close_qty = quantity or abs(position_qty)

            # Determine side (opposite of current position)
            current_side = current_position.get("side", "long")
            close_side = "SELL" if current_side == "long" else "BUY"

            # Execute closing trade
            result = await self.execute_trade(
                symbol=symbol,
                action=close_side,
                quantity=close_qty,
                order_type="market",
            )

            if result["success"]:
                logger.info(
                    "Position closed",
                    symbol=symbol,
                    quantity=close_qty,
                    order_id=result["order_id"],
                )

            return result

        except Exception as e:
            logger.error("Failed to close position", symbol=symbol, error=str(e))
            return {"success": False, "error": str(e), "symbol": symbol}

    async def _update_positions(self):
        """Update active positions from broker"""
        try:
            positions = await self._alpaca_client.get_positions()
            self.active_positions = {}

            for pos in positions:
                self.active_positions[pos["symbol"]] = {
                    "quantity": float(pos["qty"]),
                    "side": "long" if float(pos["qty"]) > 0 else "short",
                    "entry_price": float(pos["avg_cost"]),
                    "current_price": (
                        float(pos["market_value"]) / float(pos["qty"])
                        if float(pos["qty"]) != 0
                        else 0
                    ),
                    "unrealized_pnl": float(pos["unrealized_pl"]),
                    "market_value": float(pos["market_value"]),
                }

        except Exception as e:
            logger.warning("Failed to update positions", error=str(e))

    async def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            account = await self._alpaca_client.get_account()

            self.performance_metrics = {
                "total_equity": float(account.get("equity", 0)),
                "daily_pnl": float(account.get("day_pl", 0)),
                "total_pnl": float(account.get("total_pl", 0)),
                "positions_count": len(self.active_positions),
                "last_updated": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.warning("Failed to update performance metrics", error=str(e))

    async def _assess_symbol_risk(self, symbol: str, current_price: float) -> Dict:
        """Assess risk for a specific symbol"""
        try:
            # Basic risk assessment
            position = self.active_positions.get(symbol, {})
            position_size = abs(float(position.get("quantity", 0)))

            risk_metrics = {
                "position_concentration": (
                    position_size * current_price / self.account_size
                    if self.account_size > 0
                    else 0
                ),
                "volatility_risk": "medium",  # Would calculate from historical data
                "liquidity_risk": "low",  # Would assess based on volume
                "overall_risk": "medium",
            }

            return risk_metrics

        except Exception as e:
            logger.warning("Risk assessment failed", symbol=symbol, error=str(e))
            return {"error": str(e)}

    async def _get_portfolio_risk_status(self) -> Dict:
        """Get overall portfolio risk status"""
        try:
            total_exposure = sum(
                pos.get("market_value", 0) for pos in self.active_positions.values()
            )

            risk_status = {
                "total_exposure": total_exposure,
                "exposure_ratio": (
                    total_exposure / self.account_size if self.account_size > 0 else 0
                ),
                "position_count": len(self.active_positions),
                "max_position_risk": max(
                    (
                        pos.get("market_value", 0) / self.account_size
                        for pos in self.active_positions.values()
                    ),
                    default=0,
                ),
                "risk_level": (
                    "LOW"
                    if total_exposure / self.account_size < 0.5
                    else (
                        "MEDIUM" if total_exposure / self.account_size < 0.8 else "HIGH"
                    )
                ),
            }

            return risk_status

        except Exception as e:
            logger.warning("Portfolio risk status calculation failed", error=str(e))
            return {"error": str(e)}

    async def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        try:
            quote = await self._alpaca_client.get_quote(symbol)
            return float(
                quote.get("bid", 0)
            )  # Use bid price for conservative estimates
        except Exception as e:
            logger.error("Failed to get current price", symbol=symbol, error=str(e))
            raise TradingError(f"Could not get current price for {symbol}")

    @asynccontextmanager
    async def trading_session(self, user_config: Optional[Dict] = None):
        """Context manager for trading sessions"""
        if not self._initialized:
            await self.initialize(user_config)

        try:
            logger.info("Trading session started")
            yield self
        finally:
            logger.info("Trading session ended")

    async def health_check(self) -> Dict:
        """Perform health check on all components"""
        health_status = {
            "trading_manager": "healthy",
            "components": {},
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # Check Alpaca connection
            if self._alpaca_client:
                account = await self._alpaca_client.get_account()
                health_status["components"]["alpaca"] = (
                    "connected" if account else "error"
                )
            else:
                health_status["components"]["alpaca"] = "not_initialized"

            # Check other components
            health_status["components"]["risk_manager"] = (
                "ready" if self._risk_manager else "not_initialized"
            )
            health_status["components"]["order_manager"] = (
                "ready" if self._order_manager else "not_initialized"
            )
            health_status["components"]["markov_system"] = (
                "ready" if self._markov_system else "not_initialized"
            )

        except Exception as e:
            health_status["trading_manager"] = "error"
            health_status["error"] = str(e)

        return health_status


# Convenience function for getting the singleton instance
def get_trading_manager() -> TradingManager:
    """Get the TradingManager singleton instance"""
    return TradingManager()


# Export main classes
__all__ = ["TradingManager", "get_trading_manager"]
