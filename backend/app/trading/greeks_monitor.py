"""
Real-time Greeks monitoring and alerting system for options portfolios.
Provides continuous monitoring of portfolio Greeks with customizable risk limits and alerts.
"""

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from enum import Enum

import numpy as np
import structlog

from app.core.cache import get_market_cache
from app.monitoring.alerts import AlertManager
from app.trading.options_trading import OptionsTrader, GreeksData, OptionPosition

logger = structlog.get_logger(__name__)


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class GreeksRiskLevel(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class GreeksLimits:
    """Portfolio Greeks risk limits configuration"""
    max_delta: float = 100.0           # Maximum net delta exposure
    max_gamma: float = 50.0            # Maximum gamma exposure
    max_theta: float = -500.0          # Maximum theta decay (negative)
    max_vega: float = 1000.0           # Maximum vega exposure
    max_rho: float = 500.0             # Maximum rho exposure
    max_portfolio_value: float = 100000.0  # Max portfolio value at risk

    # Warning levels (percentage of max limits)
    warning_threshold: float = 0.8      # 80% of limit triggers warning
    critical_threshold: float = 0.95    # 95% of limit triggers critical alert

    # Concentration limits
    max_single_position_pct: float = 0.20   # 20% max single position
    max_expiration_concentration: float = 0.40  # 40% max in single expiration
    max_strike_concentration: float = 0.30      # 30% max at single strike

    # Time-based limits
    min_time_to_expiry_days: int = 7    # Minimum days to expiration
    max_dte_concentration: float = 0.50  # Max concentration in short-term options


@dataclass
class PortfolioGreeks:
    """Current portfolio Greeks snapshot"""
    timestamp: datetime
    total_delta: float
    total_gamma: float
    total_theta: float
    total_vega: float
    total_rho: float
    total_vanna: float = 0.0
    total_volga: float = 0.0

    # Risk metrics
    portfolio_value: float = 0.0
    max_loss_1pct_move: float = 0.0      # Loss if underlying moves 1%
    max_loss_5pct_move: float = 0.0      # Loss if underlying moves 5%
    theta_decay_1day: float = 0.0        # Daily theta decay
    var_95: float = 0.0                  # 95% Value at Risk

    # Concentration metrics
    largest_position_pct: float = 0.0
    expiration_concentration: Dict[str, float] = field(default_factory=dict)
    strike_concentration: Dict[str, float] = field(default_factory=dict)
    symbol_concentration: Dict[str, float] = field(default_factory=dict)


@dataclass
class GreeksAlert:
    """Greeks monitoring alert"""
    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    category: str
    message: str
    metric_name: str
    current_value: float
    limit_value: float
    utilization_pct: float
    positions_affected: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    auto_close_conditions: List[str] = field(default_factory=list)


class GreeksMonitor:
    """Real-time Greeks monitoring and alerting system"""

    def __init__(
        self,
        options_trader: Optional[OptionsTrader] = None,
        alert_manager: Optional[AlertManager] = None
    ):
        self.options_trader = options_trader or OptionsTrader()
        self.alert_manager = alert_manager or AlertManager()
        self.market_cache = get_market_cache()

        # Configuration
        self.limits = GreeksLimits()
        self.monitoring_enabled = True
        self.monitoring_interval = 30  # seconds
        self.alert_cooldown = 300      # 5 minutes

        # Monitoring state
        self.current_greeks: Optional[PortfolioGreeks] = None
        self.active_alerts: Dict[str, GreeksAlert] = {}
        self.alert_history: List[GreeksAlert] = []
        self.last_alert_times: Dict[str, datetime] = {}

        # Statistics
        self.monitoring_stats = {
            'total_checks': 0,
            'alerts_triggered': 0,
            'alerts_resolved': 0,
            'avg_check_duration_ms': 0.0,
            'last_check_time': None
        }

        logger.info(
            "GreeksMonitor initialized",
            limits=self.limits.__dict__,
            monitoring_interval=self.monitoring_interval
        )

    async def start_monitoring(self):
        """Start continuous Greeks monitoring"""
        logger.info("Starting Greeks monitoring")
        self.monitoring_enabled = True

        while self.monitoring_enabled:
            try:
                start_time = datetime.now()

                # Update portfolio Greeks
                await self.update_portfolio_greeks()

                # Check risk limits
                await self.check_risk_limits()

                # Update statistics
                duration_ms = (datetime.now() - start_time).total_seconds() * 1000
                self._update_monitoring_stats(duration_ms)

                # Wait for next check
                await asyncio.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error("Greeks monitoring error", error=str(e))
                await asyncio.sleep(self.monitoring_interval * 2)  # Back off on error

    def stop_monitoring(self):
        """Stop Greeks monitoring"""
        logger.info("Stopping Greeks monitoring")
        self.monitoring_enabled = False

    async def update_portfolio_greeks(self) -> PortfolioGreeks:
        """Update current portfolio Greeks from positions"""
        try:
            # Get current positions (would normally come from portfolio manager)
            positions = await self._get_current_positions()

            if not positions:
                logger.debug("No positions found, Greeks monitoring idle")
                return PortfolioGreeks(
                    timestamp=datetime.now(),
                    total_delta=0.0, total_gamma=0.0, total_theta=0.0,
                    total_vega=0.0, total_rho=0.0
                )

            # Calculate aggregate Greeks
            portfolio_greeks = await self._calculate_portfolio_greeks(positions)

            # Calculate risk metrics
            portfolio_greeks = await self._calculate_risk_metrics(portfolio_greeks, positions)

            # Update current state
            self.current_greeks = portfolio_greeks

            # Cache for other systems
            await self.market_cache.set(
                "portfolio_greeks",
                portfolio_greeks.__dict__,
                ttl_override=60
            )

            logger.debug(
                "Portfolio Greeks updated",
                delta=portfolio_greeks.total_delta,
                gamma=portfolio_greeks.total_gamma,
                theta=portfolio_greeks.total_theta,
                vega=portfolio_greeks.total_vega,
                portfolio_value=portfolio_greeks.portfolio_value
            )

            return portfolio_greeks

        except Exception as e:
            logger.error("Failed to update portfolio Greeks", error=str(e))
            return PortfolioGreeks(
                timestamp=datetime.now(),
                total_delta=0.0, total_gamma=0.0, total_theta=0.0,
                total_vega=0.0, total_rho=0.0
            )

    async def check_risk_limits(self):
        """Check portfolio Greeks against risk limits"""
        if not self.current_greeks:
            return

        try:
            new_alerts = []
            resolved_alerts = []

            # Check each Greek against limits
            checks = [
                ("delta", abs(self.current_greeks.total_delta), self.limits.max_delta),
                ("gamma", abs(self.current_greeks.total_gamma), self.limits.max_gamma),
                ("theta", abs(self.current_greeks.total_theta), abs(self.limits.max_theta)),
                ("vega", abs(self.current_greeks.total_vega), self.limits.max_vega),
                ("rho", abs(self.current_greeks.total_rho), self.limits.max_rho),
                ("portfolio_value", self.current_greeks.portfolio_value, self.limits.max_portfolio_value),
            ]

            for metric_name, current_value, limit_value in checks:
                utilization = (current_value / limit_value) if limit_value > 0 else 0
                alert_key = f"greeks_limit_{metric_name}"

                # Check if we should trigger an alert
                if utilization >= self.limits.critical_threshold:
                    severity = AlertSeverity.CRITICAL
                elif utilization >= self.limits.warning_threshold:
                    severity = AlertSeverity.WARNING
                else:
                    # Check if we should resolve an existing alert
                    if alert_key in self.active_alerts:
                        resolved_alerts.append(alert_key)
                    continue

                # Check cooldown period
                if self._is_in_cooldown(alert_key):
                    continue

                # Create new alert
                alert = self._create_greeks_alert(
                    alert_key, severity, metric_name,
                    current_value, limit_value, utilization
                )
                new_alerts.append(alert)

            # Check concentration limits
            concentration_alerts = await self._check_concentration_limits()
            new_alerts.extend(concentration_alerts)

            # Process new alerts
            for alert in new_alerts:
                await self._process_new_alert(alert)

            # Resolve old alerts
            for alert_key in resolved_alerts:
                await self._resolve_alert(alert_key)

        except Exception as e:
            logger.error("Risk limit checking failed", error=str(e))

    async def _check_concentration_limits(self) -> List[GreeksAlert]:
        """Check portfolio concentration limits"""
        alerts = []

        if not self.current_greeks:
            return alerts

        try:
            # Check single position concentration
            if self.current_greeks.largest_position_pct > self.limits.max_single_position_pct:
                alert = GreeksAlert(
                    alert_id="concentration_single_position",
                    timestamp=datetime.now(),
                    severity=AlertSeverity.WARNING,
                    category="concentration",
                    message=f"Single position concentration {self.current_greeks.largest_position_pct:.1%} exceeds limit {self.limits.max_single_position_pct:.1%}",
                    metric_name="single_position_concentration",
                    current_value=self.current_greeks.largest_position_pct,
                    limit_value=self.limits.max_single_position_pct,
                    utilization_pct=(self.current_greeks.largest_position_pct / self.limits.max_single_position_pct) * 100,
                    recommended_actions=[
                        "Reduce size of largest position",
                        "Diversify across more positions",
                        "Consider hedging the concentrated position"
                    ]
                )
                alerts.append(alert)

            # Check expiration concentration
            for expiration, concentration in self.current_greeks.expiration_concentration.items():
                if concentration > self.limits.max_expiration_concentration:
                    alert = GreeksAlert(
                        alert_id=f"concentration_expiration_{expiration}",
                        timestamp=datetime.now(),
                        severity=AlertSeverity.WARNING,
                        category="concentration",
                        message=f"Expiration {expiration} concentration {concentration:.1%} exceeds limit {self.limits.max_expiration_concentration:.1%}",
                        metric_name="expiration_concentration",
                        current_value=concentration,
                        limit_value=self.limits.max_expiration_concentration,
                        utilization_pct=(concentration / self.limits.max_expiration_concentration) * 100,
                        recommended_actions=[
                            f"Reduce exposure in {expiration} expiration",
                            "Spread positions across multiple expirations",
                            "Close some positions expiring on this date"
                        ]
                    )
                    alerts.append(alert)

        except Exception as e:
            logger.error("Concentration limit checking failed", error=str(e))

        return alerts

    def _create_greeks_alert(
        self,
        alert_key: str,
        severity: AlertSeverity,
        metric_name: str,
        current_value: float,
        limit_value: float,
        utilization: float
    ) -> GreeksAlert:
        """Create a Greeks limit alert"""

        # Generate appropriate message and actions
        if metric_name == "delta":
            message = f"Portfolio delta exposure {current_value:.1f} ({utilization:.1%} of limit)"
            actions = [
                "Hedge delta exposure with underlying",
                "Close some directional positions",
                "Add opposite delta positions"
            ]
        elif metric_name == "gamma":
            message = f"Portfolio gamma exposure {current_value:.1f} ({utilization:.1%} of limit)"
            actions = [
                "Reduce gamma by closing ATM positions",
                "Spread gamma across different strikes",
                "Monitor for gap risk"
            ]
        elif metric_name == "theta":
            message = f"Portfolio theta decay {current_value:.1f}/day ({utilization:.1%} of limit)"
            actions = [
                "Reduce short-term positions",
                "Take profits on high-theta positions",
                "Add longer-dated positions"
            ]
        elif metric_name == "vega":
            message = f"Portfolio vega exposure {current_value:.1f} ({utilization:.1%} of limit)"
            actions = [
                "Hedge volatility exposure",
                "Reduce options positions",
                "Monitor implied volatility changes"
            ]
        else:
            message = f"Portfolio {metric_name} exposure {current_value:.1f} ({utilization:.1%} of limit)"
            actions = [
                f"Reduce {metric_name} exposure",
                "Review position sizing",
                "Consider hedging strategies"
            ]

        return GreeksAlert(
            alert_id=alert_key,
            timestamp=datetime.now(),
            severity=severity,
            category="greeks_limit",
            message=message,
            metric_name=metric_name,
            current_value=current_value,
            limit_value=limit_value,
            utilization_pct=utilization * 100,
            recommended_actions=actions,
            auto_close_conditions=[
                f"{metric_name} exposure falls below {self.limits.warning_threshold:.1%} of limit",
                "Position adjustments completed",
                "Risk limits reconfigured"
            ]
        )

    async def _process_new_alert(self, alert: GreeksAlert):
        """Process and send new alert"""
        try:
            # Add to active alerts
            self.active_alerts[alert.alert_id] = alert
            self.alert_history.append(alert)

            # Update statistics
            self.monitoring_stats['alerts_triggered'] += 1
            self.last_alert_times[alert.alert_id] = alert.timestamp

            # Send alert through alert manager
            await self.alert_manager.send_alert(
                title=f"Greeks Risk Alert: {alert.category.title()}",
                message=alert.message,
                severity=alert.severity.value,
                tags=["greeks", "risk", alert.category],
                metadata={
                    "metric_name": alert.metric_name,
                    "current_value": alert.current_value,
                    "limit_value": alert.limit_value,
                    "utilization_pct": alert.utilization_pct,
                    "recommended_actions": alert.recommended_actions
                }
            )

            logger.warning(
                "Greeks alert triggered",
                alert_id=alert.alert_id,
                severity=alert.severity.value,
                metric=alert.metric_name,
                utilization=f"{alert.utilization_pct:.1f}%",
                actions=len(alert.recommended_actions)
            )

        except Exception as e:
            logger.error("Failed to process Greeks alert", alert_id=alert.alert_id, error=str(e))

    async def _resolve_alert(self, alert_key: str):
        """Resolve an active alert"""
        try:
            if alert_key in self.active_alerts:
                alert = self.active_alerts.pop(alert_key)

                # Update statistics
                self.monitoring_stats['alerts_resolved'] += 1

                # Send resolution notification
                await self.alert_manager.send_alert(
                    title=f"Greeks Alert Resolved: {alert.category.title()}",
                    message=f"Alert for {alert.metric_name} has been resolved",
                    severity="info",
                    tags=["greeks", "risk", "resolved"],
                    metadata={
                        "original_alert_id": alert.alert_id,
                        "duration_minutes": (datetime.now() - alert.timestamp).total_seconds() / 60
                    }
                )

                logger.info(
                    "Greeks alert resolved",
                    alert_id=alert_key,
                    metric=alert.metric_name,
                    duration_minutes=(datetime.now() - alert.timestamp).total_seconds() / 60
                )

        except Exception as e:
            logger.error("Failed to resolve Greeks alert", alert_key=alert_key, error=str(e))

    def _is_in_cooldown(self, alert_key: str) -> bool:
        """Check if alert is in cooldown period"""
        if alert_key not in self.last_alert_times:
            return False

        time_since_last = datetime.now() - self.last_alert_times[alert_key]
        return time_since_last.total_seconds() < self.alert_cooldown

    async def _get_current_positions(self) -> List[OptionPosition]:
        """Get current options positions (mock implementation)"""
        try:
            # In a real implementation, this would fetch from position manager
            # For now, return empty list - would be integrated with actual portfolio
            cached_positions = await self.market_cache.get("options_positions")

            if cached_positions:
                return [OptionPosition(**pos) for pos in cached_positions]

            return []

        except Exception as e:
            logger.error("Failed to get current positions", error=str(e))
            return []

    async def _calculate_portfolio_greeks(self, positions: List[OptionPosition]) -> PortfolioGreeks:
        """Calculate aggregate portfolio Greeks from positions"""
        try:
            total_delta = sum(pos.quantity * pos.position_greeks.delta for pos in positions)
            total_gamma = sum(pos.quantity * pos.position_greeks.gamma for pos in positions)
            total_theta = sum(pos.quantity * pos.position_greeks.theta for pos in positions)
            total_vega = sum(pos.quantity * pos.position_greeks.vega for pos in positions)
            total_rho = sum(pos.quantity * pos.position_greeks.rho for pos in positions)

            # Advanced Greeks if available
            total_vanna = sum(
                pos.quantity * getattr(pos.position_greeks, 'vanna', 0)
                for pos in positions
            )
            total_volga = sum(
                pos.quantity * getattr(pos.position_greeks, 'volga', 0)
                for pos in positions
            )

            return PortfolioGreeks(
                timestamp=datetime.now(),
                total_delta=total_delta,
                total_gamma=total_gamma,
                total_theta=total_theta,
                total_vega=total_vega,
                total_rho=total_rho,
                total_vanna=total_vanna,
                total_volga=total_volga
            )

        except Exception as e:
            logger.error("Portfolio Greeks calculation failed", error=str(e))
            return PortfolioGreeks(
                timestamp=datetime.now(),
                total_delta=0.0, total_gamma=0.0, total_theta=0.0,
                total_vega=0.0, total_rho=0.0
            )

    async def _calculate_risk_metrics(
        self,
        greeks: PortfolioGreeks,
        positions: List[OptionPosition]
    ) -> PortfolioGreeks:
        """Calculate additional risk metrics"""
        try:
            if not positions:
                return greeks

            # Portfolio value
            portfolio_value = sum(
                abs(pos.quantity * pos.current_value) for pos in positions
            )
            greeks.portfolio_value = portfolio_value

            # Scenario analysis - loss for 1% and 5% moves
            greeks.max_loss_1pct_move = abs(greeks.total_delta * 0.01 * 100)  # Assuming $100 underlying
            greeks.max_loss_5pct_move = abs(greeks.total_delta * 0.05 * 100) + abs(greeks.total_gamma * 0.0025 * 100)

            # Daily theta decay
            greeks.theta_decay_1day = greeks.total_theta

            # Simple VaR estimate (would be more sophisticated in reality)
            greeks.var_95 = portfolio_value * 0.05  # 5% of portfolio value

            # Concentration analysis
            if positions:
                position_values = [abs(pos.quantity * pos.current_value) for pos in positions]
                greeks.largest_position_pct = max(position_values) / portfolio_value if portfolio_value > 0 else 0

                # Expiration concentration
                exp_values = {}
                for pos in positions:
                    exp_date = pos.expiration_date.strftime("%Y-%m-%d")
                    exp_values[exp_date] = exp_values.get(exp_date, 0) + abs(pos.quantity * pos.current_value)

                greeks.expiration_concentration = {
                    exp: value / portfolio_value for exp, value in exp_values.items()
                } if portfolio_value > 0 else {}

            return greeks

        except Exception as e:
            logger.error("Risk metrics calculation failed", error=str(e))
            return greeks

    def _update_monitoring_stats(self, duration_ms: float):
        """Update monitoring statistics"""
        self.monitoring_stats['total_checks'] += 1
        self.monitoring_stats['last_check_time'] = datetime.now()

        # Update average duration using exponential moving average
        if self.monitoring_stats['avg_check_duration_ms'] == 0:
            self.monitoring_stats['avg_check_duration_ms'] = duration_ms
        else:
            alpha = 0.1  # Smoothing factor
            self.monitoring_stats['avg_check_duration_ms'] = (
                alpha * duration_ms + (1 - alpha) * self.monitoring_stats['avg_check_duration_ms']
            )

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status and statistics"""
        return {
            "monitoring_enabled": self.monitoring_enabled,
            "monitoring_interval_seconds": self.monitoring_interval,
            "active_alerts_count": len(self.active_alerts),
            "total_alerts_triggered": self.monitoring_stats['alerts_triggered'],
            "total_alerts_resolved": self.monitoring_stats['alerts_resolved'],
            "current_greeks": self.current_greeks.__dict__ if self.current_greeks else None,
            "limits": self.limits.__dict__,
            "statistics": self.monitoring_stats,
            "active_alerts": [alert.__dict__ for alert in self.active_alerts.values()],
        }

    def update_limits(self, new_limits: GreeksLimits):
        """Update risk limits configuration"""
        old_limits = self.limits
        self.limits = new_limits

        logger.info(
            "Greeks limits updated",
            old_max_delta=old_limits.max_delta,
            new_max_delta=new_limits.max_delta,
            old_max_gamma=old_limits.max_gamma,
            new_max_gamma=new_limits.max_gamma
        )


# Global instance for singleton pattern
_greeks_monitor: Optional[GreeksMonitor] = None


def get_greeks_monitor() -> GreeksMonitor:
    """Get global Greeks monitor instance"""
    global _greeks_monitor
    if _greeks_monitor is None:
        _greeks_monitor = GreeksMonitor()
    return _greeks_monitor


# Export key classes and functions
__all__ = [
    "GreeksMonitor",
    "GreeksLimits",
    "PortfolioGreeks",
    "GreeksAlert",
    "AlertSeverity",
    "GreeksRiskLevel",
    "get_greeks_monitor"
]