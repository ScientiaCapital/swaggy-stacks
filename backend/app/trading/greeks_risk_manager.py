"""
Greeks Risk Manager

Extends the existing risk management system with options Greeks-specific
risk controls and portfolio-wide Greeks aggregation and monitoring.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import asyncio

import pandas as pd
import numpy as np
import structlog

from app.core.config import settings
from app.core.exceptions import TradingError
from app.trading.risk_manager import RiskManager
from app.trading.options_trading import OptionPosition, GreeksData, get_options_trader

logger = structlog.get_logger()


@dataclass
class GreeksLimits:
    """Greeks exposure limits configuration"""

    # Delta exposure limits (equivalent shares exposure)
    max_portfolio_delta: float = 1000.0  # Max 1000 shares equivalent
    max_single_position_delta: float = 100.0  # Max 100 shares per position
    max_sector_delta: float = 500.0  # Max 500 shares per sector

    # Gamma exposure limits (delta sensitivity)
    max_portfolio_gamma: float = 10.0  # Max gamma exposure
    max_single_position_gamma: float = 2.0  # Max gamma per position
    gamma_scalping_threshold: float = 5.0  # Threshold for gamma scalping

    # Vega exposure limits (volatility sensitivity)
    max_portfolio_vega: float = 1000.0  # Max vega exposure
    max_single_position_vega: float = 100.0  # Max vega per position
    max_iv_exposure_pct: float = 25.0  # Max 25% of portfolio to IV risk

    # Theta exposure limits (time decay)
    max_portfolio_theta: float = -100.0  # Max negative theta (time decay)
    min_theta_yield_pct: float = 0.1  # Min 0.1% daily theta yield
    max_theta_concentration: float = 0.5  # Max 50% theta from single position

    # Rho exposure limits (interest rate sensitivity)
    max_portfolio_rho: float = 500.0  # Max rho exposure
    max_duration_years: float = 1.0  # Max effective duration

    # Position sizing based on Greeks
    gamma_position_multiplier: float = 0.1  # Reduce size for high gamma
    vega_position_multiplier: float = 0.2  # Reduce size for high vega


@dataclass
class GreeksRiskMetrics:
    """Current Greeks risk metrics"""

    # Portfolio-level Greeks
    portfolio_delta: float = 0.0
    portfolio_gamma: float = 0.0
    portfolio_theta: float = 0.0
    portfolio_vega: float = 0.0
    portfolio_rho: float = 0.0

    # Risk ratios (current / limit)
    delta_utilization: float = 0.0
    gamma_utilization: float = 0.0
    vega_utilization: float = 0.0
    theta_utilization: float = 0.0

    # Concentration metrics
    largest_position_delta: float = 0.0
    delta_concentration: float = 0.0  # Largest position as % of total
    gamma_concentration: float = 0.0
    vega_concentration: float = 0.0

    # Risk flags
    delta_limit_breach: bool = False
    gamma_limit_breach: bool = False
    vega_limit_breach: bool = False
    concentration_warning: bool = False

    # Timestamps
    last_updated: datetime = None
    next_rebalance: datetime = None


class GreeksRiskManager(RiskManager):
    """
    Greeks-aware risk manager extending the base RiskManager

    Provides portfolio-wide Greeks aggregation, exposure limits,
    and Greeks-based position sizing and risk adjustments.
    """

    def __init__(
        self,
        user_id: int,
        user_risk_params: Optional[Dict] = None,
        greeks_limits: Optional[GreeksLimits] = None,
    ):
        # Initialize base risk manager
        super().__init__(user_id, user_risk_params)

        # Greeks-specific configuration
        self.greeks_limits = greeks_limits or GreeksLimits()
        self.current_metrics = GreeksRiskMetrics()

        # Options trader instance for Greeks calculations
        self.options_trader = get_options_trader()

        # Greeks monitoring configuration
        self.greeks_update_interval = timedelta(minutes=5)
        self.last_greeks_update = None
        self.monitoring_task: Optional[asyncio.Task] = None

        # Position tracking
        self.current_positions: List[OptionPosition] = []
        self.sector_exposure: Dict[str, float] = {}

        logger.info("Greeks risk manager initialized",
                   user_id=user_id, limits=self.greeks_limits)

    async def update_portfolio_greeks(
        self, positions: List[OptionPosition] = None
    ) -> GreeksRiskMetrics:
        """Update portfolio-wide Greeks aggregation"""
        try:
            # Use provided positions or get current positions
            if positions is None:
                positions = self.current_positions

            if not positions:
                # Reset metrics if no positions
                self.current_metrics = GreeksRiskMetrics(last_updated=datetime.now())
                return self.current_metrics

            # Calculate portfolio Greeks using existing method
            portfolio_greeks = await self.options_trader.get_portfolio_greeks(positions)

            # Extract Greeks values
            portfolio_delta = portfolio_greeks.get("portfolio_delta", 0.0)
            portfolio_gamma = portfolio_greeks.get("portfolio_gamma", 0.0)
            portfolio_theta = portfolio_greeks.get("portfolio_theta", 0.0)
            portfolio_vega = portfolio_greeks.get("portfolio_vega", 0.0)
            portfolio_rho = portfolio_greeks.get("portfolio_rho", 0.0)

            # Calculate utilization ratios
            delta_utilization = abs(portfolio_delta) / self.greeks_limits.max_portfolio_delta
            gamma_utilization = abs(portfolio_gamma) / self.greeks_limits.max_portfolio_gamma
            vega_utilization = abs(portfolio_vega) / self.greeks_limits.max_portfolio_vega
            theta_utilization = abs(portfolio_theta) / abs(self.greeks_limits.max_portfolio_theta)

            # Calculate concentration metrics
            largest_delta, delta_concentration = self._calculate_delta_concentration(positions)
            gamma_concentration = self._calculate_gamma_concentration(positions)
            vega_concentration = self._calculate_vega_concentration(positions)

            # Check risk limit breaches
            delta_breach = abs(portfolio_delta) > self.greeks_limits.max_portfolio_delta
            gamma_breach = abs(portfolio_gamma) > self.greeks_limits.max_portfolio_gamma
            vega_breach = abs(portfolio_vega) > self.greeks_limits.max_portfolio_vega
            concentration_warning = (delta_concentration > 0.5 or
                                   gamma_concentration > 0.5 or
                                   vega_concentration > 0.5)

            # Update metrics
            self.current_metrics = GreeksRiskMetrics(
                portfolio_delta=portfolio_delta,
                portfolio_gamma=portfolio_gamma,
                portfolio_theta=portfolio_theta,
                portfolio_vega=portfolio_vega,
                portfolio_rho=portfolio_rho,
                delta_utilization=delta_utilization,
                gamma_utilization=gamma_utilization,
                vega_utilization=vega_utilization,
                theta_utilization=theta_utilization,
                largest_position_delta=largest_delta,
                delta_concentration=delta_concentration,
                gamma_concentration=gamma_concentration,
                vega_concentration=vega_concentration,
                delta_limit_breach=delta_breach,
                gamma_limit_breach=gamma_breach,
                vega_limit_breach=vega_breach,
                concentration_warning=concentration_warning,
                last_updated=datetime.now(),
                next_rebalance=datetime.now() + self.greeks_update_interval,
            )

            # Log warnings if limits are breached
            if delta_breach or gamma_breach or vega_breach or concentration_warning:
                logger.warning("Greeks risk limits breached",
                             delta_breach=delta_breach,
                             gamma_breach=gamma_breach,
                             vega_breach=vega_breach,
                             concentration_warning=concentration_warning,
                             metrics=self.current_metrics)

            self.last_greeks_update = datetime.now()
            return self.current_metrics

        except Exception as e:
            logger.error("Failed to update portfolio Greeks", error=str(e))
            raise TradingError(f"Greeks update failed: {e}")

    def validate_greeks_order(
        self,
        option_position: OptionPosition,
        side: str,  # "buy" or "sell"
        quantity: int,
    ) -> Tuple[bool, str, Optional[Dict]]:
        """
        Validate order against Greeks limits and suggest adjustments

        Args:
            option_position: The option position to validate
            side: Order side (buy/sell)
            quantity: Number of contracts

        Returns:
            Tuple of (is_valid, reason, suggested_adjustments)
        """
        try:
            if not option_position.position_greeks:
                return False, "Greeks data not available for position", None

            greeks = option_position.position_greeks

            # Calculate impact of new position
            position_multiplier = 1 if side == "buy" else -1
            effective_quantity = quantity * position_multiplier

            delta_impact = effective_quantity * greeks.delta
            gamma_impact = effective_quantity * greeks.gamma
            vega_impact = effective_quantity * greeks.vega
            theta_impact = effective_quantity * greeks.theta

            # Check portfolio-level limits
            new_portfolio_delta = self.current_metrics.portfolio_delta + delta_impact
            new_portfolio_gamma = self.current_metrics.portfolio_gamma + gamma_impact
            new_portfolio_vega = self.current_metrics.portfolio_vega + vega_impact

            # Delta limit check
            if abs(new_portfolio_delta) > self.greeks_limits.max_portfolio_delta:
                max_allowed_delta = self.greeks_limits.max_portfolio_delta - abs(self.current_metrics.portfolio_delta)
                max_quantity = int(max_allowed_delta / abs(greeks.delta)) if greeks.delta != 0 else 0
                return False, f"Portfolio delta limit exceeded", {
                    "max_quantity": max_quantity,
                    "reason": "delta_limit",
                    "current_delta": self.current_metrics.portfolio_delta,
                    "impact": delta_impact,
                    "limit": self.greeks_limits.max_portfolio_delta
                }

            # Gamma limit check
            if abs(new_portfolio_gamma) > self.greeks_limits.max_portfolio_gamma:
                max_allowed_gamma = self.greeks_limits.max_portfolio_gamma - abs(self.current_metrics.portfolio_gamma)
                max_quantity = int(max_allowed_gamma / abs(greeks.gamma)) if greeks.gamma != 0 else 0
                return False, f"Portfolio gamma limit exceeded", {
                    "max_quantity": max_quantity,
                    "reason": "gamma_limit",
                    "current_gamma": self.current_metrics.portfolio_gamma,
                    "impact": gamma_impact,
                    "limit": self.greeks_limits.max_portfolio_gamma
                }

            # Vega limit check
            if abs(new_portfolio_vega) > self.greeks_limits.max_portfolio_vega:
                max_allowed_vega = self.greeks_limits.max_portfolio_vega - abs(self.current_metrics.portfolio_vega)
                max_quantity = int(max_allowed_vega / abs(greeks.vega)) if greeks.vega != 0 else 0
                return False, f"Portfolio vega limit exceeded", {
                    "max_quantity": max_quantity,
                    "reason": "vega_limit",
                    "current_vega": self.current_metrics.portfolio_vega,
                    "impact": vega_impact,
                    "limit": self.greeks_limits.max_portfolio_vega
                }

            # Single position limits
            position_delta = abs(effective_quantity * greeks.delta)
            if position_delta > self.greeks_limits.max_single_position_delta:
                max_quantity = int(self.greeks_limits.max_single_position_delta / abs(greeks.delta))
                return False, f"Single position delta limit exceeded", {
                    "max_quantity": max_quantity,
                    "reason": "position_delta_limit",
                    "position_delta": position_delta,
                    "limit": self.greeks_limits.max_single_position_delta
                }

            # All checks passed
            return True, "Greeks validation passed", {
                "delta_impact": delta_impact,
                "gamma_impact": gamma_impact,
                "vega_impact": vega_impact,
                "theta_impact": theta_impact,
                "utilization_after": {
                    "delta": abs(new_portfolio_delta) / self.greeks_limits.max_portfolio_delta,
                    "gamma": abs(new_portfolio_gamma) / self.greeks_limits.max_portfolio_gamma,
                    "vega": abs(new_portfolio_vega) / self.greeks_limits.max_portfolio_vega,
                }
            }

        except Exception as e:
            logger.error("Greeks validation error", error=str(e))
            return False, f"Validation error: {e}", None

    def calculate_greeks_adjusted_position_size(
        self,
        base_quantity: int,
        greeks: GreeksData,
        account_value: float,
    ) -> Tuple[int, str]:
        """
        Calculate position size adjusted for Greeks risk

        Args:
            base_quantity: Base position size
            greeks: Greeks data for the position
            account_value: Current account value

        Returns:
            Tuple of (adjusted_quantity, adjustment_reason)
        """
        try:
            adjusted_quantity = base_quantity
            adjustments = []

            # Gamma adjustment
            if abs(greeks.gamma) > 1.0:  # High gamma
                gamma_multiplier = self.greeks_limits.gamma_position_multiplier
                gamma_adjustment = int(base_quantity * gamma_multiplier)
                adjusted_quantity = min(adjusted_quantity, gamma_adjustment)
                adjustments.append(f"gamma({gamma_adjustment})")

            # Vega adjustment
            if abs(greeks.vega) > 0.1:  # High vega sensitivity
                vega_multiplier = self.greeks_limits.vega_position_multiplier
                vega_adjustment = int(base_quantity * vega_multiplier)
                adjusted_quantity = min(adjusted_quantity, vega_adjustment)
                adjustments.append(f"vega({vega_adjustment})")

            # Portfolio utilization adjustment
            if self.current_metrics.delta_utilization > 0.8:  # High delta utilization
                utilization_adjustment = int(base_quantity * 0.5)  # 50% reduction
                adjusted_quantity = min(adjusted_quantity, utilization_adjustment)
                adjustments.append(f"utilization({utilization_adjustment})")

            # Account size adjustment
            position_value = adjusted_quantity * 100 * abs(greeks.delta)  # Approximate position value
            max_position_pct = 0.05  # Max 5% of account per position
            max_value = account_value * max_position_pct
            if position_value > max_value:
                account_adjustment = int(max_value / (100 * abs(greeks.delta)))
                adjusted_quantity = min(adjusted_quantity, account_adjustment)
                adjustments.append(f"account_size({account_adjustment})")

            # Ensure minimum viable position
            adjusted_quantity = max(1, adjusted_quantity)

            adjustment_reason = f"Greeks adjustments: {', '.join(adjustments)}" if adjustments else "No adjustments"

            logger.info("Position size adjusted for Greeks",
                       base_quantity=base_quantity,
                       adjusted_quantity=adjusted_quantity,
                       adjustments=adjustments)

            return adjusted_quantity, adjustment_reason

        except Exception as e:
            logger.error("Error calculating Greeks-adjusted position size", error=str(e))
            return base_quantity, f"Error in adjustment: {e}"

    def get_rebalancing_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for portfolio rebalancing based on Greeks"""
        recommendations = []

        try:
            # Check if rebalancing is needed
            if not (self.current_metrics.delta_limit_breach or
                   self.current_metrics.gamma_limit_breach or
                   self.current_metrics.vega_limit_breach or
                   self.current_metrics.concentration_warning):
                return []

            # Delta rebalancing
            if self.current_metrics.delta_limit_breach:
                delta_excess = abs(self.current_metrics.portfolio_delta) - self.greeks_limits.max_portfolio_delta
                recommendations.append({
                    "type": "delta_rebalance",
                    "priority": "high",
                    "action": "reduce_delta_exposure",
                    "amount": delta_excess,
                    "description": f"Reduce delta exposure by {delta_excess:.1f} shares equivalent",
                    "current": self.current_metrics.portfolio_delta,
                    "limit": self.greeks_limits.max_portfolio_delta,
                })

            # Gamma rebalancing
            if self.current_metrics.gamma_limit_breach:
                gamma_excess = abs(self.current_metrics.portfolio_gamma) - self.greeks_limits.max_portfolio_gamma
                recommendations.append({
                    "type": "gamma_rebalance",
                    "priority": "medium",
                    "action": "reduce_gamma_exposure",
                    "amount": gamma_excess,
                    "description": f"Reduce gamma exposure by {gamma_excess:.2f}",
                    "current": self.current_metrics.portfolio_gamma,
                    "limit": self.greeks_limits.max_portfolio_gamma,
                })

            # Vega rebalancing
            if self.current_metrics.vega_limit_breach:
                vega_excess = abs(self.current_metrics.portfolio_vega) - self.greeks_limits.max_portfolio_vega
                recommendations.append({
                    "type": "vega_rebalance",
                    "priority": "medium",
                    "action": "reduce_vega_exposure",
                    "amount": vega_excess,
                    "description": f"Reduce vega exposure by {vega_excess:.1f}",
                    "current": self.current_metrics.portfolio_vega,
                    "limit": self.greeks_limits.max_portfolio_vega,
                })

            # Concentration warnings
            if self.current_metrics.concentration_warning:
                recommendations.append({
                    "type": "concentration_warning",
                    "priority": "low",
                    "action": "diversify_positions",
                    "description": "Reduce position concentration across Greeks",
                    "delta_concentration": self.current_metrics.delta_concentration,
                    "gamma_concentration": self.current_metrics.gamma_concentration,
                    "vega_concentration": self.current_metrics.vega_concentration,
                })

            return recommendations

        except Exception as e:
            logger.error("Error generating rebalancing recommendations", error=str(e))
            return []

    async def start_monitoring(self):
        """Start continuous Greeks monitoring"""
        if self.monitoring_task and not self.monitoring_task.done():
            logger.warning("Greeks monitoring already active")
            return

        logger.info("Starting Greeks risk monitoring")
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def stop_monitoring(self):
        """Stop Greeks monitoring"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            logger.info("Greeks monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop for Greeks"""
        while True:
            try:
                await self.update_portfolio_greeks()

                # Check for rebalancing needs
                recommendations = self.get_rebalancing_recommendations()
                if recommendations:
                    logger.warning("Greeks rebalancing recommendations generated",
                                 count=len(recommendations),
                                 recommendations=recommendations)

                # Sleep until next update
                await asyncio.sleep(self.greeks_update_interval.total_seconds())

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in Greeks monitoring loop", error=str(e))
                await asyncio.sleep(60)  # Wait before retrying

    # Helper methods
    def _calculate_delta_concentration(self, positions: List[OptionPosition]) -> Tuple[float, float]:
        """Calculate delta concentration metrics"""
        if not positions:
            return 0.0, 0.0

        position_deltas = [abs(pos.quantity * pos.position_greeks.delta) for pos in positions if pos.position_greeks]

        if not position_deltas:
            return 0.0, 0.0

        largest_delta = max(position_deltas)
        total_delta = sum(position_deltas)
        concentration = largest_delta / total_delta if total_delta > 0 else 0.0

        return largest_delta, concentration

    def _calculate_gamma_concentration(self, positions: List[OptionPosition]) -> float:
        """Calculate gamma concentration"""
        if not positions:
            return 0.0

        position_gammas = [abs(pos.quantity * pos.position_greeks.gamma) for pos in positions if pos.position_greeks]

        if not position_gammas:
            return 0.0

        largest_gamma = max(position_gammas)
        total_gamma = sum(position_gammas)
        return largest_gamma / total_gamma if total_gamma > 0 else 0.0

    def _calculate_vega_concentration(self, positions: List[OptionPosition]) -> float:
        """Calculate vega concentration"""
        if not positions:
            return 0.0

        position_vegas = [abs(pos.quantity * pos.position_greeks.vega) for pos in positions if pos.position_greeks]

        if not position_vegas:
            return 0.0

        largest_vega = max(position_vegas)
        total_vega = sum(position_vegas)
        return largest_vega / total_vega if total_vega > 0 else 0.0

    def update_positions(self, positions: List[OptionPosition]):
        """Update current positions for Greeks calculations"""
        self.current_positions = positions
        logger.debug("Updated positions for Greeks tracking", count=len(positions))

    def get_current_metrics(self) -> GreeksRiskMetrics:
        """Get current Greeks risk metrics"""
        return self.current_metrics

    def get_greeks_limits(self) -> GreeksLimits:
        """Get current Greeks limits configuration"""
        return self.greeks_limits

    def update_greeks_limits(self, new_limits: GreeksLimits):
        """Update Greeks limits configuration"""
        old_limits = self.greeks_limits
        self.greeks_limits = new_limits
        logger.info("Greeks limits updated", old_limits=old_limits, new_limits=new_limits)

    def get_greeks_summary(self) -> Dict[str, Any]:
        """Get comprehensive Greeks risk summary"""
        return {
            "current_metrics": self.current_metrics,
            "limits": self.greeks_limits,
            "utilization": {
                "delta": self.current_metrics.delta_utilization,
                "gamma": self.current_metrics.gamma_utilization,
                "vega": self.current_metrics.vega_utilization,
                "theta": self.current_metrics.theta_utilization,
            },
            "risk_status": {
                "overall": "HIGH" if any([
                    self.current_metrics.delta_limit_breach,
                    self.current_metrics.gamma_limit_breach,
                    self.current_metrics.vega_limit_breach
                ]) else "MEDIUM" if self.current_metrics.concentration_warning else "LOW",
                "breaches": {
                    "delta": self.current_metrics.delta_limit_breach,
                    "gamma": self.current_metrics.gamma_limit_breach,
                    "vega": self.current_metrics.vega_limit_breach,
                    "concentration": self.current_metrics.concentration_warning,
                }
            },
            "recommendations": self.get_rebalancing_recommendations(),
            "last_updated": self.current_metrics.last_updated,
        }