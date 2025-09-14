"""
Position sizing optimization using Kelly Criterion, volatility adjustment,
and portfolio heat management
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

logger = structlog.get_logger()


class PositionOptimizer:
    """
    Advanced position sizing optimizer with Kelly Criterion,
    volatility adjustment, and portfolio heat management
    """

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.max_portfolio_heat = 0.02  # 2% max portfolio risk
        self.max_position_heat = 0.005  # 0.5% max single position risk
        self.correlation_threshold = 0.7  # Maximum correlation between positions
        self.min_kelly_fraction = 0.01  # Minimum Kelly fraction (1%)
        self.max_kelly_fraction = 0.15  # Maximum Kelly fraction (15%)

        # Historical performance tracking
        self.trade_history = []
        self.win_rate_lookback = 50  # Number of trades for win rate calculation

    def calculate_kelly_criterion(
        self,
        win_probability: float,
        avg_win: float,
        avg_loss: float,
        confidence_multiplier: float = 1.0,
    ) -> float:
        """
        Calculate Kelly Criterion fraction

        Kelly% = (bp - q) / b
        where:
        b = odds received on the wager (avg_win / avg_loss)
        p = probability of winning
        q = probability of losing (1-p)
        """
        try:
            if avg_loss <= 0 or win_probability <= 0 or win_probability >= 1:
                return self.min_kelly_fraction

            # Calculate odds (reward to risk ratio)
            b = avg_win / abs(avg_loss)
            p = win_probability
            q = 1 - p

            # Kelly fraction
            kelly_fraction = (b * p - q) / b

            # Apply confidence multiplier
            kelly_fraction *= confidence_multiplier

            # Cap the Kelly fraction
            kelly_fraction = max(self.min_kelly_fraction, kelly_fraction)
            kelly_fraction = min(self.max_kelly_fraction, kelly_fraction)

            logger.info(
                "Kelly Criterion calculated",
                kelly_fraction=kelly_fraction,
                win_prob=win_probability,
                avg_win=avg_win,
                avg_loss=avg_loss,
                odds_ratio=b,
            )

            return kelly_fraction

        except Exception as e:
            logger.error("Error calculating Kelly Criterion", error=str(e))
            return self.min_kelly_fraction

    def calculate_volatility_adjusted_size(
        self,
        base_position_size: float,
        symbol_volatility: float,
        portfolio_volatility: float = 0.15,
        volatility_target: float = 0.20,
    ) -> float:
        """
        Adjust position size based on volatility
        Higher volatility = smaller position size
        """
        try:
            # Volatility adjustment factor
            vol_adjustment = min(1.0, volatility_target / max(symbol_volatility, 0.01))

            # Portfolio volatility adjustment
            portfolio_vol_adjustment = min(
                1.0, portfolio_volatility / max(symbol_volatility, 0.01)
            )

            # Combined adjustment (geometric mean)
            combined_adjustment = np.sqrt(vol_adjustment * portfolio_vol_adjustment)

            adjusted_size = base_position_size * combined_adjustment

            logger.info(
                "Volatility adjustment applied",
                base_size=base_position_size,
                adjusted_size=adjusted_size,
                symbol_vol=symbol_volatility,
                adjustment_factor=combined_adjustment,
            )

            return adjusted_size

        except Exception as e:
            logger.error("Error calculating volatility adjustment", error=str(e))
            return base_position_size * 0.5  # Conservative fallback

    def calculate_optimal_position_size(
        self,
        symbol: str,
        current_price: float,
        account_value: float,
        signal_confidence: float,
        stop_loss_price: float,
        symbol_volatility: Optional[float] = None,
        historical_performance: Optional[Dict] = None,
        current_positions: Optional[List[Dict]] = None,
    ) -> Tuple[float, Dict]:
        """
        Calculate optimal position size using multiple factors

        Returns:
            Tuple[float, Dict]: (position_size_dollars, calculation_details)
        """
        try:
            # Calculate risk per share
            risk_per_share = abs(current_price - stop_loss_price)
            if risk_per_share <= 0:
                return 0.0, {"error": "Invalid stop loss price"}

            # Get historical performance metrics
            if historical_performance:
                win_rate = historical_performance.get("win_rate", 0.5)
                avg_win = historical_performance.get("avg_win", 0.05)
                avg_loss = historical_performance.get("avg_loss", 0.03)
            else:
                # Use default conservative estimates
                win_rate = 0.5
                avg_win = 0.05  # 5% average win
                avg_loss = 0.03  # 3% average loss

            # Calculate Kelly fraction
            kelly_fraction = self.calculate_kelly_criterion(
                win_probability=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                confidence_multiplier=signal_confidence,
            )

            # Base position size using Kelly Criterion
            base_kelly_size = account_value * kelly_fraction

            # Position size based on portfolio heat (risk-based)
            max_risk_amount = account_value * self.max_position_heat
            heat_based_size = max_risk_amount / (risk_per_share / current_price)

            # Use the smaller of Kelly and heat-based sizing
            base_position_size = min(base_kelly_size, heat_based_size)

            # Apply volatility adjustment
            if symbol_volatility:
                portfolio_vol = self._estimate_portfolio_volatility(current_positions)
                position_size = self.calculate_volatility_adjusted_size(
                    base_position_size, symbol_volatility, portfolio_vol
                )
            else:
                position_size = base_position_size

            # Apply correlation adjustment
            if current_positions:
                correlation_adjustment = self._calculate_correlation_adjustment(
                    symbol, current_positions
                )
                position_size *= correlation_adjustment

            # Apply final constraints
            position_size = self._apply_position_constraints(
                position_size, account_value, current_positions
            )

            # Calculate number of shares
            shares = int(position_size / current_price)
            final_position_size = shares * current_price

            calculation_details = {
                "symbol": symbol,
                "current_price": current_price,
                "stop_loss_price": stop_loss_price,
                "risk_per_share": risk_per_share,
                "kelly_fraction": kelly_fraction,
                "base_kelly_size": base_kelly_size,
                "heat_based_size": heat_based_size,
                "volatility_adjusted": symbol_volatility is not None,
                "correlation_adjusted": current_positions is not None,
                "final_shares": shares,
                "final_position_size": final_position_size,
                "position_heat": (risk_per_share * shares) / account_value,
                "win_rate": win_rate,
                "signal_confidence": signal_confidence,
            }

            logger.info(
                "Optimal position size calculated",
                symbol=symbol,
                shares=shares,
                position_size=final_position_size,
                kelly_fraction=kelly_fraction,
                position_heat=calculation_details["position_heat"],
            )

            return final_position_size, calculation_details

        except Exception as e:
            logger.error(
                "Error calculating optimal position size", error=str(e), symbol=symbol
            )
            return 0.0, {"error": str(e)}

    def calculate_portfolio_heat(
        self, positions: List[Dict], account_value: float
    ) -> Dict[str, float]:
        """
        Calculate portfolio heat (risk exposure)
        """
        try:
            total_risk = 0.0
            position_risks = {}

            for position in positions:
                symbol = position.get("symbol")
                shares = position.get("shares", 0)
                entry_price = position.get("entry_price", 0)
                stop_loss = position.get("stop_loss")

                if stop_loss and shares and entry_price:
                    risk_per_share = abs(entry_price - stop_loss)
                    position_risk = risk_per_share * abs(shares)
                    position_heat = position_risk / account_value

                    position_risks[symbol] = position_heat
                    total_risk += position_risk

            portfolio_heat = total_risk / account_value if account_value > 0 else 0

            return {
                "portfolio_heat": portfolio_heat,
                "position_risks": position_risks,
                "total_risk_amount": total_risk,
                "heat_utilization": portfolio_heat / self.max_portfolio_heat,
                "remaining_capacity": max(0, self.max_portfolio_heat - portfolio_heat),
            }

        except Exception as e:
            logger.error("Error calculating portfolio heat", error=str(e))
            return {"error": str(e)}

    def update_performance_history(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        shares: float,
        trade_result: str,  # 'win' or 'loss'
    ):
        """Update historical performance for future Kelly calculations"""
        try:
            pnl = (exit_price - entry_price) * shares
            pnl_percent = (exit_price - entry_price) / entry_price

            trade_record = {
                "symbol": symbol,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "shares": shares,
                "pnl": pnl,
                "pnl_percent": pnl_percent,
                "result": trade_result,
                "timestamp": datetime.now(),
            }

            self.trade_history.append(trade_record)

            # Keep only recent trades
            if len(self.trade_history) > self.win_rate_lookback * 2:
                self.trade_history = self.trade_history[-self.win_rate_lookback :]

            logger.info(
                "Trade performance updated",
                symbol=symbol,
                pnl=pnl,
                result=trade_result,
                total_trades=len(self.trade_history),
            )

        except Exception as e:
            logger.error("Error updating performance history", error=str(e))

    def get_historical_performance(self, symbol: Optional[str] = None) -> Dict:
        """Get historical performance metrics"""
        try:
            # Filter trades by symbol if specified
            trades = self.trade_history
            if symbol:
                trades = [t for t in trades if t["symbol"] == symbol]

            if not trades:
                return {
                    "win_rate": 0.5,
                    "avg_win": 0.05,
                    "avg_loss": 0.03,
                    "total_trades": 0,
                    "profit_factor": 1.0,
                }

            # Calculate metrics
            wins = [t for t in trades if t["result"] == "win"]
            losses = [t for t in trades if t["result"] == "loss"]

            win_rate = len(wins) / len(trades)
            avg_win = np.mean([t["pnl_percent"] for t in wins]) if wins else 0.0
            avg_loss = (
                abs(np.mean([t["pnl_percent"] for t in losses])) if losses else 0.0
            )

            total_wins = sum(t["pnl"] for t in wins)
            total_losses = abs(sum(t["pnl"] for t in losses))
            profit_factor = total_wins / total_losses if total_losses > 0 else 1.0

            return {
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "total_trades": len(trades),
                "winning_trades": len(wins),
                "losing_trades": len(losses),
                "profit_factor": profit_factor,
                "total_pnl": sum(t["pnl"] for t in trades),
            }

        except Exception as e:
            logger.error("Error calculating historical performance", error=str(e))
            return {"error": str(e)}

    def _estimate_portfolio_volatility(self, positions: Optional[List[Dict]]) -> float:
        """Estimate current portfolio volatility"""
        if not positions:
            return 0.15  # Default assumption

        # Simplified portfolio volatility estimation
        # In practice, would use correlation matrix and individual volatilities
        return 0.15  # Placeholder

    def _calculate_correlation_adjustment(
        self, new_symbol: str, current_positions: List[Dict]
    ) -> float:
        """
        Adjust position size based on correlation with existing positions
        Higher correlation = smaller additional position
        """
        try:
            # Simplified correlation adjustment
            # In practice, would calculate actual correlations
            existing_symbols = {pos.get("symbol") for pos in current_positions}

            # Same sector stocks (rough heuristic)
            tech_stocks = {"AAPL", "GOOGL", "MSFT", "NVDA", "TSLA"}
            finance_stocks = {"JPM", "BAC", "GS", "WFC"}

            correlation_penalty = 1.0

            if new_symbol in tech_stocks:
                tech_count = len([s for s in existing_symbols if s in tech_stocks])
                correlation_penalty = max(0.5, 1.0 - (tech_count * 0.2))
            elif new_symbol in finance_stocks:
                finance_count = len(
                    [s for s in existing_symbols if s in finance_stocks]
                )
                correlation_penalty = max(0.5, 1.0 - (finance_count * 0.2))

            logger.info(
                "Correlation adjustment applied",
                symbol=new_symbol,
                adjustment=correlation_penalty,
                existing_positions=len(current_positions),
            )

            return correlation_penalty

        except Exception as e:
            logger.error("Error calculating correlation adjustment", error=str(e))
            return 0.8  # Conservative adjustment

    def _apply_position_constraints(
        self,
        position_size: float,
        account_value: float,
        current_positions: Optional[List[Dict]],
    ) -> float:
        """Apply final position size constraints"""
        try:
            # Maximum position size (15% of account)
            max_position = account_value * 0.15
            position_size = min(position_size, max_position)

            # Maximum portfolio exposure (95%)
            if current_positions:
                current_exposure = sum(
                    pos.get("market_value", 0) for pos in current_positions
                )
                available_capital = account_value * 0.95 - current_exposure
                position_size = min(position_size, max(0, available_capital))

            # Minimum position size ($100)
            min_position = 100.0
            if position_size < min_position:
                return 0.0

            return position_size

        except Exception as e:
            logger.error("Error applying position constraints", error=str(e))
            return position_size * 0.5  # Conservative fallback

    def get_optimization_summary(self) -> Dict:
        """Get summary of position optimization settings and performance"""
        performance = self.get_historical_performance()

        return {
            "settings": {
                "max_portfolio_heat": self.max_portfolio_heat,
                "max_position_heat": self.max_position_heat,
                "kelly_fraction_range": [
                    self.min_kelly_fraction,
                    self.max_kelly_fraction,
                ],
                "correlation_threshold": self.correlation_threshold,
            },
            "performance": performance,
            "trade_count": len(self.trade_history),
            "optimizer_status": "active",
        }
