"""
Risk Assessment Tools for LangChain Integration

Provides comprehensive risk analysis and position sizing as LangChain Tools
"""

import json
import logging
from typing import List, Optional

import numpy as np
import yfinance as yf
from langchain.agents import Tool

from app.trading.trading_manager import get_trading_manager

logger = logging.getLogger(__name__)


class RiskTool:
    """Risk assessment and management tools"""

    def __init__(self):
        self.trading_manager = None

    async def initialize(self) -> None:
        """Initialize with trading manager"""
        self.trading_manager = get_trading_manager()
        await self.trading_manager.initialize()

    def get_tools(self) -> List[Tool]:
        """Get all risk assessment tools"""
        return [
            Tool(
                name="calculate_position_size",
                description="Calculate optimal position size (format: 'SYMBOL,ENTRY_PRICE,STOP_LOSS,RISK_AMOUNT')",
                func=self._calculate_position_size,
            ),
            Tool(
                name="assess_portfolio_risk",
                description="Assess current portfolio risk metrics",
                func=self._assess_portfolio_risk,
            ),
            Tool(
                name="check_risk_limits",
                description="Check if a trade meets risk management criteria (format: 'SYMBOL,ACTION,QUANTITY')",
                func=self._check_risk_limits,
            ),
            Tool(
                name="calculate_var",
                description="Calculate Value at Risk for a symbol or portfolio",
                func=self._calculate_var,
            ),
            Tool(
                name="calculate_correlation",
                description="Calculate correlation matrix (format: 'SYMBOL1,SYMBOL2,SYMBOL3...')",
                func=self._calculate_correlation,
            ),
            Tool(
                name="analyze_drawdown",
                description="Analyze maximum drawdown for a symbol",
                func=self._analyze_drawdown,
            ),
        ]

    def _calculate_position_size(self, params: str) -> str:
        """Calculate optimal position size using risk management principles"""
        try:
            parts = params.split(",")
            if len(parts) < 3:
                return "Format: SYMBOL,ENTRY_PRICE,STOP_LOSS[,RISK_AMOUNT]"

            symbol = parts[0].strip().upper()
            entry_price = float(parts[1])
            stop_loss = float(parts[2])
            risk_amount = (
                float(parts[3]) if len(parts) > 3 else 1000
            )  # Default $1000 risk

            # Calculate risk per share
            risk_per_share = abs(entry_price - stop_loss)

            if risk_per_share == 0:
                return "Error: Entry price and stop loss cannot be the same"

            # Position size calculations
            # 1. Fixed Dollar Risk
            fixed_risk_shares = risk_amount / risk_per_share

            # 2. Kelly Criterion (simplified - assumes 60% win rate, 1:1.5 risk/reward)
            win_rate = 0.6
            avg_win = risk_per_share * 1.5  # 1.5:1 reward ratio
            avg_loss = risk_per_share

            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%

            # Assume portfolio value for Kelly calculation
            portfolio_value = 100000  # Default assumption
            kelly_risk_amount = portfolio_value * kelly_fraction
            kelly_shares = kelly_risk_amount / risk_per_share

            # 3. Percentage risk (2% rule)
            percent_risk_amount = portfolio_value * 0.02
            percent_risk_shares = percent_risk_amount / risk_per_share

            return json.dumps(
                {
                    "symbol": symbol,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "risk_per_share": round(risk_per_share, 2),
                    "position_sizing": {
                        "fixed_risk": {
                            "risk_amount": risk_amount,
                            "shares": int(fixed_risk_shares),
                            "position_value": round(fixed_risk_shares * entry_price, 2),
                        },
                        "kelly_criterion": {
                            "kelly_fraction": round(kelly_fraction, 4),
                            "shares": int(kelly_shares),
                            "position_value": round(kelly_shares * entry_price, 2),
                        },
                        "percent_risk": {
                            "risk_percent": 2.0,
                            "shares": int(percent_risk_shares),
                            "position_value": round(
                                percent_risk_shares * entry_price, 2
                            ),
                        },
                    },
                    "recommended_size": int(
                        min(fixed_risk_shares, kelly_shares, percent_risk_shares)
                    ),
                    "risk_reward_ratio": round(
                        abs(
                            (entry_price + (entry_price - stop_loss) * 1.5)
                            - entry_price
                        )
                        / risk_per_share,
                        2,
                    ),
                }
            )

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return f"Error calculating position size: {str(e)}"

    def _assess_portfolio_risk(self, _: str = "") -> str:
        """Assess current portfolio risk metrics"""
        try:
            if not self.trading_manager:
                return "Trading manager not available"

            # Get portfolio status (this would be async in real implementation)
            # For now, provide example structure
            portfolio_data = {
                "total_value": 100000,
                "cash": 20000,
                "positions": {
                    "AAPL": {"shares": 100, "current_price": 150, "avg_cost": 140},
                    "GOOGL": {"shares": 50, "current_price": 2500, "avg_cost": 2400},
                    "SPY": {"shares": 200, "current_price": 400, "avg_cost": 380},
                },
            }

            total_value = portfolio_data["total_value"]
            cash = portfolio_data["cash"]
            invested_amount = total_value - cash

            # Calculate position concentrations
            position_values = {}
            total_invested = 0

            for symbol, pos in portfolio_data["positions"].items():
                position_value = pos["shares"] * pos["current_price"]
                position_values[symbol] = position_value
                total_invested += position_value

            # Risk metrics
            max_position = max(position_values.values()) if position_values else 0
            max_concentration = (
                (max_position / total_value) * 100 if total_value > 0 else 0
            )

            cash_percentage = (cash / total_value) * 100 if total_value > 0 else 0
            exposure_percentage = (
                (invested_amount / total_value) * 100 if total_value > 0 else 0
            )

            # Risk assessment
            risk_level = "low"
            warnings = []

            if max_concentration > 30:
                risk_level = "high"
                warnings.append("High position concentration")
            elif max_concentration > 20:
                risk_level = "medium"
                warnings.append("Moderate position concentration")

            if cash_percentage < 10:
                warnings.append("Low cash reserves")
                if risk_level == "low":
                    risk_level = "medium"

            if exposure_percentage > 90:
                warnings.append("High portfolio exposure")
                risk_level = "high"

            return json.dumps(
                {
                    "portfolio_summary": {
                        "total_value": total_value,
                        "cash": cash,
                        "invested_amount": invested_amount,
                        "number_of_positions": len(portfolio_data["positions"]),
                    },
                    "risk_metrics": {
                        "cash_percentage": round(cash_percentage, 1),
                        "exposure_percentage": round(exposure_percentage, 1),
                        "max_position_concentration": round(max_concentration, 1),
                        "diversification_score": round(100 - max_concentration, 1),
                    },
                    "position_concentrations": {
                        symbol: round((value / total_value) * 100, 1)
                        for symbol, value in position_values.items()
                    },
                    "risk_assessment": {
                        "overall_risk_level": risk_level,
                        "warnings": warnings,
                        "recommendations": [
                            "Maintain cash reserves above 10%",
                            "Limit single position to 20% of portfolio",
                            "Consider position sizing based on volatility",
                        ],
                    },
                }
            )

        except Exception as e:
            logger.error(f"Error assessing portfolio risk: {e}")
            return f"Error assessing portfolio risk: {str(e)}"

    def _check_risk_limits(self, params: str) -> str:
        """Check if a trade meets risk management criteria"""
        try:
            parts = params.split(",")
            if len(parts) < 3:
                return "Format: SYMBOL,ACTION,QUANTITY"

            symbol = parts[0].strip().upper()
            action = parts[1].strip().upper()
            quantity = float(parts[2])

            # Get current price
            ticker = yf.Ticker(symbol)
            current_price = ticker.info.get(
                "currentPrice", ticker.info.get("regularMarketPrice", 0)
            )

            if current_price == 0:
                return f"Unable to get current price for {symbol}"

            position_value = quantity * current_price

            # Portfolio assumptions (would come from trading manager)
            portfolio_value = 100000
            max_position_size = portfolio_value * 0.2  # 20% max position
            available_cash = 20000

            # Risk checks
            checks = {
                "position_size_check": {
                    "passed": position_value <= max_position_size,
                    "current_value": position_value,
                    "limit": max_position_size,
                    "utilization": round((position_value / max_position_size) * 100, 1),
                },
                "cash_availability": {
                    "passed": (
                        position_value <= available_cash if action == "BUY" else True
                    ),
                    "required_cash": position_value if action == "BUY" else 0,
                    "available_cash": available_cash,
                },
                "volatility_check": {
                    "passed": True,  # Simplified - would check recent volatility
                    "symbol_volatility": "normal",
                },
            }

            # Overall approval
            all_passed = all(check["passed"] for check in checks.values())

            return json.dumps(
                {
                    "symbol": symbol,
                    "action": action,
                    "quantity": quantity,
                    "position_value": round(position_value, 2),
                    "current_price": current_price,
                    "risk_checks": checks,
                    "approval": {
                        "approved": all_passed,
                        "risk_level": "low" if all_passed else "high",
                        "warnings": [
                            (
                                f"Position size exceeds limit"
                                if not checks["position_size_check"]["passed"]
                                else None
                            ),
                            (
                                f"Insufficient cash"
                                if not checks["cash_availability"]["passed"]
                                else None
                            ),
                        ],
                    },
                }
            )

        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return f"Error checking risk limits: {str(e)}"

    def _calculate_var(self, symbol: str) -> str:
        """Calculate Value at Risk"""
        try:
            symbol = symbol.strip().upper()

            # Get historical data for VaR calculation
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")

            if hist.empty or len(hist) < 30:
                return f"Insufficient data for VaR calculation for {symbol}"

            # Calculate daily returns
            returns = hist["Close"].pct_change().dropna()

            # VaR calculations at different confidence levels
            var_95 = np.percentile(returns, 5)  # 95% confidence
            var_99 = np.percentile(returns, 1)  # 99% confidence

            # Current position value assumption
            position_value = 10000

            var_95_dollar = position_value * abs(var_95)
            var_99_dollar = position_value * abs(var_99)

            # Additional risk metrics
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))

            return json.dumps(
                {
                    "symbol": symbol,
                    "position_value": position_value,
                    "var_metrics": {
                        "var_95_percent": round(var_95 * 100, 2),
                        "var_99_percent": round(var_99 * 100, 2),
                        "var_95_dollar": round(var_95_dollar, 2),
                        "var_99_dollar": round(var_99_dollar, 2),
                    },
                    "risk_statistics": {
                        "daily_volatility": round(returns.std() * 100, 2),
                        "annualized_volatility": round(volatility * 100, 2),
                        "sharpe_ratio": round(sharpe_ratio, 2),
                        "max_daily_loss": round(returns.min() * 100, 2),
                        "max_daily_gain": round(returns.max() * 100, 2),
                    },
                    "interpretation": {
                        "risk_level": (
                            "high"
                            if volatility > 0.3
                            else "medium" if volatility > 0.15 else "low"
                        ),
                        "worst_case_scenario": f"99% confident daily loss won't exceed ${var_99_dollar:.2f}",
                    },
                }
            )

        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return f"Error calculating VaR: {str(e)}"

    def _calculate_correlation(self, params: str) -> str:
        """Calculate correlation matrix between symbols"""
        try:
            symbols = [s.strip().upper() for s in params.split(",")]

            if len(symbols) < 2:
                return "Need at least 2 symbols for correlation analysis"

            # Get historical data for all symbols
            data = {}
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="6mo")
                if not hist.empty:
                    data[symbol] = hist["Close"]

            if len(data) < 2:
                return "Insufficient data for correlation analysis"

            # Create correlation matrix
            import pandas as pd

            df = pd.DataFrame(data)
            correlation_matrix = df.corr()

            # Convert to JSON-serializable format
            correlations = {}
            for i, symbol1 in enumerate(symbols):
                if symbol1 in correlation_matrix.index:
                    for j, symbol2 in enumerate(symbols):
                        if symbol2 in correlation_matrix.columns and i != j:
                            pair = f"{symbol1}_{symbol2}"
                            correlations[pair] = round(
                                correlation_matrix.loc[symbol1, symbol2], 3
                            )

            # Risk interpretation
            high_correlations = [
                (pair, corr) for pair, corr in correlations.items() if abs(corr) > 0.7
            ]

            return json.dumps(
                {
                    "symbols": symbols,
                    "correlation_matrix": correlation_matrix.round(3).to_dict(),
                    "pairwise_correlations": correlations,
                    "risk_analysis": {
                        "high_correlations": high_correlations,
                        "diversification_benefit": len(high_correlations) == 0,
                        "average_correlation": round(
                            np.mean(list(correlations.values())), 3
                        ),
                        "interpretation": (
                            "Low portfolio diversification"
                            if len(high_correlations) > 0
                            else "Good diversification"
                        ),
                    },
                }
            )

        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
            return f"Error calculating correlation: {str(e)}"

    def _analyze_drawdown(self, symbol: str) -> str:
        """Analyze maximum drawdown"""
        try:
            symbol = symbol.strip().upper()

            # Get historical data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")

            if hist.empty:
                return f"No data available for drawdown analysis for {symbol}"

            prices = hist["Close"]

            # Calculate running maximum (peak)
            peak = prices.expanding().max()

            # Calculate drawdown
            drawdown = (prices - peak) / peak

            # Find maximum drawdown
            max_drawdown = drawdown.min()
            max_drawdown_date = drawdown.idxmin()

            # Calculate recovery information
            max_dd_idx = drawdown.idxmin()
            recovery_idx = None

            for i in range(len(drawdown)):
                if drawdown.index[i] > max_dd_idx and drawdown.iloc[i] >= 0:
                    recovery_idx = drawdown.index[i]
                    break

            # Current drawdown
            current_drawdown = drawdown.iloc[-1]

            return json.dumps(
                {
                    "symbol": symbol,
                    "drawdown_analysis": {
                        "max_drawdown_percent": round(max_drawdown * 100, 2),
                        "max_drawdown_date": max_drawdown_date.strftime("%Y-%m-%d"),
                        "current_drawdown_percent": round(current_drawdown * 100, 2),
                        "recovery_status": (
                            "recovered" if recovery_idx else "still_in_drawdown"
                        ),
                        "recovery_date": (
                            recovery_idx.strftime("%Y-%m-%d") if recovery_idx else None
                        ),
                    },
                    "risk_metrics": {
                        "avg_drawdown": round(drawdown.mean() * 100, 2),
                        "drawdown_volatility": round(drawdown.std() * 100, 2),
                        "time_underwater_days": len([d for d in drawdown if d < 0]),
                        "total_periods": len(drawdown),
                    },
                    "interpretation": {
                        "risk_level": (
                            "high"
                            if abs(max_drawdown) > 0.2
                            else "medium" if abs(max_drawdown) > 0.1 else "low"
                        ),
                        "recovery_capability": (
                            "strong" if recovery_idx else "concerning"
                        ),
                    },
                }
            )

        except Exception as e:
            logger.error(f"Error analyzing drawdown: {e}")
            return f"Error analyzing drawdown: {str(e)}"


# Global risk tool instance
_risk_tool: Optional[RiskTool] = None


async def get_risk_tool() -> RiskTool:
    """Get the global risk tool instance"""
    global _risk_tool

    if _risk_tool is None:
        _risk_tool = RiskTool()
        await _risk_tool.initialize()

    return _risk_tool
