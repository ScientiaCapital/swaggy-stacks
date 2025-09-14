"""
Performance Analytics Service - Extracted from BacktestEngine

Handles advanced performance metrics, risk calculations, and result analysis
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict

from app.backtesting.portfolio_manager import PortfolioManager, TradeRecord

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Comprehensive backtest results"""
    backtest_id: str
    config: Dict[str, Any]
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    alpha_generated: float
    beta: float
    volatility: float
    var_95: float  # Value at Risk 95%
    calmar_ratio: float
    sortino_ratio: float
    portfolio_value_history: List[Dict[str, Any]]
    trade_history: List[Dict[str, Any]]
    correlation_matrix: Optional[Dict[str, Any]] = None
    performance_attribution: Optional[Dict[str, float]] = None
    execution_time_seconds: float = 0.0
    memory_used_mb: float = 0.0


class PerformanceAnalyzer:
    """
    Advanced performance analytics for backtesting results

    Calculates comprehensive metrics including:
    - Risk-adjusted returns (Sharpe, Sortino, Calmar)
    - Drawdown analysis
    - Portfolio correlation analysis
    - Performance attribution
    - Statistical significance tests
    """

    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        logger.info(f"PerformanceAnalyzer initialized with {risk_free_rate:.2%} risk-free rate")

    def calculate_comprehensive_metrics(
        self,
        backtest_id: str,
        symbols: List[str],
        portfolio_manager: PortfolioManager,
        market_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
        execution_start_time: float,
        config: Dict[str, Any] = None
    ) -> BacktestResult:
        """Calculate comprehensive backtest performance metrics"""
        try:
            # Basic portfolio metrics
            portfolio_metrics = portfolio_manager.calculate_performance_metrics()
            portfolio_history = portfolio_manager.get_portfolio_history()
            trade_history = portfolio_manager.get_trade_history()

            if 'error' in portfolio_metrics:
                logger.error(f"Portfolio metrics calculation failed: {portfolio_metrics['error']}")
                return self._create_error_result(backtest_id, symbols, str(portfolio_metrics['error']))

            # Extract portfolio value series
            portfolio_values = [snapshot['total_value'] for snapshot in portfolio_history]
            if len(portfolio_values) < 2:
                logger.error("Insufficient portfolio history for analysis")
                return self._create_error_result(backtest_id, symbols, "Insufficient data")

            # Calculate returns
            returns_series = pd.Series(portfolio_values).pct_change().dropna()

            # Core performance metrics
            total_return = portfolio_metrics['total_return']
            trading_days = len(returns_series)
            annualized_return = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0

            # Risk metrics
            volatility = returns_series.std() * np.sqrt(252) if len(returns_series) > 1 else 0

            sharpe_ratio = self._calculate_sharpe_ratio(returns_series, self.risk_free_rate)
            sortino_ratio = self._calculate_sortino_ratio(returns_series, self.risk_free_rate)
            max_drawdown = self._calculate_max_drawdown(portfolio_values)
            calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0

            var_95 = self._calculate_var(returns_series, confidence_level=0.95)

            # Market comparison metrics (simplified)
            alpha_generated, beta = self._calculate_alpha_beta(returns_series, market_data, symbols)

            # Execution metrics
            execution_time = time.time() - execution_start_time

            # Convert trade history to dictionaries
            trade_dicts = [asdict(trade) for trade in trade_history]

            # Serialize datetime objects
            for trade_dict in trade_dicts:
                trade_dict['timestamp'] = trade_dict['timestamp'].isoformat()

            result = BacktestResult(
                backtest_id=backtest_id,
                config=config or {},
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                total_trades=portfolio_metrics['total_trades'],
                winning_trades=portfolio_metrics['winning_trades'],
                losing_trades=portfolio_metrics['losing_trades'],
                win_rate=portfolio_metrics['win_rate'],
                total_return=total_return,
                annualized_return=annualized_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                alpha_generated=alpha_generated,
                beta=beta,
                volatility=volatility,
                var_95=var_95,
                calmar_ratio=calmar_ratio,
                sortino_ratio=sortino_ratio,
                portfolio_value_history=portfolio_history,
                trade_history=trade_dicts,
                execution_time_seconds=execution_time,
                memory_used_mb=0.0  # Could implement memory tracking
            )

            logger.info(f"Performance analysis completed for {backtest_id}: "
                       f"{total_return:.2%} return, {sharpe_ratio:.2f} Sharpe, {max_drawdown:.2%} max drawdown")

            return result

        except Exception as e:
            logger.error(f"Comprehensive metrics calculation failed: {e}")
            return self._create_error_result(backtest_id, symbols, str(e))

    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(returns) < 2:
                return 0.0

            excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate

            if excess_returns.std() == 0:
                return 0.0

            return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        except Exception as e:
            logger.warning(f"Sharpe ratio calculation failed: {e}")
            return 0.0

    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        try:
            if len(returns) < 2:
                return 0.0

            excess_returns = returns - (risk_free_rate / 252)
            downside_returns = excess_returns[excess_returns < 0]

            if len(downside_returns) == 0 or downside_returns.std() == 0:
                return float('inf') if excess_returns.mean() > 0 else 0.0

            downside_deviation = downside_returns.std() * np.sqrt(252)
            return (excess_returns.mean() * 252) / downside_deviation
        except Exception as e:
            logger.warning(f"Sortino ratio calculation failed: {e}")
            return 0.0

    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown"""
        try:
            if len(portfolio_values) < 2:
                return 0.0

            values = np.array(portfolio_values)
            running_max = np.maximum.accumulate(values)
            drawdowns = (values - running_max) / running_max

            return abs(np.min(drawdowns))
        except Exception as e:
            logger.warning(f"Max drawdown calculation failed: {e}")
            return 0.0

    def _calculate_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk"""
        try:
            if len(returns) < 10:
                return 0.0

            return abs(np.percentile(returns, (1 - confidence_level) * 100))
        except Exception as e:
            logger.warning(f"VaR calculation failed: {e}")
            return 0.0

    def _calculate_alpha_beta(self, portfolio_returns: pd.Series, market_data: Dict[str, pd.DataFrame], symbols: List[str]) -> tuple:
        """Calculate alpha and beta vs market (simplified benchmark)"""
        try:
            if len(portfolio_returns) < 10 or not market_data:
                return 0.0, 1.0

            # Create simple benchmark from available symbols
            benchmark_returns = []

            for symbol in symbols[:3]:  # Use first 3 symbols as benchmark
                if symbol in market_data:
                    symbol_data = market_data[symbol]
                    symbol_returns = symbol_data['Close'].pct_change().dropna()
                    if len(symbol_returns) > 0:
                        benchmark_returns.append(symbol_returns.tail(len(portfolio_returns)))

            if not benchmark_returns:
                return 0.0, 1.0

            # Average benchmark returns
            benchmark = pd.concat(benchmark_returns, axis=1).mean(axis=1)

            # Align lengths
            min_length = min(len(portfolio_returns), len(benchmark))
            if min_length < 5:
                return 0.0, 1.0

            port_returns = portfolio_returns.tail(min_length)
            bench_returns = benchmark.tail(min_length)

            # Calculate beta
            covariance = np.cov(port_returns, bench_returns)[0, 1]
            market_variance = np.var(bench_returns)

            beta = covariance / market_variance if market_variance > 0 else 1.0

            # Calculate alpha (simplified)
            portfolio_mean = port_returns.mean() * 252
            benchmark_mean = bench_returns.mean() * 252
            alpha = portfolio_mean - (self.risk_free_rate + beta * (benchmark_mean - self.risk_free_rate))

            return alpha, beta

        except Exception as e:
            logger.warning(f"Alpha/Beta calculation failed: {e}")
            return 0.0, 1.0

    def calculate_correlation_matrix(self, symbols: List[str], market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate correlation matrix for portfolio symbols"""
        try:
            if len(symbols) < 2:
                return {}

            # Extract returns for each symbol
            returns_data = {}
            for symbol in symbols:
                if symbol in market_data:
                    returns = market_data[symbol]['Close'].pct_change().dropna()
                    if len(returns) > 20:  # Minimum data requirement
                        returns_data[symbol] = returns

            if len(returns_data) < 2:
                return {}

            # Create correlation matrix
            returns_df = pd.DataFrame(returns_data)
            correlation_matrix = returns_df.corr()

            # Convert to dictionary format
            result = {
                'correlation_matrix': correlation_matrix.to_dict(),
                'average_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, 1)].mean(),
                'max_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, 1)].max(),
                'min_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, 1)].min()
            }

            return result

        except Exception as e:
            logger.warning(f"Correlation matrix calculation failed: {e}")
            return {}

    def calculate_performance_attribution(self, symbols: List[str], trade_history: List[TradeRecord]) -> Dict[str, float]:
        """Calculate performance attribution by symbol"""
        try:
            attribution = {}

            for symbol in symbols:
                symbol_trades = [t for t in trade_history if t.symbol == symbol]
                if not symbol_trades:
                    attribution[symbol] = 0.0
                    continue

                # Simple P&L calculation for this symbol
                symbol_pnl = 0.0
                buy_trades = [t for t in symbol_trades if t.action == 'BUY']
                sell_trades = [t for t in symbol_trades if t.action == 'SELL']

                # Match buys and sells (simplified FIFO)
                for sell in sell_trades:
                    for buy in buy_trades:
                        if buy.timestamp <= sell.timestamp:
                            pnl = (sell.price - buy.price) * min(buy.quantity, sell.quantity)
                            symbol_pnl += pnl

                attribution[symbol] = symbol_pnl

            return attribution

        except Exception as e:
            logger.warning(f"Performance attribution calculation failed: {e}")
            return {}

    def _create_error_result(self, backtest_id: str, symbols: List[str], error_message: str) -> BacktestResult:
        """Create error result when analysis fails"""
        return BacktestResult(
            backtest_id=backtest_id,
            config={},
            symbols=symbols,
            start_date=datetime.now(),
            end_date=datetime.now(),
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_return=0.0,
            annualized_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            alpha_generated=0.0,
            beta=1.0,
            volatility=0.0,
            var_95=0.0,
            calmar_ratio=0.0,
            sortino_ratio=0.0,
            portfolio_value_history=[],
            trade_history=[],
            execution_time_seconds=0.0,
            memory_used_mb=0.0
        )

    def compare_strategies(self, results: Dict[str, BacktestResult]) -> Dict[str, Any]:
        """Compare multiple strategy results"""
        try:
            if len(results) < 2:
                return {'error': 'Need at least 2 strategies to compare'}

            comparison = {
                'strategy_count': len(results),
                'metrics_comparison': {},
                'rankings': {},
                'summary': {}
            }

            # Extract key metrics for each strategy
            metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'volatility']

            for metric in metrics:
                comparison['metrics_comparison'][metric] = {
                    strategy: getattr(result, metric) for strategy, result in results.items()
                }

            # Rank strategies by different metrics
            for metric in metrics:
                if metric == 'max_drawdown':  # Lower is better
                    ranking = sorted(results.items(), key=lambda x: getattr(x[1], metric))
                else:  # Higher is better
                    ranking = sorted(results.items(), key=lambda x: getattr(x[1], metric), reverse=True)

                comparison['rankings'][metric] = [strategy for strategy, _ in ranking]

            # Overall best strategy (weighted score)
            strategy_scores = {}
            for strategy, result in results.items():
                score = (
                    result.total_return * 0.3 +
                    result.sharpe_ratio * 0.3 +
                    result.win_rate * 0.2 +
                    (1 - result.max_drawdown) * 0.2  # Invert drawdown
                )
                strategy_scores[strategy] = score

            best_strategy = max(strategy_scores.items(), key=lambda x: x[1])
            comparison['summary'] = {
                'best_overall_strategy': best_strategy[0],
                'best_overall_score': best_strategy[1],
                'strategy_scores': strategy_scores
            }

            return comparison

        except Exception as e:
            logger.error(f"Strategy comparison failed: {e}")
            return {'error': str(e)}