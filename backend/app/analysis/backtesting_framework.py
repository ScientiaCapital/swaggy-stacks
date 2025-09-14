"""
Comprehensive Backtesting Framework for Unsupervised Learning Validation
Production-grade backtesting to measure 30%+ improvement requirement
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import asyncio
import time
from dataclasses import dataclass
from enum import Enum

from app.core.logging import get_logger
from app.ml.unsupervised.clustering import MarketDataClusterer
from app.ml.unsupervised.market_regime import MarketRegimeDetector
from app.ml.unsupervised.anomaly_detector import EnsembleAnomalyDetector
from app.ml.unsupervised.pattern_memory import PatternMemorySystem
from app.monitoring.metrics import PrometheusMetrics

logger = get_logger(__name__)


class StrategyType(Enum):
    """Strategy types for backtesting comparison"""
    BASELINE = "baseline"
    UNSUPERVISED_ENHANCED = "unsupervised_enhanced"
    CLUSTERING_ONLY = "clustering_only"
    REGIME_ONLY = "regime_only"
    ANOMALY_ONLY = "anomaly_only"


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001  # 0.1% per trade
    max_position_size: float = 0.1  # 10% of portfolio
    rebalance_frequency: str = '1H'  # Hourly rebalancing
    lookback_window: int = 100  # Data points for analysis
    min_confidence_threshold: float = 0.7
    risk_free_rate: float = 0.02  # Annual risk-free rate


@dataclass
class TradeSignal:
    """Individual trade signal"""
    timestamp: datetime
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float
    position_size: float
    reasoning: Dict[str, Any]


@dataclass
class BacktestResults:
    """Comprehensive backtesting results"""
    strategy_type: StrategyType
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    performance_metrics: Dict[str, float]
    trade_history: List[Dict[str, Any]]
    equity_curve: pd.Series
    detailed_analysis: Dict[str, Any]


class BaselineStrategy:
    """Baseline trading strategy without unsupervised learning"""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.logger = get_logger(f"{__name__}.BaselineStrategy")

    def generate_signals(self, data: pd.DataFrame) -> List[TradeSignal]:
        """Generate trading signals using traditional technical analysis"""
        signals = []

        if len(data) < self.config.lookback_window:
            return signals

        # Simple moving average crossover strategy
        short_window = 20
        long_window = 50

        if len(data) < long_window:
            return signals

        # Calculate moving averages
        data = data.copy()
        data['sma_short'] = data['price'].rolling(window=short_window).mean()
        data['sma_long'] = data['price'].rolling(window=long_window).mean()

        # RSI for additional confirmation
        data['rsi'] = self._calculate_rsi(data['price'], 14)

        for i in range(long_window, len(data)):
            current_row = data.iloc[i]
            prev_row = data.iloc[i-1]

            # Signal generation logic
            action = 'hold'
            confidence = 0.5
            position_size = 0.0

            # Buy signal: short MA crosses above long MA and RSI < 70
            if (prev_row['sma_short'] <= prev_row['sma_long'] and
                current_row['sma_short'] > current_row['sma_long'] and
                current_row['rsi'] < 70):
                action = 'buy'
                confidence = 0.6
                position_size = self.config.max_position_size

            # Sell signal: short MA crosses below long MA and RSI > 30
            elif (prev_row['sma_short'] >= prev_row['sma_long'] and
                  current_row['sma_short'] < current_row['sma_long'] and
                  current_row['rsi'] > 30):
                action = 'sell'
                confidence = 0.6
                position_size = self.config.max_position_size

            if action != 'hold':
                signal = TradeSignal(
                    timestamp=current_row['timestamp'],
                    symbol='BACKTEST',
                    action=action,
                    confidence=confidence,
                    position_size=position_size,
                    reasoning={
                        'strategy': 'baseline_ma_crossover',
                        'sma_short': current_row['sma_short'],
                        'sma_long': current_row['sma_long'],
                        'rsi': current_row['rsi']
                    }
                )
                signals.append(signal)

        return signals

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class UnsupervisedEnhancedStrategy:
    """Trading strategy enhanced with unsupervised learning insights"""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.clusterer = MarketDataClusterer()
        self.regime_detector = MarketRegimeDetector()
        self.anomaly_detector = EnsembleAnomalyDetector()
        self.pattern_memory = PatternMemorySystem()
        self.baseline_strategy = BaselineStrategy(config)
        self.logger = get_logger(f"{__name__}.UnsupervisedEnhancedStrategy")

    def generate_signals(self, data: pd.DataFrame) -> List[TradeSignal]:
        """Generate enhanced trading signals using unsupervised learning"""
        if len(data) < self.config.lookback_window:
            return []

        # Get baseline signals first
        baseline_signals = self.baseline_strategy.generate_signals(data)

        # Apply unsupervised learning enhancements
        enhanced_signals = []

        try:
            # Generate unsupervised insights
            clustering_result = self.clusterer.cluster_market_data('BACKTEST', data)
            regime_result = self.regime_detector.detect_regime(data)
            anomaly_result = self.anomaly_detector.detect_anomalies(data)

            # Enhance each baseline signal
            for signal in baseline_signals:
                enhanced_signal = self._enhance_signal_with_unsupervised_insights(
                    signal, data, clustering_result, regime_result, anomaly_result
                )
                enhanced_signals.append(enhanced_signal)

            # Generate additional signals from unsupervised insights
            additional_signals = self._generate_unsupervised_signals(
                data, clustering_result, regime_result, anomaly_result
            )
            enhanced_signals.extend(additional_signals)

        except Exception as e:
            self.logger.warning(f"Unsupervised enhancement failed: {e}")
            return baseline_signals

        return enhanced_signals

    def _enhance_signal_with_unsupervised_insights(
        self,
        signal: TradeSignal,
        data: pd.DataFrame,
        clustering_result: Dict[str, Any],
        regime_result: Dict[str, Any],
        anomaly_result: Dict[str, Any]
    ) -> TradeSignal:
        """Enhance baseline signal with unsupervised learning insights"""

        # Find signal timestamp index
        signal_idx = None
        for i, row in data.iterrows():
            if row['timestamp'] == signal.timestamp:
                signal_idx = i
                break

        if signal_idx is None:
            return signal

        enhanced_signal = signal
        confidence_adjustments = []

        # Market regime enhancement
        current_regime = regime_result['regime']
        regime_confidence = regime_result['confidence']

        if current_regime == 'bull' and signal.action == 'buy':
            confidence_adjustments.append(0.15)  # Boost buy signals in bull market
        elif current_regime == 'bear' and signal.action == 'sell':
            confidence_adjustments.append(0.15)  # Boost sell signals in bear market
        elif current_regime == 'volatile':
            confidence_adjustments.append(-0.1)  # Reduce confidence in volatile markets

        # Anomaly detection enhancement
        if signal_idx < len(anomaly_result['anomaly_flags']):
            is_anomaly = anomaly_result['anomaly_flags'][signal_idx]
            anomaly_score = anomaly_result['anomaly_scores'][signal_idx]

            if is_anomaly and anomaly_score > 0.8:
                # High anomaly - be cautious
                confidence_adjustments.append(-0.2)
            elif anomaly_score < 0.3:
                # Normal conditions - slight boost
                confidence_adjustments.append(0.05)

        # Clustering enhancement
        if signal_idx < len(clustering_result['kmeans_clusters']):
            cluster_id = clustering_result['kmeans_clusters'][signal_idx]
            # Analyze cluster characteristics for position sizing
            cluster_volatility = self._analyze_cluster_volatility(data, clustering_result, cluster_id)

            if cluster_volatility < 0.01:  # Low volatility cluster
                enhanced_signal.position_size *= 1.2  # Increase position size
                confidence_adjustments.append(0.1)
            elif cluster_volatility > 0.03:  # High volatility cluster
                enhanced_signal.position_size *= 0.7  # Reduce position size
                confidence_adjustments.append(-0.1)

        # Apply confidence adjustments
        total_adjustment = sum(confidence_adjustments)
        enhanced_signal.confidence = min(1.0, max(0.0, signal.confidence + total_adjustment))

        # Update reasoning
        enhanced_signal.reasoning.update({
            'unsupervised_enhancement': True,
            'regime': current_regime,
            'regime_confidence': regime_confidence,
            'confidence_adjustments': confidence_adjustments,
            'total_adjustment': total_adjustment
        })

        return enhanced_signal

    def _generate_unsupervised_signals(
        self,
        data: pd.DataFrame,
        clustering_result: Dict[str, Any],
        regime_result: Dict[str, Any],
        anomaly_result: Dict[str, Any]
    ) -> List[TradeSignal]:
        """Generate additional signals based purely on unsupervised insights"""
        signals = []

        # Regime transition signals
        if regime_result['stability'] < 0.5:  # Unstable regime - potential transition
            signal = TradeSignal(
                timestamp=data['timestamp'].iloc[-1],
                symbol='BACKTEST',
                action='hold',  # Conservative during regime uncertainty
                confidence=0.8,
                position_size=0.0,
                reasoning={
                    'strategy': 'regime_transition',
                    'regime_stability': regime_result['stability'],
                    'action_rationale': 'regime_uncertainty'
                }
            )
            signals.append(signal)

        # Anomaly-based signals
        recent_anomalies = anomaly_result['anomaly_flags'][-10:]  # Last 10 points
        if np.sum(recent_anomalies) >= 3:  # Multiple recent anomalies
            signal = TradeSignal(
                timestamp=data['timestamp'].iloc[-1],
                symbol='BACKTEST',
                action='sell',  # Defensive action
                confidence=0.75,
                position_size=self.config.max_position_size * 0.5,
                reasoning={
                    'strategy': 'anomaly_defense',
                    'recent_anomaly_count': np.sum(recent_anomalies),
                    'action_rationale': 'anomaly_cluster_detected'
                }
            )
            signals.append(signal)

        return signals

    def _analyze_cluster_volatility(
        self,
        data: pd.DataFrame,
        clustering_result: Dict[str, Any],
        cluster_id: int
    ) -> float:
        """Analyze volatility characteristics of a specific cluster"""
        clusters = clustering_result['kmeans_clusters']
        cluster_mask = np.array(clusters) == cluster_id

        if np.sum(cluster_mask) < 5:  # Too few points
            return data['volatility'].mean()

        cluster_volatility = data.loc[cluster_mask, 'volatility'].mean()
        return cluster_volatility


class BacktestingEngine:
    """Main backtesting engine for strategy comparison"""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.strategies = {
            StrategyType.BASELINE: BaselineStrategy(config),
            StrategyType.UNSUPERVISED_ENHANCED: UnsupervisedEnhancedStrategy(config)
        }
        self.logger = get_logger(f"{__name__}.BacktestingEngine")

    async def run_comprehensive_backtest(
        self,
        data: pd.DataFrame,
        strategies: Optional[List[StrategyType]] = None
    ) -> Dict[StrategyType, BacktestResults]:
        """Run comprehensive backtesting across multiple strategies"""

        if strategies is None:
            strategies = [StrategyType.BASELINE, StrategyType.UNSUPERVISED_ENHANCED]

        results = {}

        for strategy_type in strategies:
            self.logger.info(f"Running backtest for {strategy_type.value}")

            try:
                result = await self._run_single_strategy_backtest(strategy_type, data)
                results[strategy_type] = result
                self.logger.info(f"Completed {strategy_type.value}: {result.total_return:.2%} return")

            except Exception as e:
                self.logger.error(f"Backtest failed for {strategy_type.value}: {e}")
                continue

        return results

    async def _run_single_strategy_backtest(
        self,
        strategy_type: StrategyType,
        data: pd.DataFrame
    ) -> BacktestResults:
        """Run backtest for a single strategy"""

        strategy = self.strategies[strategy_type]

        # Initialize portfolio
        portfolio_value = self.config.initial_capital
        positions = {}
        cash = self.config.initial_capital
        trade_history = []
        equity_curve = []

        # Portfolio tracking
        equity_timestamps = []
        equity_values = []

        # Performance metrics tracking
        daily_returns = []
        trades_executed = 0
        winning_trades = 0
        total_profit = 0.0
        total_loss = 0.0
        max_equity = self.config.initial_capital
        max_drawdown = 0.0

        # Generate all signals first
        self.logger.info(f"Generating signals for {strategy_type.value}")
        signals = strategy.generate_signals(data)
        self.logger.info(f"Generated {len(signals)} signals")

        # Execute signals chronologically
        signal_idx = 0
        for i, row in data.iterrows():
            current_time = row['timestamp']
            current_price = row['price']

            # Execute any signals for this timestamp
            while (signal_idx < len(signals) and
                   signals[signal_idx].timestamp <= current_time):

                signal = signals[signal_idx]

                # Execute trade
                trade_result = self._execute_trade(
                    signal, current_price, cash, positions, portfolio_value
                )

                if trade_result['executed']:
                    cash = trade_result['new_cash']
                    positions = trade_result['new_positions']
                    trade_history.append(trade_result['trade_record'])
                    trades_executed += 1

                    # Update performance metrics
                    if trade_result['pnl'] > 0:
                        winning_trades += 1
                        total_profit += trade_result['pnl']
                    else:
                        total_loss += abs(trade_result['pnl'])

                signal_idx += 1

            # Update portfolio value
            position_value = sum(
                pos['quantity'] * current_price for pos in positions.values()
            )
            portfolio_value = cash + position_value

            # Track equity curve
            equity_timestamps.append(current_time)
            equity_values.append(portfolio_value)

            # Update max drawdown
            if portfolio_value > max_equity:
                max_equity = portfolio_value

            current_drawdown = (max_equity - portfolio_value) / max_equity
            if current_drawdown > max_drawdown:
                max_drawdown = current_drawdown

            # Daily returns for Sharpe calculation
            if len(equity_values) > 1:
                daily_return = (equity_values[-1] - equity_values[-2]) / equity_values[-2]
                daily_returns.append(daily_return)

        # Calculate final performance metrics
        total_return = (portfolio_value - self.config.initial_capital) / self.config.initial_capital

        # Annualized return
        total_days = (data['timestamp'].iloc[-1] - data['timestamp'].iloc[0]).days
        annualized_return = (1 + total_return) ** (365 / max(total_days, 1)) - 1

        # Volatility and Sharpe ratio
        volatility = np.std(daily_returns) * np.sqrt(252) if daily_returns else 0
        excess_return = annualized_return - self.config.risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0

        # Win rate and profit factor
        win_rate = winning_trades / trades_executed if trades_executed > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        # Average trade duration
        avg_trade_duration = self._calculate_avg_trade_duration(trade_history)

        # Create equity curve series
        equity_curve = pd.Series(equity_values, index=equity_timestamps)

        # Detailed analysis
        detailed_analysis = self._generate_detailed_analysis(
            trade_history, equity_curve, data
        )

        return BacktestResults(
            strategy_type=strategy_type,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=trades_executed,
            avg_trade_duration=avg_trade_duration,
            performance_metrics={
                'total_profit': total_profit,
                'total_loss': total_loss,
                'final_portfolio_value': portfolio_value,
                'max_equity': max_equity
            },
            trade_history=trade_history,
            equity_curve=equity_curve,
            detailed_analysis=detailed_analysis
        )

    def _execute_trade(
        self,
        signal: TradeSignal,
        current_price: float,
        cash: float,
        positions: Dict[str, Dict],
        portfolio_value: float
    ) -> Dict[str, Any]:
        """Execute a single trade"""

        if signal.confidence < self.config.min_confidence_threshold:
            return {'executed': False, 'reason': 'low_confidence'}

        symbol = signal.symbol
        action = signal.action
        position_size = signal.position_size

        # Calculate trade amount
        if action == 'buy':
            max_trade_value = portfolio_value * position_size
            trade_quantity = max_trade_value / current_price
            trade_cost = trade_quantity * current_price * (1 + self.config.transaction_cost)

            if trade_cost <= cash:
                # Execute buy
                new_cash = cash - trade_cost
                new_positions = positions.copy()

                if symbol in new_positions:
                    new_positions[symbol]['quantity'] += trade_quantity
                    new_positions[symbol]['avg_price'] = (
                        (new_positions[symbol]['avg_price'] * new_positions[symbol]['quantity'] +
                         current_price * trade_quantity) /
                        (new_positions[symbol]['quantity'] + trade_quantity)
                    )
                else:
                    new_positions[symbol] = {
                        'quantity': trade_quantity,
                        'avg_price': current_price
                    }

                return {
                    'executed': True,
                    'new_cash': new_cash,
                    'new_positions': new_positions,
                    'pnl': -self.config.transaction_cost * trade_quantity * current_price,
                    'trade_record': {
                        'timestamp': signal.timestamp,
                        'symbol': symbol,
                        'action': action,
                        'quantity': trade_quantity,
                        'price': current_price,
                        'value': trade_quantity * current_price,
                        'confidence': signal.confidence,
                        'reasoning': signal.reasoning
                    }
                }

        elif action == 'sell' and symbol in positions:
            # Execute sell
            current_position = positions[symbol]
            sell_quantity = min(current_position['quantity'],
                              portfolio_value * position_size / current_price)

            if sell_quantity > 0:
                trade_value = sell_quantity * current_price * (1 - self.config.transaction_cost)
                pnl = sell_quantity * (current_price - current_position['avg_price'])

                new_cash = cash + trade_value
                new_positions = positions.copy()

                if sell_quantity >= current_position['quantity']:
                    del new_positions[symbol]
                else:
                    new_positions[symbol]['quantity'] -= sell_quantity

                return {
                    'executed': True,
                    'new_cash': new_cash,
                    'new_positions': new_positions,
                    'pnl': pnl - self.config.transaction_cost * sell_quantity * current_price,
                    'trade_record': {
                        'timestamp': signal.timestamp,
                        'symbol': symbol,
                        'action': action,
                        'quantity': sell_quantity,
                        'price': current_price,
                        'value': sell_quantity * current_price,
                        'confidence': signal.confidence,
                        'reasoning': signal.reasoning
                    }
                }

        return {'executed': False, 'reason': 'insufficient_funds_or_position'}

    def _calculate_avg_trade_duration(self, trade_history: List[Dict]) -> float:
        """Calculate average trade duration in hours"""
        if len(trade_history) < 2:
            return 0.0

        durations = []
        open_trades = {}

        for trade in trade_history:
            symbol = trade['symbol']
            action = trade['action']
            timestamp = trade['timestamp']

            if action == 'buy':
                open_trades[symbol] = timestamp
            elif action == 'sell' and symbol in open_trades:
                duration = (timestamp - open_trades[symbol]).total_seconds() / 3600
                durations.append(duration)
                del open_trades[symbol]

        return np.mean(durations) if durations else 0.0

    def _generate_detailed_analysis(
        self,
        trade_history: List[Dict],
        equity_curve: pd.Series,
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate detailed performance analysis"""

        analysis = {
            'trade_analysis': self._analyze_trades(trade_history),
            'risk_analysis': self._analyze_risk(equity_curve),
            'market_correlation': self._analyze_market_correlation(equity_curve, market_data),
            'monthly_performance': self._analyze_monthly_performance(equity_curve)
        }

        return analysis

    def _analyze_trades(self, trade_history: List[Dict]) -> Dict[str, Any]:
        """Analyze trade patterns"""
        if not trade_history:
            return {}

        buy_trades = [t for t in trade_history if t['action'] == 'buy']
        sell_trades = [t for t in trade_history if t['action'] == 'sell']

        return {
            'total_trades': len(trade_history),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'avg_trade_size': np.mean([t['value'] for t in trade_history]),
            'avg_confidence': np.mean([t['confidence'] for t in trade_history]),
            'confidence_distribution': {
                'high': len([t for t in trade_history if t['confidence'] > 0.8]),
                'medium': len([t for t in trade_history if 0.6 <= t['confidence'] <= 0.8]),
                'low': len([t for t in trade_history if t['confidence'] < 0.6])
            }
        }

    def _analyze_risk(self, equity_curve: pd.Series) -> Dict[str, Any]:
        """Analyze risk metrics"""
        if len(equity_curve) < 2:
            return {}

        returns = equity_curve.pct_change().dropna()

        return {
            'var_95': np.percentile(returns, 5),
            'var_99': np.percentile(returns, 1),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'downside_deviation': np.std(returns[returns < 0]),
            'upside_deviation': np.std(returns[returns > 0])
        }

    def _analyze_market_correlation(
        self,
        equity_curve: pd.Series,
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze correlation with market"""
        if len(equity_curve) < 2:
            return {}

        # Align timestamps
        market_returns = market_data.set_index('timestamp')['price'].pct_change().dropna()
        strategy_returns = equity_curve.pct_change().dropna()

        # Find common timestamps
        common_times = market_returns.index.intersection(strategy_returns.index)

        if len(common_times) < 10:
            return {'correlation': 0.0, 'beta': 0.0}

        market_aligned = market_returns[common_times]
        strategy_aligned = strategy_returns[common_times]

        correlation = np.corrcoef(market_aligned, strategy_aligned)[0, 1]

        # Calculate beta
        market_var = np.var(market_aligned)
        covariance = np.cov(market_aligned, strategy_aligned)[0, 1]
        beta = covariance / market_var if market_var > 0 else 0

        return {
            'correlation': correlation,
            'beta': beta,
            'alpha': np.mean(strategy_aligned) - beta * np.mean(market_aligned)
        }

    def _analyze_monthly_performance(self, equity_curve: pd.Series) -> Dict[str, Any]:
        """Analyze monthly performance breakdown"""
        if len(equity_curve) < 2:
            return {}

        monthly_returns = equity_curve.resample('M').last().pct_change().dropna()

        return {
            'monthly_returns': monthly_returns.to_dict(),
            'best_month': float(monthly_returns.max()),
            'worst_month': float(monthly_returns.min()),
            'positive_months': int((monthly_returns > 0).sum()),
            'negative_months': int((monthly_returns < 0).sum())
        }

    def generate_comparison_report(
        self,
        results: Dict[StrategyType, BacktestResults]
    ) -> Dict[str, Any]:
        """Generate comprehensive comparison report"""

        if StrategyType.BASELINE not in results or StrategyType.UNSUPERVISED_ENHANCED not in results:
            raise ValueError("Both baseline and enhanced strategy results required for comparison")

        baseline = results[StrategyType.BASELINE]
        enhanced = results[StrategyType.UNSUPERVISED_ENHANCED]

        # Calculate improvement metrics
        return_improvement = (enhanced.total_return - baseline.total_return) / abs(baseline.total_return) if baseline.total_return != 0 else 0
        sharpe_improvement = (enhanced.sharpe_ratio - baseline.sharpe_ratio) / abs(baseline.sharpe_ratio) if baseline.sharpe_ratio != 0 else 0

        report = {
            'improvement_analysis': {
                'return_improvement_pct': return_improvement * 100,
                'sharpe_improvement_pct': sharpe_improvement * 100,
                'drawdown_improvement': baseline.max_drawdown - enhanced.max_drawdown,
                'win_rate_improvement': enhanced.win_rate - baseline.win_rate,
                'meets_30_percent_target': return_improvement >= 0.30
            },
            'comparative_metrics': {
                'baseline': {
                    'total_return': baseline.total_return,
                    'sharpe_ratio': baseline.sharpe_ratio,
                    'max_drawdown': baseline.max_drawdown,
                    'win_rate': baseline.win_rate,
                    'total_trades': baseline.total_trades
                },
                'enhanced': {
                    'total_return': enhanced.total_return,
                    'sharpe_ratio': enhanced.sharpe_ratio,
                    'max_drawdown': enhanced.max_drawdown,
                    'win_rate': enhanced.win_rate,
                    'total_trades': enhanced.total_trades
                }
            },
            'statistical_significance': self._test_statistical_significance(baseline, enhanced),
            'risk_adjusted_performance': {
                'baseline_risk_adjusted_return': baseline.total_return / baseline.volatility if baseline.volatility > 0 else 0,
                'enhanced_risk_adjusted_return': enhanced.total_return / enhanced.volatility if enhanced.volatility > 0 else 0
            }
        }

        return report

    def _test_statistical_significance(
        self,
        baseline: BacktestResults,
        enhanced: BacktestResults
    ) -> Dict[str, Any]:
        """Test statistical significance of performance difference"""

        baseline_returns = baseline.equity_curve.pct_change().dropna()
        enhanced_returns = enhanced.equity_curve.pct_change().dropna()

        # Align returns
        min_length = min(len(baseline_returns), len(enhanced_returns))
        baseline_aligned = baseline_returns.iloc[:min_length]
        enhanced_aligned = enhanced_returns.iloc[:min_length]

        # T-test for mean difference
        from scipy import stats
        t_stat, p_value = stats.ttest_rel(enhanced_aligned, baseline_aligned)

        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant_at_95': p_value < 0.05,
            'significant_at_99': p_value < 0.01
        }


async def run_comprehensive_validation(
    market_data: pd.DataFrame,
    config: Optional[BacktestConfig] = None
) -> Dict[str, Any]:
    """Run comprehensive validation to meet 30%+ improvement requirement"""

    if config is None:
        config = BacktestConfig(
            start_date=market_data['timestamp'].min(),
            end_date=market_data['timestamp'].max()
        )

    engine = BacktestingEngine(config)

    # Run backtesting
    logger.info("Starting comprehensive backtesting validation")
    results = await engine.run_comprehensive_backtest(market_data)

    # Generate comparison report
    if len(results) >= 2:
        comparison_report = engine.generate_comparison_report(results)

        # Log key findings
        improvement = comparison_report['improvement_analysis']['return_improvement_pct']
        meets_target = comparison_report['improvement_analysis']['meets_30_percent_target']

        logger.info(f"Backtesting complete: {improvement:.1f}% improvement")
        logger.info(f"Meets 30% target: {meets_target}")

        return {
            'backtest_results': results,
            'comparison_report': comparison_report,
            'validation_passed': meets_target
        }
    else:
        logger.error("Insufficient backtest results for comparison")
        return {
            'backtest_results': results,
            'validation_passed': False,
            'error': 'Insufficient strategies completed'
        }


if __name__ == "__main__":
    # Example usage
    logger.info("Backtesting framework ready for production validation")