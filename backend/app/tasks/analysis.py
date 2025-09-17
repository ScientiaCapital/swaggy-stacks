"""
Analysis and calculation background tasks
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from celery import current_task
from app.core.celery_app import celery_app
from app.core.cache import strategy_cache, market_data_cache
from app.analysis.consolidated_markov_system import ConsolidatedMarkovSystem
from app.indicators.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, name='app.tasks.analysis.calculate_portfolio_metrics')
def calculate_portfolio_metrics(self, portfolio_id: str = 'default') -> Dict[str, Any]:
    """Calculate comprehensive portfolio metrics"""
    try:
        current_task.update_state(
            state='PROGRESS',
            meta={'portfolio_id': portfolio_id, 'progress': 0}
        )

        # Check cache first
        cached_metrics = strategy_cache.get_portfolio_metrics(portfolio_id)
        if cached_metrics:
            return {
                'portfolio_id': portfolio_id,
                'status': 'cached',
                'metrics': cached_metrics,
                'calculated_at': datetime.utcnow().isoformat()
            }

        current_task.update_state(
            state='PROGRESS',
            meta={'portfolio_id': portfolio_id, 'progress': 25}
        )

        # Mock portfolio metrics calculation
        # In production, this would fetch actual portfolio data from database
        metrics = {
            'total_value': 100000.0,
            'daily_pnl': 250.0,
            'daily_return_percent': 0.25,
            'total_return_percent': 5.0,
            'volatility': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.05,
            'positions_count': 8,
            'cash_balance': 25000.0,
            'margin_used': 0.0,
            'buying_power': 200000.0
        }

        current_task.update_state(
            state='PROGRESS',
            meta={'portfolio_id': portfolio_id, 'progress': 75}
        )

        # Cache the results
        strategy_cache.set_portfolio_metrics(portfolio_id, metrics, ttl=300)

        current_task.update_state(
            state='PROGRESS',
            meta={'portfolio_id': portfolio_id, 'progress': 100}
        )

        return {
            'portfolio_id': portfolio_id,
            'status': 'calculated',
            'metrics': metrics,
            'calculated_at': datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error calculating portfolio metrics for {portfolio_id}: {e}")
        return {
            'portfolio_id': portfolio_id,
            'status': 'error',
            'error': str(e),
            'calculated_at': datetime.utcnow().isoformat()
        }

@celery_app.task(bind=True, name='app.tasks.analysis.calculate_technical_indicators')
def calculate_technical_indicators(self, symbol: str, indicators: List[str] = None) -> Dict[str, Any]:
    """Calculate technical indicators for a symbol"""
    try:
        if indicators is None:
            indicators = ['rsi', 'macd', 'bollinger_bands', 'sma_20', 'ema_12']

        current_task.update_state(
            state='PROGRESS',
            meta={'symbol': symbol, 'indicators': indicators, 'progress': 0}
        )

        # Get historical data
        historical_data = market_data_cache.get_historical_data(symbol, '1Day', 30)
        if not historical_data:
            return {
                'symbol': symbol,
                'status': 'no_data',
                'calculated_at': datetime.utcnow().isoformat()
            }

        current_task.update_state(
            state='PROGRESS',
            meta={'symbol': symbol, 'indicators': indicators, 'progress': 25}
        )

        # Convert to price arrays
        prices = [float(bar['close']) for bar in historical_data]
        high_prices = [float(bar['high']) for bar in historical_data]
        low_prices = [float(bar['low']) for bar in historical_data]
        volumes = [int(bar['volume']) for bar in historical_data]

        technical = TechnicalIndicators()
        results = {}

        current_task.update_state(
            state='PROGRESS',
            meta={'symbol': symbol, 'indicators': indicators, 'progress': 50}
        )

        # Calculate indicators
        for i, indicator in enumerate(indicators):
            try:
                if indicator == 'rsi':
                    rsi_values = technical.rsi(prices, period=14)
                    results['rsi'] = {
                        'current': rsi_values[-1] if rsi_values else None,
                        'values': rsi_values[-5:] if rsi_values else []
                    }

                elif indicator == 'macd':
                    macd_line, macd_signal, macd_histogram = technical.macd(prices)
                    results['macd'] = {
                        'line': macd_line[-1] if macd_line else None,
                        'signal': macd_signal[-1] if macd_signal else None,
                        'histogram': macd_histogram[-1] if macd_histogram else None
                    }

                elif indicator == 'bollinger_bands':
                    bb_upper, bb_middle, bb_lower = technical.bollinger_bands(prices, period=20)
                    results['bollinger_bands'] = {
                        'upper': bb_upper[-1] if bb_upper else None,
                        'middle': bb_middle[-1] if bb_middle else None,
                        'lower': bb_lower[-1] if bb_lower else None
                    }

                elif indicator == 'sma_20':
                    sma = technical.sma(prices, period=20)
                    results['sma_20'] = {
                        'current': sma[-1] if sma else None,
                        'values': sma[-5:] if sma else []
                    }

                elif indicator == 'ema_12':
                    ema = technical.ema(prices, period=12)
                    results['ema_12'] = {
                        'current': ema[-1] if ema else None,
                        'values': ema[-5:] if ema else []
                    }

                # Update progress
                progress = 50 + (i + 1) / len(indicators) * 40
                current_task.update_state(
                    state='PROGRESS',
                    meta={'symbol': symbol, 'indicator': indicator, 'progress': progress}
                )

            except Exception as e:
                logger.warning(f"Error calculating {indicator} for {symbol}: {e}")
                results[indicator] = {'error': str(e)}

        # Cache the results
        for indicator, values in results.items():
            if 'error' not in values:
                market_data_cache.set_technical_indicators(
                    symbol, indicator, {}, values, ttl=300
                )

        current_task.update_state(
            state='PROGRESS',
            meta={'symbol': symbol, 'progress': 100}
        )

        return {
            'symbol': symbol,
            'status': 'calculated',
            'indicators': results,
            'data_points': len(historical_data),
            'calculated_at': datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error calculating technical indicators for {symbol}: {e}")
        return {
            'symbol': symbol,
            'status': 'error',
            'error': str(e),
            'calculated_at': datetime.utcnow().isoformat()
        }

@celery_app.task(bind=True, name='app.tasks.analysis.run_markov_analysis')
def run_markov_analysis(self, symbol: str, lookback: int = 100, n_states: int = 5) -> Dict[str, Any]:
    """Run Markov chain analysis for a symbol"""
    try:
        current_task.update_state(
            state='PROGRESS',
            meta={'symbol': symbol, 'lookback': lookback, 'n_states': n_states, 'progress': 0}
        )

        # Check cache first
        cached_analysis = strategy_cache.get_markov_analysis(symbol, lookback, n_states)
        if cached_analysis:
            return {
                'symbol': symbol,
                'status': 'cached',
                'analysis': cached_analysis,
                'calculated_at': datetime.utcnow().isoformat()
            }

        current_task.update_state(
            state='PROGRESS',
            meta={'symbol': symbol, 'progress': 25}
        )

        # Get historical data
        historical_data = market_data_cache.get_historical_data(symbol, '1Day', lookback + 10)
        if not historical_data or len(historical_data) < lookback:
            return {
                'symbol': symbol,
                'status': 'insufficient_data',
                'required_points': lookback,
                'available_points': len(historical_data) if historical_data else 0,
                'calculated_at': datetime.utcnow().isoformat()
            }

        current_task.update_state(
            state='PROGRESS',
            meta={'symbol': symbol, 'progress': 50}
        )

        # Initialize Markov system
        markov_system = ConsolidatedMarkovSystem(
            lookback_period=lookback,
            n_states=n_states
        )

        # Convert historical data to required format
        prices = [float(bar['close']) for bar in historical_data]
        volumes = [int(bar['volume']) for bar in historical_data]

        current_task.update_state(
            state='PROGRESS',
            meta={'symbol': symbol, 'progress': 75}
        )

        # Run analysis
        analysis_result = markov_system.analyze_symbol(symbol, prices, volumes)

        # Cache the results
        strategy_cache.set_markov_analysis(symbol, lookback, n_states, analysis_result, ttl=1800)

        current_task.update_state(
            state='PROGRESS',
            meta={'symbol': symbol, 'progress': 100}
        )

        return {
            'symbol': symbol,
            'status': 'calculated',
            'analysis': analysis_result,
            'lookback': lookback,
            'n_states': n_states,
            'data_points': len(historical_data),
            'calculated_at': datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error running Markov analysis for {symbol}: {e}")
        return {
            'symbol': symbol,
            'status': 'error',
            'error': str(e),
            'lookback': lookback,
            'n_states': n_states,
            'calculated_at': datetime.utcnow().isoformat()
        }