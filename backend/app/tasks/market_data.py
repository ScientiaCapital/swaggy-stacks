"""
Market data background tasks
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

from celery import current_task
from app.core.celery_app import celery_app
from app.core.database import get_db_session
from app.core.cache import market_data_cache
from app.trading.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, name='app.tasks.market_data.update_symbol')
def update_symbol_data(self, symbol: str) -> Dict[str, Any]:
    """Update market data for a single symbol"""
    try:
        # Update task progress
        current_task.update_state(state='PROGRESS', meta={'symbol': symbol, 'progress': 0})

        client = AlpacaClient()

        # Get latest quote
        quote = client.get_latest_quote(symbol)
        if quote:
            # Cache the quote data
            quote_data = {
                'symbol': symbol,
                'bid_price': float(quote.bid_price),
                'ask_price': float(quote.ask_price),
                'bid_size': int(quote.bid_size),
                'ask_size': int(quote.ask_size),
                'timestamp': quote.timestamp.isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            }
            market_data_cache.set_quote(symbol, quote_data, ttl=60)

            current_task.update_state(state='PROGRESS', meta={'symbol': symbol, 'progress': 50})

        # Get latest trade
        trade = client.get_latest_trade(symbol)
        if trade:
            trade_data = {
                'symbol': symbol,
                'price': float(trade.price),
                'size': int(trade.size),
                'timestamp': trade.timestamp.isoformat(),
                'conditions': trade.conditions,
                'updated_at': datetime.utcnow().isoformat()
            }

            # Cache trade data with quote data
            if quote:
                quote_data.update(trade_data)
                market_data_cache.set_quote(symbol, quote_data, ttl=60)

        current_task.update_state(state='PROGRESS', meta={'symbol': symbol, 'progress': 100})

        return {
            'symbol': symbol,
            'status': 'success',
            'updated_at': datetime.utcnow().isoformat(),
            'has_quote': bool(quote),
            'has_trade': bool(trade)
        }

    except Exception as e:
        logger.error(f"Error updating {symbol}: {e}")
        return {
            'symbol': symbol,
            'status': 'error',
            'error': str(e),
            'updated_at': datetime.utcnow().isoformat()
        }

@celery_app.task(bind=True, name='app.tasks.market_data.update_all_symbols')
def update_all_symbols(self, symbols: List[str] = None) -> Dict[str, Any]:
    """Update market data for all tracked symbols"""
    try:
        if symbols is None:
            # Default symbols to track
            symbols = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
                'META', 'NVDA', 'NFLX', 'CRM', 'ADBE'
            ]

        current_task.update_state(
            state='PROGRESS',
            meta={'total_symbols': len(symbols), 'processed': 0}
        )

        results = []
        for i, symbol in enumerate(symbols):
            result = update_symbol_data(symbol)
            results.append(result)

            # Update progress
            current_task.update_state(
                state='PROGRESS',
                meta={
                    'total_symbols': len(symbols),
                    'processed': i + 1,
                    'current_symbol': symbol
                }
            )

        success_count = sum(1 for r in results if r['status'] == 'success')

        return {
            'status': 'completed',
            'total_symbols': len(symbols),
            'successful': success_count,
            'failed': len(symbols) - success_count,
            'results': results,
            'completed_at': datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error updating all symbols: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'completed_at': datetime.utcnow().isoformat()
        }

@celery_app.task(bind=True, name='app.tasks.market_data.fetch_historical_data')
def fetch_historical_data(self, symbol: str, timeframe: str = '1Day',
                         lookback_days: int = 30) -> Dict[str, Any]:
    """Fetch and cache historical data for a symbol"""
    try:
        current_task.update_state(
            state='PROGRESS',
            meta={'symbol': symbol, 'timeframe': timeframe, 'progress': 0}
        )

        client = AlpacaClient()

        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=lookback_days)

        # Check cache first
        cached_data = market_data_cache.get_historical_data(
            symbol, timeframe, lookback_days
        )

        if cached_data:
            return {
                'symbol': symbol,
                'status': 'cached',
                'data_points': len(cached_data),
                'timeframe': timeframe,
                'lookback_days': lookback_days,
                'retrieved_at': datetime.utcnow().isoformat()
            }

        current_task.update_state(
            state='PROGRESS',
            meta={'symbol': symbol, 'timeframe': timeframe, 'progress': 25}
        )

        # Fetch from Alpaca
        bars = client.get_bars(
            symbol,
            timeframe=timeframe,
            start=start_date,
            end=end_date
        )

        current_task.update_state(
            state='PROGRESS',
            meta={'symbol': symbol, 'timeframe': timeframe, 'progress': 75}
        )

        if bars:
            # Convert to cache format
            historical_data = []
            for bar in bars:
                historical_data.append({
                    'timestamp': bar.timestamp.isoformat(),
                    'open': float(bar.open),
                    'high': float(bar.high),
                    'low': float(bar.low),
                    'close': float(bar.close),
                    'volume': int(bar.volume),
                    'vwap': float(bar.vwap) if bar.vwap else None,
                    'trade_count': int(bar.trade_count) if bar.trade_count else None
                })

            # Cache the data
            market_data_cache.set_historical_data(
                symbol, timeframe, lookback_days, historical_data, ttl=300
            )

            return {
                'symbol': symbol,
                'status': 'success',
                'data_points': len(historical_data),
                'timeframe': timeframe,
                'lookback_days': lookback_days,
                'date_range': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                'retrieved_at': datetime.utcnow().isoformat()
            }
        else:
            return {
                'symbol': symbol,
                'status': 'no_data',
                'timeframe': timeframe,
                'lookback_days': lookback_days,
                'retrieved_at': datetime.utcnow().isoformat()
            }

    except Exception as e:
        logger.error(f"Error fetching historical data for {symbol}: {e}")
        return {
            'symbol': symbol,
            'status': 'error',
            'error': str(e),
            'timeframe': timeframe,
            'lookback_days': lookback_days,
            'retrieved_at': datetime.utcnow().isoformat()
        }

@celery_app.task(bind=True, name='app.tasks.market_data.warm_cache')
def warm_market_data_cache(self, symbols: List[str] = None) -> Dict[str, Any]:
    """Warm up the market data cache with commonly used data"""
    try:
        if symbols is None:
            symbols = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
                'META', 'NVDA', 'NFLX', 'CRM', 'ADBE'
            ]

        current_task.update_state(
            state='PROGRESS',
            meta={'total_symbols': len(symbols), 'processed': 0, 'phase': 'quotes'}
        )

        # Warm up current quotes
        for i, symbol in enumerate(symbols):
            update_symbol_data(symbol)
            current_task.update_state(
                state='PROGRESS',
                meta={
                    'total_symbols': len(symbols),
                    'processed': i + 1,
                    'phase': 'quotes',
                    'current_symbol': symbol
                }
            )

        # Warm up historical data
        current_task.update_state(
            state='PROGRESS',
            meta={'total_symbols': len(symbols), 'processed': 0, 'phase': 'historical'}
        )

        for i, symbol in enumerate(symbols):
            fetch_historical_data(symbol, '1Day', 30)
            fetch_historical_data(symbol, '1Hour', 7)
            current_task.update_state(
                state='PROGRESS',
                meta={
                    'total_symbols': len(symbols),
                    'processed': i + 1,
                    'phase': 'historical',
                    'current_symbol': symbol
                }
            )

        return {
            'status': 'completed',
            'symbols_warmed': len(symbols),
            'completed_at': datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error warming cache: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'completed_at': datetime.utcnow().isoformat()
        }