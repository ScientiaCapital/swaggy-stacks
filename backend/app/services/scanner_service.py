"""
Scanner Service for integrating multi-symbol scanning into the trading system
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import BackgroundTasks

from app.scanners.symbol_scanner import SymbolScanner
from app.scanners.symbol_universe import AssetClass
from app.core.config import settings
from app.tasks.market_data import update_all_symbols, warm_market_data_cache

logger = logging.getLogger(__name__)

class ScannerService:
    """Production-ready scanner service with background task integration"""

    def __init__(self):
        self.scanner = SymbolScanner(max_concurrent=settings.MAX_SYMBOLS)
        self.is_initialized = False
        self.scan_running = False

    async def initialize(self) -> Dict[str, Any]:
        """Initialize the scanner service"""
        if self.is_initialized:
            return {'status': 'already_initialized'}

        try:
            logger.info("Initializing Scanner Service...")

            # Initialize the scanner
            init_result = await self.scanner.initialize()

            # Warm up cache with priority symbols
            warm_result = warm_market_data_cache.delay()

            self.is_initialized = True

            result = {
                'status': 'initialized',
                'scanner_result': init_result,
                'cache_warming': 'started',
                'max_concurrent': self.scanner.max_concurrent,
                'universe_size': init_result.get('total_symbols', 0),
                'initialized_at': datetime.utcnow().isoformat()
            }

            logger.info(f"Scanner Service initialized: {result}")
            return result

        except Exception as e:
            logger.error(f"Error initializing Scanner Service: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'initialized_at': datetime.utcnow().isoformat()
            }

    async def scan_opportunities(self,
                               asset_classes: List[str] = None,
                               max_symbols: int = 1000,
                               min_opportunity_score: float = 0.6) -> Dict[str, Any]:
        """Scan for trading opportunities across symbol universe"""
        if not self.is_initialized:
            await self.initialize()

        if self.scan_running:
            return {
                'status': 'scan_in_progress',
                'message': 'Another scan is currently running'
            }

        try:
            self.scan_running = True
            logger.info(f"Starting opportunity scan: max_symbols={max_symbols}, "
                       f"min_score={min_opportunity_score}")

            # Convert asset class strings to enums
            target_asset_classes = None
            if asset_classes:
                target_asset_classes = []
                for ac in asset_classes:
                    try:
                        target_asset_classes.append(AssetClass(ac.lower()))
                    except ValueError:
                        logger.warning(f"Invalid asset class: {ac}")

            # Perform the scan
            scan_result = await self.scanner.full_universe_scan(target_asset_classes)

            # Filter results by minimum score
            top_opportunities = scan_result.get('top_opportunities', [])
            filtered_opportunities = [
                opp for opp in top_opportunities
                if opp.get('opportunity_score', 0) >= min_opportunity_score
            ]

            # Enhanced result with filtering
            result = {
                'status': 'completed',
                'scan_summary': {
                    'total_symbols_scanned': scan_result.get('total_symbols_scanned', 0),
                    'scan_duration_ms': scan_result.get('scan_duration_ms', 0),
                    'total_opportunities': len(top_opportunities),
                    'filtered_opportunities': len(filtered_opportunities),
                    'min_score_threshold': min_opportunity_score
                },
                'opportunities': filtered_opportunities,
                'scan_statistics': scan_result.get('scan_statistics', {}),
                'scanned_at': scan_result.get('timestamp')
            }

            logger.info(f"Opportunity scan completed: {len(filtered_opportunities)} "
                       f"opportunities found from {scan_result.get('total_symbols_scanned', 0)} symbols")

            return result

        except Exception as e:
            logger.error(f"Error during opportunity scan: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'scanned_at': datetime.utcnow().isoformat()
            }
        finally:
            self.scan_running = False

    async def scan_specific_symbols(self, symbols: List[str]) -> Dict[str, Any]:
        """Scan specific symbols for opportunities"""
        if not self.is_initialized:
            await self.initialize()

        try:
            logger.info(f"Scanning specific symbols: {symbols}")

            # Validate symbols exist in universe
            valid_symbols = []
            for symbol in symbols:
                if self.scanner.universe_manager.is_symbol_active(symbol):
                    valid_symbols.append(symbol)
                else:
                    logger.warning(f"Symbol {symbol} not found in universe")

            if not valid_symbols:
                return {
                    'status': 'no_valid_symbols',
                    'message': 'No valid symbols provided',
                    'requested_symbols': symbols
                }

            # Scan the symbols
            analyses = await self.scanner.scan_multiple_symbols(valid_symbols)

            # Convert to opportunity format
            opportunities = []
            for analysis in analyses:
                opportunities.append({
                    'symbol': analysis.symbol,
                    'opportunity_score': analysis.opportunity_score,
                    'scan_result': analysis.scan_result.value,
                    'confidence': analysis.confidence,
                    'entry_signal': analysis.entry_signal,
                    'exit_signal': analysis.exit_signal,
                    'current_price': analysis.current_price,
                    'price_change_percent': analysis.price_change_percent,
                    'volume_spike': analysis.volume_spike,
                    'position_size_suggestion': analysis.position_size_suggestion,
                    'technical_indicators': {
                        'rsi': analysis.rsi,
                        'macd_signal': analysis.macd_signal,
                        'bollinger_position': analysis.bollinger_position
                    },
                    'scan_duration_ms': analysis.scan_duration_ms,
                    'errors': analysis.errors
                })

            result = {
                'status': 'completed',
                'requested_symbols': symbols,
                'valid_symbols': valid_symbols,
                'scanned_symbols': len(analyses),
                'opportunities': opportunities,
                'scanned_at': datetime.utcnow().isoformat()
            }

            logger.info(f"Specific symbol scan completed: {len(analyses)} symbols analyzed")
            return result

        except Exception as e:
            logger.error(f"Error scanning specific symbols: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'requested_symbols': symbols,
                'scanned_at': datetime.utcnow().isoformat()
            }

    def get_scanner_status(self) -> Dict[str, Any]:
        """Get current scanner status and statistics"""
        if not self.is_initialized:
            return {
                'status': 'not_initialized',
                'initialized': False
            }

        stats = self.scanner.get_scan_statistics()
        return {
            'status': 'ready',
            'initialized': True,
            'scan_running': self.scan_running,
            'scanner_statistics': stats,
            'universe_stats': self.scanner.universe_manager.get_universe_stats(),
            'ranking_criteria': self.scanner.opportunity_ranker.get_ranking_criteria()
        }

    async def start_continuous_scanning(self, interval_minutes: int = 5) -> Dict[str, Any]:
        """Start continuous scanning in background"""
        if not self.is_initialized:
            await self.initialize()

        try:
            # This would typically be implemented with Celery beat or similar
            # For now, return configuration for manual scheduling
            result = {
                'status': 'configured',
                'message': 'Continuous scanning can be started with Celery beat',
                'recommended_tasks': [
                    {
                        'task': 'scan_high_volume_symbols',
                        'interval_minutes': interval_minutes,
                        'symbols': self.scanner.universe_manager.get_high_volume_symbols(100)
                    },
                    {
                        'task': 'scan_crypto_24_7',
                        'interval_minutes': 1,  # More frequent for crypto
                        'symbols': self.scanner.universe_manager.get_crypto_symbols_24_7()
                    }
                ],
                'configured_at': datetime.utcnow().isoformat()
            }

            logger.info(f"Continuous scanning configured: {result}")
            return result

        except Exception as e:
            logger.error(f"Error configuring continuous scanning: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'configured_at': datetime.utcnow().isoformat()
            }

# Global scanner service instance
scanner_service = ScannerService()