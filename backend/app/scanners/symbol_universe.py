"""
Symbol Universe Manager for handling thousands of symbols across asset classes
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import aiohttp
from app.core.cache import market_data_cache
from app.core.config import settings

logger = logging.getLogger(__name__)

class AssetClass(Enum):
    """Asset class enumeration"""
    EQUITY = "equity"
    CRYPTO = "crypto"
    ETF = "etf"
    FOREX = "forex"

@dataclass
class SymbolInfo:
    """Symbol information container"""
    symbol: str
    name: str
    asset_class: AssetClass
    market_cap: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    exchange: Optional[str] = None
    active: bool = True
    last_updated: Optional[datetime] = None

class SymbolUniverseManager:
    """Manages universe of tradeable symbols across multiple asset classes"""

    def __init__(self):
        self.symbols: Dict[str, SymbolInfo] = {}
        self.asset_class_symbols: Dict[AssetClass, Set[str]] = {
            asset_class: set() for asset_class in AssetClass
        }
        self.market_cap_tiers: Dict[str, Set[str]] = {
            'large_cap': set(),    # > 10B
            'mid_cap': set(),      # 2B - 10B
            'small_cap': set(),    # 300M - 2B
            'micro_cap': set(),    # < 300M
            'unknown': set()
        }
        self.sector_symbols: Dict[str, Set[str]] = {}
        self.last_universe_update = None

    async def initialize_universe(self) -> Dict[str, int]:
        """Initialize the complete symbol universe"""
        logger.info("Initializing symbol universe...")

        results = {
            'equity_symbols': 0,
            'crypto_symbols': 0,
            'etf_symbols': 0,
            'total_symbols': 0,
            'errors': 0
        }

        try:
            # Load equity symbols (S&P 500, Russell 1000, etc.)
            equity_count = await self._load_equity_symbols()
            results['equity_symbols'] = equity_count

            # Load crypto symbols (top tradeable cryptocurrencies)
            crypto_count = await self._load_crypto_symbols()
            results['crypto_symbols'] = crypto_count

            # Load ETF symbols (popular ETFs)
            etf_count = await self._load_etf_symbols()
            results['etf_symbols'] = etf_count

            # Update market cap tiers
            self._categorize_by_market_cap()

            # Cache the universe
            await self._cache_universe()

            results['total_symbols'] = len(self.symbols)
            self.last_universe_update = datetime.utcnow()

            logger.info(f"Universe initialized: {results}")
            return results

        except Exception as e:
            logger.error(f"Error initializing universe: {e}")
            results['errors'] = 1
            return results

    async def _load_equity_symbols(self) -> int:
        """Load equity symbols from various sources"""
        equity_symbols = []

        # S&P 500 symbols (static list for reliability)
        sp500_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK.B',
            'UNH', 'JNJ', 'V', 'PG', 'JPM', 'HD', 'CVX', 'MA', 'BAC', 'ABBV',
            'PFE', 'AVGO', 'KO', 'COST', 'DIS', 'TMO', 'WMT', 'DHR', 'VZ',
            'PEP', 'ABT', 'ADBE', 'CRM', 'NFLX', 'CMCSA', 'NKE', 'INTC',
            'T', 'MRK', 'AMD', 'TXN', 'LIN', 'QCOM', 'WFC', 'UPS', 'PM',
            'HON', 'ORCL', 'IBM', 'CVS', 'NEE', 'RTX', 'LOW', 'SPGI', 'INTU',
            'AMAT', 'GS', 'COP', 'CAT', 'MDT', 'DE', 'ISRG', 'AXP', 'TJX',
            'BKNG', 'NOW', 'SCHW', 'GILD', 'MU', 'AMT', 'SYK', 'PLD', 'MDLZ',
            'CI', 'CB', 'MO', 'ZTS', 'FIS', 'MMC', 'BDX', 'C', 'USB', 'CSX',
            'TGT', 'MCD', 'SO', 'BSX', 'AON', 'CL', 'DUK', 'ICE', 'PNC'
        ]

        # Popular tech stocks
        tech_symbols = [
            'CRM', 'SNOW', 'DDOG', 'OKTA', 'ZM', 'SHOP', 'SQ', 'ROKU',
            'TWLO', 'NET', 'CRWD', 'PLTR', 'U', 'DOCU', 'ZS', 'COUP'
        ]

        # High-volume stocks for trading
        volume_leaders = [
            'SPY', 'QQQ', 'IWM', 'GLD', 'SLV', 'USO', 'TLT', 'HYG',
            'LQD', 'EEM', 'FXI', 'EWZ', 'EFA', 'VEA', 'VWO'
        ]

        all_equity_symbols = set(sp500_symbols + tech_symbols + volume_leaders)

        for symbol in all_equity_symbols:
            symbol_info = SymbolInfo(
                symbol=symbol,
                name=f"{symbol} Inc.",  # Simplified for demo
                asset_class=AssetClass.EQUITY,
                exchange="NYSE" if symbol not in tech_symbols else "NASDAQ",
                active=True,
                last_updated=datetime.utcnow()
            )

            self.symbols[symbol] = symbol_info
            self.asset_class_symbols[AssetClass.EQUITY].add(symbol)
            equity_symbols.append(symbol)

        logger.info(f"Loaded {len(equity_symbols)} equity symbols")
        return len(equity_symbols)

    async def _load_crypto_symbols(self) -> int:
        """Load cryptocurrency symbols"""
        crypto_symbols = []

        # Top cryptocurrencies available on Alpaca
        crypto_list = [
            'BTCUSD', 'ETHUSD', 'ADAUSD', 'SOLUSD', 'DOTUSD', 'AVAXUSD',
            'MATICUSD', 'LINKUSD', 'LTCUSD', 'BCHUSD', 'XLMUSD', 'ALGOUSD',
            'AAVEUSD', 'COMPUSD', 'UNIUSD', 'SUSHIUSD', 'YFIUSD', 'SNXUSD',
            'CRVUSD', 'BATUSD', 'GRTUSD', 'MKRUSD', 'ZRXUSD', 'MANAUSD'
        ]

        for symbol in crypto_list:
            symbol_info = SymbolInfo(
                symbol=symbol,
                name=symbol.replace('USD', ' USD'),
                asset_class=AssetClass.CRYPTO,
                exchange="CRYPTO",
                active=True,
                last_updated=datetime.utcnow()
            )

            self.symbols[symbol] = symbol_info
            self.asset_class_symbols[AssetClass.CRYPTO].add(symbol)
            crypto_symbols.append(symbol)

        logger.info(f"Loaded {len(crypto_symbols)} crypto symbols")
        return len(crypto_symbols)

    async def _load_etf_symbols(self) -> int:
        """Load ETF symbols"""
        etf_symbols = []

        # Popular ETFs for trading
        etf_list = [
            'SPY', 'QQQ', 'IWM', 'VTI', 'VEA', 'VWO', 'GLD', 'SLV',
            'TLT', 'HYG', 'LQD', 'EEM', 'FXI', 'EWZ', 'EFA', 'XLF',
            'XLK', 'XLE', 'XLV', 'XLI', 'XLP', 'XLU', 'XLB', 'XLRE',
            'VNQ', 'ARKK', 'ARKQ', 'ARKW', 'ARKG', 'ARKF'
        ]

        for symbol in etf_list:
            symbol_info = SymbolInfo(
                symbol=symbol,
                name=f"{symbol} ETF",
                asset_class=AssetClass.ETF,
                exchange="NYSE" if symbol.startswith(('S', 'V', 'G', 'T', 'H', 'L', 'E', 'X')) else "NASDAQ",
                active=True,
                last_updated=datetime.utcnow()
            )

            self.symbols[symbol] = symbol_info
            self.asset_class_symbols[AssetClass.ETF].add(symbol)
            etf_symbols.append(symbol)

        logger.info(f"Loaded {len(etf_symbols)} ETF symbols")
        return len(etf_symbols)

    def _categorize_by_market_cap(self):
        """Categorize symbols by market cap (mock implementation)"""
        large_cap_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA',
            'BRK.B', 'UNH', 'JNJ', 'V', 'PG', 'JPM', 'HD', 'CVX'
        ]

        mid_cap_symbols = [
            'AMD', 'NFLX', 'CRM', 'ADBE', 'PYPL', 'INTC', 'CMCSA',
            'NKE', 'T', 'VZ', 'MRK', 'PFE', 'KO', 'PEP', 'WMT'
        ]

        for symbol in large_cap_symbols:
            if symbol in self.symbols:
                self.market_cap_tiers['large_cap'].add(symbol)

        for symbol in mid_cap_symbols:
            if symbol in self.symbols:
                self.market_cap_tiers['mid_cap'].add(symbol)

        # Everything else goes to unknown for now
        for symbol in self.symbols:
            if (symbol not in self.market_cap_tiers['large_cap'] and
                symbol not in self.market_cap_tiers['mid_cap'] and
                self.symbols[symbol].asset_class == AssetClass.EQUITY):
                self.market_cap_tiers['small_cap'].add(symbol)

    async def _cache_universe(self):
        """Cache the symbol universe for quick access"""
        cache_data = {
            'symbols': {symbol: {
                'name': info.name,
                'asset_class': info.asset_class.value,
                'exchange': info.exchange,
                'active': info.active
            } for symbol, info in self.symbols.items()},
            'asset_classes': {
                asset_class.value: list(symbols)
                for asset_class, symbols in self.asset_class_symbols.items()
            },
            'market_cap_tiers': {
                tier: list(symbols) for tier, symbols in self.market_cap_tiers.items()
            },
            'last_updated': self.last_universe_update.isoformat() if self.last_universe_update else None
        }

        # Cache for 1 hour
        market_data_cache.cache.set('symbol_universe', cache_data, ttl=3600)

    def get_symbols_by_asset_class(self, asset_class: AssetClass) -> List[str]:
        """Get symbols by asset class"""
        return list(self.asset_class_symbols.get(asset_class, set()))

    def get_symbols_by_market_cap(self, tier: str) -> List[str]:
        """Get symbols by market cap tier"""
        return list(self.market_cap_tiers.get(tier, set()))

    def get_high_volume_symbols(self, limit: int = 100) -> List[str]:
        """Get high-volume symbols for scanning priority"""
        # Priority symbols for active trading
        priority_symbols = [
            'SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA', 'NVDA', 'META', 'GOOGL',
            'AMZN', 'AMD', 'NFLX', 'CRM', 'ADBE', 'PYPL', 'ARKK', 'IWM',
            'BTCUSD', 'ETHUSD', 'SOLUSD', 'ADAUSD', 'AVAXUSD'
        ]

        # Add remaining symbols to reach limit
        remaining_symbols = [
            symbol for symbol in self.symbols.keys()
            if symbol not in priority_symbols and self.symbols[symbol].active
        ]

        result = priority_symbols[:limit]
        if len(result) < limit:
            result.extend(remaining_symbols[:limit - len(result)])

        return result[:limit]

    def get_crypto_symbols_24_7(self) -> List[str]:
        """Get cryptocurrency symbols for 24/7 trading"""
        return self.get_symbols_by_asset_class(AssetClass.CRYPTO)

    def get_market_hours_symbols(self) -> List[str]:
        """Get symbols that trade during market hours only"""
        market_hours_classes = [AssetClass.EQUITY, AssetClass.ETF]
        symbols = []
        for asset_class in market_hours_classes:
            symbols.extend(self.get_symbols_by_asset_class(asset_class))
        return symbols

    def is_symbol_active(self, symbol: str) -> bool:
        """Check if symbol is active for trading"""
        symbol_info = self.symbols.get(symbol)
        return symbol_info.active if symbol_info else False

    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """Get detailed information about a symbol"""
        return self.symbols.get(symbol)

    def get_total_symbols(self) -> int:
        """Get total number of symbols in universe"""
        return len(self.symbols)

    def get_universe_stats(self) -> Dict[str, int]:
        """Get statistics about the symbol universe"""
        return {
            'total_symbols': len(self.symbols),
            'equity_symbols': len(self.asset_class_symbols[AssetClass.EQUITY]),
            'crypto_symbols': len(self.asset_class_symbols[AssetClass.CRYPTO]),
            'etf_symbols': len(self.asset_class_symbols[AssetClass.ETF]),
            'large_cap': len(self.market_cap_tiers['large_cap']),
            'mid_cap': len(self.market_cap_tiers['mid_cap']),
            'small_cap': len(self.market_cap_tiers['small_cap']),
            'active_symbols': sum(1 for info in self.symbols.values() if info.active)
        }