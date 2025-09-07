"""
Market Data Tool for retrieving market information
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import logging

from .base_tool import AgentTool, ToolResult, ToolParameter
from app.trading.alpaca_client import AlpacaClient
from app.core.exceptions import MarketDataError

logger = logging.getLogger(__name__)


class MarketDataTool(AgentTool):
    """Tool for retrieving market data using Alpaca API"""
    
    def __init__(self):
        super().__init__(
            name="market_data",
            description="Retrieve market data including historical bars, quotes, and trades"
        )
        self.category = "market_data"
        self.alpaca_client = None
    
    async def _get_client(self) -> AlpacaClient:
        """Get or create Alpaca client"""
        if self.alpaca_client is None:
            self.alpaca_client = AlpacaClient()
        return self.alpaca_client
    
    def get_parameters(self) -> List[ToolParameter]:
        """Get tool parameter definitions"""
        return [
            ToolParameter(
                name="action",
                type="str",
                description="Action to perform: 'historical', 'quotes', 'trades'",
                required=True
            ),
            ToolParameter(
                name="symbols",
                type="list",
                description="List of stock symbols (e.g., ['AAPL', 'GOOGL'])",
                required=True
            ),
            ToolParameter(
                name="timeframe",
                type="str",
                description="Timeframe for historical data: '1Min', '5Min', '15Min', '1H', '1D'",
                required=False,
                default="1D"
            ),
            ToolParameter(
                name="start_date",
                type="str",
                description="Start date in ISO format (e.g., '2024-01-01')",
                required=False
            ),
            ToolParameter(
                name="end_date",
                type="str",
                description="End date in ISO format (e.g., '2024-01-31')",
                required=False
            ),
            ToolParameter(
                name="limit",
                type="int",
                description="Maximum number of data points to retrieve",
                required=False,
                default=1000
            )
        ]
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute market data retrieval"""
        try:
            client = await self._get_client()
            action = parameters["action"].lower()
            symbols = parameters["symbols"]
            
            if not isinstance(symbols, list):
                symbols = [symbols] if isinstance(symbols, str) else []
            
            if action == "historical":
                return await self._get_historical_data(client, parameters, symbols)
            elif action == "quotes":
                return await self._get_quotes(client, symbols)
            elif action == "trades":
                return await self._get_trades(client, symbols)
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Unknown action: {action}. Supported actions: 'historical', 'quotes', 'trades'"
                )
                
        except MarketDataError as e:
            logger.error(f"Market data error: {e}")
            return ToolResult(success=False, data=None, error=str(e))
        except Exception as e:
            logger.error(f"Unexpected error in market data tool: {e}")
            return ToolResult(success=False, data=None, error=f"Unexpected error: {str(e)}")
    
    async def _get_historical_data(self, client: AlpacaClient, parameters: Dict[str, Any], symbols: List[str]) -> ToolResult:
        """Get historical market data"""
        try:
            timeframe = parameters.get("timeframe", "1D")
            start_date = parameters.get("start_date")
            end_date = parameters.get("end_date")
            limit = parameters.get("limit", 1000)
            
            # Set default dates if not provided
            if not end_date:
                end_date = datetime.now().isoformat()
            if not start_date:
                start_date = (datetime.now() - timedelta(days=30)).isoformat()
            
            data = await client.get_market_data(
                symbols=symbols,
                timeframe=timeframe,
                start=start_date,
                end=end_date,
                limit=limit
            )
            
            # Add metadata
            metadata = {
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date,
                "symbols_requested": symbols,
                "symbols_returned": list(data.keys()),
                "data_points": {symbol: len(bars) for symbol, bars in data.items()}
            }
            
            return ToolResult(
                success=True,
                data=data,
                metadata=metadata
            )
            
        except Exception as e:
            raise MarketDataError(f"Failed to get historical data: {str(e)}")
    
    async def _get_quotes(self, client: AlpacaClient, symbols: List[str]) -> ToolResult:
        """Get latest quotes"""
        try:
            quotes = await client.get_latest_quotes(symbols)
            
            metadata = {
                "symbols_requested": symbols,
                "symbols_returned": list(quotes.keys()),
                "retrieved_at": datetime.now().isoformat()
            }
            
            return ToolResult(
                success=True,
                data=quotes,
                metadata=metadata
            )
            
        except Exception as e:
            raise MarketDataError(f"Failed to get quotes: {str(e)}")
    
    async def _get_trades(self, client: AlpacaClient, symbols: List[str]) -> ToolResult:
        """Get latest trades"""
        try:
            trades = await client.get_latest_trades(symbols)
            
            metadata = {
                "symbols_requested": symbols,
                "symbols_returned": list(trades.keys()),
                "retrieved_at": datetime.now().isoformat()
            }
            
            return ToolResult(
                success=True,
                data=trades,
                metadata=metadata
            )
            
        except Exception as e:
            raise MarketDataError(f"Failed to get trades: {str(e)}")