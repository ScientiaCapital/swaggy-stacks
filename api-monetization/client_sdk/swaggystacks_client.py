"""
SwaggyStacks Client SDK
Python SDK for easy integration with SwaggyStacks Trading Intelligence API
"""

import aiohttp
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json

class SwaggyStacksClient:
    """
    Official SwaggyStacks Trading Intelligence API Client
    
    Provides easy access to all API endpoints with automatic error handling,
    rate limiting, and response parsing.
    """
    
    def __init__(self, api_key: str, base_url: str = "https://api.swaggystacks.com", 
                 timeout: int = 30):
        """
        Initialize the SwaggyStacks client
        
        Args:
            api_key: Your SwaggyStacks API key
            base_url: API base URL (default: production)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = None
        
        # Headers for all requests
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json",
            "User-Agent": "SwaggyStacks-Python-SDK/2.0.0"
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make an HTTP request to the API
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            **kwargs: Additional request parameters
            
        Returns:
            API response as dictionary
            
        Raises:
            SwaggyStacksAPIError: For API errors
            SwaggyStacksRateLimitError: For rate limit errors
            SwaggyStacksQuotaError: For quota exceeded errors
        """
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with self.session.request(method, url, **kwargs) as response:
                data = await response.json()
                
                if response.status == 200:
                    return data
                elif response.status == 401:
                    raise SwaggyStacksAPIError("Invalid API key", response.status)
                elif response.status == 402:
                    raise SwaggyStacksQuotaError("Monthly quota exceeded", response.status)
                elif response.status == 429:
                    raise SwaggyStacksRateLimitError("Rate limit exceeded", response.status)
                else:
                    error_msg = data.get("detail", f"API error: {response.status}")
                    raise SwaggyStacksAPIError(error_msg, response.status)
                    
        except aiohttp.ClientError as e:
            raise SwaggyStacksAPIError(f"Network error: {str(e)}", 0)
    
    # Stock Analysis Methods
    async def analyze_stock(self, symbol: str, depth: str = "standard", 
                          include_technical: bool = True, 
                          include_sentiment: bool = False) -> Dict[str, Any]:
        """
        Analyze a stock using AI models
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            depth: Analysis depth ('basic', 'standard', 'advanced')
            include_technical: Include technical analysis
            include_sentiment: Include sentiment analysis
            
        Returns:
            Stock analysis results
        """
        params = {
            "timeframe": "1d",
            "depth": depth,
            "include_technical": include_technical,
            "include_sentiment": include_sentiment
        }
        
        return await self._make_request("GET", f"/v1/analyze/stock/{symbol}", params=params)
    
    async def analyze_stock_advanced(self, symbol: str) -> Dict[str, Any]:
        """
        Perform advanced stock analysis (convenience method)
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Advanced stock analysis results
        """
        return await self.analyze_stock(symbol, depth="advanced", 
                                      include_technical=True, include_sentiment=True)
    
    # Portfolio Analysis Methods
    async def analyze_portfolio(self, symbols: List[str], 
                              weights: Optional[Dict[str, float]] = None,
                              risk_tolerance: str = "medium",
                              investment_horizon: str = "medium") -> Dict[str, Any]:
        """
        Analyze a portfolio using multi-model AI
        
        Args:
            symbols: List of stock symbols
            weights: Portfolio weights (optional)
            risk_tolerance: Risk tolerance level
            investment_horizon: Investment horizon
            
        Returns:
            Portfolio analysis results
        """
        data = {
            "symbols": symbols,
            "risk_tolerance": risk_tolerance,
            "investment_horizon": investment_horizon
        }
        
        if weights:
            data["weights"] = weights
        
        return await self._make_request("POST", "/v1/analyze/portfolio", json=data)
    
    # Trading Signals Methods
    async def generate_signals(self, symbol: str, strategy: str = "multi_model") -> Dict[str, Any]:
        """
        Generate trading signals using specialized AI models
        
        Args:
            symbol: Stock symbol
            strategy: Trading strategy ('fibonacci', 'elliott_wave', 'wyckoff', 'markov', 'multi_model')
            
        Returns:
            Trading signals
        """
        data = {
            "symbol": symbol,
            "strategy": strategy
        }
        
        return await self._make_request("POST", "/v1/signals/generate", json=data)
    
    async def get_fibonacci_signals(self, symbol: str) -> Dict[str, Any]:
        """
        Get Fibonacci-based trading signals (convenience method)
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Fibonacci trading signals
        """
        return await self.generate_signals(symbol, strategy="fibonacci")
    
    async def get_elliott_wave_signals(self, symbol: str) -> Dict[str, Any]:
        """
        Get Elliott Wave trading signals (convenience method)
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Elliott Wave trading signals
        """
        return await self.generate_signals(symbol, strategy="elliott_wave")
    
    # Backtesting Methods
    async def backtest_strategy(self, strategy_config: Dict[str, Any],
                              start_date: str, end_date: str,
                              initial_capital: float = 10000) -> Dict[str, Any]:
        """
        Backtest a trading strategy using historical data
        
        Args:
            strategy_config: Strategy configuration
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Initial capital amount
            
        Returns:
            Backtest results
        """
        data = {
            "strategy_config": strategy_config,
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": initial_capital
        }
        
        return await self._make_request("POST", "/v1/backtest", json=data)
    
    # Market Analysis Methods
    async def get_market_regime(self) -> Dict[str, Any]:
        """
        Get current market regime analysis
        
        Returns:
            Market regime analysis
        """
        return await self._make_request("GET", "/v1/market/regime")
    
    # Account and Usage Methods
    async def get_usage_info(self) -> Dict[str, Any]:
        """
        Get current usage information
        
        Returns:
            Usage information
        """
        return await self._make_request("GET", "/v1/billing/usage")
    
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information (alias for get_usage_info)
        
        Returns:
            Account information
        """
        return await self.get_usage_info()
    
    # Health Check
    async def health_check(self) -> Dict[str, Any]:
        """
        Check API health status
        
        Returns:
            Health status
        """
        return await self._make_request("GET", "/health")
    
    # Batch Operations
    async def analyze_multiple_stocks(self, symbols: List[str], 
                                    depth: str = "standard") -> Dict[str, Any]:
        """
        Analyze multiple stocks in parallel
        
        Args:
            symbols: List of stock symbols
            depth: Analysis depth
            
        Returns:
            Dictionary with symbol as key and analysis as value
        """
        tasks = [self.analyze_stock(symbol, depth) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            symbol: result if not isinstance(result, Exception) else {"error": str(result)}
            for symbol, result in zip(symbols, results)
        }
    
    async def get_signals_for_portfolio(self, symbols: List[str], 
                                      strategy: str = "multi_model") -> Dict[str, Any]:
        """
        Get trading signals for multiple stocks in parallel
        
        Args:
            symbols: List of stock symbols
            strategy: Trading strategy
            
        Returns:
            Dictionary with symbol as key and signals as value
        """
        tasks = [self.generate_signals(symbol, strategy) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            symbol: result if not isinstance(result, Exception) else {"error": str(result)}
            for symbol, result in zip(symbols, results)
        }

# Custom Exceptions
class SwaggyStacksAPIError(Exception):
    """Base exception for SwaggyStacks API errors"""
    def __init__(self, message: str, status_code: int = 0):
        self.message = message
        self.status_code = status_code
        super().__init__(f"SwaggyStacks API Error ({status_code}): {message}")

class SwaggyStacksRateLimitError(SwaggyStacksAPIError):
    """Exception for rate limit errors"""
    pass

class SwaggyStacksQuotaError(SwaggyStacksAPIError):
    """Exception for quota exceeded errors"""
    pass

# Convenience Functions
async def quick_analysis(symbol: str, api_key: str) -> Dict[str, Any]:
    """
    Quick stock analysis function for simple use cases
    
    Args:
        symbol: Stock symbol
        api_key: API key
        
    Returns:
        Stock analysis results
    """
    async with SwaggyStacksClient(api_key) as client:
        return await client.analyze_stock_advanced(symbol)

async def portfolio_health_check(symbols: List[str], api_key: str) -> Dict[str, Any]:
    """
    Quick portfolio health check
    
    Args:
        symbols: List of stock symbols
        api_key: API key
        
    Returns:
        Portfolio analysis results
    """
    async with SwaggyStacksClient(api_key) as client:
        return await client.analyze_portfolio(symbols)

# Example Usage
async def example_usage():
    """Example usage of the SwaggyStacks client"""
    api_key = "your_api_key_here"
    
    async with SwaggyStacksClient(api_key) as client:
        try:
            # Health check
            health = await client.health_check()
            print(f"API Status: {health['status']}")
            
            # Analyze a stock
            analysis = await client.analyze_stock_advanced("AAPL")
            print(f"AAPL Analysis: {analysis['analysis']['recommendation']}")
            
            # Get trading signals
            signals = await client.get_fibonacci_signals("AAPL")
            print(f"Fibonacci Signals: {len(signals['signals'])} signals generated")
            
            # Analyze portfolio
            portfolio = await client.analyze_portfolio(["AAPL", "MSFT", "GOOGL"])
            print(f"Portfolio Sharpe Ratio: {portfolio['portfolio_analysis']['performance_metrics']['sharpe_ratio']}")
            
            # Get market regime
            regime = await client.get_market_regime()
            print(f"Current Market Regime: {regime['market_regime']['current_regime']}")
            
            # Check usage
            usage = await client.get_usage_info()
            print(f"Remaining Credits: {usage.get('remaining_credits', 'N/A')}")
            
        except SwaggyStacksAPIError as e:
            print(f"API Error: {e}")
        except SwaggyStacksRateLimitError as e:
            print(f"Rate Limit Error: {e}")
        except SwaggyStacksQuotaError as e:
            print(f"Quota Error: {e}")

if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
