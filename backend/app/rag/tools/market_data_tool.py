"""
Market Data Tools for LangChain Integration

Provides real-time and historical market data access as LangChain Tools
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import yfinance as yf
from langchain.agents import Tool

from app.trading.trading_manager import get_trading_manager

logger = logging.getLogger(__name__)


class MarketDataTool:
    """Market data tools for real-time and historical data access"""
    
    def __init__(self):
        self.trading_manager = None
        
    async def initialize(self) -> None:
        """Initialize with trading manager"""
        self.trading_manager = get_trading_manager()
        await self.trading_manager.initialize()
    
    def get_tools(self) -> List[Tool]:
        """Get all market data tools"""
        return [
            Tool(
                name="get_current_price",
                description="Get current price for a symbol (e.g., 'AAPL')",
                func=self._get_current_price
            ),
            Tool(
                name="get_market_depth",
                description="Get market depth/order book for a symbol",
                func=self._get_market_depth
            ),
            Tool(
                name="get_historical_data",
                description="Get historical price data (format: 'SYMBOL,DAYS' e.g., 'AAPL,30')",
                func=self._get_historical_data
            ),
            Tool(
                name="get_market_sentiment",
                description="Get market sentiment indicators for a symbol",
                func=self._get_market_sentiment
            ),
            Tool(
                name="get_volume_profile",
                description="Get volume profile analysis for a symbol",
                func=self._get_volume_profile
            ),
            Tool(
                name="get_premarket_data",
                description="Get pre-market trading data for a symbol",
                func=self._get_premarket_data
            )
        ]
    
    def _get_current_price(self, symbol: str) -> str:
        """Get current price for a symbol"""
        try:
            symbol = symbol.strip().upper()
            
            # Use yfinance for quick price data
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if 'currentPrice' in info:
                current_price = info['currentPrice']
            elif 'regularMarketPrice' in info:
                current_price = info['regularMarketPrice']
            else:
                return f"Unable to get current price for {symbol}"
            
            # Get additional context
            change = info.get('regularMarketChange', 0)
            change_percent = info.get('regularMarketChangePercent', 0) * 100
            volume = info.get('regularMarketVolume', 0)
            
            return json.dumps({
                "symbol": symbol,
                "price": current_price,
                "change": change,
                "change_percent": round(change_percent, 2),
                "volume": volume,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return f"Error getting price for {symbol}: {str(e)}"
    
    def _get_market_depth(self, symbol: str) -> str:
        """Get market depth/order book"""
        try:
            symbol = symbol.strip().upper()
            
            # Note: Real order book data requires premium API access
            # This provides basic bid/ask information
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            bid = info.get('bid', 0)
            ask = info.get('ask', 0)
            bid_size = info.get('bidSize', 0)
            ask_size = info.get('askSize', 0)
            
            spread = ask - bid if ask > 0 and bid > 0 else 0
            spread_percent = (spread / ((ask + bid) / 2)) * 100 if spread > 0 else 0
            
            return json.dumps({
                "symbol": symbol,
                "bid": bid,
                "ask": ask,
                "bid_size": bid_size,
                "ask_size": ask_size,
                "spread": round(spread, 4),
                "spread_percent": round(spread_percent, 4),
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error getting market depth for {symbol}: {e}")
            return f"Error getting market depth for {symbol}: {str(e)}"
    
    def _get_historical_data(self, params: str) -> str:
        """Get historical price data"""
        try:
            parts = params.split(',')
            symbol = parts[0].strip().upper()
            days = int(parts[1]) if len(parts) > 1 else 30
            
            # Limit to reasonable range
            days = min(max(days, 1), 365)
            
            ticker = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            hist = ticker.history(start=start_date, end=end_date)
            
            if hist.empty:
                return f"No historical data available for {symbol}"
            
            # Convert to JSON-serializable format
            data = {
                "symbol": symbol,
                "period_days": days,
                "data_points": len(hist),
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "prices": {
                    "open": hist['Open'].tolist()[-10:],  # Last 10 days
                    "high": hist['High'].tolist()[-10:],
                    "low": hist['Low'].tolist()[-10:],
                    "close": hist['Close'].tolist()[-10:],
                    "volume": hist['Volume'].tolist()[-10:]
                },
                "summary": {
                    "current_price": float(hist['Close'].iloc[-1]),
                    "period_high": float(hist['High'].max()),
                    "period_low": float(hist['Low'].min()),
                    "avg_volume": float(hist['Volume'].mean()),
                    "total_return": float((hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100)
                }
            }
            
            return json.dumps(data)
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return f"Error getting historical data: {str(e)}"
    
    def _get_market_sentiment(self, symbol: str) -> str:
        """Get market sentiment indicators"""
        try:
            symbol = symbol.strip().upper()
            ticker = yf.Ticker(symbol)
            
            # Get basic sentiment indicators from available data
            info = ticker.info
            
            # Price momentum (short-term trend)
            hist = ticker.history(period="5d")
            if not hist.empty:
                recent_prices = hist['Close'].tolist()
                if len(recent_prices) >= 2:
                    momentum = (recent_prices[-1] / recent_prices[0] - 1) * 100
                else:
                    momentum = 0
            else:
                momentum = 0
            
            # Volume trend
            if not hist.empty and len(hist) >= 2:
                recent_volumes = hist['Volume'].tolist()
                avg_volume = sum(recent_volumes) / len(recent_volumes)
                latest_volume = recent_volumes[-1] if recent_volumes else 0
                volume_ratio = latest_volume / avg_volume if avg_volume > 0 else 1
            else:
                volume_ratio = 1
            
            # Basic sentiment scoring
            sentiment_score = 0.5  # Neutral base
            
            if momentum > 5:
                sentiment_score += 0.2
            elif momentum < -5:
                sentiment_score -= 0.2
            
            if volume_ratio > 1.5:
                sentiment_score += 0.1
            elif volume_ratio < 0.5:
                sentiment_score -= 0.1
            
            sentiment_score = max(0, min(1, sentiment_score))
            
            # Sentiment classification
            if sentiment_score > 0.7:
                sentiment = "bullish"
            elif sentiment_score < 0.3:
                sentiment = "bearish"
            else:
                sentiment = "neutral"
            
            return json.dumps({
                "symbol": symbol,
                "sentiment": sentiment,
                "sentiment_score": round(sentiment_score, 3),
                "momentum_5d": round(momentum, 2),
                "volume_ratio": round(volume_ratio, 2),
                "indicators": {
                    "price_momentum": "positive" if momentum > 0 else "negative",
                    "volume_trend": "high" if volume_ratio > 1.2 else "normal" if volume_ratio > 0.8 else "low"
                },
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error getting market sentiment for {symbol}: {e}")
            return f"Error getting sentiment for {symbol}: {str(e)}"
    
    def _get_volume_profile(self, symbol: str) -> str:
        """Get volume profile analysis"""
        try:
            symbol = symbol.strip().upper()
            ticker = yf.Ticker(symbol)
            
            # Get recent data for volume analysis
            hist = ticker.history(period="30d")
            
            if hist.empty:
                return f"No volume data available for {symbol}"
            
            # Basic volume profile analysis
            volumes = hist['Volume'].tolist()
            prices = hist['Close'].tolist()
            
            avg_volume = sum(volumes) / len(volumes)
            max_volume = max(volumes)
            min_volume = min(volumes)
            
            # Volume-weighted average price (VWAP) approximation
            total_volume = sum(volumes)
            volume_price_sum = sum(v * p for v, p in zip(volumes, prices))
            vwap = volume_price_sum / total_volume if total_volume > 0 else 0
            
            current_price = prices[-1] if prices else 0
            
            return json.dumps({
                "symbol": symbol,
                "vwap": round(vwap, 2),
                "current_price": round(current_price, 2),
                "price_vs_vwap": round((current_price / vwap - 1) * 100, 2) if vwap > 0 else 0,
                "volume_stats": {
                    "avg_volume": int(avg_volume),
                    "max_volume": int(max_volume),
                    "min_volume": int(min_volume),
                    "latest_volume": int(volumes[-1]) if volumes else 0
                },
                "volume_trend": "increasing" if volumes[-1] > avg_volume else "decreasing",
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error getting volume profile for {symbol}: {e}")
            return f"Error getting volume profile for {symbol}: {str(e)}"
    
    def _get_premarket_data(self, symbol: str) -> str:
        """Get pre-market trading data"""
        try:
            symbol = symbol.strip().upper()
            
            # Note: Pre-market data requires real-time API access
            # This provides basic information from available data
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            premarket_price = info.get('preMarketPrice', 0)
            premarket_change = info.get('preMarketChange', 0)
            premarket_change_percent = info.get('preMarketChangePercent', 0) * 100
            
            regular_price = info.get('regularMarketPreviousClose', 0)
            
            return json.dumps({
                "symbol": symbol,
                "premarket_price": premarket_price,
                "premarket_change": premarket_change,
                "premarket_change_percent": round(premarket_change_percent, 2),
                "previous_close": regular_price,
                "gap_from_close": round(premarket_price - regular_price, 2) if premarket_price > 0 and regular_price > 0 else 0,
                "has_premarket_data": premarket_price > 0,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error getting premarket data for {symbol}: {e}")
            return f"Error getting premarket data for {symbol}: {str(e)}"


# Global market data tool instance
_market_data_tool: Optional[MarketDataTool] = None


async def get_market_data_tool() -> MarketDataTool:
    """Get the global market data tool instance"""
    global _market_data_tool
    
    if _market_data_tool is None:
        _market_data_tool = MarketDataTool()
        await _market_data_tool.initialize()
    
    return _market_data_tool