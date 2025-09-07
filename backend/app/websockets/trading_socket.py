"""
Real-time WebSocket trading dashboard implementation.
Provides live market data, trading signals, portfolio updates, and system health monitoring.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, asdict

from fastapi import WebSocket, WebSocketDisconnect
import structlog

from app.core.config import get_settings
from app.rag.agents.consolidated_strategy_agent import ConsolidatedStrategyAgent
from app.trading.alpaca_client import AlpacaClient
from app.core.cache import get_market_cache

logger = structlog.get_logger(__name__)
settings = get_settings()


@dataclass
class MarketUpdate:
    """Real-time market data update"""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: str
    bid: float = 0.0
    ask: float = 0.0
    high: float = 0.0
    low: float = 0.0


@dataclass
class TradingSignalUpdate:
    """Real-time trading signal update"""
    symbol: str
    signal_type: str  # BUY, SELL, HOLD
    confidence: float
    reasoning: str
    entry_price: Optional[float] = None
    timestamp: str = ""
    strategy: str = ""
    metadata: Dict[str, Any] = None


@dataclass
class PortfolioUpdate:
    """Real-time portfolio update"""
    total_value: float
    cash_balance: float
    day_change: float
    day_change_percent: float
    positions_count: int
    buying_power: float
    timestamp: str
    positions: List[Dict[str, Any]] = None


@dataclass 
class SystemHealthUpdate:
    """Real-time system health update"""
    status: str  # healthy, degraded, unhealthy
    services: Dict[str, bool]
    cache_metrics: Dict[str, Any]
    trading_enabled: bool
    last_update: str
    uptime: str


class ConnectionManager:
    """Manage WebSocket connections and broadcasting"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.subscriptions: Dict[WebSocket, Dict[str, Set[str]]] = {}
        self.last_updates: Dict[str, Any] = {}
        
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.subscriptions[websocket] = {
            "market_data": set(),
            "trading_signals": set(), 
            "portfolio": {"enabled"},
            "system_health": {"enabled"}
        }
        logger.info("WebSocket connection established", connections=len(self.active_connections))
        
        # Send initial data
        await self._send_initial_data(websocket)
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        self.active_connections.discard(websocket)
        self.subscriptions.pop(websocket, None)
        logger.info("WebSocket connection closed", connections=len(self.active_connections))
    
    async def subscribe(self, websocket: WebSocket, data_type: str, symbols: List[str] = None):
        """Subscribe to specific data types and symbols"""
        if websocket not in self.subscriptions:
            return
            
        if data_type in ["market_data", "trading_signals"] and symbols:
            self.subscriptions[websocket][data_type].update(symbols)
        elif data_type in ["portfolio", "system_health"]:
            self.subscriptions[websocket][data_type] = {"enabled"}
            
        logger.info(f"WebSocket subscribed to {data_type}", symbols=symbols)
    
    async def unsubscribe(self, websocket: WebSocket, data_type: str, symbols: List[str] = None):
        """Unsubscribe from specific data types and symbols"""
        if websocket not in self.subscriptions:
            return
            
        if data_type in ["market_data", "trading_signals"] and symbols:
            for symbol in symbols:
                self.subscriptions[websocket][data_type].discard(symbol)
        elif data_type in ["portfolio", "system_health"]:
            self.subscriptions[websocket][data_type] = set()
            
        logger.info(f"WebSocket unsubscribed from {data_type}", symbols=symbols)
    
    async def broadcast_market_update(self, update: MarketUpdate):
        """Broadcast market data to subscribed clients"""
        message = {
            "type": "market_update",
            "data": asdict(update),
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache the update
        self.last_updates[f"market_{update.symbol}"] = update
        
        await self._broadcast_to_subscribers("market_data", update.symbol, message)
    
    async def broadcast_trading_signal(self, signal: TradingSignalUpdate):
        """Broadcast trading signals to subscribed clients"""
        message = {
            "type": "trading_signal", 
            "data": asdict(signal),
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache the signal
        self.last_updates[f"signal_{signal.symbol}"] = signal
        
        await self._broadcast_to_subscribers("trading_signals", signal.symbol, message)
    
    async def broadcast_portfolio_update(self, update: PortfolioUpdate):
        """Broadcast portfolio updates to subscribed clients"""
        message = {
            "type": "portfolio_update",
            "data": asdict(update),
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache the update
        self.last_updates["portfolio"] = update
        
        await self._broadcast_to_subscribers("portfolio", "enabled", message)
    
    async def broadcast_system_health(self, health: SystemHealthUpdate):
        """Broadcast system health updates to subscribed clients"""
        message = {
            "type": "system_health",
            "data": asdict(health),
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache the health update
        self.last_updates["system_health"] = health
        
        await self._broadcast_to_subscribers("system_health", "enabled", message)
    
    async def _broadcast_to_subscribers(self, data_type: str, identifier: str, message: dict):
        """Broadcast message to subscribers of specific data type"""
        disconnected = []
        
        for websocket in self.active_connections.copy():
            if (websocket in self.subscriptions and 
                identifier in self.subscriptions[websocket].get(data_type, set())):
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.warning("Failed to send message to client", error=str(e))
                    disconnected.append(websocket)
        
        # Clean up disconnected clients
        for ws in disconnected:
            self.disconnect(ws)
    
    async def _send_initial_data(self, websocket: WebSocket):
        """Send cached data to newly connected client"""
        try:
            # Send recent updates if available
            for key, update in self.last_updates.items():
                if key.startswith("market_"):
                    message = {
                        "type": "market_update",
                        "data": asdict(update),
                        "timestamp": datetime.now().isoformat()
                    }
                elif key.startswith("signal_"):
                    message = {
                        "type": "trading_signal", 
                        "data": asdict(update),
                        "timestamp": datetime.now().isoformat()
                    }
                elif key == "portfolio":
                    message = {
                        "type": "portfolio_update",
                        "data": asdict(update),
                        "timestamp": datetime.now().isoformat()
                    }
                elif key == "system_health":
                    message = {
                        "type": "system_health",
                        "data": asdict(update),
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    continue
                    
                await websocket.send_text(json.dumps(message))
                    
        except Exception as e:
            logger.warning("Failed to send initial data", error=str(e))


class TradingDashboardWebSocket:
    """Real-time trading dashboard WebSocket service"""
    
    def __init__(self):
        self.connection_manager = ConnectionManager()
        self.trading_agent: Optional[ConsolidatedStrategyAgent] = None
        self.alpaca_client: Optional[AlpacaClient] = None
        self.market_cache = get_market_cache()
        self.running = False
        self.update_intervals = {
            "market_data": 1.0,  # 1 second
            "trading_signals": 5.0,  # 5 seconds
            "portfolio": 10.0,  # 10 seconds
            "system_health": 30.0,  # 30 seconds
        }
        self.last_updates = {
            "market_data": datetime.now(),
            "trading_signals": datetime.now(), 
            "portfolio": datetime.now(),
            "system_health": datetime.now(),
        }
        
        # Default symbols to track
        self.tracked_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
        
    async def initialize(self):
        """Initialize trading components"""
        try:
            # Initialize trading agent
            self.trading_agent = ConsolidatedStrategyAgent(
                use_market_research=True,
                use_ai_advisor=True
            )
            
            # Initialize Alpaca client
            self.alpaca_client = AlpacaClient()
            
            logger.info("TradingDashboardWebSocket initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize TradingDashboardWebSocket", error=str(e))
            raise
    
    async def connect(self, websocket: WebSocket):
        """Handle new WebSocket connection"""
        await self.connection_manager.connect(websocket)
        
        # Start background tasks if not running
        if not self.running:
            self.running = True
            asyncio.create_task(self._background_updates())
    
    async def disconnect(self, websocket: WebSocket):
        """Handle WebSocket disconnection"""
        self.connection_manager.disconnect(websocket)
    
    async def handle_message(self, websocket: WebSocket, message: str):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            action = data.get("action")
            
            if action == "subscribe":
                data_type = data.get("data_type")
                symbols = data.get("symbols", [])
                await self.connection_manager.subscribe(websocket, data_type, symbols)
                
            elif action == "unsubscribe":
                data_type = data.get("data_type")
                symbols = data.get("symbols", [])
                await self.connection_manager.unsubscribe(websocket, data_type, symbols)
                
            elif action == "add_symbol":
                symbol = data.get("symbol", "").upper()
                if symbol and symbol not in self.tracked_symbols:
                    self.tracked_symbols.append(symbol)
                    logger.info(f"Added symbol {symbol} to tracking list")
                    
            elif action == "remove_symbol":
                symbol = data.get("symbol", "").upper()
                if symbol in self.tracked_symbols:
                    self.tracked_symbols.remove(symbol)
                    logger.info(f"Removed symbol {symbol} from tracking list")
            
        except Exception as e:
            logger.warning("Failed to handle WebSocket message", message=message, error=str(e))
    
    async def _background_updates(self):
        """Background task for periodic updates"""
        logger.info("Started background updates for WebSocket dashboard")
        
        while self.running and self.connection_manager.active_connections:
            try:
                now = datetime.now()
                
                # Market data updates
                if (now - self.last_updates["market_data"]).total_seconds() >= self.update_intervals["market_data"]:
                    await self._update_market_data()
                    self.last_updates["market_data"] = now
                
                # Trading signals updates
                if (now - self.last_updates["trading_signals"]).total_seconds() >= self.update_intervals["trading_signals"]:
                    await self._update_trading_signals()
                    self.last_updates["trading_signals"] = now
                
                # Portfolio updates
                if (now - self.last_updates["portfolio"]).total_seconds() >= self.update_intervals["portfolio"]:
                    await self._update_portfolio()
                    self.last_updates["portfolio"] = now
                
                # System health updates
                if (now - self.last_updates["system_health"]).total_seconds() >= self.update_intervals["system_health"]:
                    await self._update_system_health()
                    self.last_updates["system_health"] = now
                
                await asyncio.sleep(0.5)  # Small delay to prevent CPU spinning
                
            except Exception as e:
                logger.error("Error in background updates", error=str(e))
                await asyncio.sleep(5)  # Wait longer on error
        
        self.running = False
        logger.info("Stopped background updates for WebSocket dashboard")
    
    async def _update_market_data(self):
        """Update market data for tracked symbols"""
        if not self.alpaca_client:
            return
            
        try:
            # Get latest market data from Alpaca
            for symbol in self.tracked_symbols:
                # Check cache first
                cache_key = f"market_data_{symbol}"
                cached_data = await self.market_cache.get(cache_key)
                
                if not cached_data:
                    # Fetch from Alpaca
                    try:
                        quote = await self.alpaca_client.get_latest_quote(symbol)
                        if quote:
                            market_update = MarketUpdate(
                                symbol=symbol,
                                price=quote.get("price", 0.0),
                                change=quote.get("change", 0.0),
                                change_percent=quote.get("change_percent", 0.0),
                                volume=quote.get("volume", 0),
                                bid=quote.get("bid", 0.0),
                                ask=quote.get("ask", 0.0),
                                high=quote.get("high", 0.0),
                                low=quote.get("low", 0.0),
                                timestamp=datetime.now().isoformat()
                            )
                            
                            # Cache for 30 seconds
                            await self.market_cache.set(cache_key, market_update, ttl_override=30)
                            
                            # Broadcast update
                            await self.connection_manager.broadcast_market_update(market_update)
                            
                    except Exception as e:
                        logger.warning(f"Failed to fetch market data for {symbol}", error=str(e))
                else:
                    # Use cached data
                    await self.connection_manager.broadcast_market_update(cached_data)
                    
        except Exception as e:
            logger.error("Failed to update market data", error=str(e))
    
    async def _update_trading_signals(self):
        """Update trading signals for tracked symbols"""
        if not self.trading_agent:
            return
            
        try:
            # Generate signals for tracked symbols
            for symbol in self.tracked_symbols[:3]:  # Limit to prevent overload
                try:
                    # Create market data for analysis
                    market_data = {
                        "symbol": symbol,
                        "current_price": 100.0,  # This would come from real market data
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Get trading signal
                    signal = await self.trading_agent.analyze_market(market_data)
                    
                    signal_update = TradingSignalUpdate(
                        symbol=symbol,
                        signal_type=signal.action,
                        confidence=signal.confidence,
                        reasoning=signal.reasoning,
                        entry_price=signal.entry_price,
                        strategy=signal.strategy_name,
                        timestamp=datetime.now().isoformat(),
                        metadata=signal.metadata
                    )
                    
                    await self.connection_manager.broadcast_trading_signal(signal_update)
                    
                except Exception as e:
                    logger.warning(f"Failed to generate signal for {symbol}", error=str(e))
                    
        except Exception as e:
            logger.error("Failed to update trading signals", error=str(e))
    
    async def _update_portfolio(self):
        """Update portfolio information"""
        if not self.alpaca_client:
            return
            
        try:
            # Get account info from Alpaca
            account = await self.alpaca_client.get_account()
            positions = await self.alpaca_client.get_positions()
            
            if account:
                portfolio_update = PortfolioUpdate(
                    total_value=float(account.get("portfolio_value", 0.0)),
                    cash_balance=float(account.get("cash", 0.0)),
                    day_change=float(account.get("day_change", 0.0)),
                    day_change_percent=float(account.get("day_change_percent", 0.0)),
                    positions_count=len(positions) if positions else 0,
                    buying_power=float(account.get("buying_power", 0.0)),
                    timestamp=datetime.now().isoformat(),
                    positions=positions[:10] if positions else []  # Limit positions
                )
                
                await self.connection_manager.broadcast_portfolio_update(portfolio_update)
                
        except Exception as e:
            logger.error("Failed to update portfolio", error=str(e))
    
    async def _update_system_health(self):
        """Update system health status"""
        try:
            # Check various system components
            services = {
                "trading_agent": self.trading_agent is not None,
                "alpaca_client": self.alpaca_client is not None,
                "cache": True,  # Assume cache is working if we got here
                "database": True,  # Would check database connection
            }
            
            # Get cache metrics
            cache_health = await self.market_cache.health_check()
            
            # Determine overall status
            healthy_services = sum(services.values())
            total_services = len(services)
            
            if healthy_services == total_services:
                status = "healthy"
            elif healthy_services >= total_services * 0.7:
                status = "degraded" 
            else:
                status = "unhealthy"
            
            health_update = SystemHealthUpdate(
                status=status,
                services=services,
                cache_metrics=cache_health.get("metrics", {}),
                trading_enabled=services.get("alpaca_client", False),
                last_update=datetime.now().isoformat(),
                uptime="N/A"  # Would calculate actual uptime
            )
            
            await self.connection_manager.broadcast_system_health(health_update)
            
        except Exception as e:
            logger.error("Failed to update system health", error=str(e))


# Global instance
dashboard_websocket = TradingDashboardWebSocket()