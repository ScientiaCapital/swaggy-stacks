"""
Message handlers for processing RabbitMQ messages in the trading system.
Handles trading signals, portfolio updates, market data, and system events.
"""

import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

import structlog

from app.messaging.rabbitmq_client import TradingMessage, MessageType, get_rabbitmq_client
from app.websockets.trading_socket import dashboard_websocket
from app.core.cache import get_market_cache

logger = structlog.get_logger(__name__)


class TradingMessageHandler:
    """Main message handler for trading system events"""
    
    def __init__(self):
        self.market_cache = get_market_cache()
        self.processing_stats = {
            "messages_processed": 0,
            "messages_failed": 0,
            "last_processed": None,
        }

    async def handle_trading_signal(self, message: TradingMessage):
        """Handle trading signal messages"""
        try:
            payload = message.payload
            logger.info(
                "Processing trading signal",
                symbol=payload.get("symbol"),
                action=payload.get("action"),
                confidence=payload.get("confidence")
            )
            
            # Broadcast to WebSocket clients
            if dashboard_websocket.connection_manager.active_connections:
                from app.websockets.trading_socket import TradingSignalUpdate
                
                signal_update = TradingSignalUpdate(
                    symbol=payload.get("symbol", "UNKNOWN"),
                    signal_type=payload.get("action", "HOLD"),
                    confidence=payload.get("confidence", 0.0),
                    reasoning=payload.get("reasoning", "No reason provided"),
                    entry_price=payload.get("entry_price"),
                    timestamp=message.timestamp,
                    strategy=payload.get("strategy", "unknown"),
                    metadata=payload.get("metadata", {})
                )
                
                await dashboard_websocket.connection_manager.broadcast_trading_signal(signal_update)
            
            # Cache the signal for historical analysis
            cache_key = f"signal_{payload.get('symbol')}_{message.timestamp}"
            await self.market_cache.set(cache_key, payload, ttl_override=3600)  # 1 hour
            
            self.processing_stats["messages_processed"] += 1
            self.processing_stats["last_processed"] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error("Failed to handle trading signal", error=str(e))
            self.processing_stats["messages_failed"] += 1
            raise

    async def handle_portfolio_update(self, message: TradingMessage):
        """Handle portfolio update messages"""
        try:
            payload = message.payload
            logger.info("Processing portfolio update", total_value=payload.get("total_value"))
            
            # Broadcast to WebSocket clients
            if dashboard_websocket.connection_manager.active_connections:
                from app.websockets.trading_socket import PortfolioUpdate
                
                portfolio_update = PortfolioUpdate(
                    total_value=payload.get("total_value", 0.0),
                    cash_balance=payload.get("cash_balance", 0.0),
                    day_change=payload.get("day_change", 0.0),
                    day_change_percent=payload.get("day_change_percent", 0.0),
                    positions_count=payload.get("positions_count", 0),
                    buying_power=payload.get("buying_power", 0.0),
                    timestamp=message.timestamp,
                    positions=payload.get("positions", [])
                )
                
                await dashboard_websocket.connection_manager.broadcast_portfolio_update(portfolio_update)
            
            # Cache portfolio state
            await self.market_cache.set("portfolio_latest", payload, ttl_override=300)  # 5 minutes
            
            self.processing_stats["messages_processed"] += 1
            self.processing_stats["last_processed"] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error("Failed to handle portfolio update", error=str(e))
            self.processing_stats["messages_failed"] += 1
            raise

    async def handle_market_data(self, message: TradingMessage):
        """Handle market data messages"""
        try:
            payload = message.payload
            symbol = payload.get("symbol", "UNKNOWN")
            logger.debug("Processing market data", symbol=symbol, price=payload.get("price"))
            
            # Broadcast to WebSocket clients
            if dashboard_websocket.connection_manager.active_connections:
                from app.websockets.trading_socket import MarketUpdate
                
                market_update = MarketUpdate(
                    symbol=symbol,
                    price=payload.get("price", 0.0),
                    change=payload.get("change", 0.0),
                    change_percent=payload.get("change_percent", 0.0),
                    volume=payload.get("volume", 0),
                    timestamp=message.timestamp,
                    bid=payload.get("bid", 0.0),
                    ask=payload.get("ask", 0.0),
                    high=payload.get("high", 0.0),
                    low=payload.get("low", 0.0)
                )
                
                await dashboard_websocket.connection_manager.broadcast_market_update(market_update)
            
            # Cache market data
            cache_key = f"market_data_{symbol}"
            await self.market_cache.set(cache_key, payload, ttl_override=60)  # 1 minute
            
            self.processing_stats["messages_processed"] += 1
            self.processing_stats["last_processed"] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error("Failed to handle market data", error=str(e), symbol=symbol)
            self.processing_stats["messages_failed"] += 1
            raise

    async def handle_risk_alert(self, message: TradingMessage):
        """Handle risk management alerts"""
        try:
            payload = message.payload
            alert_type = payload.get("alert_type", "unknown")
            severity = payload.get("severity", "medium")
            
            logger.warning(
                "Risk alert received",
                alert_type=alert_type,
                severity=severity,
                message=payload.get("message", "No message")
            )
            
            # Send urgent notification to WebSocket clients
            if dashboard_websocket.connection_manager.active_connections:
                alert_message = {
                    "type": "risk_alert",
                    "data": payload,
                    "timestamp": message.timestamp,
                    "urgent": severity == "high"
                }
                
                # Broadcast to all connected clients immediately
                await dashboard_websocket.connection_manager._broadcast_to_subscribers(
                    "system_health", "enabled", alert_message
                )
            
            # Cache alert for historical tracking
            cache_key = f"risk_alert_{alert_type}_{message.timestamp}"
            await self.market_cache.set(cache_key, payload, ttl_override=86400)  # 24 hours
            
            self.processing_stats["messages_processed"] += 1
            self.processing_stats["last_processed"] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error("Failed to handle risk alert", error=str(e))
            self.processing_stats["messages_failed"] += 1
            raise

    async def handle_system_health(self, message: TradingMessage):
        """Handle system health messages"""
        try:
            payload = message.payload
            service_name = payload.get("service_name", "unknown")
            status = payload.get("status", "unknown")
            
            logger.info("System health update", service=service_name, status=status)
            
            # Update cached health status
            health_cache_key = f"health_{service_name}"
            await self.market_cache.set(health_cache_key, payload, ttl_override=300)  # 5 minutes
            
            # Broadcast to WebSocket if significant status change
            if status in ["unhealthy", "degraded"]:
                if dashboard_websocket.connection_manager.active_connections:
                    health_message = {
                        "type": "system_health",
                        "data": payload,
                        "timestamp": message.timestamp
                    }
                    
                    await dashboard_websocket.connection_manager._broadcast_to_subscribers(
                        "system_health", "enabled", health_message
                    )
            
            self.processing_stats["messages_processed"] += 1
            self.processing_stats["last_processed"] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error("Failed to handle system health", error=str(e))
            self.processing_stats["messages_failed"] += 1
            raise

    async def handle_websocket_broadcast(self, message: TradingMessage):
        """Handle messages intended for WebSocket broadcast"""
        try:
            payload = message.payload
            broadcast_type = payload.get("broadcast_type", "general")
            
            logger.debug("Processing WebSocket broadcast", type=broadcast_type)
            
            # Direct broadcast to WebSocket clients
            if dashboard_websocket.connection_manager.active_connections:
                ws_message = {
                    "type": broadcast_type,
                    "data": payload.get("data", {}),
                    "timestamp": message.timestamp
                }
                
                # Determine subscription type
                subscription_type = payload.get("subscription_type", "system_health")
                subscription_key = payload.get("subscription_key", "enabled")
                
                await dashboard_websocket.connection_manager._broadcast_to_subscribers(
                    subscription_type, subscription_key, ws_message
                )
            
            self.processing_stats["messages_processed"] += 1
            self.processing_stats["last_processed"] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error("Failed to handle WebSocket broadcast", error=str(e))
            self.processing_stats["messages_failed"] += 1
            raise

    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get message processing statistics"""
        return {
            **self.processing_stats,
            "timestamp": datetime.now().isoformat()
        }


class MessageRouter:
    """Routes messages to appropriate handlers based on message type"""
    
    def __init__(self):
        self.handler = TradingMessageHandler()
        
        # Message type routing map
        self.route_map = {
            MessageType.TRADING_SIGNAL.value: self.handler.handle_trading_signal,
            MessageType.PORTFOLIO_UPDATE.value: self.handler.handle_portfolio_update,
            MessageType.MARKET_DATA.value: self.handler.handle_market_data,
            MessageType.RISK_ALERT.value: self.handler.handle_risk_alert,
            MessageType.SYSTEM_HEALTH.value: self.handler.handle_system_health,
            MessageType.WEBSOCKET_BROADCAST.value: self.handler.handle_websocket_broadcast,
        }

    async def route_message(self, message: TradingMessage):
        """Route message to appropriate handler"""
        try:
            handler = self.route_map.get(message.message_type)
            
            if handler:
                await handler(message)
                logger.debug("Message routed successfully", message_type=message.message_type)
            else:
                logger.warning("No handler for message type", message_type=message.message_type)
                
        except Exception as e:
            logger.error(
                "Failed to route message",
                error=str(e),
                message_type=message.message_type
            )
            raise


# Global message router instance
message_router = MessageRouter()


async def setup_message_consumers():
    """Set up RabbitMQ message consumers"""
    try:
        rabbitmq_client = await get_rabbitmq_client()
        
        # Subscribe to all trading-related messages
        await rabbitmq_client.subscribe_to_messages(
            routing_patterns=[
                "trading.*",      # All trading signals
                "portfolio.*",    # Portfolio updates
                "market.*",       # Market data
                "risk.*",         # Risk alerts
                "system.*",       # System health
                "websocket.*",    # WebSocket broadcasts
            ],
            queue_name="swaggy_stacks_main_queue",
            handler=message_router.route_message,
            durable=True
        )
        
        logger.info("Message consumers set up successfully")
        
    except Exception as e:
        logger.error("Failed to set up message consumers", error=str(e))
        raise