"""
RabbitMQ message queue integration for distributed trading system.
Provides reliable message passing for trading signals, portfolio updates, and system events.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

import aio_pika
from aio_pika import Message, DeliveryMode, ExchangeType, connect_robust
from aio_pika.abc import AbstractIncomingMessage
import structlog

from app.core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class MessageType(Enum):
    """Message types for routing"""
    TRADING_SIGNAL = "trading.signal"
    PORTFOLIO_UPDATE = "portfolio.update"
    MARKET_DATA = "market.data"
    SYSTEM_HEALTH = "system.health"
    ORDER_EXECUTION = "order.execution"
    RISK_ALERT = "risk.alert"
    AI_ANALYSIS = "ai.analysis"
    WEBSOCKET_BROADCAST = "websocket.broadcast"


@dataclass
class TradingMessage:
    """Standard message format for trading system"""
    message_type: str
    payload: Dict[str, Any]
    timestamp: str
    source_service: str
    correlation_id: Optional[str] = None
    priority: int = 0  # 0 = normal, 1 = high, 2 = urgent
    retry_count: int = 0
    metadata: Dict[str, Any] = None

    def to_json(self) -> str:
        """Convert message to JSON string"""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> 'TradingMessage':
        """Create message from JSON string"""
        data = json.loads(json_str)
        return cls(**data)


class RabbitMQClient:
    """Async RabbitMQ client for trading system messaging"""

    def __init__(
        self,
        connection_url: Optional[str] = None,
        exchange_name: str = "swaggy_stacks_exchange",
        max_retries: int = 3,
        retry_delay: float = 5.0
    ):
        self.connection_url = connection_url or getattr(settings, 'RABBITMQ_URL', 'amqp://localhost:5672/')
        self.exchange_name = exchange_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Connection components
        self.connection: Optional[aio_pika.abc.AbstractConnection] = None
        self.channel: Optional[aio_pika.abc.AbstractChannel] = None
        self.exchange: Optional[aio_pika.abc.AbstractExchange] = None
        
        # Message handlers
        self.message_handlers: Dict[str, List[Callable]] = {}
        self.dead_letter_queue: Optional[aio_pika.abc.AbstractQueue] = None
        
        # Connection state
        self.connected = False
        self.reconnect_task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        """Establish connection to RabbitMQ"""
        try:
            # Create robust connection with auto-reconnect
            self.connection = await connect_robust(
                self.connection_url,
                heartbeat=300,
                blocked_connection_timeout=300,
            )
            
            # Create channel
            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=10)  # Limit concurrent messages
            
            # Declare exchange
            self.exchange = await self.channel.declare_exchange(
                self.exchange_name,
                ExchangeType.TOPIC,
                durable=True
            )
            
            # Set up dead letter queue
            await self._setup_dead_letter_queue()
            
            self.connected = True
            logger.info("RabbitMQ connection established", exchange=self.exchange_name)
            
            return True
            
        except Exception as e:
            logger.error("Failed to connect to RabbitMQ", error=str(e))
            self.connected = False
            return False

    async def disconnect(self):
        """Close RabbitMQ connection"""
        try:
            if self.reconnect_task:
                self.reconnect_task.cancel()
                
            if self.connection and not self.connection.is_closed:
                await self.connection.close()
                
            self.connected = False
            logger.info("RabbitMQ connection closed")
            
        except Exception as e:
            logger.error("Error during RabbitMQ disconnect", error=str(e))

    async def _setup_dead_letter_queue(self):
        """Set up dead letter queue for failed messages"""
        try:
            # Declare dead letter exchange
            dl_exchange = await self.channel.declare_exchange(
                f"{self.exchange_name}.dlx",
                ExchangeType.DIRECT,
                durable=True
            )
            
            # Declare dead letter queue
            self.dead_letter_queue = await self.channel.declare_queue(
                f"{self.exchange_name}.dlq",
                durable=True,
                arguments={
                    "x-message-ttl": 86400000,  # 24 hours
                    "x-max-length": 1000,
                }
            )
            
            # Bind dead letter queue to exchange
            await self.dead_letter_queue.bind(dl_exchange, routing_key="failed")
            
            logger.info("Dead letter queue configured")
            
        except Exception as e:
            logger.error("Failed to setup dead letter queue", error=str(e))

    async def publish_message(
        self,
        message: TradingMessage,
        routing_key: Optional[str] = None
    ) -> bool:
        """Publish message to exchange"""
        if not self.connected or not self.exchange:
            logger.warning("Cannot publish message - not connected to RabbitMQ")
            return False
            
        try:
            # Use message type as routing key if not provided
            routing_key = routing_key or message.message_type
            
            # Create AMQP message
            amqp_message = Message(
                message.to_json().encode(),
                delivery_mode=DeliveryMode.PERSISTENT,
                priority=message.priority,
                headers={
                    "message_type": message.message_type,
                    "source_service": message.source_service,
                    "correlation_id": message.correlation_id,
                    "timestamp": message.timestamp,
                },
                expiration=300000,  # 5 minutes TTL
            )
            
            # Publish message
            await self.exchange.publish(amqp_message, routing_key=routing_key)
            
            logger.debug(
                "Message published",
                routing_key=routing_key,
                message_type=message.message_type,
                correlation_id=message.correlation_id
            )
            
            return True
            
        except Exception as e:
            logger.error("Failed to publish message", error=str(e), routing_key=routing_key)
            return False

    async def publish_trading_signal(
        self,
        signal_data: Dict[str, Any],
        source_service: str = "trading_engine"
    ) -> bool:
        """Publish trading signal message"""
        message = TradingMessage(
            message_type=MessageType.TRADING_SIGNAL.value,
            payload=signal_data,
            timestamp=datetime.now().isoformat(),
            source_service=source_service,
            priority=1,  # High priority for trading signals
        )
        
        return await self.publish_message(message)

    async def publish_portfolio_update(
        self,
        portfolio_data: Dict[str, Any],
        source_service: str = "portfolio_service"
    ) -> bool:
        """Publish portfolio update message"""
        message = TradingMessage(
            message_type=MessageType.PORTFOLIO_UPDATE.value,
            payload=portfolio_data,
            timestamp=datetime.now().isoformat(),
            source_service=source_service,
            priority=0,  # Normal priority
        )
        
        return await self.publish_message(message)

    async def publish_market_data(
        self,
        market_data: Dict[str, Any],
        source_service: str = "market_data_service"
    ) -> bool:
        """Publish market data update"""
        message = TradingMessage(
            message_type=MessageType.MARKET_DATA.value,
            payload=market_data,
            timestamp=datetime.now().isoformat(),
            source_service=source_service,
            priority=0,
        )
        
        return await self.publish_message(message)

    async def publish_risk_alert(
        self,
        alert_data: Dict[str, Any],
        source_service: str = "risk_manager"
    ) -> bool:
        """Publish risk management alert"""
        message = TradingMessage(
            message_type=MessageType.RISK_ALERT.value,
            payload=alert_data,
            timestamp=datetime.now().isoformat(),
            source_service=source_service,
            priority=2,  # Urgent priority for risk alerts
        )
        
        return await self.publish_message(message)

    async def subscribe_to_messages(
        self,
        routing_patterns: List[str],
        queue_name: str,
        handler: Callable[[TradingMessage], None],
        durable: bool = True
    ) -> bool:
        """Subscribe to messages matching routing patterns"""
        if not self.connected or not self.exchange:
            logger.warning("Cannot subscribe - not connected to RabbitMQ")
            return False
            
        try:
            # Declare queue with dead letter configuration
            queue_args = {
                "x-dead-letter-exchange": f"{self.exchange_name}.dlx",
                "x-dead-letter-routing-key": "failed",
                "x-max-retries": self.max_retries,
            }
            
            queue = await self.channel.declare_queue(
                queue_name,
                durable=durable,
                arguments=queue_args
            )
            
            # Bind queue to exchange with routing patterns
            for pattern in routing_patterns:
                await queue.bind(self.exchange, routing_key=pattern)
                logger.info(f"Subscribed to pattern: {pattern}")
            
            # Set up message consumer
            async def message_consumer(message: AbstractIncomingMessage):
                async with message.process():
                    try:
                        # Parse message
                        trading_message = TradingMessage.from_json(message.body.decode())
                        
                        # Call handler
                        await handler(trading_message)
                        
                        logger.debug(
                            "Message processed",
                            message_type=trading_message.message_type,
                            queue=queue_name
                        )
                        
                    except Exception as e:
                        logger.error(
                            "Error processing message",
                            error=str(e),
                            queue=queue_name
                        )
                        raise  # Re-raise to trigger dead letter queue
            
            # Start consuming
            await queue.consume(message_consumer)
            
            logger.info(f"Started consuming from queue: {queue_name}")
            return True
            
        except Exception as e:
            logger.error("Failed to subscribe to messages", error=str(e), queue=queue_name)
            return False

    async def get_queue_stats(self, queue_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a queue"""
        if not self.connected or not self.channel:
            return None
            
        try:
            queue = await self.channel.declare_queue(queue_name, passive=True)
            
            # Get queue information
            queue_info = await queue.get_info()
            
            return {
                "name": queue_name,
                "message_count": queue_info.message_count,
                "consumer_count": queue_info.consumer_count,
            }
            
        except Exception as e:
            logger.error("Failed to get queue stats", error=str(e), queue=queue_name)
            return None

    async def health_check(self) -> Dict[str, Any]:
        """Check RabbitMQ connection health"""
        try:
            status = "healthy" if self.connected else "disconnected"
            
            health_data = {
                "status": status,
                "connected": self.connected,
                "exchange_name": self.exchange_name,
                "connection_url": self.connection_url.replace("amqp://", "amqp://***:***@"),  # Hide credentials
                "timestamp": datetime.now().isoformat(),
            }
            
            # Get additional stats if connected
            if self.connected and self.channel:
                try:
                    # Test message publish/consume
                    test_message = TradingMessage(
                        message_type="health.check",
                        payload={"test": True},
                        timestamp=datetime.now().isoformat(),
                        source_service="health_check"
                    )
                    
                    await self.publish_message(test_message, routing_key="health.check")
                    health_data["last_publish"] = "success"
                    
                except Exception as e:
                    health_data["last_publish"] = f"failed: {str(e)}"
                    status = "degraded"
            
            health_data["status"] = status
            return health_data
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }


# Global RabbitMQ client instance
_rabbitmq_client: Optional[RabbitMQClient] = None


async def get_rabbitmq_client() -> RabbitMQClient:
    """Get or create RabbitMQ client instance"""
    global _rabbitmq_client
    
    if _rabbitmq_client is None:
        _rabbitmq_client = RabbitMQClient()
        await _rabbitmq_client.connect()
    
    return _rabbitmq_client


async def close_rabbitmq_client():
    """Close RabbitMQ client connection"""
    global _rabbitmq_client
    
    if _rabbitmq_client:
        await _rabbitmq_client.disconnect()
        _rabbitmq_client = None