"""
Event-Driven Trigger Engine for waking agents based on market conditions
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import heapq
from collections import defaultdict, deque
import structlog

from app.core.config import settings

logger = structlog.get_logger(__name__)


class TriggerType(Enum):
    """Types of triggers"""
    PRICE_MOVE = "price_move"
    VOLUME_SPIKE = "volume_spike"
    VOLATILITY_SPIKE = "volatility_spike"
    TECHNICAL_SIGNAL = "technical_signal"
    TIME_BASED = "time_based"
    PATTERN_DETECTION = "pattern_detection"
    CUSTOM = "custom"


class TriggerPriority(Enum):
    """Trigger priority levels"""
    CRITICAL = 1    # Immediate action required
    HIGH = 2        # High-value opportunities
    MEDIUM = 3      # Standard opportunities
    LOW = 4         # Background monitoring


@dataclass
class TriggerCondition:
    """Individual trigger condition"""
    trigger_id: str
    trigger_type: TriggerType
    symbol: str
    condition: str
    threshold: float
    priority: TriggerPriority
    target_agents: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    cooldown_seconds: int = 60
    last_triggered: Optional[datetime] = None
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TriggerEvent:
    """Triggered event ready for processing"""
    event_id: str
    trigger_condition: TriggerCondition
    current_value: float
    triggered_at: datetime
    market_data: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    def __lt__(self, other):
        """For priority queue ordering"""
        return self.trigger_condition.priority.value < other.trigger_condition.priority.value


class EventTriggerEngine:
    """
    Comprehensive event trigger system for agent coordination:
    - Rule-based triggers for market conditions
    - Priority queue for event processing
    - Support for thousands of concurrent triggers
    - Intelligent cooldown management
    """

    def __init__(self):
        self.triggers: Dict[str, TriggerCondition] = {}
        self.symbol_triggers: Dict[str, Set[str]] = defaultdict(set)
        self.event_queue: List[TriggerEvent] = []
        self.event_handlers: Dict[TriggerType, List[Callable]] = defaultdict(list)

        # Performance tracking
        self.stats = {
            'triggers_registered': 0,
            'events_triggered': 0,
            'events_processed': 0,
            'events_skipped_cooldown': 0,
            'processing_time_ms': deque(maxlen=1000),
            'queue_size_history': deque(maxlen=100)
        }

        # Market data cache for trigger evaluation
        self.market_data_cache: Dict[str, Dict[str, Any]] = {}
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))

        # Processing control
        self.is_running = False
        self.processing_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        logger.info("EventTriggerEngine initialized")

    async def start(self):
        """Start the trigger engine"""
        if self.is_running:
            logger.warning("Trigger engine already running")
            return

        self.is_running = True
        self.processing_task = asyncio.create_task(self._process_events())
        logger.info("EventTriggerEngine started")

    async def stop(self):
        """Stop the trigger engine"""
        self.is_running = False

        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass

        logger.info("EventTriggerEngine stopped")

    async def register_trigger(self, trigger: TriggerCondition) -> str:
        """Register a new trigger condition"""
        async with self._lock:
            self.triggers[trigger.trigger_id] = trigger
            self.symbol_triggers[trigger.symbol].add(trigger.trigger_id)
            self.stats['triggers_registered'] += 1

        logger.info("Trigger registered",
                   trigger_id=trigger.trigger_id,
                   symbol=trigger.symbol,
                   type=trigger.trigger_type.value)

        return trigger.trigger_id

    async def unregister_trigger(self, trigger_id: str):
        """Unregister a trigger condition"""
        async with self._lock:
            if trigger_id in self.triggers:
                trigger = self.triggers[trigger_id]
                self.symbol_triggers[trigger.symbol].discard(trigger_id)
                del self.triggers[trigger_id]

                logger.info("Trigger unregistered", trigger_id=trigger_id)

    async def register_event_handler(self, trigger_type: TriggerType, handler: Callable):
        """Register an event handler for specific trigger types"""
        self.event_handlers[trigger_type].append(handler)
        logger.info("Event handler registered", trigger_type=trigger_type.value)

    async def process_market_data(self, symbol: str, market_data: Dict[str, Any]):
        """Process incoming market data and evaluate triggers"""
        try:
            # Update market data cache
            self.market_data_cache[symbol] = {
                **market_data,
                'timestamp': datetime.utcnow(),
                'received_at': time.time()
            }

            # Update historical data
            if 'price' in market_data:
                self.price_history[symbol].append({
                    'price': market_data['price'],
                    'timestamp': datetime.utcnow()
                })

            if 'volume' in market_data:
                self.volume_history[symbol].append({
                    'volume': market_data['volume'],
                    'timestamp': datetime.utcnow()
                })

            # Evaluate triggers for this symbol
            await self._evaluate_symbol_triggers(symbol, market_data)

        except Exception as e:
            logger.error("Failed to process market data",
                        symbol=symbol, error=str(e))

    async def _evaluate_symbol_triggers(self, symbol: str, market_data: Dict[str, Any]):
        """Evaluate all triggers for a specific symbol"""
        if symbol not in self.symbol_triggers:
            return

        triggered_events = []

        async with self._lock:
            for trigger_id in self.symbol_triggers[symbol]:
                trigger = self.triggers.get(trigger_id)
                if not trigger or not trigger.is_active:
                    continue

                # Check cooldown
                if self._is_in_cooldown(trigger):
                    self.stats['events_skipped_cooldown'] += 1
                    continue

                # Evaluate trigger condition
                if await self._evaluate_trigger(trigger, symbol, market_data):
                    event = TriggerEvent(
                        event_id=f"{trigger_id}_{int(time.time())}",
                        trigger_condition=trigger,
                        current_value=market_data.get('price', 0),
                        triggered_at=datetime.utcnow(),
                        market_data=market_data.copy()
                    )
                    triggered_events.append(event)
                    trigger.last_triggered = datetime.utcnow()

        # Add events to queue
        for event in triggered_events:
            await self._add_event_to_queue(event)

    async def _evaluate_trigger(self, trigger: TriggerCondition, symbol: str, market_data: Dict[str, Any]) -> bool:
        """Evaluate if a trigger condition is met"""
        try:
            current_price = market_data.get('price', 0)
            current_volume = market_data.get('volume', 0)

            if trigger.trigger_type == TriggerType.PRICE_MOVE:
                return await self._evaluate_price_move(trigger, symbol, current_price)

            elif trigger.trigger_type == TriggerType.VOLUME_SPIKE:
                return await self._evaluate_volume_spike(trigger, symbol, current_volume)

            elif trigger.trigger_type == TriggerType.VOLATILITY_SPIKE:
                return await self._evaluate_volatility_spike(trigger, symbol, market_data)

            elif trigger.trigger_type == TriggerType.TECHNICAL_SIGNAL:
                return await self._evaluate_technical_signal(trigger, symbol, market_data)

            elif trigger.trigger_type == TriggerType.PATTERN_DETECTION:
                return await self._evaluate_pattern_detection(trigger, symbol, market_data)

            elif trigger.trigger_type == TriggerType.CUSTOM:
                return await self._evaluate_custom_trigger(trigger, symbol, market_data)

            return False

        except Exception as e:
            logger.error("Failed to evaluate trigger",
                        trigger_id=trigger.trigger_id, error=str(e))
            return False

    async def _evaluate_price_move(self, trigger: TriggerCondition, symbol: str, current_price: float) -> bool:
        """Evaluate price movement triggers"""
        if not self.price_history[symbol] or len(self.price_history[symbol]) < 2:
            return False

        # Get price from specified time ago
        lookback_minutes = trigger.metadata.get('lookback_minutes', 5)
        lookback_time = datetime.utcnow() - timedelta(minutes=lookback_minutes)

        # Find price at lookback time
        reference_price = None
        for price_data in self.price_history[symbol]:
            if price_data['timestamp'] <= lookback_time:
                reference_price = price_data['price']
                break

        if reference_price is None:
            reference_price = self.price_history[symbol][0]['price']

        # Calculate percentage change
        if reference_price > 0:
            price_change_pct = abs((current_price - reference_price) / reference_price) * 100
            return price_change_pct >= trigger.threshold

        return False

    async def _evaluate_volume_spike(self, trigger: TriggerCondition, symbol: str, current_volume: float) -> bool:
        """Evaluate volume spike triggers"""
        if not self.volume_history[symbol] or len(self.volume_history[symbol]) < 10:
            return False

        # Calculate average volume
        volumes = [v['volume'] for v in self.volume_history[symbol]]
        avg_volume = sum(volumes) / len(volumes)

        if avg_volume > 0:
            volume_ratio = current_volume / avg_volume
            return volume_ratio >= trigger.threshold

        return False

    async def _evaluate_volatility_spike(self, trigger: TriggerCondition, symbol: str, market_data: Dict[str, Any]) -> bool:
        """Evaluate volatility spike triggers"""
        # Simple volatility estimation using price history
        if len(self.price_history[symbol]) < 20:
            return False

        prices = [p['price'] for p in self.price_history[symbol]]

        # Calculate simple volatility (standard deviation of returns)
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                returns.append((prices[i] - prices[i-1]) / prices[i-1])

        if len(returns) < 10:
            return False

        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        current_volatility = variance ** 0.5 * 100  # Convert to percentage

        return current_volatility >= trigger.threshold

    async def _evaluate_technical_signal(self, trigger: TriggerCondition, symbol: str, market_data: Dict[str, Any]) -> bool:
        """Evaluate technical indicator signals"""
        # Placeholder for technical indicators
        # This would integrate with technical analysis modules
        signal_type = trigger.metadata.get('signal_type', 'rsi')

        if signal_type == 'rsi':
            rsi_value = market_data.get('rsi', 50)
            if trigger.condition == 'oversold':
                return rsi_value <= trigger.threshold
            elif trigger.condition == 'overbought':
                return rsi_value >= trigger.threshold

        return False

    async def _evaluate_pattern_detection(self, trigger: TriggerCondition, symbol: str, market_data: Dict[str, Any]) -> bool:
        """Evaluate pattern detection triggers"""
        # Placeholder for pattern detection
        # This would integrate with pattern recognition modules
        pattern_type = trigger.metadata.get('pattern_type', 'breakout')
        confidence = market_data.get('pattern_confidence', 0)

        return confidence >= trigger.threshold

    async def _evaluate_custom_trigger(self, trigger: TriggerCondition, symbol: str, market_data: Dict[str, Any]) -> bool:
        """Evaluate custom trigger conditions"""
        # Execute custom condition logic
        custom_function = trigger.metadata.get('custom_function')
        if custom_function and callable(custom_function):
            try:
                return await custom_function(symbol, market_data, trigger)
            except Exception as e:
                logger.error("Custom trigger function failed",
                           trigger_id=trigger.trigger_id, error=str(e))

        return False

    def _is_in_cooldown(self, trigger: TriggerCondition) -> bool:
        """Check if trigger is in cooldown period"""
        if trigger.last_triggered is None:
            return False

        time_since_trigger = (datetime.utcnow() - trigger.last_triggered).total_seconds()
        return time_since_trigger < trigger.cooldown_seconds

    async def _add_event_to_queue(self, event: TriggerEvent):
        """Add event to priority queue"""
        heapq.heappush(self.event_queue, event)
        self.stats['events_triggered'] += 1
        self.stats['queue_size_history'].append(len(self.event_queue))

        logger.info("Event triggered",
                   event_id=event.event_id,
                   symbol=event.trigger_condition.symbol,
                   type=event.trigger_condition.trigger_type.value,
                   priority=event.trigger_condition.priority.value)

    async def _process_events(self):
        """Main event processing loop"""
        logger.info("Event processing loop started")

        while self.is_running:
            try:
                if not self.event_queue:
                    await asyncio.sleep(0.1)
                    continue

                # Process events in priority order
                start_time = time.time()

                # Get highest priority event
                event = heapq.heappop(self.event_queue)

                # Process the event
                await self._handle_event(event)

                # Track processing time
                processing_time = (time.time() - start_time) * 1000
                self.stats['processing_time_ms'].append(processing_time)
                self.stats['events_processed'] += 1

            except Exception as e:
                logger.error("Error in event processing loop", error=str(e))
                await asyncio.sleep(1)

    async def _handle_event(self, event: TriggerEvent):
        """Handle a triggered event"""
        try:
            trigger_type = event.trigger_condition.trigger_type

            # Call registered handlers
            for handler in self.event_handlers[trigger_type]:
                try:
                    await handler(event)
                except Exception as e:
                    logger.error("Event handler failed",
                               event_id=event.event_id,
                               handler=str(handler),
                               error=str(e))

            # Log event processing
            logger.info("Event processed",
                       event_id=event.event_id,
                       symbol=event.trigger_condition.symbol,
                       agents=event.trigger_condition.target_agents)

        except Exception as e:
            logger.error("Failed to handle event",
                        event_id=event.event_id, error=str(e))

    async def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics"""
        async with self._lock:
            avg_processing_time = 0
            if self.stats['processing_time_ms']:
                avg_processing_time = sum(self.stats['processing_time_ms']) / len(self.stats['processing_time_ms'])

            return {
                'triggers_registered': self.stats['triggers_registered'],
                'active_triggers': len(self.triggers),
                'events_triggered': self.stats['events_triggered'],
                'events_processed': self.stats['events_processed'],
                'events_skipped_cooldown': self.stats['events_skipped_cooldown'],
                'current_queue_size': len(self.event_queue),
                'avg_processing_time_ms': round(avg_processing_time, 2),
                'symbols_monitored': len(self.symbol_triggers),
                'is_running': self.is_running
            }

    async def get_active_triggers(self, symbol: str = None) -> List[Dict[str, Any]]:
        """Get list of active triggers"""
        async with self._lock:
            triggers = []

            for trigger_id, trigger in self.triggers.items():
                if symbol and trigger.symbol != symbol:
                    continue

                if trigger.is_active:
                    triggers.append({
                        'trigger_id': trigger.trigger_id,
                        'symbol': trigger.symbol,
                        'type': trigger.trigger_type.value,
                        'condition': trigger.condition,
                        'threshold': trigger.threshold,
                        'priority': trigger.priority.value,
                        'target_agents': trigger.target_agents,
                        'cooldown_seconds': trigger.cooldown_seconds,
                        'last_triggered': trigger.last_triggered.isoformat() if trigger.last_triggered else None,
                        'in_cooldown': self._is_in_cooldown(trigger)
                    })

            return triggers


# Global trigger engine instance
trigger_engine: Optional[EventTriggerEngine] = None


async def get_trigger_engine() -> EventTriggerEngine:
    """Get or create global trigger engine instance"""
    global trigger_engine

    if trigger_engine is None:
        trigger_engine = EventTriggerEngine()
        if settings.EVENT_TRIGGERS_ENABLED:
            await trigger_engine.start()

    return trigger_engine