"""
Market condition triggers for event-driven agent system
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import structlog

from app.events.trigger_engine import (
    TriggerCondition, TriggerType, TriggerPriority, TriggerEvent, get_trigger_engine
)
from app.core.config import settings

logger = structlog.get_logger(__name__)


class MarketTriggerManager:
    """
    Manages market-specific triggers:
    - Price movement triggers
    - Volume spike detection
    - Volatility alerts
    - Technical indicator signals
    """

    def __init__(self):
        self.trigger_engine = None
        self.default_triggers_created = False

    async def initialize(self):
        """Initialize market trigger manager"""
        self.trigger_engine = await get_trigger_engine()
        await self._register_event_handlers()
        await self._create_default_triggers()
        logger.info("MarketTriggerManager initialized")

    async def _register_event_handlers(self):
        """Register handlers for market events"""
        await self.trigger_engine.register_event_handler(
            TriggerType.PRICE_MOVE,
            self._handle_price_move_event
        )
        await self.trigger_engine.register_event_handler(
            TriggerType.VOLUME_SPIKE,
            self._handle_volume_spike_event
        )
        await self.trigger_engine.register_event_handler(
            TriggerType.VOLATILITY_SPIKE,
            self._handle_volatility_spike_event
        )
        await self.trigger_engine.register_event_handler(
            TriggerType.TECHNICAL_SIGNAL,
            self._handle_technical_signal_event
        )

    async def _create_default_triggers(self):
        """Create default market triggers for common scenarios"""
        if self.default_triggers_created:
            return

        # High-priority symbols for enhanced monitoring
        high_priority_symbols = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "GOOGL", "AMZN"]

        # Medium-priority symbols
        medium_priority_symbols = ["META", "NFLX", "AMD", "INTC", "CRM", "ORCL", "ADBE", "NOW"]

        # Create price movement triggers
        await self._create_price_movement_triggers(high_priority_symbols, medium_priority_symbols)

        # Create volume spike triggers
        await self._create_volume_spike_triggers(high_priority_symbols, medium_priority_symbols)

        # Create volatility spike triggers
        await self._create_volatility_spike_triggers(high_priority_symbols)

        # Create technical signal triggers
        await self._create_technical_signal_triggers(high_priority_symbols)

        self.default_triggers_created = True
        logger.info("Default market triggers created")

    async def _create_price_movement_triggers(self, high_priority: List[str], medium_priority: List[str]):
        """Create price movement triggers for different symbols and thresholds"""

        # Critical price moves (3%+ in 5 minutes) - High priority symbols
        for symbol in high_priority:
            trigger = TriggerCondition(
                trigger_id=f"price_critical_{symbol}",
                trigger_type=TriggerType.PRICE_MOVE,
                symbol=symbol,
                condition="large_move",
                threshold=3.0,  # 3% move
                priority=TriggerPriority.CRITICAL,
                target_agents=["analyst", "risk", "strategist"],
                metadata={"lookback_minutes": 5, "move_type": "any"},
                cooldown_seconds=300  # 5 minutes
            )
            await self.trigger_engine.register_trigger(trigger)

        # Significant price moves (2%+ in 10 minutes) - High priority symbols
        for symbol in high_priority:
            trigger = TriggerCondition(
                trigger_id=f"price_high_{symbol}",
                trigger_type=TriggerType.PRICE_MOVE,
                symbol=symbol,
                condition="significant_move",
                threshold=2.0,  # 2% move
                priority=TriggerPriority.HIGH,
                target_agents=["analyst", "strategist"],
                metadata={"lookback_minutes": 10, "move_type": "any"},
                cooldown_seconds=600  # 10 minutes
            )
            await self.trigger_engine.register_trigger(trigger)

        # Medium price moves for medium priority symbols
        for symbol in medium_priority:
            trigger = TriggerCondition(
                trigger_id=f"price_medium_{symbol}",
                trigger_type=TriggerType.PRICE_MOVE,
                symbol=symbol,
                condition="medium_move",
                threshold=2.5,  # 2.5% move
                priority=TriggerPriority.MEDIUM,
                target_agents=["analyst"],
                metadata={"lookback_minutes": 15, "move_type": "any"},
                cooldown_seconds=900  # 15 minutes
            )
            await self.trigger_engine.register_trigger(trigger)

    async def _create_volume_spike_triggers(self, high_priority: List[str], medium_priority: List[str]):
        """Create volume spike triggers"""

        # Critical volume spikes (5x average) - High priority symbols
        for symbol in high_priority:
            trigger = TriggerCondition(
                trigger_id=f"volume_critical_{symbol}",
                trigger_type=TriggerType.VOLUME_SPIKE,
                symbol=symbol,
                condition="extreme_volume",
                threshold=5.0,  # 5x average volume
                priority=TriggerPriority.CRITICAL,
                target_agents=["analyst", "risk"],
                metadata={"comparison_period": "20_bars"},
                cooldown_seconds=180  # 3 minutes
            )
            await self.trigger_engine.register_trigger(trigger)

        # High volume spikes (3x average)
        for symbol in high_priority + medium_priority:
            trigger = TriggerCondition(
                trigger_id=f"volume_high_{symbol}",
                trigger_type=TriggerType.VOLUME_SPIKE,
                symbol=symbol,
                condition="high_volume",
                threshold=3.0,  # 3x average volume
                priority=TriggerPriority.HIGH,
                target_agents=["analyst"],
                metadata={"comparison_period": "20_bars"},
                cooldown_seconds=300  # 5 minutes
            )
            await self.trigger_engine.register_trigger(trigger)

    async def _create_volatility_spike_triggers(self, high_priority: List[str]):
        """Create volatility spike triggers"""

        for symbol in high_priority:
            # High volatility trigger
            trigger = TriggerCondition(
                trigger_id=f"volatility_high_{symbol}",
                trigger_type=TriggerType.VOLATILITY_SPIKE,
                symbol=symbol,
                condition="high_volatility",
                threshold=25.0,  # 25% volatility
                priority=TriggerPriority.HIGH,
                target_agents=["risk", "strategist"],
                metadata={"calculation_method": "rolling_std", "period": 20},
                cooldown_seconds=600  # 10 minutes
            )
            await self.trigger_engine.register_trigger(trigger)

    async def _create_technical_signal_triggers(self, high_priority: List[str]):
        """Create technical indicator signal triggers"""

        for symbol in high_priority:
            # RSI oversold trigger
            trigger = TriggerCondition(
                trigger_id=f"rsi_oversold_{symbol}",
                trigger_type=TriggerType.TECHNICAL_SIGNAL,
                symbol=symbol,
                condition="oversold",
                threshold=30.0,  # RSI < 30
                priority=TriggerPriority.MEDIUM,
                target_agents=["analyst", "strategist"],
                metadata={"signal_type": "rsi", "period": 14},
                cooldown_seconds=1800  # 30 minutes
            )
            await self.trigger_engine.register_trigger(trigger)

            # RSI overbought trigger
            trigger = TriggerCondition(
                trigger_id=f"rsi_overbought_{symbol}",
                trigger_type=TriggerType.TECHNICAL_SIGNAL,
                symbol=symbol,
                condition="overbought",
                threshold=70.0,  # RSI > 70
                priority=TriggerPriority.MEDIUM,
                target_agents=["analyst", "strategist"],
                metadata={"signal_type": "rsi", "period": 14},
                cooldown_seconds=1800  # 30 minutes
            )
            await self.trigger_engine.register_trigger(trigger)

    # Event Handlers
    async def _handle_price_move_event(self, event: TriggerEvent):
        """Handle price movement events"""
        try:
            symbol = event.trigger_condition.symbol
            current_value = event.current_value
            threshold = event.trigger_condition.threshold

            logger.info("Price move detected",
                       symbol=symbol,
                       current_price=current_value,
                       threshold=threshold,
                       priority=event.trigger_condition.priority.value)

            # Wake up target agents
            await self._wake_agents(event.trigger_condition.target_agents, {
                'event_type': 'price_move',
                'symbol': symbol,
                'current_price': current_value,
                'threshold': threshold,
                'market_data': event.market_data,
                'priority': event.trigger_condition.priority.value
            })

        except Exception as e:
            logger.error("Failed to handle price move event",
                        event_id=event.event_id, error=str(e))

    async def _handle_volume_spike_event(self, event: TriggerEvent):
        """Handle volume spike events"""
        try:
            symbol = event.trigger_condition.symbol
            current_volume = event.market_data.get('volume', 0)

            logger.info("Volume spike detected",
                       symbol=symbol,
                       current_volume=current_volume,
                       threshold=event.trigger_condition.threshold)

            # Wake up target agents
            await self._wake_agents(event.trigger_condition.target_agents, {
                'event_type': 'volume_spike',
                'symbol': symbol,
                'current_volume': current_volume,
                'threshold': event.trigger_condition.threshold,
                'market_data': event.market_data,
                'priority': event.trigger_condition.priority.value
            })

        except Exception as e:
            logger.error("Failed to handle volume spike event",
                        event_id=event.event_id, error=str(e))

    async def _handle_volatility_spike_event(self, event: TriggerEvent):
        """Handle volatility spike events"""
        try:
            symbol = event.trigger_condition.symbol

            logger.info("Volatility spike detected",
                       symbol=symbol,
                       threshold=event.trigger_condition.threshold)

            # Wake up target agents
            await self._wake_agents(event.trigger_condition.target_agents, {
                'event_type': 'volatility_spike',
                'symbol': symbol,
                'threshold': event.trigger_condition.threshold,
                'market_data': event.market_data,
                'priority': event.trigger_condition.priority.value
            })

        except Exception as e:
            logger.error("Failed to handle volatility spike event",
                        event_id=event.event_id, error=str(e))

    async def _handle_technical_signal_event(self, event: TriggerEvent):
        """Handle technical signal events"""
        try:
            symbol = event.trigger_condition.symbol
            signal_type = event.trigger_condition.metadata.get('signal_type', 'unknown')

            logger.info("Technical signal detected",
                       symbol=symbol,
                       signal_type=signal_type,
                       condition=event.trigger_condition.condition)

            # Wake up target agents
            await self._wake_agents(event.trigger_condition.target_agents, {
                'event_type': 'technical_signal',
                'symbol': symbol,
                'signal_type': signal_type,
                'condition': event.trigger_condition.condition,
                'threshold': event.trigger_condition.threshold,
                'market_data': event.market_data,
                'priority': event.trigger_condition.priority.value
            })

        except Exception as e:
            logger.error("Failed to handle technical signal event",
                        event_id=event.event_id, error=str(e))

    async def _wake_agents(self, agent_names: List[str], event_data: Dict[str, Any]):
        """Wake up specified agents with event data"""
        try:
            # Import here to avoid circular imports
            from app.ai.coordination_hub import get_coordination_hub

            hub = await get_coordination_hub()
            await hub.broadcast_market_event(agent_names, event_data)

            logger.info("Agents awakened",
                       agents=agent_names,
                       event_type=event_data.get('event_type'))

        except Exception as e:
            logger.error("Failed to wake agents",
                        agents=agent_names, error=str(e))

    # Public API
    async def create_custom_price_trigger(
        self,
        symbol: str,
        threshold: float,
        lookback_minutes: int = 5,
        priority: TriggerPriority = TriggerPriority.MEDIUM,
        target_agents: List[str] = None,
        cooldown_seconds: int = 300
    ) -> str:
        """Create a custom price movement trigger"""

        if target_agents is None:
            target_agents = ["analyst"]

        trigger = TriggerCondition(
            trigger_id=f"custom_price_{symbol}_{int(datetime.utcnow().timestamp())}",
            trigger_type=TriggerType.PRICE_MOVE,
            symbol=symbol,
            condition="custom_move",
            threshold=threshold,
            priority=priority,
            target_agents=target_agents,
            metadata={"lookback_minutes": lookback_minutes},
            cooldown_seconds=cooldown_seconds
        )

        return await self.trigger_engine.register_trigger(trigger)

    async def create_custom_volume_trigger(
        self,
        symbol: str,
        volume_multiplier: float,
        priority: TriggerPriority = TriggerPriority.MEDIUM,
        target_agents: List[str] = None,
        cooldown_seconds: int = 300
    ) -> str:
        """Create a custom volume spike trigger"""

        if target_agents is None:
            target_agents = ["analyst"]

        trigger = TriggerCondition(
            trigger_id=f"custom_volume_{symbol}_{int(datetime.utcnow().timestamp())}",
            trigger_type=TriggerType.VOLUME_SPIKE,
            symbol=symbol,
            condition="custom_volume",
            threshold=volume_multiplier,
            priority=priority,
            target_agents=target_agents,
            metadata={"comparison_period": "20_bars"},
            cooldown_seconds=cooldown_seconds
        )

        return await self.trigger_engine.register_trigger(trigger)

    async def get_market_trigger_status(self) -> Dict[str, Any]:
        """Get status of market triggers"""
        stats = await self.trigger_engine.get_statistics()
        active_triggers = await self.trigger_engine.get_active_triggers()

        # Categorize triggers by type
        trigger_counts = {}
        for trigger in active_triggers:
            trigger_type = trigger['type']
            trigger_counts[trigger_type] = trigger_counts.get(trigger_type, 0) + 1

        return {
            'total_triggers': len(active_triggers),
            'trigger_counts_by_type': trigger_counts,
            'engine_stats': stats,
            'default_triggers_created': self.default_triggers_created
        }


# Global market trigger manager instance
market_trigger_manager: Optional[MarketTriggerManager] = None


async def get_market_trigger_manager() -> MarketTriggerManager:
    """Get or create global market trigger manager instance"""
    global market_trigger_manager

    if market_trigger_manager is None:
        market_trigger_manager = MarketTriggerManager()
        await market_trigger_manager.initialize()

    return market_trigger_manager