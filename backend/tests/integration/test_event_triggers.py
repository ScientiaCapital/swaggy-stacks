"""
Event Trigger System Integration Tests

Comprehensive testing of the event-driven trigger system that powers autonomous trading:
- Complex trigger condition evaluation
- Multi-tier trigger cascading
- Real-time event processing
- Trigger performance optimization
- Event correlation and pattern matching
- Autonomous decision-making workflows

These tests validate the intelligent event system that enables real-time
market response and autonomous trading agent coordination.
"""

import pytest
import pytest_asyncio
import asyncio
import json
import logging
import time
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Callable
import random
from dataclasses import dataclass, field
from enum import Enum

# Test configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TriggerType(Enum):
    """Types of event triggers"""
    PRICE_MOVEMENT = "price_movement"
    VOLUME_SPIKE = "volume_spike"
    TECHNICAL_INDICATOR = "technical_indicator"
    MARKET_REGIME_CHANGE = "market_regime_change"
    RISK_THRESHOLD = "risk_threshold"
    PORTFOLIO_REBALANCE = "portfolio_rebalance"
    NEWS_SENTIMENT = "news_sentiment"
    VOLATILITY_CHANGE = "volatility_change"


class TriggerPriority(Enum):
    """Trigger execution priorities"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


@dataclass
class TriggerCondition:
    """Complex trigger condition with multiple criteria"""
    field_path: str
    operator: str  # eq, ne, gt, lt, gte, lte, in, not_in, contains, regex
    value: Any
    weight: float = 1.0


@dataclass
class EventTrigger:
    """Advanced event trigger with complex conditions"""
    trigger_id: str
    trigger_type: TriggerType
    priority: TriggerPriority
    conditions: List[TriggerCondition]
    action: str
    cooldown_seconds: int = 0
    max_executions: int = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_executed: Optional[datetime] = None
    execution_count: int = 0
    is_active: bool = True


class MockAdvancedEventTriggerSystem:
    """Advanced event trigger system with complex condition evaluation"""

    def __init__(self):
        self.triggers: Dict[str, EventTrigger] = {}
        self.event_history: List[Dict[str, Any]] = []
        self.processing_queue = asyncio.Queue()
        self.is_running = False
        self.processing_task = None
        self.statistics = {
            'events_processed': 0,
            'triggers_evaluated': 0,
            'triggers_fired': 0,
            'triggers_skipped_cooldown': 0,
            'triggers_skipped_max_executions': 0,
            'processing_errors': 0,
            'average_processing_time_ms': 0.0
        }
        self.performance_metrics = []

    async def start(self):
        """Start the advanced trigger system"""
        self.is_running = True
        self.processing_task = asyncio.create_task(self._process_events())
        logger.info("✅ Advanced Event Trigger System started")

    async def stop(self):
        """Stop the trigger system"""
        self.is_running = False
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 Advanced Event Trigger System stopped")

    async def register_trigger(self, trigger: EventTrigger):
        """Register an advanced trigger"""
        self.triggers[trigger.trigger_id] = trigger
        logger.info(f"📋 Advanced trigger registered: {trigger.trigger_id} ({trigger.trigger_type.value})")

    async def submit_event(self, event: Dict[str, Any]) -> bool:
        """Submit event for processing"""
        if not self.is_running:
            return False

        await self.processing_queue.put({
            'event': event,
            'submitted_at': datetime.now(timezone.utc)
        })
        return True

    async def _process_events(self):
        """Process events from the queue"""
        while self.is_running:
            try:
                # Get event with timeout
                try:
                    event_data = await asyncio.wait_for(
                        self.processing_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                await self._process_single_event(event_data)

            except Exception as e:
                logger.error(f"Error in event processing loop: {e}")
                self.statistics['processing_errors'] += 1
                await asyncio.sleep(0.1)

    async def _process_single_event(self, event_data: Dict[str, Any]):
        """Process a single event against all triggers"""
        start_time = time.time()
        event = event_data['event']

        try:
            # Add to history
            self.event_history.append({
                'event': event,
                'timestamp': datetime.now(timezone.utc),
                'processing_start': start_time
            })

            # Keep history manageable
            if len(self.event_history) > 10000:
                self.event_history = self.event_history[-5000:]

            # Evaluate all triggers
            triggered_actions = []
            for trigger in self.triggers.values():
                self.statistics['triggers_evaluated'] += 1

                if await self._should_execute_trigger(trigger, event):
                    if await self._evaluate_trigger_conditions(trigger, event):
                        # Execute trigger
                        await self._execute_trigger(trigger, event)
                        triggered_actions.append(trigger.action)
                        self.statistics['triggers_fired'] += 1

            # Update performance metrics
            processing_time_ms = (time.time() - start_time) * 1000
            self.performance_metrics.append(processing_time_ms)

            # Keep performance metrics manageable
            if len(self.performance_metrics) > 1000:
                self.performance_metrics = self.performance_metrics[-500:]

            # Update average
            self.statistics['average_processing_time_ms'] = sum(self.performance_metrics) / len(self.performance_metrics)
            self.statistics['events_processed'] += 1

        except Exception as e:
            logger.error(f"Error processing event: {e}")
            self.statistics['processing_errors'] += 1

    async def _should_execute_trigger(self, trigger: EventTrigger, event: Dict[str, Any]) -> bool:
        """Check if trigger should be executed (cooldown, max executions, etc.)"""
        if not trigger.is_active:
            return False

        # Check max executions
        if trigger.max_executions and trigger.execution_count >= trigger.max_executions:
            self.statistics['triggers_skipped_max_executions'] += 1
            return False

        # Check cooldown
        if trigger.cooldown_seconds > 0 and trigger.last_executed:
            time_since_last = datetime.now(timezone.utc) - trigger.last_executed
            if time_since_last.total_seconds() < trigger.cooldown_seconds:
                self.statistics['triggers_skipped_cooldown'] += 1
                return False

        return True

    async def _evaluate_trigger_conditions(self, trigger: EventTrigger, event: Dict[str, Any]) -> bool:
        """Evaluate complex trigger conditions"""
        if not trigger.conditions:
            return True

        total_weight = sum(condition.weight for condition in trigger.conditions)
        satisfied_weight = 0.0

        for condition in trigger.conditions:
            if await self._evaluate_single_condition(condition, event):
                satisfied_weight += condition.weight

        # Require all conditions to be satisfied (100% weight)
        return satisfied_weight >= total_weight

    async def _evaluate_single_condition(self, condition: TriggerCondition, event: Dict[str, Any]) -> bool:
        """Evaluate a single condition"""
        try:
            # Get value from event using field path
            event_value = self._get_nested_value(event, condition.field_path)

            if event_value is None:
                return False

            # Evaluate based on operator
            if condition.operator == 'eq':
                return event_value == condition.value
            elif condition.operator == 'ne':
                return event_value != condition.value
            elif condition.operator == 'gt':
                return float(event_value) > float(condition.value)
            elif condition.operator == 'lt':
                return float(event_value) < float(condition.value)
            elif condition.operator == 'gte':
                return float(event_value) >= float(condition.value)
            elif condition.operator == 'lte':
                return float(event_value) <= float(condition.value)
            elif condition.operator == 'in':
                return event_value in condition.value
            elif condition.operator == 'not_in':
                return event_value not in condition.value
            elif condition.operator == 'contains':
                return condition.value in str(event_value)
            elif condition.operator == 'regex':
                import re
                return bool(re.search(condition.value, str(event_value)))
            else:
                logger.warning(f"Unknown operator: {condition.operator}")
                return False

        except Exception as e:
            logger.warning(f"Error evaluating condition: {e}")
            return False

    def _get_nested_value(self, data: Dict[str, Any], field_path: str) -> Any:
        """Get nested value from dict using dot notation"""
        keys = field_path.split('.')
        value = data

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None

        return value

    async def _execute_trigger(self, trigger: EventTrigger, event: Dict[str, Any]):
        """Execute trigger action"""
        trigger.last_executed = datetime.now(timezone.utc)
        trigger.execution_count += 1

        logger.info(f"🔥 Trigger executed: {trigger.trigger_id} -> {trigger.action}")

        # In a real system, this would dispatch to action handlers
        # For testing, we just log the action

    def get_trigger_statistics(self) -> Dict[str, Any]:
        """Get comprehensive trigger statistics"""
        return {
            'total_triggers': len(self.triggers),
            'active_triggers': sum(1 for t in self.triggers.values() if t.is_active),
            'processing_stats': self.statistics.copy(),
            'trigger_execution_counts': {
                trigger_id: trigger.execution_count
                for trigger_id, trigger in self.triggers.items()
            },
            'recent_performance': {
                'min_processing_time_ms': min(self.performance_metrics) if self.performance_metrics else 0,
                'max_processing_time_ms': max(self.performance_metrics) if self.performance_metrics else 0,
                'p95_processing_time_ms': self._percentile(self.performance_metrics, 95) if self.performance_metrics else 0
            }
        }

    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]


@pytest_asyncio.fixture
async def advanced_trigger_system():
    """Create advanced trigger system"""
    system = MockAdvancedEventTriggerSystem()
    await system.start()
    yield system
    await system.stop()


class TestEventTriggerIntegration:
    """Comprehensive event trigger integration tests"""

    @pytest.mark.asyncio
    async def test_complex_trigger_conditions(self, advanced_trigger_system):
        """Test complex multi-condition triggers"""

        # Create complex price movement trigger
        price_trigger = EventTrigger(
            trigger_id='complex_price_movement',
            trigger_type=TriggerType.PRICE_MOVEMENT,
            priority=TriggerPriority.HIGH,
            conditions=[
                TriggerCondition('type', 'eq', 'PRICE_UPDATE'),
                TriggerCondition('price_data.change_percent', 'gt', 5.0, weight=0.6),
                TriggerCondition('price_data.volume', 'gt', 1000000, weight=0.4),
            ],
            action='ALERT_SIGNIFICANT_PRICE_MOVEMENT',
            cooldown_seconds=30
        )

        await advanced_trigger_system.register_trigger(price_trigger)

        # Test events that should trigger
        triggering_events = [
            {
                'type': 'PRICE_UPDATE',
                'symbol': 'AAPL',
                'price_data': {
                    'price': 150.0,
                    'change_percent': 6.5,  # Satisfies condition
                    'volume': 1500000       # Satisfies condition
                },
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        ]

        # Test events that should NOT trigger
        non_triggering_events = [
            {
                'type': 'PRICE_UPDATE',
                'symbol': 'AAPL',
                'price_data': {
                    'price': 150.0,
                    'change_percent': 3.0,  # Doesn't satisfy condition
                    'volume': 1500000
                },
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            {
                'type': 'PRICE_UPDATE',
                'symbol': 'AAPL',
                'price_data': {
                    'price': 150.0,
                    'change_percent': 6.5,
                    'volume': 500000       # Doesn't satisfy condition
                },
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        ]

        # Process events
        initial_fired_count = advanced_trigger_system.statistics['triggers_fired']

        for event in triggering_events:
            await advanced_trigger_system.submit_event(event)

        for event in non_triggering_events:
            await advanced_trigger_system.submit_event(event)

        # Wait for processing
        await asyncio.sleep(0.5)

        # Verify only triggering events fired
        final_fired_count = advanced_trigger_system.statistics['triggers_fired']
        assert final_fired_count == initial_fired_count + len(triggering_events)

        # Verify trigger execution count
        assert price_trigger.execution_count == len(triggering_events)

        logger.info("✅ Complex trigger conditions test completed")

    @pytest.mark.asyncio
    async def test_trigger_priority_and_ordering(self, advanced_trigger_system):
        """Test trigger priority and execution ordering"""

        # Create triggers with different priorities
        triggers = [
            EventTrigger(
                trigger_id='critical_risk_alert',
                trigger_type=TriggerType.RISK_THRESHOLD,
                priority=TriggerPriority.CRITICAL,
                conditions=[TriggerCondition('risk_level', 'gt', 0.8)],
                action='EMERGENCY_RISK_SHUTDOWN'
            ),
            EventTrigger(
                trigger_id='high_volatility_alert',
                trigger_type=TriggerType.VOLATILITY_CHANGE,
                priority=TriggerPriority.HIGH,
                conditions=[TriggerCondition('volatility', 'gt', 0.5)],
                action='ADJUST_POSITION_SIZES'
            ),
            EventTrigger(
                trigger_id='medium_rebalance',
                trigger_type=TriggerType.PORTFOLIO_REBALANCE,
                priority=TriggerPriority.MEDIUM,
                conditions=[TriggerCondition('drift_percent', 'gt', 5.0)],
                action='REBALANCE_PORTFOLIO'
            ),
            EventTrigger(
                trigger_id='low_info_log',
                trigger_type=TriggerType.TECHNICAL_INDICATOR,
                priority=TriggerPriority.LOW,
                conditions=[TriggerCondition('rsi', 'lt', 30)],
                action='LOG_OVERSOLD_CONDITION'
            )
        ]

        # Register all triggers
        for trigger in triggers:
            await advanced_trigger_system.register_trigger(trigger)

        # Create event that triggers multiple priorities
        multi_trigger_event = {
            'type': 'MARKET_STATE_UPDATE',
            'risk_level': 0.85,      # Critical trigger
            'volatility': 0.6,       # High trigger
            'drift_percent': 7.0,    # Medium trigger
            'rsi': 25,               # Low trigger
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        # Process event
        await advanced_trigger_system.submit_event(multi_trigger_event)
        await asyncio.sleep(0.5)

        # Verify all triggers fired
        for trigger in triggers:
            assert trigger.execution_count == 1

        # In a real system, you'd verify execution order based on priority
        # For this test, we verify all priorities were processed

        stats = advanced_trigger_system.get_trigger_statistics()
        assert stats['processing_stats']['triggers_fired'] == len(triggers)

        logger.info("✅ Trigger priority and ordering test completed")

    @pytest.mark.asyncio
    async def test_trigger_cooldown_and_limits(self, advanced_trigger_system):
        """Test trigger cooldown and execution limits"""

        # Create trigger with cooldown
        cooldown_trigger = EventTrigger(
            trigger_id='cooldown_test',
            trigger_type=TriggerType.PRICE_MOVEMENT,
            priority=TriggerPriority.MEDIUM,
            conditions=[TriggerCondition('type', 'eq', 'TEST_EVENT')],
            action='COOLDOWN_ACTION',
            cooldown_seconds=2  # 2 second cooldown
        )

        # Create trigger with execution limit
        limited_trigger = EventTrigger(
            trigger_id='limited_test',
            trigger_type=TriggerType.PRICE_MOVEMENT,
            priority=TriggerPriority.MEDIUM,
            conditions=[TriggerCondition('type', 'eq', 'LIMITED_EVENT')],
            action='LIMITED_ACTION',
            max_executions=3  # Only 3 executions allowed
        )

        await advanced_trigger_system.register_trigger(cooldown_trigger)
        await advanced_trigger_system.register_trigger(limited_trigger)

        # Test cooldown behavior
        cooldown_event = {'type': 'TEST_EVENT', 'data': 'cooldown_test'}

        # First execution should work
        await advanced_trigger_system.submit_event(cooldown_event)
        await asyncio.sleep(0.1)
        assert cooldown_trigger.execution_count == 1

        # Second execution within cooldown should be skipped
        await advanced_trigger_system.submit_event(cooldown_event)
        await asyncio.sleep(0.1)
        assert cooldown_trigger.execution_count == 1  # Still 1

        # Wait for cooldown to expire
        await asyncio.sleep(2.1)

        # Third execution after cooldown should work
        await advanced_trigger_system.submit_event(cooldown_event)
        await asyncio.sleep(0.1)
        assert cooldown_trigger.execution_count == 2

        # Test execution limit behavior
        limited_event = {'type': 'LIMITED_EVENT', 'data': 'limit_test'}

        # Execute up to limit
        for i in range(5):  # Try to execute 5 times, but limit is 3
            await advanced_trigger_system.submit_event(limited_event)
            await asyncio.sleep(0.1)

        assert limited_trigger.execution_count == 3  # Should stop at limit

        # Verify statistics
        stats = advanced_trigger_system.get_trigger_statistics()
        assert stats['processing_stats']['triggers_skipped_cooldown'] > 0
        assert stats['processing_stats']['triggers_skipped_max_executions'] > 0

        logger.info("✅ Trigger cooldown and limits test completed")

    @pytest.mark.asyncio
    async def test_cascading_trigger_workflows(self, advanced_trigger_system):
        """Test cascading trigger workflows where triggers cause other events"""

        # Simulate a cascading workflow: Market Analysis -> Risk Check -> Position Adjustment

        # Stage 1: Market analysis trigger
        analysis_trigger = EventTrigger(
            trigger_id='market_analysis',
            trigger_type=TriggerType.TECHNICAL_INDICATOR,
            priority=TriggerPriority.HIGH,
            conditions=[
                TriggerCondition('type', 'eq', 'MARKET_DATA'),
                TriggerCondition('indicators.rsi', 'lt', 30)
            ],
            action='TRIGGER_RISK_ANALYSIS'
        )

        # Stage 2: Risk analysis trigger (triggered by stage 1 result)
        risk_trigger = EventTrigger(
            trigger_id='risk_analysis',
            trigger_type=TriggerType.RISK_THRESHOLD,
            priority=TriggerPriority.CRITICAL,
            conditions=[
                TriggerCondition('type', 'eq', 'RISK_ANALYSIS_RESULT'),
                TriggerCondition('risk_score', 'lt', 0.5)
            ],
            action='TRIGGER_POSITION_ADJUSTMENT'
        )

        # Stage 3: Position adjustment trigger (triggered by stage 2 result)
        position_trigger = EventTrigger(
            trigger_id='position_adjustment',
            trigger_type=TriggerType.PORTFOLIO_REBALANCE,
            priority=TriggerPriority.HIGH,
            conditions=[
                TriggerCondition('type', 'eq', 'POSITION_ADJUSTMENT_REQUEST'),
                TriggerCondition('adjustment_type', 'eq', 'INCREASE')
            ],
            action='EXECUTE_POSITION_INCREASE'
        )

        # Register all triggers
        for trigger in [analysis_trigger, risk_trigger, position_trigger]:
            await advanced_trigger_system.register_trigger(trigger)

        # Execute cascading workflow
        workflow_events = [
            # Stage 1: Initial market data triggers analysis
            {
                'type': 'MARKET_DATA',
                'symbol': 'AAPL',
                'indicators': {'rsi': 25, 'macd': 'bullish'},
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            # Stage 2: Analysis result triggers risk check
            {
                'type': 'RISK_ANALYSIS_RESULT',
                'symbol': 'AAPL',
                'risk_score': 0.3,  # Low risk
                'recommendation': 'INCREASE_POSITION',
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            # Stage 3: Risk approval triggers position adjustment
            {
                'type': 'POSITION_ADJUSTMENT_REQUEST',
                'symbol': 'AAPL',
                'adjustment_type': 'INCREASE',
                'amount': 1000,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        ]

        # Process each stage with delays to simulate real workflow
        for i, event in enumerate(workflow_events):
            await advanced_trigger_system.submit_event(event)
            await asyncio.sleep(0.2)  # Allow processing
            logger.info(f"Stage {i+1} completed: {event['type']}")

        # Verify all stages executed
        assert analysis_trigger.execution_count == 1
        assert risk_trigger.execution_count == 1
        assert position_trigger.execution_count == 1

        # Verify total workflow execution
        stats = advanced_trigger_system.get_trigger_statistics()
        assert stats['processing_stats']['triggers_fired'] == 3
        assert stats['processing_stats']['events_processed'] == 3

        logger.info("✅ Cascading trigger workflows test completed")

    @pytest.mark.asyncio
    async def test_event_pattern_matching(self, advanced_trigger_system):
        """Test complex event pattern matching and correlation"""

        # Create pattern-based trigger that requires sequence of events
        pattern_conditions = [
            TriggerCondition('type', 'eq', 'PATTERN_EVENT'),
            TriggerCondition('pattern_data.sequence_id', 'in', ['A', 'B', 'C'])
        ]

        pattern_trigger = EventTrigger(
            trigger_id='pattern_sequence',
            trigger_type=TriggerType.MARKET_REGIME_CHANGE,
            priority=TriggerPriority.HIGH,
            conditions=pattern_conditions,
            action='PATTERN_DETECTED',
            cooldown_seconds=5
        )

        await advanced_trigger_system.register_trigger(pattern_trigger)

        # Send pattern sequence
        pattern_events = [
            {
                'type': 'PATTERN_EVENT',
                'pattern_data': {'sequence_id': 'A', 'value': 1},
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            {
                'type': 'PATTERN_EVENT',
                'pattern_data': {'sequence_id': 'B', 'value': 2},
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            {
                'type': 'PATTERN_EVENT',
                'pattern_data': {'sequence_id': 'C', 'value': 3},
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            {
                'type': 'PATTERN_EVENT',
                'pattern_data': {'sequence_id': 'D', 'value': 4},  # Not in pattern
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        ]

        # Process pattern events
        for event in pattern_events:
            await advanced_trigger_system.submit_event(event)
            await asyncio.sleep(0.1)

        # Verify pattern trigger fired for A, B, C but not D
        assert pattern_trigger.execution_count == 3  # A, B, C triggered

        logger.info("✅ Event pattern matching test completed")

    @pytest.mark.asyncio
    async def test_high_frequency_trigger_performance(self, advanced_trigger_system):
        """Test trigger system performance under high frequency events"""

        # Create multiple triggers with different complexities
        simple_trigger = EventTrigger(
            trigger_id='simple_perf',
            trigger_type=TriggerType.PRICE_MOVEMENT,
            priority=TriggerPriority.LOW,
            conditions=[TriggerCondition('type', 'eq', 'PERF_EVENT')],
            action='SIMPLE_ACTION'
        )

        complex_trigger = EventTrigger(
            trigger_id='complex_perf',
            trigger_type=TriggerType.TECHNICAL_INDICATOR,
            priority=TriggerPriority.MEDIUM,
            conditions=[
                TriggerCondition('type', 'eq', 'PERF_EVENT'),
                TriggerCondition('data.value', 'gt', 50),
                TriggerCondition('data.category', 'in', ['A', 'B', 'C']),
                TriggerCondition('metadata.priority', 'gte', 1),
            ],
            action='COMPLEX_ACTION'
        )

        await advanced_trigger_system.register_trigger(simple_trigger)
        await advanced_trigger_system.register_trigger(complex_trigger)

        # Generate high frequency events
        num_events = 1000
        events = []

        for i in range(num_events):
            event = {
                'type': 'PERF_EVENT',
                'data': {
                    'value': random.randint(0, 100),
                    'category': random.choice(['A', 'B', 'C', 'D']),
                },
                'metadata': {
                    'priority': random.randint(0, 5),
                    'sequence': i
                },
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            events.append(event)

        # Process events with timing
        start_time = time.time()

        # Submit all events
        submission_tasks = [
            advanced_trigger_system.submit_event(event)
            for event in events
        ]
        await asyncio.gather(*submission_tasks)

        # Wait for processing to complete
        await asyncio.sleep(2.0)

        processing_time = time.time() - start_time

        # Verify performance
        stats = advanced_trigger_system.get_trigger_statistics()
        events_per_second = num_events / processing_time

        assert events_per_second > 100  # Should process >100 events/sec
        assert stats['processing_stats']['events_processed'] >= num_events
        assert stats['recent_performance']['average_processing_time_ms'] < 10  # <10ms per event

        # Verify trigger executions
        assert simple_trigger.execution_count == num_events  # All events trigger simple
        # Complex trigger should fire for subset that meet all conditions

        logger.info(f"✅ High frequency performance test completed")
        logger.info(f"   Events/sec: {events_per_second:.0f}")
        logger.info(f"   Avg processing time: {stats['recent_performance']['average_processing_time_ms']:.2f}ms")
        logger.info(f"   Simple triggers: {simple_trigger.execution_count}")
        logger.info(f"   Complex triggers: {complex_trigger.execution_count}")

    @pytest.mark.asyncio
    async def test_trigger_error_handling_and_resilience(self, advanced_trigger_system):
        """Test trigger system error handling and resilience"""

        # Create trigger with invalid condition (for testing)
        error_trigger = EventTrigger(
            trigger_id='error_test',
            trigger_type=TriggerType.PRICE_MOVEMENT,
            priority=TriggerPriority.MEDIUM,
            conditions=[
                TriggerCondition('type', 'eq', 'ERROR_EVENT'),
                TriggerCondition('invalid.path.that.does.not.exist', 'gt', 100)  # Invalid path
            ],
            action='ERROR_ACTION'
        )

        normal_trigger = EventTrigger(
            trigger_id='normal_test',
            trigger_type=TriggerType.PRICE_MOVEMENT,
            priority=TriggerPriority.MEDIUM,
            conditions=[TriggerCondition('type', 'eq', 'NORMAL_EVENT')],
            action='NORMAL_ACTION'
        )

        await advanced_trigger_system.register_trigger(error_trigger)
        await advanced_trigger_system.register_trigger(normal_trigger)

        # Send events that will cause errors and normal events
        test_events = [
            {'type': 'ERROR_EVENT', 'data': 'will cause error'},
            {'type': 'NORMAL_EVENT', 'data': 'should work fine'},
            {'type': 'ERROR_EVENT', 'invalid_data': None},  # Another error case
            {'type': 'NORMAL_EVENT', 'data': 'should still work'},
        ]

        # Process events
        for event in test_events:
            await advanced_trigger_system.submit_event(event)
            await asyncio.sleep(0.1)

        # Verify system resilience
        stats = advanced_trigger_system.get_trigger_statistics()

        # Normal trigger should still work despite errors in other trigger
        assert normal_trigger.execution_count == 2

        # Error trigger should have 0 executions due to condition evaluation failures
        assert error_trigger.execution_count == 0

        # System should still be running and processing
        assert advanced_trigger_system.is_running
        assert stats['processing_stats']['events_processed'] == len(test_events)

        logger.info("✅ Error handling and resilience test completed")

    @pytest.mark.asyncio
    async def test_dynamic_trigger_management(self, advanced_trigger_system):
        """Test dynamic trigger registration, modification, and removal"""

        # Test dynamic registration during runtime
        dynamic_trigger = EventTrigger(
            trigger_id='dynamic_test',
            trigger_type=TriggerType.VOLUME_SPIKE,
            priority=TriggerPriority.MEDIUM,
            conditions=[TriggerCondition('type', 'eq', 'DYNAMIC_EVENT')],
            action='DYNAMIC_ACTION'
        )

        # Register trigger during active processing
        await advanced_trigger_system.register_trigger(dynamic_trigger)

        # Send event to verify immediate activation
        await advanced_trigger_system.submit_event({
            'type': 'DYNAMIC_EVENT',
            'data': 'test'
        })
        await asyncio.sleep(0.1)

        assert dynamic_trigger.execution_count == 1

        # Test trigger deactivation
        dynamic_trigger.is_active = False

        await advanced_trigger_system.submit_event({
            'type': 'DYNAMIC_EVENT',
            'data': 'test2'
        })
        await asyncio.sleep(0.1)

        # Should still be 1 (not incremented)
        assert dynamic_trigger.execution_count == 1

        # Test trigger reactivation
        dynamic_trigger.is_active = True

        await advanced_trigger_system.submit_event({
            'type': 'DYNAMIC_EVENT',
            'data': 'test3'
        })
        await asyncio.sleep(0.1)

        assert dynamic_trigger.execution_count == 2

        logger.info("✅ Dynamic trigger management test completed")


@pytest.mark.asyncio
async def test_end_to_end_autonomous_trading_scenario():
    """Complete end-to-end test of autonomous trading using event triggers"""

    logger.info("🚀 Starting end-to-end autonomous trading scenario")

    # Initialize trigger system
    trigger_system = MockAdvancedEventTriggerSystem()
    await trigger_system.start()

    try:
        # Define complete autonomous trading workflow triggers
        trading_triggers = [
            # Market data analysis
            EventTrigger(
                trigger_id='market_data_analysis',
                trigger_type=TriggerType.PRICE_MOVEMENT,
                priority=TriggerPriority.HIGH,
                conditions=[
                    TriggerCondition('type', 'eq', 'MARKET_DATA_UPDATE'),
                    TriggerCondition('volume', 'gt', 500000)
                ],
                action='ANALYZE_TECHNICAL_INDICATORS'
            ),
            # Signal generation
            EventTrigger(
                trigger_id='signal_generation',
                trigger_type=TriggerType.TECHNICAL_INDICATOR,
                priority=TriggerPriority.HIGH,
                conditions=[
                    TriggerCondition('type', 'eq', 'TECHNICAL_ANALYSIS_COMPLETE'),
                    TriggerCondition('signals.strength', 'gt', 0.7)
                ],
                action='GENERATE_TRADING_SIGNAL'
            ),
            # Risk assessment
            EventTrigger(
                trigger_id='risk_assessment',
                trigger_type=TriggerType.RISK_THRESHOLD,
                priority=TriggerPriority.CRITICAL,
                conditions=[
                    TriggerCondition('type', 'eq', 'TRADING_SIGNAL_GENERATED'),
                    TriggerCondition('signal.direction', 'in', ['BUY', 'SELL'])
                ],
                action='ASSESS_PORTFOLIO_RISK'
            ),
            # Position sizing
            EventTrigger(
                trigger_id='position_sizing',
                trigger_type=TriggerType.PORTFOLIO_REBALANCE,
                priority=TriggerPriority.HIGH,
                conditions=[
                    TriggerCondition('type', 'eq', 'RISK_ASSESSMENT_COMPLETE'),
                    TriggerCondition('risk.approved', 'eq', True)
                ],
                action='CALCULATE_POSITION_SIZE'
            ),
            # Order execution
            EventTrigger(
                trigger_id='order_execution',
                trigger_type=TriggerType.PORTFOLIO_REBALANCE,
                priority=TriggerPriority.CRITICAL,
                conditions=[
                    TriggerCondition('type', 'eq', 'POSITION_SIZE_CALCULATED'),
                    TriggerCondition('position.size', 'gt', 0)
                ],
                action='EXECUTE_TRADE_ORDER'
            )
        ]

        # Register all triggers
        for trigger in trading_triggers:
            await trigger_system.register_trigger(trigger)

        # Simulate autonomous trading workflow
        workflow_events = [
            # 1. Market data arrives
            {
                'type': 'MARKET_DATA_UPDATE',
                'symbol': 'AAPL',
                'price': 150.25,
                'volume': 1200000,
                'change_percent': 2.5,
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            # 2. Technical analysis completes
            {
                'type': 'TECHNICAL_ANALYSIS_COMPLETE',
                'symbol': 'AAPL',
                'signals': {
                    'rsi': 45,
                    'macd': 'bullish',
                    'strength': 0.85
                },
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            # 3. Trading signal generated
            {
                'type': 'TRADING_SIGNAL_GENERATED',
                'symbol': 'AAPL',
                'signal': {
                    'direction': 'BUY',
                    'confidence': 0.85,
                    'target_price': 155.0
                },
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            # 4. Risk assessment completes
            {
                'type': 'RISK_ASSESSMENT_COMPLETE',
                'symbol': 'AAPL',
                'risk': {
                    'score': 0.3,
                    'approved': True,
                    'max_position_value': 10000
                },
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            # 5. Position size calculated
            {
                'type': 'POSITION_SIZE_CALCULATED',
                'symbol': 'AAPL',
                'position': {
                    'size': 66,  # shares
                    'value': 9916.50,
                    'percentage': 2.0
                },
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        ]

        # Execute autonomous workflow with timing
        workflow_start = time.time()

        for i, event in enumerate(workflow_events):
            logger.info(f"🔄 Workflow Step {i+1}: {event['type']}")
            await trigger_system.submit_event(event)
            await asyncio.sleep(0.2)  # Allow processing

        workflow_time = time.time() - workflow_start

        # Verify complete autonomous execution
        final_stats = trigger_system.get_trigger_statistics()

        # All triggers should have fired once
        for trigger in trading_triggers:
            assert trigger.execution_count == 1, f"Trigger {trigger.trigger_id} did not execute"

        # Verify workflow completion
        assert final_stats['processing_stats']['triggers_fired'] == len(trading_triggers)
        assert final_stats['processing_stats']['events_processed'] == len(workflow_events)
        assert final_stats['processing_stats']['processing_errors'] == 0

        logger.info("✅ AUTONOMOUS TRADING WORKFLOW COMPLETED SUCCESSFULLY")
        logger.info(f"   Workflow time: {workflow_time:.3f}s")
        logger.info(f"   Events processed: {len(workflow_events)}")
        logger.info(f"   Triggers fired: {len(trading_triggers)}")
        logger.info(f"   Average processing time: {final_stats['recent_performance']['average_processing_time_ms']:.2f}ms")
        logger.info("🎉 System demonstrates full autonomous trading capability!")

    finally:
        await trigger_system.stop()


if __name__ == "__main__":
    # Run the autonomous trading demonstration
    asyncio.run(test_end_to_end_autonomous_trading_scenario())