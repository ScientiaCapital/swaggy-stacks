"""
Time-based and scheduled triggers for trading events
"""

import asyncio
from datetime import datetime, time, timedelta
from typing import Dict, List, Any, Optional, Callable
import pytz
from dataclasses import dataclass
import structlog

from app.events.trigger_engine import (
    TriggerCondition, TriggerType, TriggerPriority, TriggerEvent, get_trigger_engine
)
from app.core.config import settings

logger = structlog.get_logger(__name__)


@dataclass
class ScheduledEvent:
    """Scheduled event configuration"""
    event_id: str
    name: str
    trigger_time: time
    days_of_week: List[int]  # 0=Monday, 6=Sunday
    timezone: str
    target_agents: List[str]
    event_data: Dict[str, Any]
    is_active: bool = True
    last_triggered: Optional[datetime] = None


class ScheduleTriggerManager:
    """
    Manages time-based and scheduled triggers:
    - Market open/close events
    - Pre-market preparation
    - After-hours analysis
    - Weekly/monthly events
    - Custom scheduled triggers
    """

    def __init__(self):
        self.trigger_engine = None
        self.scheduled_events: Dict[str, ScheduledEvent] = {}
        self.scheduler_task: Optional[asyncio.Task] = None
        self.is_running = False

        # Market hours (Eastern Time)
        self.market_timezone = pytz.timezone('America/New_York')
        self.market_open_time = time(9, 30)  # 9:30 AM ET
        self.market_close_time = time(16, 0)  # 4:00 PM ET
        self.pre_market_start = time(4, 0)   # 4:00 AM ET
        self.after_hours_end = time(20, 0)   # 8:00 PM ET

    async def initialize(self):
        """Initialize schedule trigger manager"""
        self.trigger_engine = await get_trigger_engine()
        await self._register_event_handlers()
        await self._create_default_scheduled_events()
        await self.start_scheduler()
        logger.info("ScheduleTriggerManager initialized")

    async def _register_event_handlers(self):
        """Register handlers for time-based events"""
        await self.trigger_engine.register_event_handler(
            TriggerType.TIME_BASED,
            self._handle_time_based_event
        )

    async def _create_default_scheduled_events(self):
        """Create default scheduled events for trading"""

        # Pre-market preparation (8:00 AM ET, Monday-Friday)
        await self.add_scheduled_event(ScheduledEvent(
            event_id="pre_market_prep",
            name="Pre-Market Preparation",
            trigger_time=time(8, 0),
            days_of_week=[0, 1, 2, 3, 4],  # Monday-Friday
            timezone="America/New_York",
            target_agents=["analyst", "risk", "strategist"],
            event_data={
                "event_type": "pre_market_prep",
                "description": "Prepare for market open",
                "tasks": [
                    "validate_connections",
                    "check_overnight_news",
                    "update_watchlists",
                    "review_positions"
                ]
            }
        ))

        # Market open (9:30 AM ET, Monday-Friday)
        await self.add_scheduled_event(ScheduledEvent(
            event_id="market_open",
            name="Market Open",
            trigger_time=self.market_open_time,
            days_of_week=[0, 1, 2, 3, 4],
            timezone="America/New_York",
            target_agents=["analyst", "risk", "strategist", "chat"],
            event_data={
                "event_type": "market_open",
                "description": "Market is now open",
                "priority": "high",
                "enable_trading": True
            }
        ))

        # Mid-day review (12:00 PM ET, Monday-Friday)
        await self.add_scheduled_event(ScheduledEvent(
            event_id="midday_review",
            name="Mid-Day Review",
            trigger_time=time(12, 0),
            days_of_week=[0, 1, 2, 3, 4],
            timezone="America/New_York",
            target_agents=["analyst", "risk"],
            event_data={
                "event_type": "midday_review",
                "description": "Mid-day portfolio review",
                "tasks": [
                    "review_performance",
                    "check_risk_exposure",
                    "adjust_positions"
                ]
            }
        ))

        # Market close (4:00 PM ET, Monday-Friday)
        await self.add_scheduled_event(ScheduledEvent(
            event_id="market_close",
            name="Market Close",
            trigger_time=self.market_close_time,
            days_of_week=[0, 1, 2, 3, 4],
            timezone="America/New_York",
            target_agents=["analyst", "risk", "strategist"],
            event_data={
                "event_type": "market_close",
                "description": "Market is now closed",
                "priority": "high",
                "disable_trading": True,
                "tasks": [
                    "calculate_daily_pnl",
                    "update_positions",
                    "generate_daily_report"
                ]
            }
        ))

        # After-hours analysis (5:00 PM ET, Monday-Friday)
        await self.add_scheduled_event(ScheduledEvent(
            event_id="after_hours_analysis",
            name="After-Hours Analysis",
            trigger_time=time(17, 0),
            days_of_week=[0, 1, 2, 3, 4],
            timezone="America/New_York",
            target_agents=["analyst", "reasoning"],
            event_data={
                "event_type": "after_hours_analysis",
                "description": "Analyze day's performance and plan for tomorrow",
                "tasks": [
                    "analyze_trades",
                    "review_strategy_performance",
                    "plan_tomorrow_trades",
                    "update_learning_models"
                ]
            }
        ))

        # Weekend portfolio review (Saturday 10:00 AM ET)
        await self.add_scheduled_event(ScheduledEvent(
            event_id="weekend_review",
            name="Weekend Portfolio Review",
            trigger_time=time(10, 0),
            days_of_week=[5],  # Saturday
            timezone="America/New_York",
            target_agents=["analyst", "risk", "reasoning"],
            event_data={
                "event_type": "weekend_review",
                "description": "Weekly portfolio and strategy review",
                "tasks": [
                    "weekly_performance_analysis",
                    "strategy_optimization",
                    "risk_assessment",
                    "model_retraining"
                ]
            }
        ))

        # 24/7 Crypto monitoring checks (every 4 hours)
        for hour in [0, 4, 8, 12, 16, 20]:
            await self.add_scheduled_event(ScheduledEvent(
                event_id=f"crypto_check_{hour}",
                name=f"Crypto Check {hour}:00",
                trigger_time=time(hour, 0),
                days_of_week=[0, 1, 2, 3, 4, 5, 6],  # Every day
                timezone="UTC",
                target_agents=["analyst", "strategist"],
                event_data={
                    "event_type": "crypto_monitoring",
                    "description": f"24/7 crypto market check at {hour}:00 UTC",
                    "crypto_focus": True,
                    "tasks": [
                        "check_crypto_opportunities",
                        "monitor_crypto_positions",
                        "analyze_crypto_trends"
                    ]
                }
            ))

        logger.info("Default scheduled events created")

    async def start_scheduler(self):
        """Start the scheduler task"""
        if self.is_running:
            return

        self.is_running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Schedule trigger manager started")

    async def stop_scheduler(self):
        """Stop the scheduler task"""
        self.is_running = False

        if self.scheduler_task and not self.scheduler_task.done():
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass

        logger.info("Schedule trigger manager stopped")

    async def _scheduler_loop(self):
        """Main scheduler loop"""
        logger.info("Scheduler loop started")

        while self.is_running:
            try:
                current_time = datetime.now()

                # Check all scheduled events
                for event in self.scheduled_events.values():
                    if await self._should_trigger_event(event, current_time):
                        await self._trigger_scheduled_event(event)

                # Sleep for 30 seconds before next check
                await asyncio.sleep(30)

            except Exception as e:
                logger.error("Error in scheduler loop", error=str(e))
                await asyncio.sleep(60)

    async def _should_trigger_event(self, event: ScheduledEvent, current_time: datetime) -> bool:
        """Check if scheduled event should be triggered"""
        if not event.is_active:
            return False

        # Convert to event timezone
        event_tz = pytz.timezone(event.timezone)
        current_time_tz = current_time.astimezone(event_tz)

        # Check day of week
        if current_time_tz.weekday() not in event.days_of_week:
            return False

        # Check if we're within the trigger time window (30 seconds)
        event_time_today = event_tz.localize(
            datetime.combine(current_time_tz.date(), event.trigger_time)
        )

        time_diff = (current_time_tz - event_time_today).total_seconds()

        # Trigger if within 30 seconds of scheduled time
        if 0 <= time_diff <= 30:
            # Check if already triggered today
            if event.last_triggered:
                last_trigger_date = event.last_triggered.astimezone(event_tz).date()
                if last_trigger_date == current_time_tz.date():
                    return False

            return True

        return False

    async def _trigger_scheduled_event(self, event: ScheduledEvent):
        """Trigger a scheduled event"""
        try:
            logger.info("Triggering scheduled event",
                       event_id=event.event_id,
                       name=event.name,
                       agents=event.target_agents)

            # Create time-based trigger event
            trigger = TriggerCondition(
                trigger_id=f"schedule_{event.event_id}_{int(datetime.utcnow().timestamp())}",
                trigger_type=TriggerType.TIME_BASED,
                symbol="SCHEDULE",
                condition="scheduled_time",
                threshold=0,
                priority=TriggerPriority.HIGH,
                target_agents=event.target_agents,
                metadata=event.event_data
            )

            # Process the event
            trigger_event = TriggerEvent(
                event_id=trigger.trigger_id,
                trigger_condition=trigger,
                current_value=0,
                triggered_at=datetime.utcnow(),
                market_data=event.event_data
            )

            await self._handle_time_based_event(trigger_event)

            # Update last triggered time
            event.last_triggered = datetime.utcnow()

        except Exception as e:
            logger.error("Failed to trigger scheduled event",
                        event_id=event.event_id, error=str(e))

    async def _handle_time_based_event(self, event: TriggerEvent):
        """Handle time-based trigger events"""
        try:
            event_type = event.market_data.get('event_type', 'unknown')

            logger.info("Time-based event triggered",
                       event_type=event_type,
                       agents=event.trigger_condition.target_agents)

            # Wake up target agents
            await self._wake_agents(
                event.trigger_condition.target_agents,
                event.market_data
            )

        except Exception as e:
            logger.error("Failed to handle time-based event",
                        event_id=event.event_id, error=str(e))

    async def _wake_agents(self, agent_names: List[str], event_data: Dict[str, Any]):
        """Wake up specified agents with event data"""
        try:
            # Import here to avoid circular imports
            from app.ai.coordination_hub import get_coordination_hub

            hub = await get_coordination_hub()
            await hub.broadcast_scheduled_event(agent_names, event_data)

            logger.info("Agents awakened for scheduled event",
                       agents=agent_names,
                       event_type=event_data.get('event_type'))

        except Exception as e:
            logger.error("Failed to wake agents for scheduled event",
                        agents=agent_names, error=str(e))

    # Public API
    async def add_scheduled_event(self, event: ScheduledEvent) -> str:
        """Add a new scheduled event"""
        self.scheduled_events[event.event_id] = event
        logger.info("Scheduled event added",
                   event_id=event.event_id,
                   name=event.name,
                   trigger_time=event.trigger_time.strftime("%H:%M"))
        return event.event_id

    async def remove_scheduled_event(self, event_id: str):
        """Remove a scheduled event"""
        if event_id in self.scheduled_events:
            del self.scheduled_events[event_id]
            logger.info("Scheduled event removed", event_id=event_id)

    async def get_scheduled_events(self) -> List[Dict[str, Any]]:
        """Get list of all scheduled events"""
        events = []

        for event in self.scheduled_events.values():
            events.append({
                'event_id': event.event_id,
                'name': event.name,
                'trigger_time': event.trigger_time.strftime("%H:%M"),
                'days_of_week': event.days_of_week,
                'timezone': event.timezone,
                'target_agents': event.target_agents,
                'is_active': event.is_active,
                'last_triggered': event.last_triggered.isoformat() if event.last_triggered else None,
                'event_type': event.event_data.get('event_type', 'unknown')
            })

        return events

    async def get_next_scheduled_events(self, hours_ahead: int = 24) -> List[Dict[str, Any]]:
        """Get upcoming scheduled events within specified hours"""
        current_time = datetime.now()
        upcoming_events = []

        for event in self.scheduled_events.values():
            if not event.is_active:
                continue

            # Calculate next occurrence
            next_occurrence = await self._calculate_next_occurrence(event, current_time)

            if next_occurrence:
                time_until = (next_occurrence - current_time).total_seconds() / 3600

                if 0 <= time_until <= hours_ahead:
                    upcoming_events.append({
                        'event_id': event.event_id,
                        'name': event.name,
                        'next_occurrence': next_occurrence.isoformat(),
                        'hours_until': round(time_until, 2),
                        'target_agents': event.target_agents,
                        'event_type': event.event_data.get('event_type', 'unknown')
                    })

        # Sort by time until occurrence
        upcoming_events.sort(key=lambda x: x['hours_until'])
        return upcoming_events

    async def _calculate_next_occurrence(self, event: ScheduledEvent, from_time: datetime) -> Optional[datetime]:
        """Calculate next occurrence of a scheduled event"""
        try:
            event_tz = pytz.timezone(event.timezone)
            from_time_tz = from_time.astimezone(event_tz)

            # Check today first
            today_occurrence = event_tz.localize(
                datetime.combine(from_time_tz.date(), event.trigger_time)
            )

            if (today_occurrence > from_time_tz and
                from_time_tz.weekday() in event.days_of_week):
                return today_occurrence

            # Check next 7 days
            for days_ahead in range(1, 8):
                future_date = from_time_tz.date() + timedelta(days=days_ahead)
                future_occurrence = event_tz.localize(
                    datetime.combine(future_date, event.trigger_time)
                )

                if future_occurrence.weekday() in event.days_of_week:
                    return future_occurrence

            return None

        except Exception as e:
            logger.error("Failed to calculate next occurrence",
                        event_id=event.event_id, error=str(e))
            return None

    async def is_market_hours(self, timezone: str = "America/New_York") -> bool:
        """Check if current time is within market hours"""
        tz = pytz.timezone(timezone)
        current_time = datetime.now(tz).time()

        return (self.market_open_time <= current_time <= self.market_close_time and
                datetime.now(tz).weekday() < 5)  # Monday-Friday

    async def is_pre_market(self, timezone: str = "America/New_York") -> bool:
        """Check if current time is pre-market"""
        tz = pytz.timezone(timezone)
        current_time = datetime.now(tz).time()

        return (self.pre_market_start <= current_time < self.market_open_time and
                datetime.now(tz).weekday() < 5)  # Monday-Friday

    async def is_after_hours(self, timezone: str = "America/New_York") -> bool:
        """Check if current time is after-hours"""
        tz = pytz.timezone(timezone)
        current_time = datetime.now(tz).time()

        return (self.market_close_time < current_time <= self.after_hours_end and
                datetime.now(tz).weekday() < 5)  # Monday-Friday


# Global schedule trigger manager instance
schedule_trigger_manager: Optional[ScheduleTriggerManager] = None


async def get_schedule_trigger_manager() -> ScheduleTriggerManager:
    """Get or create global schedule trigger manager instance"""
    global schedule_trigger_manager

    if schedule_trigger_manager is None:
        schedule_trigger_manager = ScheduleTriggerManager()
        await schedule_trigger_manager.initialize()

    return schedule_trigger_manager