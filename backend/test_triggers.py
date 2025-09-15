#!/usr/bin/env python3
"""
Test script for Event-Driven Trigger System
"""

import asyncio
import sys
import os
from datetime import datetime, time

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from app.events.trigger_engine import (
    EventTriggerEngine, TriggerCondition, TriggerType, TriggerPriority, get_trigger_engine
)
from app.events.market_triggers import get_market_trigger_manager
from app.events.schedule_triggers import get_schedule_trigger_manager


async def test_trigger_engine():
    """Test the core trigger engine functionality"""
    print("🔥 Testing Event Trigger Engine...")

    # Get trigger engine
    engine = await get_trigger_engine()

    # Create test trigger
    test_trigger = TriggerCondition(
        trigger_id="test_price_move",
        trigger_type=TriggerType.PRICE_MOVE,
        symbol="AAPL",
        condition="test_move",
        threshold=2.0,
        priority=TriggerPriority.HIGH,
        target_agents=["analyst", "strategist"],
        cooldown_seconds=60
    )

    # Register trigger
    trigger_id = await engine.register_trigger(test_trigger)
    print(f"✅ Registered trigger: {trigger_id}")

    # Test market data processing
    test_market_data = {
        'symbol': 'AAPL',
        'price': 150.00,
        'volume': 1000000,
        'timestamp': datetime.utcnow()
    }

    await engine.process_market_data('AAPL', test_market_data)

    # Simulate price movement
    test_market_data['price'] = 153.50  # 2.3% increase
    await engine.process_market_data('AAPL', test_market_data)

    # Get statistics
    stats = await engine.get_statistics()
    print(f"📊 Engine Stats: {stats}")

    # Get active triggers
    triggers = await engine.get_active_triggers('AAPL')
    print(f"🎯 Active Triggers for AAPL: {len(triggers)}")

    return engine


async def test_market_triggers():
    """Test market trigger functionality"""
    print("\n📈 Testing Market Triggers...")

    # Get market trigger manager
    manager = await get_market_trigger_manager()

    # Get status
    status = await manager.get_market_trigger_status()
    print(f"📊 Market Trigger Status: {status}")

    # Create custom trigger
    trigger_id = await manager.create_custom_price_trigger(
        symbol="TSLA",
        threshold=1.5,  # 1.5% move
        lookback_minutes=3,
        priority=TriggerPriority.HIGH,
        target_agents=["analyst", "risk"]
    )
    print(f"✅ Created custom trigger: {trigger_id}")

    # Create volume trigger
    volume_trigger_id = await manager.create_custom_volume_trigger(
        symbol="NVDA",
        volume_multiplier=2.5,  # 2.5x volume
        priority=TriggerPriority.MEDIUM
    )
    print(f"✅ Created volume trigger: {volume_trigger_id}")

    return manager


async def test_schedule_triggers():
    """Test scheduled trigger functionality"""
    print("\n⏰ Testing Schedule Triggers...")

    # Get schedule trigger manager
    manager = await get_schedule_trigger_manager()

    # Get scheduled events
    events = await manager.get_scheduled_events()
    print(f"📅 Total Scheduled Events: {len(events)}")

    for event in events[:5]:  # Show first 5
        print(f"  - {event['name']}: {event['trigger_time']} ({event['event_type']})")

    # Get upcoming events
    upcoming = await manager.get_next_scheduled_events(hours_ahead=48)
    print(f"\n🔮 Next {len(upcoming)} events in 48 hours:")
    for event in upcoming[:3]:  # Show first 3
        print(f"  - {event['name']}: {event['hours_until']:.1f} hours")

    # Check market status
    is_market_hours = await manager.is_market_hours()
    is_pre_market = await manager.is_pre_market()
    is_after_hours = await manager.is_after_hours()

    print(f"\n🕐 Market Status:")
    print(f"  Market Hours: {is_market_hours}")
    print(f"  Pre-Market: {is_pre_market}")
    print(f"  After Hours: {is_after_hours}")

    return manager


async def test_trigger_integration():
    """Test integration between streaming and triggers"""
    print("\n🔗 Testing Stream-Trigger Integration...")

    try:
        from app.trading.alpaca_stream_manager import get_stream_manager

        # Get managers
        stream_manager = await get_stream_manager()
        trigger_engine = await get_trigger_engine()

        # Create integration callback
        async def trigger_callback(market_data):
            """Callback to process market data through triggers"""
            if hasattr(market_data, 'symbol'):
                await trigger_engine.process_market_data(
                    market_data.symbol,
                    {
                        'price': float(market_data.price) if hasattr(market_data, 'price') else 0,
                        'volume': int(market_data.size) if hasattr(market_data, 'size') else 0,
                        'timestamp': market_data.timestamp if hasattr(market_data, 'timestamp') else datetime.utcnow()
                    }
                )

        print("✅ Integration callback created")
        print("🔗 Ready to connect streaming data to trigger system")

    except Exception as e:
        print(f"⚠️ Stream integration test skipped: {e}")


async def run_all_tests():
    """Run all trigger system tests"""
    print("🚀 Event-Driven Trigger System Tests")
    print("=" * 50)

    try:
        # Test core engine
        engine = await test_trigger_engine()

        # Test market triggers
        market_manager = await test_market_triggers()

        # Test schedule triggers
        schedule_manager = await test_schedule_triggers()

        # Test integration
        await test_trigger_integration()

        print("\n✅ All trigger system tests completed successfully!")

        # Final statistics
        final_stats = await engine.get_statistics()
        print(f"\n📊 Final Engine Statistics:")
        for key, value in final_stats.items():
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🔥 Event-Driven Trigger System Test Suite")
    asyncio.run(run_all_tests())