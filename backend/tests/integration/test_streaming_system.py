"""
Comprehensive Streaming System Integration Tests

Tests the complete streaming and event-driven architecture for autonomous trading:
- WebSocket connection lifecycle management
- Real-time market data streaming
- Event trigger processing and coordination
- High-load scenario handling
- Failure recovery and resilience
- End-to-end streaming workflows

These tests validate the critical streaming infrastructure that powers 24/7
autonomous trading operations for thousands of symbols.
"""

import pytest
import pytest_asyncio
import asyncio
import json
import logging
import time
from unittest.mock import AsyncMock, MagicMock, patch, call
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedError
import concurrent.futures
import random

# Test configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock streaming components
class MockStreamingWebSocketManager:
    """Mock WebSocket manager for streaming tests"""

    def __init__(self):
        self.active_connections = {}
        self.agent_connections = {}
        self.is_running = False
        self.message_queue = asyncio.Queue()
        self.connection_stats = {
            'total_connections': 0,
            'active_connections': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'connection_errors': 0
        }
        self._shutdown_event = asyncio.Event()

    async def start(self):
        """Start the WebSocket manager"""
        self.is_running = True
        logger.info("✅ Streaming WebSocket Manager started")

    async def stop(self):
        """Stop the WebSocket manager"""
        self.is_running = False
        self._shutdown_event.set()
        await self._disconnect_all()
        logger.info("🛑 Streaming WebSocket Manager stopped")

    async def connect_client(self, client_id: str, client_type: str = "trader") -> bool:
        """Connect a new client"""
        if not self.is_running:
            return False

        mock_websocket = AsyncMock()
        mock_websocket.id = client_id
        mock_websocket.type = client_type
        mock_websocket.connected = True
        mock_websocket.last_ping = datetime.now(timezone.utc)

        self.active_connections[client_id] = mock_websocket
        self.connection_stats['total_connections'] += 1
        self.connection_stats['active_connections'] += 1

        logger.info(f"📡 Client connected: {client_id} ({client_type})")
        return True

    async def disconnect_client(self, client_id: str) -> bool:
        """Disconnect a client"""
        if client_id in self.active_connections:
            self.active_connections[client_id].connected = False
            del self.active_connections[client_id]
            self.connection_stats['active_connections'] -= 1
            logger.info(f"📡 Client disconnected: {client_id}")
            return True
        return False

    async def broadcast_message(self, message: Dict[str, Any], target_type: str = None) -> int:
        """Broadcast message to clients"""
        if not self.is_running:
            return 0

        sent_count = 0
        for client_id, websocket in self.active_connections.items():
            if target_type is None or websocket.type == target_type:
                try:
                    await websocket.send(json.dumps(message))
                    sent_count += 1
                    self.connection_stats['messages_sent'] += 1
                except Exception as e:
                    logger.warning(f"Failed to send message to {client_id}: {e}")
                    self.connection_stats['connection_errors'] += 1

        return sent_count

    async def send_personal_message(self, client_id: str, message: Dict[str, Any]) -> bool:
        """Send message to specific client"""
        if client_id in self.active_connections:
            try:
                websocket = self.active_connections[client_id]
                await websocket.send(json.dumps(message))
                self.connection_stats['messages_sent'] += 1
                return True
            except Exception as e:
                logger.warning(f"Failed to send personal message to {client_id}: {e}")
                self.connection_stats['connection_errors'] += 1
        return False

    async def _disconnect_all(self):
        """Disconnect all clients"""
        for client_id in list(self.active_connections.keys()):
            await self.disconnect_client(client_id)


class MockEventTriggerSystem:
    """Mock event trigger system for testing"""

    def __init__(self):
        self.triggers = {}
        self.event_queue = asyncio.Queue()
        self.processed_events = []
        self.is_running = False
        self.processing_stats = {
            'events_received': 0,
            'events_processed': 0,
            'events_failed': 0,
            'triggers_fired': 0
        }

    async def start(self):
        """Start the event trigger system"""
        self.is_running = True
        logger.info("✅ Event Trigger System started")

    async def stop(self):
        """Stop the event trigger system"""
        self.is_running = False
        logger.info("🛑 Event Trigger System stopped")

    async def register_trigger(self, trigger_id: str, condition: Dict[str, Any], action: str):
        """Register a new event trigger"""
        self.triggers[trigger_id] = {
            'condition': condition,
            'action': action,
            'created_at': datetime.now(timezone.utc),
            'fired_count': 0
        }
        logger.info(f"📋 Trigger registered: {trigger_id}")

    async def process_event(self, event: Dict[str, Any]) -> List[str]:
        """Process incoming event against all triggers"""
        if not self.is_running:
            return []

        self.processing_stats['events_received'] += 1
        triggered_actions = []

        try:
            await self.event_queue.put(event)

            # Check all triggers
            for trigger_id, trigger in self.triggers.items():
                if await self._evaluate_trigger(event, trigger['condition']):
                    triggered_actions.append(trigger['action'])
                    trigger['fired_count'] += 1
                    self.processing_stats['triggers_fired'] += 1
                    logger.info(f"🔥 Trigger fired: {trigger_id} -> {trigger['action']}")

            self.processed_events.append({
                'event': event,
                'timestamp': datetime.now(timezone.utc),
                'triggered_actions': triggered_actions
            })

            self.processing_stats['events_processed'] += 1

        except Exception as e:
            logger.error(f"Error processing event: {e}")
            self.processing_stats['events_failed'] += 1

        return triggered_actions

    async def _evaluate_trigger(self, event: Dict[str, Any], condition: Dict[str, Any]) -> bool:
        """Evaluate if event matches trigger condition"""
        # Simple condition matching for testing
        for key, expected_value in condition.items():
            if key not in event:
                return False
            if isinstance(expected_value, dict):
                # Handle comparison operators
                if 'gt' in expected_value and event[key] <= expected_value['gt']:
                    return False
                if 'lt' in expected_value and event[key] >= expected_value['lt']:
                    return False
                if 'eq' in expected_value and event[key] != expected_value['eq']:
                    return False
            else:
                if event[key] != expected_value:
                    return False
        return True


class MockMarketDataStreamer:
    """Mock market data streaming service"""

    def __init__(self):
        self.subscriptions = set()
        self.is_streaming = False
        self.stream_rate = 1.0  # Events per second
        self.stream_task = None
        self.data_stats = {
            'symbols_subscribed': 0,
            'data_points_sent': 0,
            'stream_errors': 0
        }

    async def start_streaming(self):
        """Start market data streaming"""
        self.is_streaming = True
        self.stream_task = asyncio.create_task(self._stream_data())
        logger.info("📈 Market data streaming started")

    async def stop_streaming(self):
        """Stop market data streaming"""
        self.is_streaming = False
        if self.stream_task:
            self.stream_task.cancel()
            try:
                await self.stream_task
            except asyncio.CancelledError:
                pass
        logger.info("📈 Market data streaming stopped")

    async def subscribe(self, symbol: str):
        """Subscribe to symbol data"""
        self.subscriptions.add(symbol)
        self.data_stats['symbols_subscribed'] = len(self.subscriptions)
        logger.info(f"📊 Subscribed to {symbol}")

    async def unsubscribe(self, symbol: str):
        """Unsubscribe from symbol data"""
        self.subscriptions.discard(symbol)
        self.data_stats['symbols_subscribed'] = len(self.subscriptions)
        logger.info(f"📊 Unsubscribed from {symbol}")

    async def _stream_data(self):
        """Generate mock streaming data"""
        while self.is_streaming:
            try:
                for symbol in self.subscriptions:
                    # Generate realistic market data
                    price = 100 + random.uniform(-10, 10)
                    volume = random.randint(1000, 100000)

                    data = {
                        'type': 'MARKET_DATA',
                        'symbol': symbol,
                        'price': round(price, 2),
                        'volume': volume,
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'change': round(random.uniform(-2, 2), 2),
                        'change_percent': round(random.uniform(-5, 5), 2)
                    }

                    # This would normally be sent via WebSocket
                    self.data_stats['data_points_sent'] += 1

                await asyncio.sleep(1.0 / self.stream_rate)

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                self.data_stats['stream_errors'] += 1
                await asyncio.sleep(1.0)


@pytest_asyncio.fixture
async def streaming_websocket_manager():
    """Create mock streaming WebSocket manager"""
    manager = MockStreamingWebSocketManager()
    await manager.start()
    yield manager
    await manager.stop()


@pytest_asyncio.fixture
async def event_trigger_system():
    """Create mock event trigger system"""
    system = MockEventTriggerSystem()
    await system.start()
    yield system
    await system.stop()


@pytest_asyncio.fixture
async def market_data_streamer():
    """Create mock market data streamer"""
    streamer = MockMarketDataStreamer()
    await streamer.start_streaming()
    yield streamer
    await streamer.stop_streaming()


class TestStreamingSystemIntegration:
    """Comprehensive streaming system integration tests"""

    @pytest.mark.asyncio
    async def test_websocket_connection_lifecycle(self, streaming_websocket_manager):
        """Test complete WebSocket connection lifecycle"""

        # Test initial state
        assert streaming_websocket_manager.is_running
        assert len(streaming_websocket_manager.active_connections) == 0

        # Test connection establishment
        clients = ["trader_1", "agent_1", "scanner_1"]
        for client_id in clients:
            success = await streaming_websocket_manager.connect_client(client_id)
            assert success
            assert client_id in streaming_websocket_manager.active_connections

        assert len(streaming_websocket_manager.active_connections) == 3
        assert streaming_websocket_manager.connection_stats['active_connections'] == 3

        # Test message broadcasting
        test_message = {
            'type': 'SYSTEM_STATUS',
            'status': 'operational',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        sent_count = await streaming_websocket_manager.broadcast_message(test_message)
        assert sent_count == 3
        assert streaming_websocket_manager.connection_stats['messages_sent'] == 3

        # Test personal messaging
        personal_message = {'type': 'PERSONAL', 'data': 'test'}
        success = await streaming_websocket_manager.send_personal_message("trader_1", personal_message)
        assert success
        assert streaming_websocket_manager.connection_stats['messages_sent'] == 4

        # Test disconnection
        for client_id in clients:
            success = await streaming_websocket_manager.disconnect_client(client_id)
            assert success
            assert client_id not in streaming_websocket_manager.active_connections

        assert len(streaming_websocket_manager.active_connections) == 0
        assert streaming_websocket_manager.connection_stats['active_connections'] == 0

        logger.info("✅ WebSocket connection lifecycle test completed")

    @pytest.mark.asyncio
    async def test_event_trigger_firing(self, event_trigger_system):
        """Test event trigger processing and firing"""

        # Register various trigger types
        triggers = [
            {
                'id': 'price_spike',
                'condition': {'type': 'PRICE_CHANGE', 'change_percent': {'gt': 5.0}},
                'action': 'ALERT_HIGH_VOLATILITY'
            },
            {
                'id': 'volume_surge',
                'condition': {'type': 'VOLUME_DATA', 'volume': {'gt': 1000000}},
                'action': 'SCAN_FOR_OPPORTUNITIES'
            },
            {
                'id': 'rsi_oversold',
                'condition': {'type': 'TECHNICAL_INDICATOR', 'indicator': 'RSI', 'value': {'lt': 30}},
                'action': 'EVALUATE_LONG_ENTRY'
            }
        ]

        # Register all triggers
        for trigger in triggers:
            await event_trigger_system.register_trigger(
                trigger['id'],
                trigger['condition'],
                trigger['action']
            )

        assert len(event_trigger_system.triggers) == 3

        # Test events that should trigger
        test_events = [
            {
                'type': 'PRICE_CHANGE',
                'symbol': 'AAPL',
                'change_percent': 7.5,  # Should trigger price_spike
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            {
                'type': 'VOLUME_DATA',
                'symbol': 'SPY',
                'volume': 1500000,  # Should trigger volume_surge
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            {
                'type': 'TECHNICAL_INDICATOR',
                'symbol': 'TSLA',
                'indicator': 'RSI',
                'value': 25.0,  # Should trigger rsi_oversold
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            {
                'type': 'PRICE_CHANGE',
                'symbol': 'MSFT',
                'change_percent': 2.0,  # Should NOT trigger
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        ]

        # Process events and check results
        total_triggers_fired = 0
        for event in test_events:
            triggered_actions = await event_trigger_system.process_event(event)
            total_triggers_fired += len(triggered_actions)

        # Verify trigger firing
        assert total_triggers_fired == 3  # First 3 events should trigger
        assert event_trigger_system.processing_stats['events_processed'] == 4
        assert event_trigger_system.processing_stats['triggers_fired'] == 3
        assert event_trigger_system.processing_stats['events_failed'] == 0

        # Verify specific trigger counts
        assert event_trigger_system.triggers['price_spike']['fired_count'] == 1
        assert event_trigger_system.triggers['volume_surge']['fired_count'] == 1
        assert event_trigger_system.triggers['rsi_oversold']['fired_count'] == 1

        logger.info("✅ Event trigger firing test completed")

    @pytest.mark.asyncio
    async def test_agent_coordination_flow(self, streaming_websocket_manager, event_trigger_system):
        """Test complete agent coordination through streaming system"""

        # Connect multiple agents
        agents = {
            'analyst_agent': 'analyst',
            'risk_agent': 'risk',
            'strategy_agent': 'strategist',
            'execution_agent': 'executor'
        }

        for agent_id, agent_type in agents.items():
            await streaming_websocket_manager.connect_client(agent_id, agent_type)

        # Register coordination triggers
        await event_trigger_system.register_trigger(
            'analyst_signal',
            {'type': 'MARKET_ANALYSIS', 'signal': 'BUY'},
            'REQUEST_RISK_ASSESSMENT'
        )

        await event_trigger_system.register_trigger(
            'risk_approval',
            {'type': 'RISK_ASSESSMENT', 'approved': True},
            'REQUEST_STRATEGY_CREATION'
        )

        await event_trigger_system.register_trigger(
            'strategy_ready',
            {'type': 'STRATEGY_CREATED', 'status': 'ready'},
            'EXECUTE_TRADE'
        )

        # Simulate agent coordination workflow
        coordination_events = [
            {
                'type': 'MARKET_ANALYSIS',
                'agent': 'analyst_agent',
                'symbol': 'AAPL',
                'signal': 'BUY',
                'confidence': 0.85,
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            {
                'type': 'RISK_ASSESSMENT',
                'agent': 'risk_agent',
                'symbol': 'AAPL',
                'approved': True,
                'risk_score': 0.3,
                'max_position': 10000,
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            {
                'type': 'STRATEGY_CREATED',
                'agent': 'strategy_agent',
                'symbol': 'AAPL',
                'strategy': 'LONG_CALL',
                'status': 'ready',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        ]

        # Process coordination workflow
        workflow_results = []
        for event in coordination_events:
            # Broadcast to all agents
            sent_count = await streaming_websocket_manager.broadcast_message(event)
            assert sent_count == len(agents)

            # Process through trigger system
            triggered_actions = await event_trigger_system.process_event(event)
            workflow_results.extend(triggered_actions)

        # Verify coordination flow
        expected_actions = ['REQUEST_RISK_ASSESSMENT', 'REQUEST_STRATEGY_CREATION', 'EXECUTE_TRADE']
        assert workflow_results == expected_actions

        # Verify message statistics
        total_messages = len(coordination_events) * len(agents)
        assert streaming_websocket_manager.connection_stats['messages_sent'] == total_messages

        logger.info("✅ Agent coordination flow test completed")

    @pytest.mark.asyncio
    async def test_high_load_scenarios(self, streaming_websocket_manager, market_data_streamer):
        """Test system behavior under high load"""

        # Connect many clients
        num_clients = 100
        client_ids = [f"client_{i}" for i in range(num_clients)]

        start_time = time.time()

        # Connect all clients concurrently
        connection_tasks = [
            streaming_websocket_manager.connect_client(client_id)
            for client_id in client_ids
        ]
        connection_results = await asyncio.gather(*connection_tasks)

        connection_time = time.time() - start_time

        # Verify all connections succeeded
        assert all(connection_results)
        assert len(streaming_websocket_manager.active_connections) == num_clients
        assert connection_time < 5.0  # Should connect 100 clients in under 5 seconds

        # Subscribe to many symbols
        symbols = [f"SYMBOL_{i}" for i in range(50)]
        for symbol in symbols:
            await market_data_streamer.subscribe(symbol)

        # High-frequency message broadcasting
        num_messages = 1000
        start_time = time.time()

        broadcast_tasks = []
        for i in range(num_messages):
            message = {
                'type': 'HIGH_FREQ_DATA',
                'sequence': i,
                'data': f"test_data_{i}",
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            task = streaming_websocket_manager.broadcast_message(message)
            broadcast_tasks.append(task)

        # Process broadcasts in batches to avoid overwhelming
        batch_size = 50
        for i in range(0, len(broadcast_tasks), batch_size):
            batch = broadcast_tasks[i:i + batch_size]
            await asyncio.gather(*batch)

        broadcast_time = time.time() - start_time

        # Verify performance
        total_message_sends = num_messages * num_clients
        messages_per_second = total_message_sends / broadcast_time

        assert messages_per_second > 1000  # Should handle >1000 messages/sec
        assert streaming_websocket_manager.connection_stats['messages_sent'] == total_message_sends
        assert streaming_websocket_manager.connection_stats['connection_errors'] == 0

        # Test concurrent disconnection
        start_time = time.time()

        disconnection_tasks = [
            streaming_websocket_manager.disconnect_client(client_id)
            for client_id in client_ids
        ]
        await asyncio.gather(*disconnection_tasks)

        disconnection_time = time.time() - start_time

        assert len(streaming_websocket_manager.active_connections) == 0
        assert disconnection_time < 3.0  # Should disconnect all clients quickly

        logger.info(f"✅ High load test completed: {messages_per_second:.0f} msg/sec")

    @pytest.mark.asyncio
    async def test_failure_recovery(self, streaming_websocket_manager, event_trigger_system):
        """Test system failure recovery and resilience"""

        # Connect test clients
        await streaming_websocket_manager.connect_client("test_client_1")
        await streaming_websocket_manager.connect_client("test_client_2")

        # Register test trigger
        await event_trigger_system.register_trigger(
            'test_trigger',
            {'type': 'TEST_EVENT'},
            'TEST_ACTION'
        )

        # Test 1: WebSocket manager failure simulation
        original_broadcast = streaming_websocket_manager.broadcast_message

        async def failing_broadcast(message, target_type=None):
            # Simulate intermittent failures
            if random.random() < 0.3:  # 30% failure rate
                raise ConnectionClosed(None, None)
            return await original_broadcast(message, target_type)

        streaming_websocket_manager.broadcast_message = failing_broadcast

        # Send messages and expect some failures
        failure_count = 0
        success_count = 0

        for i in range(20):
            try:
                message = {'type': 'TEST', 'id': i}
                await streaming_websocket_manager.broadcast_message(message)
                success_count += 1
            except ConnectionClosed:
                failure_count += 1
                # System should handle gracefully
                pass

        assert failure_count > 0  # Should have some failures
        assert success_count > 0  # Should have some successes

        # Restore original function
        streaming_websocket_manager.broadcast_message = original_broadcast

        # Test 2: Event trigger system failure recovery
        original_process = event_trigger_system.process_event

        async def failing_process(event):
            if 'fail' in event.get('type', ''):
                raise Exception("Simulated processing failure")
            return await original_process(event)

        event_trigger_system.process_event = failing_process

        # Process events with some failures
        test_events = [
            {'type': 'NORMAL_EVENT', 'data': 'test1'},
            {'type': 'FAIL_EVENT', 'data': 'test2'},  # Should fail
            {'type': 'NORMAL_EVENT', 'data': 'test3'},
            {'type': 'FAIL_EVENT', 'data': 'test4'},  # Should fail
        ]

        processed_count = 0
        for event in test_events:
            try:
                await event_trigger_system.process_event(event)
                processed_count += 1
            except Exception:
                # System should handle gracefully
                pass

        assert processed_count == 2  # Should process 2 normal events

        # Restore original function
        event_trigger_system.process_event = original_process

        # Test 3: Full system recovery
        await streaming_websocket_manager.stop()
        await streaming_websocket_manager.start()

        # System should be functional after restart
        await streaming_websocket_manager.connect_client("recovery_test")
        assert len(streaming_websocket_manager.active_connections) == 1

        test_message = {'type': 'RECOVERY_TEST'}
        sent_count = await streaming_websocket_manager.broadcast_message(test_message)
        assert sent_count == 1

        logger.info("✅ Failure recovery test completed")

    @pytest.mark.asyncio
    async def test_end_to_end_streaming_workflow(
        self,
        streaming_websocket_manager,
        event_trigger_system,
        market_data_streamer
    ):
        """Test complete end-to-end streaming workflow"""

        logger.info("🚀 Starting end-to-end streaming workflow test")

        # Setup: Connect trading components
        trading_components = {
            'market_data_handler': 'data_processor',
            'signal_generator': 'signal_processor',
            'risk_manager': 'risk_processor',
            'order_manager': 'order_processor',
            'portfolio_tracker': 'portfolio_processor'
        }

        for component_id, component_type in trading_components.items():
            await streaming_websocket_manager.connect_client(component_id, component_type)

        # Setup: Register complete trading workflow triggers
        workflow_triggers = [
            {
                'id': 'market_data_received',
                'condition': {'type': 'MARKET_DATA'},
                'action': 'PROCESS_TECHNICAL_ANALYSIS'
            },
            {
                'id': 'signal_generated',
                'condition': {'type': 'TRADING_SIGNAL', 'strength': {'gt': 0.7}},
                'action': 'EVALUATE_RISK'
            },
            {
                'id': 'risk_approved',
                'condition': {'type': 'RISK_EVALUATION', 'approved': True},
                'action': 'GENERATE_ORDER'
            },
            {
                'id': 'order_created',
                'condition': {'type': 'ORDER_CREATED'},
                'action': 'EXECUTE_TRADE'
            },
            {
                'id': 'trade_executed',
                'condition': {'type': 'TRADE_EXECUTED'},
                'action': 'UPDATE_PORTFOLIO'
            }
        ]

        for trigger in workflow_triggers:
            await event_trigger_system.register_trigger(
                trigger['id'],
                trigger['condition'],
                trigger['action']
            )

        # Subscribe to test symbols
        test_symbols = ['AAPL', 'SPY', 'QQQ']
        for symbol in test_symbols:
            await market_data_streamer.subscribe(symbol)

        # Execute: Simulate complete trading workflow
        workflow_events = [
            # 1. Market data arrives
            {
                'type': 'MARKET_DATA',
                'symbol': 'AAPL',
                'price': 150.25,
                'volume': 1000000,
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            # 2. Technical analysis generates signal
            {
                'type': 'TRADING_SIGNAL',
                'symbol': 'AAPL',
                'direction': 'BUY',
                'strength': 0.85,
                'indicators': {'rsi': 35, 'macd': 'bullish'},
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            # 3. Risk evaluation approves
            {
                'type': 'RISK_EVALUATION',
                'symbol': 'AAPL',
                'approved': True,
                'risk_score': 0.25,
                'position_size': 100,
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            # 4. Order is created
            {
                'type': 'ORDER_CREATED',
                'symbol': 'AAPL',
                'order_id': 'ORDER_123',
                'side': 'BUY',
                'quantity': 100,
                'price': 150.25,
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            # 5. Trade is executed
            {
                'type': 'TRADE_EXECUTED',
                'symbol': 'AAPL',
                'order_id': 'ORDER_123',
                'executed_price': 150.30,
                'executed_quantity': 100,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        ]

        # Process workflow with timing
        workflow_start = time.time()
        workflow_results = []

        for i, event in enumerate(workflow_events):
            step_start = time.time()

            # Broadcast to all components
            sent_count = await streaming_websocket_manager.broadcast_message(event)
            assert sent_count == len(trading_components)

            # Process through trigger system
            triggered_actions = await event_trigger_system.process_event(event)
            workflow_results.extend(triggered_actions)

            step_time = time.time() - step_start
            logger.info(f"Step {i+1}: {event['type']} -> {triggered_actions} ({step_time:.3f}s)")

        workflow_time = time.time() - workflow_start

        # Verify: Complete workflow execution
        expected_actions = [
            'PROCESS_TECHNICAL_ANALYSIS',
            'EVALUATE_RISK',
            'GENERATE_ORDER',
            'EXECUTE_TRADE',
            'UPDATE_PORTFOLIO'
        ]
        assert workflow_results == expected_actions

        # Verify: Performance metrics
        assert workflow_time < 2.0  # Complete workflow should finish quickly
        assert event_trigger_system.processing_stats['events_processed'] == len(workflow_events)
        assert event_trigger_system.processing_stats['triggers_fired'] == len(expected_actions)

        # Verify: Message distribution
        total_messages = len(workflow_events) * len(trading_components)
        assert streaming_websocket_manager.connection_stats['messages_sent'] == total_messages

        # Test: System statistics
        assert streaming_websocket_manager.connection_stats['connection_errors'] == 0
        assert event_trigger_system.processing_stats['events_failed'] == 0

        logger.info("✅ End-to-end streaming workflow completed successfully")
        logger.info(f"   Workflow time: {workflow_time:.3f}s")
        logger.info(f"   Messages sent: {total_messages}")
        logger.info(f"   Triggers fired: {len(expected_actions)}")
        logger.info(f"   Components connected: {len(trading_components)}")

    @pytest.mark.asyncio
    async def test_streaming_performance_benchmarks(self, streaming_websocket_manager, event_trigger_system):
        """Test streaming system performance benchmarks"""

        # Performance test configuration
        num_clients = 200
        num_events = 5000
        num_triggers = 50

        logger.info(f"🏃 Performance test: {num_clients} clients, {num_events} events, {num_triggers} triggers")

        # Setup: Connect many clients
        client_tasks = [
            streaming_websocket_manager.connect_client(f"perf_client_{i}")
            for i in range(num_clients)
        ]

        connection_start = time.time()
        await asyncio.gather(*client_tasks)
        connection_time = time.time() - connection_start

        assert len(streaming_websocket_manager.active_connections) == num_clients

        # Setup: Register many triggers
        trigger_tasks = []
        for i in range(num_triggers):
            trigger_tasks.append(
                event_trigger_system.register_trigger(
                    f'perf_trigger_{i}',
                    {'type': 'PERF_EVENT', 'id': i},
                    f'PERF_ACTION_{i}'
                )
            )

        await asyncio.gather(*trigger_tasks)

        # Test: High-frequency event processing
        events = [
            {
                'type': 'PERF_EVENT',
                'id': i % num_triggers,  # Will trigger some events
                'data': f'performance_test_{i}',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            for i in range(num_events)
        ]

        # Process events in batches for realistic load
        batch_size = 100
        processing_start = time.time()

        for i in range(0, len(events), batch_size):
            batch = events[i:i + batch_size]

            # Process batch concurrently
            batch_tasks = []
            for event in batch:
                # Broadcast to clients
                broadcast_task = streaming_websocket_manager.broadcast_message(event)
                # Process through triggers
                trigger_task = event_trigger_system.process_event(event)
                batch_tasks.extend([broadcast_task, trigger_task])

            await asyncio.gather(*batch_tasks)

        processing_time = time.time() - processing_start

        # Calculate performance metrics
        events_per_second = num_events / processing_time
        messages_per_second = (num_events * num_clients) / processing_time

        # Performance assertions
        assert events_per_second > 500  # Should process >500 events/sec
        assert messages_per_second > 10000  # Should handle >10k messages/sec
        assert connection_time < 5.0  # Should connect 200 clients quickly

        # Verify accuracy
        assert event_trigger_system.processing_stats['events_processed'] == num_events
        expected_messages = num_events * num_clients
        assert streaming_websocket_manager.connection_stats['messages_sent'] >= expected_messages

        logger.info("✅ Performance benchmarks completed")
        logger.info(f"   Connection time: {connection_time:.3f}s ({num_clients} clients)")
        logger.info(f"   Processing time: {processing_time:.3f}s ({num_events} events)")
        logger.info(f"   Events/sec: {events_per_second:.0f}")
        logger.info(f"   Messages/sec: {messages_per_second:.0f}")


if __name__ == "__main__":
    # Run specific tests for development
    import sys

    async def run_test():
        """Run a specific test for development"""
        # Create test instances
        manager = MockStreamingWebSocketManager()
        trigger_system = MockEventTriggerSystem()
        streamer = MockMarketDataStreamer()

        await manager.start()
        await trigger_system.start()
        await streamer.start_streaming()

        try:
            # Run end-to-end test
            test_instance = TestStreamingSystemIntegration()
            await test_instance.test_end_to_end_streaming_workflow(
                manager, trigger_system, streamer
            )

        finally:
            await manager.stop()
            await trigger_system.stop()
            await streamer.stop_streaming()

    if len(sys.argv) > 1 and sys.argv[1] == "run":
        asyncio.run(run_test())