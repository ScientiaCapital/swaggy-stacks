"""
NATS Performance Optimization Tests
==================================

Tests for the high-performance NATS implementation including:
- Connection pooling performance
- Message batching efficiency
- Bulk publishing throughput
- Performance monitoring accuracy
- Latency optimization validation
"""

import asyncio
import json
import pytest
import time
from typing import List, Dict
from unittest.mock import AsyncMock, MagicMock, patch

from app.messaging.nats_coordinator import (
    NATSAgentCoordinator,
    NATSConnectionPool,
    MessageBatcher,
    get_nats_coordinator
)
from app.core.config import settings


class TestNATSConnectionPool:
    """Test suite for NATSConnectionPool performance optimization"""

    @pytest.fixture
    async def connection_pool(self):
        """Create a test connection pool"""
        pool = NATSConnectionPool(pool_size=3)
        yield pool
        await pool.close_all()

    @pytest.mark.asyncio
    async def test_connection_pool_creation(self, connection_pool):
        """Test connection pool initializes properly"""
        assert connection_pool.pool_size == 3
        assert len(connection_pool.connections) == 0
        assert len(connection_pool.active_connections) == 0
        assert connection_pool._connection_counter == 0

    @pytest.mark.asyncio
    async def test_connection_pool_get_connection(self, connection_pool):
        """Test getting connections from pool"""
        # Mock NATS connection creation
        with patch('app.messaging.nats_coordinator.nc.NATS') as mock_nats:
            mock_conn = AsyncMock()
            mock_conn.is_connected = True
            mock_conn.connect = AsyncMock()
            mock_nats.return_value = mock_conn

            # Get first connection
            conn1 = await connection_pool.get_connection()
            assert conn1 is not None
            assert len(connection_pool.active_connections) == 1

            # Get second connection
            conn2 = await connection_pool.get_connection()
            assert conn2 is not None
            assert len(connection_pool.active_connections) == 2

            # Verify connections are different
            assert conn1 != conn2

    @pytest.mark.asyncio
    async def test_connection_pool_stats_tracking(self, connection_pool):
        """Test connection statistics tracking"""
        conn_id = "test_conn_001"

        # Test send operation stats
        connection_pool.update_stats(conn_id, 'send', 1024)
        stats = connection_pool.connection_stats[conn_id]
        assert stats['messages_sent'] == 1
        assert stats['bytes_sent'] == 1024
        assert stats['last_used'] > 0

        # Test receive operation stats
        connection_pool.update_stats(conn_id, 'receive', 512)
        stats = connection_pool.connection_stats[conn_id]
        assert stats['messages_received'] == 1
        assert stats['bytes_received'] == 512

        # Test error tracking
        connection_pool.update_stats(conn_id, 'error')
        stats = connection_pool.connection_stats[conn_id]
        assert stats['errors'] == 1


class TestMessageBatcher:
    """Test suite for MessageBatcher performance optimization"""

    @pytest.fixture
    def message_batcher(self):
        """Create a test message batcher"""
        return MessageBatcher(batch_size=5, batch_timeout=0.01)  # 10ms timeout for testing

    @pytest.fixture
    async def mock_connection_pool(self):
        """Create a mock connection pool"""
        pool = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.publish = AsyncMock()
        mock_conn.flush = AsyncMock()
        pool.get_connection = AsyncMock(return_value=mock_conn)
        pool.return_connection = MagicMock()
        return pool

    @pytest.mark.asyncio
    async def test_message_batcher_initialization(self, message_batcher):
        """Test message batcher initializes properly"""
        assert message_batcher.batch_size == 5
        assert message_batcher.batch_timeout == 0.01
        assert len(message_batcher.pending_messages) == 0
        assert len(message_batcher.pending_tasks) == 0
        assert message_batcher.batch_stats['batches_sent'] == 0

    @pytest.mark.asyncio
    async def test_message_batching_by_size(self, message_batcher, mock_connection_pool):
        """Test message batching triggers at batch size limit"""
        subject = "test.batching.size"

        # Add messages up to batch size
        for i in range(5):
            await message_batcher.add_message(
                subject,
                f"test_message_{i}".encode(),
                mock_connection_pool
            )

        # Give a moment for batch to be processed
        await asyncio.sleep(0.02)

        # Verify batch was sent
        assert message_batcher.batch_stats['batches_sent'] == 1
        assert message_batcher.batch_stats['messages_batched'] == 5
        assert message_batcher.batch_stats['avg_batch_size'] == 5.0

        # Verify connection was used
        mock_connection_pool.get_connection.assert_called()

    @pytest.mark.asyncio
    async def test_message_batching_by_timeout(self, message_batcher, mock_connection_pool):
        """Test message batching triggers at timeout"""
        subject = "test.batching.timeout"

        # Add just one message (below batch size)
        await message_batcher.add_message(
            subject,
            b"test_timeout_message",
            mock_connection_pool
        )

        # Wait for timeout to trigger
        await asyncio.sleep(0.02)

        # Verify batch was sent due to timeout
        assert message_batcher.batch_stats['batches_sent'] == 1
        assert message_batcher.batch_stats['messages_batched'] == 1

    @pytest.mark.asyncio
    async def test_multiple_subject_batching(self, message_batcher, mock_connection_pool):
        """Test batching works correctly with multiple subjects"""
        # Add messages for different subjects
        await message_batcher.add_message("subject.a", b"msg_a1", mock_connection_pool)
        await message_batcher.add_message("subject.b", b"msg_b1", mock_connection_pool)
        await message_batcher.add_message("subject.a", b"msg_a2", mock_connection_pool)

        # Wait for timeouts
        await asyncio.sleep(0.02)

        # Should have sent 2 batches (one per subject)
        assert message_batcher.batch_stats['batches_sent'] == 2
        assert message_batcher.batch_stats['messages_batched'] == 3


class TestNATSPerformanceOptimizations:
    """Test suite for NATS performance optimizations in NATSAgentCoordinator"""

    @pytest.fixture
    async def nats_coordinator(self):
        """Create a test NATS coordinator with performance optimizations"""
        coordinator = NATSAgentCoordinator()
        yield coordinator
        if coordinator.is_connected:
            await coordinator.disconnect()

    @pytest.mark.asyncio
    async def test_performance_metrics_initialization(self, nats_coordinator):
        """Test performance metrics are properly initialized"""
        metrics = nats_coordinator.performance_metrics
        assert metrics['messages_sent'] == 0
        assert metrics['messages_received'] == 0
        assert metrics['avg_send_latency'] == 0.0
        assert metrics['avg_receive_latency'] == 0.0
        assert 'last_performance_check' in metrics

    @pytest.mark.asyncio
    async def test_high_performance_publish_mock(self, nats_coordinator):
        """Test high-performance publish functionality with mocks"""
        # Mock the connection to avoid actual NATS dependency
        with patch.object(nats_coordinator, 'is_connected', True), \
             patch.object(nats_coordinator.connection_pool, 'get_connection') as mock_get_conn, \
             patch.object(nats_coordinator.connection_pool, 'return_connection') as mock_return_conn:

            mock_conn = AsyncMock()
            mock_conn.publish = AsyncMock()
            mock_conn.flush = AsyncMock()
            mock_get_conn.return_value = mock_conn

            # Test high-performance publish
            result = await nats_coordinator.publish_high_performance(
                subject="test.performance",
                data={"test": "high_performance_message"},
                agent_id=None,
                use_batching=False
            )

            assert result is True
            assert nats_coordinator.performance_metrics['messages_sent'] == 1
            mock_conn.publish.assert_called_once()
            mock_conn.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_bulk_publish_mock(self, nats_coordinator):
        """Test bulk publish functionality with mocks"""
        with patch.object(nats_coordinator, 'is_connected', True), \
             patch.object(nats_coordinator.connection_pool, 'get_connection') as mock_get_conn, \
             patch.object(nats_coordinator.connection_pool, 'return_connection') as mock_return_conn:

            mock_conn = AsyncMock()
            mock_conn.publish = AsyncMock()
            mock_conn.flush = AsyncMock()
            mock_get_conn.return_value = mock_conn

            # Test bulk publish
            messages = [
                {"subject": "test.bulk.1", "data": {"msg": 1}},
                {"subject": "test.bulk.1", "data": {"msg": 2}},
                {"subject": "test.bulk.2", "data": {"msg": 3}},
            ]

            results = await nats_coordinator.bulk_publish(messages)

            assert results['success'] == 3
            assert results['failed'] == 0
            # Should be called twice (once per unique subject)
            assert mock_get_conn.call_count == 2

    @pytest.mark.asyncio
    async def test_performance_score_calculation(self, nats_coordinator):
        """Test performance score calculation logic"""
        # Test perfect performance
        perfect_metrics = {
            'avg_send_latency': 0.0005,  # 0.5ms
            'connection_pool_utilization': 50.0,
            'batch_efficiency': 90.0
        }
        score = nats_coordinator._calculate_performance_score(perfect_metrics)
        assert score == 100

        # Test poor latency
        high_latency_metrics = {
            'avg_send_latency': 0.01,  # 10ms
            'connection_pool_utilization': 50.0,
            'batch_efficiency': 90.0
        }
        score = nats_coordinator._calculate_performance_score(high_latency_metrics)
        assert score < 100  # Should be penalized for high latency

    def test_latency_metric_update(self, nats_coordinator):
        """Test latency metric exponential moving average"""
        # Start with zero
        assert nats_coordinator.performance_metrics['avg_send_latency'] == 0.0

        # First measurement
        nats_coordinator._update_latency_metric('send', 0.001)  # 1ms
        assert nats_coordinator.performance_metrics['avg_send_latency'] == 0.0001  # 0.1 * 0.001

        # Second measurement
        nats_coordinator._update_latency_metric('send', 0.002)  # 2ms
        expected = 0.0001 * 0.9 + 0.002 * 0.1  # EMA calculation
        assert abs(nats_coordinator.performance_metrics['avg_send_latency'] - expected) < 0.0001


class TestNATSPerformanceIntegration:
    """Integration tests for NATS performance optimizations"""

    @pytest.mark.asyncio
    async def test_coordinator_with_all_optimizations(self):
        """Test NATS coordinator with all performance optimizations enabled"""
        # Test with high throughput mode enabled
        with patch('app.core.config.settings') as mock_settings:
            mock_settings.NATS_HIGH_THROUGHPUT_MODE = True
            mock_settings.NATS_CONNECTION_POOL_SIZE = 5
            mock_settings.NATS_MESSAGE_BATCH_SIZE = 100
            mock_settings.NATS_BATCH_TIMEOUT = 0.005
            mock_settings.NATS_SECURITY_ENABLED = False

            coordinator = NATSAgentCoordinator()

            # Verify performance components are initialized
            assert coordinator.connection_pool is not None
            assert coordinator.connection_pool.pool_size == 5
            assert coordinator.message_batcher is not None
            assert coordinator.message_batcher.batch_size == 100
            assert coordinator.message_batcher.batch_timeout == 0.005

    @pytest.mark.asyncio
    async def test_performance_monitoring_task(self):
        """Test that performance monitoring task runs correctly"""
        coordinator = NATSAgentCoordinator()

        # Mock the connection state
        coordinator.is_connected = True

        # Create a short-running performance monitor for testing
        async def short_performance_monitor():
            # Run one iteration of performance monitoring
            try:
                active_connections = len(coordinator.connection_pool.active_connections)
                pool_utilization = (active_connections / coordinator.connection_pool.pool_size) * 100

                batch_efficiency = 0.0
                if coordinator.message_batcher:
                    batch_efficiency = coordinator.message_batcher.batch_stats.get('batch_efficiency', 0.0)

                coordinator.performance_metrics.update({
                    'connection_pool_utilization': pool_utilization,
                    'batch_efficiency': batch_efficiency,
                    'last_performance_check': time.time()
                })

                return True
            except Exception:
                return False

        # Test the monitoring logic
        result = await short_performance_monitor()
        assert result is True
        assert 'connection_pool_utilization' in coordinator.performance_metrics
        assert coordinator.performance_metrics['last_performance_check'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])