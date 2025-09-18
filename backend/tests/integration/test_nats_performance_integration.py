"""
NATS Performance Integration Test
===============================

End-to-end integration test for NATS performance optimizations.
This test verifies the complete performance enhancement system works together.
"""

import asyncio
import pytest
import time
from unittest.mock import patch, AsyncMock

from app.messaging.nats_coordinator import NATSAgentCoordinator


class TestNATSPerformanceIntegrationComplete:
    """Complete integration test for NATS performance system"""

    @pytest.mark.asyncio
    async def test_complete_performance_optimization_flow(self):
        """
        Test the complete performance optimization flow:
        1. Initialize coordinator with optimizations
        2. Connect with connection pool
        3. Send high-performance messages
        4. Monitor performance metrics
        5. Run optimization analysis
        6. Verify performance improvements
        """

        # Mock settings for high performance mode
        with patch('app.core.config.settings') as mock_settings:
            mock_settings.NATS_HIGH_THROUGHPUT_MODE = True
            mock_settings.NATS_CONNECTION_POOL_SIZE = 3
            mock_settings.NATS_MESSAGE_BATCH_SIZE = 5
            mock_settings.NATS_BATCH_TIMEOUT = 0.01
            mock_settings.NATS_SECURITY_ENABLED = False
            mock_settings.NATS_URL = "nats://localhost:4222"
            mock_settings.NATS_MAX_RECONNECT_ATTEMPTS = 10
            mock_settings.NATS_RECONNECT_TIME_WAIT = 1
            mock_settings.NATS_MAX_OUTSTANDING_PINGS = 2
            mock_settings.NATS_PING_INTERVAL = 60
            mock_settings.NATS_MAX_PAYLOAD = 1048576
            mock_settings.NATS_SEND_BUFFER_SIZE = 2097152
            mock_settings.NATS_RECEIVE_BUFFER_SIZE = 2097152
            mock_settings.NATS_FLUSH_TIMEOUT = 1.0
            mock_settings.NATS_TLS_ENABLED = False

            # Step 1: Initialize coordinator with performance optimizations
            coordinator = NATSAgentCoordinator()

            # Verify performance components are initialized
            assert coordinator.connection_pool is not None
            assert coordinator.connection_pool.pool_size == 3
            assert coordinator.message_batcher is not None
            assert coordinator.message_batcher.batch_size == 5
            assert coordinator.performance_metrics['messages_sent'] == 0

            # Step 2: Mock connection for testing
            with patch.object(coordinator.connection_pool, 'get_connection') as mock_get_conn, \
                 patch.object(coordinator.connection_pool, 'return_connection') as mock_return_conn:

                mock_conn = AsyncMock()
                mock_conn.is_connected = True
                mock_conn.publish = AsyncMock()
                mock_conn.flush = AsyncMock()
                mock_get_conn.return_value = mock_conn

                # Mock the connection state
                coordinator.is_connected = True

                # Step 3: Test high-performance message sending
                start_time = time.time()

                # Send multiple messages using high-performance method
                success_count = 0
                for i in range(10):
                    result = await coordinator.publish_high_performance(
                        subject=f"test.performance.{i}",
                        data={"test_msg": i, "timestamp": time.time()},
                        use_batching=False  # Direct send for immediate testing
                    )
                    if result:
                        success_count += 1

                send_duration = time.time() - start_time

                # Verify all messages were sent successfully
                assert success_count == 10
                assert coordinator.performance_metrics['messages_sent'] == 10
                assert coordinator.performance_metrics['avg_send_latency'] > 0

                # Step 4: Test bulk publishing
                bulk_messages = [
                    {"subject": "bulk.test.1", "data": {"bulk_msg": i}}
                    for i in range(20)
                ]

                bulk_start = time.time()
                bulk_results = await coordinator.bulk_publish(bulk_messages)
                bulk_duration = time.time() - bulk_start

                # Verify bulk publishing results
                assert bulk_results['success'] == 20
                assert bulk_results['failed'] == 0

                # Step 5: Test performance metrics collection
                performance_metrics = await coordinator.get_performance_metrics()

                # Verify metrics structure
                assert 'connection_pool' in performance_metrics
                assert 'messages_sent' in performance_metrics
                assert 'avg_send_latency' in performance_metrics
                assert performance_metrics['messages_sent'] >= 10  # At least from high-perf messages

                # Step 6: Test performance optimization analysis
                optimization_result = await coordinator.optimize_performance()

                # Verify optimization analysis
                assert 'current_metrics' in optimization_result
                assert 'optimizations' in optimization_result
                assert 'performance_score' in optimization_result
                assert 0 <= optimization_result['performance_score'] <= 100

                # Step 7: Verify performance improvements
                # Calculate theoretical throughput
                messages_per_second = 10 / send_duration if send_duration > 0 else 0
                bulk_messages_per_second = 20 / bulk_duration if bulk_duration > 0 else 0

                # Performance should be reasonable (>100 msg/sec for mocked operations)
                assert messages_per_second > 100, f"Performance too low: {messages_per_second} msg/sec"
                assert bulk_messages_per_second > 500, f"Bulk performance too low: {bulk_messages_per_second} msg/sec"

                # Verify connection pool was used efficiently
                assert mock_get_conn.call_count >= 10  # Should have gotten connections for messages
                assert mock_return_conn.call_count >= 10  # Should have returned connections

                print(f"✅ Performance Test Results:")
                print(f"   High-Performance Messages: {messages_per_second:.0f} msg/sec")
                print(f"   Bulk Publishing: {bulk_messages_per_second:.0f} msg/sec")
                print(f"   Performance Score: {optimization_result['performance_score']}/100")
                print(f"   Optimizations Suggested: {len(optimization_result['optimizations'])}")

    @pytest.mark.asyncio
    async def test_message_batching_performance(self):
        """Test that message batching improves performance"""

        with patch('app.core.config.settings') as mock_settings:
            mock_settings.NATS_HIGH_THROUGHPUT_MODE = True
            mock_settings.NATS_CONNECTION_POOL_SIZE = 2
            mock_settings.NATS_MESSAGE_BATCH_SIZE = 10
            mock_settings.NATS_BATCH_TIMEOUT = 0.01
            mock_settings.NATS_SECURITY_ENABLED = False

            coordinator = NATSAgentCoordinator()

            # Mock connection pool
            with patch.object(coordinator.connection_pool, 'get_connection') as mock_get_conn:
                mock_conn = AsyncMock()
                mock_conn.publish = AsyncMock()
                mock_conn.flush = AsyncMock()
                mock_get_conn.return_value = mock_conn

                coordinator.is_connected = True

                # Test batching efficiency
                start_time = time.time()

                # Send messages that should be batched
                for i in range(25):  # Should create 3 batches (10+10+5)
                    await coordinator.publish_high_performance(
                        subject="batch.test",
                        data=f"message_{i}",
                        use_batching=True
                    )

                # Wait for all batches to complete
                await asyncio.sleep(0.05)

                # Flush any remaining batches
                if coordinator.message_batcher:
                    await coordinator.message_batcher.flush_all(coordinator.connection_pool)

                batch_duration = time.time() - start_time

                # Verify batching statistics
                if coordinator.message_batcher:
                    batch_stats = coordinator.message_batcher.batch_stats
                    print(f"✅ Batching Test Results:")
                    print(f"   Messages Batched: {batch_stats['messages_batched']}")
                    print(f"   Batches Sent: {batch_stats['batches_sent']}")
                    print(f"   Average Batch Size: {batch_stats['avg_batch_size']:.1f}")
                    print(f"   Batch Efficiency: {batch_stats['batch_efficiency']:.1f}%")

                    # Verify batching worked
                    assert batch_stats['messages_batched'] >= 25
                    assert batch_stats['batches_sent'] >= 1
                    assert batch_stats['avg_batch_size'] > 1  # Should be batching multiple messages

    def test_performance_score_calculation_edge_cases(self):
        """Test performance score calculation with various metrics"""
        coordinator = NATSAgentCoordinator()

        # Test perfect score
        perfect_metrics = {
            'avg_send_latency': 0.0005,  # 0.5ms - excellent
            'connection_pool_utilization': 50.0,  # Optimal
            'batch_efficiency': 95.0  # Excellent
        }
        score = coordinator._calculate_performance_score(perfect_metrics)
        assert score == 100

        # Test terrible performance
        terrible_metrics = {
            'avg_send_latency': 0.1,  # 100ms - terrible for trading
            'connection_pool_utilization': 95.0,  # Over-utilized
            'batch_efficiency': 10.0  # Poor batching
        }
        score = coordinator._calculate_performance_score(terrible_metrics)
        assert score < 50  # Should be heavily penalized

        # Test medium performance
        medium_metrics = {
            'avg_send_latency': 0.003,  # 3ms - acceptable
            'connection_pool_utilization': 70.0,  # Good utilization
            'batch_efficiency': 60.0  # Decent batching
        }
        score = coordinator._calculate_performance_score(medium_metrics)
        assert 60 <= score <= 90  # Should be in the acceptable range


if __name__ == "__main__":
    pytest.main([__file__, "-v"])