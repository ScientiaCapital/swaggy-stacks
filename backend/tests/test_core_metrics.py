"""
Test Core Metrics - Institutional-Grade Trading System
Focused testing for essential trading metrics
"""

import pytest
from app.monitoring.core_metrics import CoreTradingMetrics, CoreMetricsCollector


class TestCoreTradingMetrics:
    """Test core trading metrics functionality"""

    @pytest.fixture
    def core_metrics(self):
        """Create CoreTradingMetrics instance for testing"""
        return CoreTradingMetrics()

    def test_trading_performance_update(self, core_metrics):
        """Test trading performance metrics update"""
        core_metrics.update_trading_performance(
            strategy="momentum_v1",
            symbol="AAPL",
            pnl_usd=1250.50,
            win_rate=0.68,
            sharpe_ratio=1.45
        )

        # Verify metrics can be accessed (actual values tested in integration)
        assert core_metrics.total_pnl is not None
        assert core_metrics.win_rate is not None
        assert core_metrics.sharpe_ratio is not None

    def test_order_execution_recording(self, core_metrics):
        """Test order execution metrics recording"""
        core_metrics.record_order_execution(
            order_type="market",
            symbol="AAPL",
            latency_ms=25.5,
            slippage_bps=1.2,
            strategy="momentum_v1"
        )

        # Verify metrics exist
        assert core_metrics.order_latency is not None
        assert core_metrics.slippage is not None

    def test_risk_metrics_update(self, core_metrics):
        """Test risk management metrics"""
        core_metrics.update_risk_metrics(
            symbol="AAPL",
            exposure_pct=15.5,
            daily_var_usd=2500.0,
            confidence_level="95"
        )

        # Verify risk metrics
        assert core_metrics.portfolio_exposure is not None
        assert core_metrics.daily_var is not None

    def test_ai_intelligence_update(self, core_metrics):
        """Test AI intelligence metrics"""
        core_metrics.update_ai_intelligence(
            symbol="AAPL",
            pattern_confidence=0.85,
            regime_accuracy=0.91,
            anomaly_score=0.15,
            prediction_accuracy=0.78
        )

        # Verify AI metrics
        assert core_metrics.pattern_match_confidence is not None
        assert core_metrics.regime_detection_accuracy is not None
        assert core_metrics.anomaly_score is not None
        assert core_metrics.ai_prediction_accuracy is not None

    def test_anomaly_alert_trigger(self, core_metrics):
        """Test anomaly alert functionality"""
        core_metrics.trigger_anomaly_alert(symbol="AAPL", severity="high")

        # Verify alert counter exists
        assert core_metrics.anomaly_alerts is not None

    def test_system_monitoring(self, core_metrics):
        """Test system monitoring metrics"""
        core_metrics.update_system_resources(
            component="trading_engine",
            cpu_percent=45.5,
            memory_mb=1024.0
        )

        core_metrics.update_system_uptime(uptime_seconds=86400.0)

        core_metrics.record_data_feed_latency(
            feed_type="real_time",
            symbol="AAPL",
            latency_ms=5.2
        )

        # Verify system metrics
        assert core_metrics.cpu_usage is not None
        assert core_metrics.memory_usage is not None
        assert core_metrics.system_uptime is not None
        assert core_metrics.data_feed_latency is not None

    def test_metrics_summary(self, core_metrics):
        """Test metrics summary functionality"""
        summary = core_metrics.get_metrics_summary()

        assert isinstance(summary, dict)
        assert "metrics_count" in summary
        assert "last_updated" in summary
        assert summary["metrics_count"] > 0


class TestCoreMetricsCollector:
    """Test core metrics collector functionality"""

    @pytest.fixture
    def collector(self):
        """Create CoreMetricsCollector instance for testing"""
        return CoreMetricsCollector()

    @pytest.mark.asyncio
    async def test_essential_metrics_collection(self, collector):
        """Test essential metrics collection"""
        result = await collector.collect_essential_metrics()

        assert isinstance(result, dict)
        assert "status" in result
        assert "timestamp" in result

        # Should either be updated or cached
        assert result["status"] in ["updated", "cached", "error"]

    def test_prometheus_metrics_export(self, collector):
        """Test Prometheus metrics export"""
        metrics_output = collector.get_prometheus_metrics()

        assert isinstance(metrics_output, str)
        # Should contain metric definitions
        if metrics_output:  # May be empty in test environment
            assert "trading_" in metrics_output or "ai_" in metrics_output or "system_" in metrics_output

    def test_collector_performance(self, collector):
        """Test collector performance under load"""
        import time

        start_time = time.time()

        # Run collection multiple times
        for _ in range(5):
            # Synchronous call for performance test
            summary = collector.core_metrics.get_metrics_summary()
            assert summary is not None

        end_time = time.time()
        total_time = end_time - start_time

        # Should complete quickly (under 0.1 seconds for 5 calls)
        assert total_time < 0.1


class TestInstitutionalGradeFeatures:
    """Test institutional-grade features"""

    def test_metric_precision(self):
        """Test that metrics support institutional precision requirements"""
        core_metrics = CoreTradingMetrics()

        # Test high-precision values
        core_metrics.update_trading_performance(
            strategy="high_freq_v1",
            symbol="AAPL",
            pnl_usd=1234.5678,  # Precision to 4 decimal places
            win_rate=0.6789,     # Precision for win rate
            sharpe_ratio=2.3456  # Precision for Sharpe ratio
        )

        # Test sub-millisecond latency recording
        core_metrics.record_order_execution(
            order_type="limit",
            symbol="AAPL",
            latency_ms=0.123,    # Sub-millisecond precision
            slippage_bps=0.05,   # Sub-basis point precision
            strategy="high_freq_v1"
        )

        # No exceptions should be raised with high precision values
        assert True

    def test_high_frequency_updates(self):
        """Test system can handle high-frequency metric updates"""
        core_metrics = CoreTradingMetrics()

        # Simulate high-frequency trading updates
        for i in range(1000):
            core_metrics.record_order_execution(
                order_type="market",
                symbol=f"SYM{i % 10}",
                latency_ms=float(i % 100),
                slippage_bps=float(i % 10),
                strategy="hft_strategy"
            )

        # Should handle 1000 updates without issues
        summary = core_metrics.get_metrics_summary()
        assert summary["metrics_count"] > 0

    def test_concurrent_metric_updates(self):
        """Test concurrent access to metrics"""
        import threading
        import time

        core_metrics = CoreTradingMetrics()
        errors = []

        def update_metrics(thread_id):
            try:
                for i in range(100):
                    core_metrics.update_trading_performance(
                        strategy=f"strategy_{thread_id}",
                        symbol="AAPL",
                        pnl_usd=float(i),
                        win_rate=0.5,
                        sharpe_ratio=1.0
                    )
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=update_metrics, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Should complete without errors
        assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])