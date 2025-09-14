"""
Test cache metrics module
"""

import pytest
from datetime import datetime, timedelta
from app.core.cache_metrics import CacheMetrics


class TestCacheMetrics:
    """Test cache metrics tracking"""

    def test_initial_state(self):
        """Test initial metrics state"""
        metrics = CacheMetrics()

        assert metrics.l1_hits == 0
        assert metrics.l2_hits == 0
        assert metrics.misses == 0
        assert metrics.l1_sets == 0
        assert metrics.l2_sets == 0
        assert metrics.errors == 0
        assert metrics.total_requests == 0
        assert metrics.avg_l1_time == 0.0
        assert metrics.avg_l2_time == 0.0
        assert isinstance(metrics.start_time, datetime)

    def test_record_l1_hit(self):
        """Test L1 hit recording"""
        metrics = CacheMetrics()

        metrics.record_l1_hit(0.05)

        assert metrics.l1_hits == 1
        assert metrics.total_requests == 1
        assert metrics.avg_l1_time == 0.05

        # Test averaging
        metrics.record_l1_hit(0.15)
        assert metrics.l1_hits == 2
        assert metrics.total_requests == 2
        assert metrics.avg_l1_time == 0.10  # (0.05 + 0.15) / 2

    def test_record_l2_hit(self):
        """Test L2 hit recording"""
        metrics = CacheMetrics()

        metrics.record_l2_hit(0.20)

        assert metrics.l2_hits == 1
        assert metrics.total_requests == 1
        assert metrics.avg_l2_time == 0.20

    def test_record_miss(self):
        """Test miss recording"""
        metrics = CacheMetrics()

        metrics.record_miss()

        assert metrics.misses == 1
        assert metrics.total_requests == 1

    def test_record_operations(self):
        """Test set and error operations"""
        metrics = CacheMetrics()

        metrics.record_l1_set()
        metrics.record_l2_set()
        metrics.record_error()

        assert metrics.l1_sets == 1
        assert metrics.l2_sets == 1
        assert metrics.errors == 1

    def test_hit_rate_calculations(self):
        """Test hit rate calculations"""
        metrics = CacheMetrics()

        # No requests yet
        assert metrics.hit_rate == 0.0
        assert metrics.l1_hit_rate == 0.0
        assert metrics.l2_hit_rate == 0.0

        # Add some hits and misses
        metrics.record_l1_hit(0.05)  # total: 1
        metrics.record_l2_hit(0.10)  # total: 2
        metrics.record_miss()        # total: 3
        metrics.record_miss()        # total: 4

        # 2 hits out of 4 requests = 50%
        assert metrics.hit_rate == 0.5
        # 1 L1 hit out of 4 requests = 25%
        assert metrics.l1_hit_rate == 0.25
        # 1 L2 hit out of 4 requests = 25%
        assert metrics.l2_hit_rate == 0.25

    def test_error_rate_calculation(self):
        """Test error rate calculation"""
        metrics = CacheMetrics()

        metrics.record_l1_hit(0.05)  # total: 1
        metrics.record_error()       # errors: 1 (errors don't count as requests)
        metrics.record_miss()        # total: 2

        # 1 error out of 2 total requests = 50% error rate
        assert metrics.error_rate == 0.5
        assert metrics.total_requests == 2
        assert metrics.errors == 1

    def test_get_summary(self):
        """Test summary generation"""
        metrics = CacheMetrics()

        # Add some data
        metrics.record_l1_hit(0.05)
        metrics.record_l2_hit(0.20)
        metrics.record_miss()
        metrics.record_l1_set()
        metrics.record_l2_set()
        metrics.record_error()

        summary = metrics.get_summary()

        assert isinstance(summary, dict)
        assert "uptime_seconds" in summary
        assert summary["total_requests"] == 3
        assert summary["l1_hits"] == 1
        assert summary["l2_hits"] == 1
        assert summary["misses"] == 1
        assert summary["errors"] == 1
        assert summary["hit_rate"] == 2/3
        assert summary["l1_hit_rate"] == 1/3
        assert summary["l2_hit_rate"] == 1/3
        assert summary["error_rate"] == 1/3  # 1 error out of 3 requests
        assert summary["avg_l1_time_ms"] == 50.0  # 0.05 * 1000
        assert summary["avg_l2_time_ms"] == 200.0  # 0.20 * 1000

    def test_reset(self):
        """Test metrics reset"""
        metrics = CacheMetrics()

        # Add some data
        metrics.record_l1_hit(0.05)
        metrics.record_l2_hit(0.20)
        metrics.record_miss()

        # Verify data exists
        assert metrics.total_requests == 3
        assert metrics.l1_hits == 1

        # Reset
        original_start_time = metrics.start_time
        metrics.reset()

        # Verify reset
        assert metrics.total_requests == 0
        assert metrics.l1_hits == 0
        assert metrics.l2_hits == 0
        assert metrics.misses == 0
        # Start time should be updated
        assert metrics.start_time != original_start_time