"""
Cache performance metrics tracking
"""

from datetime import datetime
from typing import Dict


class CacheMetrics:
    """Track cache performance metrics"""

    def __init__(self):
        self.l1_hits = 0
        self.l2_hits = 0
        self.misses = 0
        self.l1_sets = 0
        self.l2_sets = 0
        self.errors = 0
        self.total_requests = 0
        self.avg_l1_time = 0.0
        self.avg_l2_time = 0.0
        self.start_time = datetime.now()

    def record_l1_hit(self, response_time: float):
        """Record L1 cache hit with response time"""
        self.l1_hits += 1
        self.total_requests += 1
        self._update_avg_time("l1", response_time)

    def record_l2_hit(self, response_time: float):
        """Record L2 cache hit with response time"""
        self.l2_hits += 1
        self.total_requests += 1
        self._update_avg_time("l2", response_time)

    def record_miss(self):
        """Record cache miss"""
        self.misses += 1
        self.total_requests += 1

    def record_l1_set(self):
        """Record L1 cache set operation"""
        self.l1_sets += 1

    def record_l2_set(self):
        """Record L2 cache set operation"""
        self.l2_sets += 1

    def record_error(self):
        """Record cache error"""
        self.errors += 1

    def _update_avg_time(self, cache_level: str, response_time: float):
        """Update average response time for cache level"""
        if cache_level == "l1":
            # Simple moving average approximation
            if self.l1_hits == 1:
                self.avg_l1_time = response_time
            else:
                self.avg_l1_time = (self.avg_l1_time + response_time) / 2
        elif cache_level == "l2":
            if self.l2_hits == 1:
                self.avg_l2_time = response_time
            else:
                self.avg_l2_time = (self.avg_l2_time + response_time) / 2

    @property
    def hit_rate(self) -> float:
        """Calculate overall cache hit rate"""
        if self.total_requests == 0:
            return 0.0
        return (self.l1_hits + self.l2_hits) / self.total_requests

    @property
    def l1_hit_rate(self) -> float:
        """Calculate L1 cache hit rate"""
        if self.total_requests == 0:
            return 0.0
        return self.l1_hits / self.total_requests

    @property
    def l2_hit_rate(self) -> float:
        """Calculate L2 cache hit rate"""
        if self.total_requests == 0:
            return 0.0
        return self.l2_hits / self.total_requests

    @property
    def error_rate(self) -> float:
        """Calculate error rate"""
        if self.total_requests == 0:
            return 0.0
        return self.errors / self.total_requests

    def get_summary(self) -> Dict:
        """Get metrics summary"""
        uptime = (datetime.now() - self.start_time).total_seconds()

        return {
            "uptime_seconds": uptime,
            "total_requests": self.total_requests,
            "l1_hits": self.l1_hits,
            "l2_hits": self.l2_hits,
            "misses": self.misses,
            "errors": self.errors,
            "hit_rate": self.hit_rate,
            "l1_hit_rate": self.l1_hit_rate,
            "l2_hit_rate": self.l2_hit_rate,
            "error_rate": self.error_rate,
            "avg_l1_time_ms": self.avg_l1_time * 1000,
            "avg_l2_time_ms": self.avg_l2_time * 1000,
        }

    def reset(self):
        """Reset all metrics"""
        self.__init__()