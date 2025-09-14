"""
Comprehensive Test Suite for Unsupervised Learning System
Production-grade testing for algorithmic trading platform
"""

# Import all test modules for easy access
from .test_clustering import (
    TestKMeansClusterer,
    TestHierarchicalClusterer,
    TestDBSCANClusterer,
    TestMarketDataClusterer,
    TestClusteringIntegration
)

from .test_market_regime import (
    TestMarketRegimeDetector,
    TestRegimeTransitionPredictor,
    TestRegimeStabilityAnalyzer,
    TestMarketRegimeIntegration
)

from .test_anomaly_detection import (
    TestMarketAnomalyDetector,
    TestDBSCANAnomalyDetector,
    TestIsolationForestAnomalyDetector,
    TestEnsembleAnomalyDetector,
    TestAnomalyDetectionIntegration
)

from .test_integration import (
    TestUnsupervisedSystemIntegration
)

from .test_performance_benchmarks import (
    TestPerformanceBenchmarks,
    TestProductionReadinessBenchmarks
)

__all__ = [
    # Clustering tests
    'TestKMeansClusterer',
    'TestHierarchicalClusterer',
    'TestDBSCANClusterer',
    'TestMarketDataClusterer',
    'TestClusteringIntegration',

    # Market regime tests
    'TestMarketRegimeDetector',
    'TestRegimeTransitionPredictor',
    'TestRegimeStabilityAnalyzer',
    'TestMarketRegimeIntegration',

    # Anomaly detection tests
    'TestMarketAnomalyDetector',
    'TestDBSCANAnomalyDetector',
    'TestIsolationForestAnomalyDetector',
    'TestEnsembleAnomalyDetector',
    'TestAnomalyDetectionIntegration',

    # Integration tests
    'TestUnsupervisedSystemIntegration',

    # Performance benchmarks
    'TestPerformanceBenchmarks',
    'TestProductionReadinessBenchmarks'
]