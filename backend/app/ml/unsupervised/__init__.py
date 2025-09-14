"""
Unsupervised Learning Module for SwaggyStacks Trading System

This module provides comprehensive unsupervised learning capabilities including:
- Clustering algorithms (k-means, hierarchical, DBSCAN)
- Dimensionality reduction (PCA, autoencoders, t-SNE)
- Pattern discovery and mining
- Market regime detection
- Anomaly detection

All implementations are optimized for real-time trading with incremental learning capabilities.
"""

from .base import (
    BaseUnsupervisedModel,
    ClusteringMetrics,
    IncrementalLearner,
    UnsupervisedModelError
)

from .clustering import (
    EnhancedKMeans,
    HierarchicalClusterer,
    DensityBasedDetector
)

from .reduction import (
    IncrementalPCAReducer,
    AutoencoderReducer,
    TSNEVisualizer
)

from .feature_engineer import FeatureEngineer

from .market_regime import (
    MarketRegimeDetector,
    MarketRegime
)

__all__ = [
    "BaseUnsupervisedModel",
    "ClusteringMetrics",
    "IncrementalLearner",
    "UnsupervisedModelError",
    "EnhancedKMeans",
    "HierarchicalClusterer",
    "DensityBasedDetector",
    "IncrementalPCAReducer",
    "AutoencoderReducer",
    "TSNEVisualizer",
    "FeatureEngineer",
    "MarketRegimeDetector",
    "MarketRegime"
]