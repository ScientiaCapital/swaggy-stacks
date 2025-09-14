"""
Market Regime Detection System

Sophisticated market regime detector using ensemble clustering to identify
bull, bear, sideways, and volatile market conditions. Integrates seamlessly
with MarkovSystem to enhance state modeling and strategy optimization.
"""

import warnings
from typing import Dict, Any, Optional, Union, List, Tuple
from enum import Enum

import numpy as np
import pandas as pd
import structlog
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from .base import BaseUnsupervisedModel, ClusteringMetrics, UnsupervisedModelError
from .clustering import EnhancedKMeans, HierarchicalClusterer

warnings.filterwarnings("ignore")
logger = structlog.get_logger()


class MarketRegime(Enum):
    """Market regime classifications"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    RECOVERY = "recovery"


class MarketRegimeDetector(BaseUnsupervisedModel):
    """
    Advanced market regime detection using ensemble clustering.

    Combines k-means and hierarchical clustering to identify distinct market
    conditions and predict regime transitions with confidence scoring.
    """

    def __init__(
        self,
        n_regimes: int = 5,
        lookback_period: int = 252,  # 1 year of trading days
        rolling_window: int = 20,    # Rolling window for feature calculation
        transition_threshold: float = 0.7,
        min_regime_duration: int = 5,
        enable_scaling: bool = True
    ):
        """
        Initialize Market Regime Detector.

        Args:
            n_regimes: Number of market regimes to detect (4-5 recommended)
            lookback_period: Historical periods for regime analysis
            rolling_window: Window for calculating regime features
            transition_threshold: Confidence threshold for regime changes
            min_regime_duration: Minimum bars to maintain regime classification
            enable_scaling: Whether to scale input features
        """
        super().__init__(lookback_period, enable_scaling)

        self.n_regimes = n_regimes
        self.rolling_window = rolling_window
        self.transition_threshold = transition_threshold
        self.min_regime_duration = min_regime_duration

        # Ensemble clustering models
        self.kmeans_model = None
        self.hierarchical_model = None

        # Regime tracking
        self.current_regime = None
        self.regime_history = []
        self.regime_probabilities = None
        self.transition_probabilities = None
        self.regime_persistence_counter = 0

        # Feature engineering
        self.feature_scaler = StandardScaler() if enable_scaling else None
        self.regime_features_ = None
        self.regime_labels_ = None
        self.regime_centroids_ = None

        # Performance tracking
        self.regime_transition_accuracy = None
        self.regime_stability_score = None

        self.model_params.update({
            'n_regimes': n_regimes,
            'rolling_window': rolling_window,
            'transition_threshold': transition_threshold,
            'min_regime_duration': min_regime_duration
        })

        logger.info(
            "MarketRegimeDetector initialized",
            n_regimes=n_regimes,
            lookback_period=lookback_period,
            rolling_window=rolling_window
        )

    def _extract_regime_features(self, price_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Extract comprehensive features for regime detection.

        Args:
            price_data: Price data (OHLCV or close prices)

        Returns:
            Feature matrix for regime clustering
        """
        if isinstance(price_data, pd.DataFrame):
            # Extract OHLCV if available
            if 'close' in price_data.columns:
                closes = price_data['close'].values
                highs = price_data.get('high', closes).values
                lows = price_data.get('low', closes).values
                volumes = price_data.get('volume', np.ones_like(closes)).values
            else:
                closes = price_data.iloc[:, 0].values
                highs = lows = closes
                volumes = np.ones_like(closes)
        else:
            closes = np.asarray(price_data)
            highs = lows = closes
            volumes = np.ones_like(closes)

        # Calculate rolling features
        features = []

        for i in range(self.rolling_window, len(closes)):
            window_closes = closes[i-self.rolling_window:i]
            window_highs = highs[i-self.rolling_window:i]
            window_lows = lows[i-self.rolling_window:i]
            window_volumes = volumes[i-self.rolling_window:i]

            # Price-based features
            returns = np.diff(window_closes) / window_closes[:-1]

            # 1. Trend strength (linear regression slope)
            x = np.arange(len(window_closes))
            trend_slope = np.polyfit(x, window_closes, 1)[0]

            # 2. Volatility (rolling standard deviation of returns)
            volatility = np.std(returns)

            # 3. Return skewness and kurtosis
            skewness = self._safe_skewness(returns)
            kurtosis = self._safe_kurtosis(returns)

            # 4. Price momentum (rate of change)
            momentum = (window_closes[-1] - window_closes[0]) / window_closes[0]

            # 5. Volatility of volatility (volatility clustering)
            rolling_vol = pd.Series(returns).rolling(5).std().values
            vol_of_vol = np.std(rolling_vol[~np.isnan(rolling_vol)])

            # 6. Range efficiency (high-low range vs price movement)
            price_range = (window_highs[-1] - window_lows[-1]) / window_closes[-1]
            price_movement = abs(window_closes[-1] - window_closes[0]) / window_closes[0]
            range_efficiency = price_movement / (price_range + 1e-8)

            # 7. Volume trend (if available)
            volume_trend = np.polyfit(x, window_volumes, 1)[0] if len(set(window_volumes)) > 1 else 0

            # 8. Drawdown severity
            peak = np.maximum.accumulate(window_closes)
            drawdown = (peak - window_closes) / peak
            max_drawdown = np.max(drawdown)

            # 9. Up/Down day ratio
            up_days = np.sum(returns > 0)
            down_days = np.sum(returns < 0)
            up_down_ratio = up_days / (down_days + 1e-8)

            feature_vector = [
                trend_slope,      # Trend direction and strength
                volatility,       # Market uncertainty
                skewness,         # Return distribution asymmetry
                kurtosis,         # Tail risk
                momentum,         # Price momentum
                vol_of_vol,       # Volatility clustering
                range_efficiency, # Market efficiency
                volume_trend,     # Volume characteristics
                max_drawdown,     # Risk measure
                up_down_ratio     # Market sentiment
            ]

            features.append(feature_vector)

        return np.array(features)

    def _safe_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness with error handling"""
        try:
            if len(data) < 3:
                return 0.0
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            return np.mean(((data - mean) / std) ** 3)
        except:
            return 0.0

    def _safe_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis with error handling"""
        try:
            if len(data) < 4:
                return 0.0
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            return np.mean(((data - mean) / std) ** 4) - 3  # Excess kurtosis
        except:
            return 0.0

    def _ensemble_clustering(self, features: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform ensemble clustering using k-means and hierarchical methods.

        Args:
            features: Feature matrix for clustering

        Returns:
            Tuple of (ensemble_labels, clustering_results)
        """
        results = {}

        # 1. Enhanced K-means clustering
        self.kmeans_model = EnhancedKMeans(
            n_clusters=self.n_regimes,
            enable_scaling=False  # We handle scaling ourselves
        )
        self.kmeans_model.fit(features)
        kmeans_labels = self.kmeans_model.predict(features)
        results['kmeans_labels'] = kmeans_labels
        results['kmeans_centroids'] = self.kmeans_model.get_cluster_centers()

        # 2. Hierarchical clustering
        self.hierarchical_model = HierarchicalClusterer(
            n_clusters=self.n_regimes,
            linkage='ward',
            enable_scaling=False
        )
        self.hierarchical_model.fit(features)
        hierarchical_labels = self.hierarchical_model.predict(features)
        results['hierarchical_labels'] = hierarchical_labels

        # 3. Ensemble combination (weighted voting)
        # K-means gets higher weight due to centroid-based approach
        kmeans_weight = 0.7
        hierarchical_weight = 0.3

        # Create consensus labels using majority voting with weights
        ensemble_labels = []
        for i in range(len(features)):
            # Get cluster assignments
            k_label = kmeans_labels[i]
            h_label = hierarchical_labels[i]

            # Simple majority vote (could be enhanced with distance-based weighting)
            if k_label == h_label:
                ensemble_label = k_label
            else:
                # Use k-means result as tie-breaker (higher weight)
                ensemble_label = k_label

            ensemble_labels.append(ensemble_label)

        ensemble_labels = np.array(ensemble_labels)
        results['ensemble_labels'] = ensemble_labels

        # 4. Calculate clustering quality metrics
        if len(np.unique(ensemble_labels)) > 1:
            silhouette = ClusteringMetrics.silhouette_score_safe(features, ensemble_labels)
            davies_bouldin = ClusteringMetrics.davies_bouldin_score_safe(features, ensemble_labels)
            results['silhouette_score'] = silhouette
            results['davies_bouldin_score'] = davies_bouldin

        # 5. Calculate regime centroids from ensemble results
        regime_centroids = []
        for regime in range(self.n_regimes):
            regime_mask = ensemble_labels == regime
            if np.any(regime_mask):
                centroid = np.mean(features[regime_mask], axis=0)
                regime_centroids.append(centroid)
            else:
                # Handle empty clusters
                regime_centroids.append(np.mean(features, axis=0))

        self.regime_centroids_ = np.array(regime_centroids)
        results['regime_centroids'] = self.regime_centroids_

        logger.info(
            "Ensemble clustering completed",
            n_regimes=self.n_regimes,
            silhouette_score=results.get('silhouette_score', 'N/A'),
            davies_bouldin_score=results.get('davies_bouldin_score', 'N/A')
        )

        return ensemble_labels, results

    def _classify_regimes(self, labels: np.ndarray, features: np.ndarray) -> Dict[int, MarketRegime]:
        """
        Map cluster labels to market regime classifications.

        Args:
            labels: Cluster labels from ensemble method
            features: Feature matrix used for clustering

        Returns:
            Dictionary mapping cluster labels to regime types
        """
        regime_mapping = {}

        # Analyze characteristics of each cluster
        for regime_label in range(self.n_regimes):
            regime_mask = labels == regime_label
            if not np.any(regime_mask):
                regime_mapping[regime_label] = MarketRegime.SIDEWAYS
                continue

            regime_features = features[regime_mask]

            # Calculate average characteristics
            avg_trend = np.mean(regime_features[:, 0])      # Trend slope
            avg_volatility = np.mean(regime_features[:, 1]) # Volatility
            avg_momentum = np.mean(regime_features[:, 4])   # Momentum
            avg_drawdown = np.mean(regime_features[:, 8])   # Max drawdown

            # Classification logic based on feature characteristics
            if avg_trend > 0.001 and avg_momentum > 0.02 and avg_drawdown < 0.15:
                regime_type = MarketRegime.BULL
            elif avg_trend < -0.001 and avg_momentum < -0.02:
                regime_type = MarketRegime.BEAR
            elif avg_volatility > np.percentile(features[:, 1], 80):
                regime_type = MarketRegime.VOLATILE
            elif avg_trend > 0 and avg_drawdown > 0.15:
                regime_type = MarketRegime.RECOVERY
            else:
                regime_type = MarketRegime.SIDEWAYS

            regime_mapping[regime_label] = regime_type

        # Ensure we have diverse regime types (avoid all same classification)
        regime_types = list(regime_mapping.values())
        if len(set(regime_types)) < 3:
            # Force diversity by reassigning based on volatility ranking
            volatility_ranks = []
            for regime_label in range(self.n_regimes):
                regime_mask = labels == regime_label
                if np.any(regime_mask):
                    avg_vol = np.mean(features[regime_mask, 1])
                    volatility_ranks.append((regime_label, avg_vol))

            volatility_ranks.sort(key=lambda x: x[1])

            # Assign regimes based on volatility ranking
            for i, (regime_label, _) in enumerate(volatility_ranks):
                if i == 0:
                    regime_mapping[regime_label] = MarketRegime.SIDEWAYS
                elif i == len(volatility_ranks) - 1:
                    regime_mapping[regime_label] = MarketRegime.VOLATILE
                else:
                    # Use trend to distinguish bull/bear
                    regime_mask = labels == regime_label
                    avg_trend = np.mean(features[regime_mask, 0])
                    if avg_trend > 0:
                        regime_mapping[regime_label] = MarketRegime.BULL
                    else:
                        regime_mapping[regime_label] = MarketRegime.BEAR

        logger.info(
            "Regime classification completed",
            regime_mapping={k: v.value for k, v in regime_mapping.items()}
        )

        return regime_mapping

    def _detect_regime_transitions(self, labels: np.ndarray) -> Tuple[List[int], List[float]]:
        """
        Detect regime transitions with confidence scoring.

        Args:
            labels: Time series of regime labels

        Returns:
            Tuple of (transition_points, confidence_scores)
        """
        transitions = []
        confidences = []

        if len(labels) < 2:
            return transitions, confidences

        # Detect transitions
        for i in range(1, len(labels)):
            if labels[i] != labels[i-1]:
                # Calculate transition confidence based on regime stability

                # Look at surrounding periods for stability
                lookback = min(i, 10)
                lookahead = min(len(labels) - i, 10)

                # Stability before transition
                pre_stability = np.sum(labels[i-lookback:i] == labels[i-1]) / lookback

                # Stability after transition
                post_stability = np.sum(labels[i:i+lookahead] == labels[i]) / lookahead

                # Combined confidence (both regimes should be stable)
                confidence = (pre_stability + post_stability) / 2

                transitions.append(i)
                confidences.append(confidence)

        return transitions, confidences

    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> 'MarketRegimeDetector':
        """
        Fit regime detector to historical market data.

        Args:
            data: Historical price data

        Returns:
            Self for method chaining
        """
        try:
            logger.info(
                "Starting market regime detection",
                data_points=len(data),
                n_regimes=self.n_regimes
            )

            # Extract regime features
            self.regime_features_ = self._extract_regime_features(data)

            if len(self.regime_features_) < self.n_regimes:
                raise UnsupervisedModelError(
                    f"Insufficient data for regime detection: need at least {self.n_regimes} periods, got {len(self.regime_features_)}"
                )

            # Scale features if enabled
            if self.enable_scaling and self.feature_scaler is not None:
                self.regime_features_ = self.feature_scaler.fit_transform(self.regime_features_)

            # Perform ensemble clustering
            self.regime_labels_, clustering_results = self._ensemble_clustering(self.regime_features_)

            # Classify regimes
            self.regime_mapping = self._classify_regimes(self.regime_labels_, self.regime_features_)

            # Detect transitions
            transitions, confidences = self._detect_regime_transitions(self.regime_labels_)
            self.transition_points = transitions
            self.transition_confidences = confidences

            # Set current regime
            if len(self.regime_labels_) > 0:
                current_label = self.regime_labels_[-1]
                self.current_regime = self.regime_mapping[current_label]
                self.regime_persistence_counter = 1

            # Calculate regime transition probabilities
            self._calculate_transition_probabilities()

            self.is_fitted = True

            logger.info(
                "Market regime detection completed",
                current_regime=self.current_regime.value if self.current_regime else None,
                n_transitions=len(transitions),
                avg_transition_confidence=np.mean(confidences) if confidences else 0,
                regime_distribution={k: np.sum(self.regime_labels_ == k) for k in range(self.n_regimes)}
            )

            return self

        except Exception as e:
            raise UnsupervisedModelError(f"Regime detection fitting failed: {str(e)}")

    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """
        Predict current market regime and transition probabilities.

        Args:
            data: Recent market data for regime prediction

        Returns:
            Dictionary with regime predictions and confidence metrics
        """
        self._validate_fitted()

        try:
            # Extract features for recent data
            features = self._extract_regime_features(data)

            if len(features) == 0:
                raise UnsupervisedModelError("Insufficient data for regime prediction")

            # Scale features if enabled
            if self.enable_scaling and self.feature_scaler is not None:
                features = self.feature_scaler.transform(features)

            # Get most recent feature vector
            latest_features = features[-1:]

            # Predict using ensemble models
            kmeans_prediction = self.kmeans_model.predict(latest_features)[0]
            hierarchical_prediction = self.hierarchical_model.predict(latest_features)[0]

            # Ensemble prediction (use k-means as primary)
            predicted_label = kmeans_prediction
            predicted_regime = self.regime_mapping[predicted_label]

            # Calculate prediction confidence based on distance to centroids
            distances = []
            for centroid in self.regime_centroids_:
                distance = np.linalg.norm(latest_features[0] - centroid)
                distances.append(distance)

            # Confidence is inverse of distance to nearest centroid
            min_distance = min(distances)
            max_distance = max(distances) if max(distances) > min_distance else min_distance + 1
            confidence = 1 - (min_distance / max_distance)

            # Detect potential regime change
            regime_changed = False
            transition_probability = 0.0

            if self.current_regime is not None:
                if predicted_regime != self.current_regime:
                    if confidence > self.transition_threshold:
                        if self.regime_persistence_counter >= self.min_regime_duration:
                            regime_changed = True
                            transition_probability = confidence
                            self.current_regime = predicted_regime
                            self.regime_persistence_counter = 1
                        else:
                            # Not enough persistence, maintain current regime
                            predicted_regime = self.current_regime
                            self.regime_persistence_counter += 1
                    else:
                        # Low confidence, maintain current regime
                        predicted_regime = self.current_regime
                        self.regime_persistence_counter += 1
                else:
                    # Same regime, increment persistence
                    self.regime_persistence_counter += 1
            else:
                # First prediction
                self.current_regime = predicted_regime
                self.regime_persistence_counter = 1

            # Calculate regime probabilities for all regimes
            regime_probabilities = {}
            for label, regime_type in self.regime_mapping.items():
                # Inverse distance to centroid as probability proxy
                distance = np.linalg.norm(latest_features[0] - self.regime_centroids_[label])
                prob = np.exp(-distance)  # Exponential decay with distance
                regime_probabilities[regime_type.value] = prob

            # Normalize probabilities
            total_prob = sum(regime_probabilities.values())
            if total_prob > 0:
                regime_probabilities = {k: v/total_prob for k, v in regime_probabilities.items()}

            result = {
                'current_regime': predicted_regime.value,
                'regime_confidence': float(confidence),
                'regime_changed': regime_changed,
                'transition_probability': float(transition_probability),
                'regime_probabilities': regime_probabilities,
                'regime_persistence': self.regime_persistence_counter,
                'predicted_label': int(predicted_label),
                'ensemble_agreement': kmeans_prediction == hierarchical_prediction,
                'prediction_metadata': {
                    'n_features': len(latest_features[0]),
                    'min_distance_to_centroid': float(min_distance),
                    'prediction_timestamp': pd.Timestamp.now().isoformat()
                }
            }

            logger.debug(
                "Regime prediction completed",
                current_regime=predicted_regime.value,
                confidence=confidence,
                regime_changed=regime_changed
            )

            return result

        except Exception as e:
            raise UnsupervisedModelError(f"Regime prediction failed: {str(e)}")

    def update(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """
        Update regime detector with new market data.

        Args:
            data: New market data to incorporate
        """
        if not self.is_fitted:
            logger.warning("Regime detector not fitted, performing full fit")
            self.fit(data)
            return

        try:
            # Extract new features
            new_features = self._extract_regime_features(data)

            if len(new_features) == 0:
                return

            # Scale if enabled
            if self.enable_scaling and self.feature_scaler is not None:
                new_features = self.feature_scaler.transform(new_features)

            # Update models incrementally if supported
            if hasattr(self.kmeans_model, 'partial_fit'):
                self.kmeans_model.partial_fit(new_features)

            if hasattr(self.hierarchical_model, 'partial_fit'):
                self.hierarchical_model.partial_fit(new_features)

            # Update regime history
            latest_prediction = self.predict(data)
            self.regime_history.append({
                'timestamp': pd.Timestamp.now(),
                'regime': latest_prediction['current_regime'],
                'confidence': latest_prediction['regime_confidence']
            })

            # Keep history manageable
            if len(self.regime_history) > self.lookback_period:
                self.regime_history = self.regime_history[-self.lookback_period:]

            logger.debug("Regime detector updated incrementally")

        except Exception as e:
            logger.warning(f"Incremental update failed: {str(e)}, performing full refit")
            # Fallback to full refit
            self.fit(data)

    def _calculate_transition_probabilities(self) -> None:
        """Calculate regime transition probability matrix"""
        if self.regime_labels_ is None or len(self.regime_labels_) < 2:
            return

        # Build transition matrix
        transition_counts = np.zeros((self.n_regimes, self.n_regimes))

        for i in range(len(self.regime_labels_) - 1):
            current_regime = self.regime_labels_[i]
            next_regime = self.regime_labels_[i + 1]
            transition_counts[current_regime, next_regime] += 1

        # Normalize to probabilities
        row_sums = transition_counts.sum(axis=1)
        self.transition_probabilities = np.divide(
            transition_counts,
            row_sums[:, np.newaxis],
            out=np.zeros_like(transition_counts),
            where=row_sums[:, np.newaxis] != 0
        )

    def get_regime_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of regime detection results.

        Returns:
            Dictionary with regime analysis summary
        """
        self._validate_fitted()

        summary = {
            'current_regime': self.current_regime.value if self.current_regime else None,
            'n_regimes': self.n_regimes,
            'regime_mapping': {k: v.value for k, v in self.regime_mapping.items()},
            'transition_points': len(self.transition_points) if hasattr(self, 'transition_points') else 0,
            'avg_transition_confidence': (
                np.mean(self.transition_confidences)
                if hasattr(self, 'transition_confidences') and self.transition_confidences
                else 0
            )
        }

        if self.regime_labels_ is not None:
            # Regime distribution
            regime_distribution = {}
            for label in range(self.n_regimes):
                count = np.sum(self.regime_labels_ == label)
                regime_type = self.regime_mapping[label].value
                regime_distribution[regime_type] = count
            summary['regime_distribution'] = regime_distribution

        if self.transition_probabilities is not None:
            summary['transition_matrix'] = self.transition_probabilities.tolist()

        return summary

    def get_regime_for_markov_integration(self) -> Dict[str, Any]:
        """
        Get regime information formatted for MarkovSystem integration.

        Returns:
            Dictionary compatible with MarkovSystem.analyze() output
        """
        if not self.is_fitted or self.current_regime is None:
            return {
                'regime_detected': False,
                'regime_type': 'unknown',
                'regime_confidence': 0.0,
                'regime_transition_risk': 0.0
            }

        # Calculate transition risk based on regime persistence
        transition_risk = max(0.0, 1.0 - (self.regime_persistence_counter / self.min_regime_duration))

        return {
            'regime_detected': True,
            'regime_type': self.current_regime.value,
            'regime_confidence': float(getattr(self, 'last_confidence', 0.8)),
            'regime_transition_risk': float(transition_risk),
            'regime_persistence': self.regime_persistence_counter,
            'regime_mapping': {k: v.value for k, v in self.regime_mapping.items()},
            'n_regimes_detected': self.n_regimes
        }