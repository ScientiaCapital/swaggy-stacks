"""
Advanced Anomaly Detection System for Trading Markets

Sophisticated anomaly detector using DBSCAN to identify unusual market conditions,
potential black swan events, and early warning signals. Supports multi-timeframe
detection and integrates with risk management systems.
"""

import warnings
from typing import Dict, Any, Optional, Union, List, Tuple
from enum import Enum
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import structlog
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors

from .base import BaseUnsupervisedModel, UnsupervisedModelError
from .clustering import DensityBasedDetector

warnings.filterwarnings("ignore")
logger = structlog.get_logger()


class AnomalySeverity(Enum):
    """Anomaly severity levels"""
    LOW = "low"           # Score 0.0-0.3: Minor deviations
    MEDIUM = "medium"     # Score 0.3-0.6: Notable anomalies
    HIGH = "high"         # Score 0.6-0.8: Significant anomalies
    CRITICAL = "critical" # Score 0.8-1.0: Extreme anomalies/black swans


class AnomalyType(Enum):
    """Types of market anomalies detected"""
    VOLUME_SPIKE = "volume_spike"
    PRICE_GAP = "price_gap"
    CORRELATION_BREAK = "correlation_break"
    VOLATILITY_BURST = "volatility_burst"
    TREND_REVERSAL = "trend_reversal"
    DENSITY_OUTLIER = "density_outlier"
    MULTI_FACTOR = "multi_factor"


class AnomalyDetector(BaseUnsupervisedModel):
    """
    Advanced anomaly detection system for financial markets.

    Uses DBSCAN-based density analysis combined with specialized detectors
    for different types of market anomalies. Provides real-time scoring
    and early warning capabilities for risk management.
    """

    def __init__(
        self,
        lookback_period: int = 252,  # 1 year of trading data
        detection_timeframes: List[str] = None,  # ['1min', '5min', '1hour']
        anomaly_threshold: float = 0.7,
        critical_threshold: float = 0.9,
        early_warning_minutes: int = 5,
        max_false_positive_rate: float = 0.05,
        enable_scaling: bool = True,

        # DBSCAN parameters
        eps: float = None,  # Auto-tuned if None
        min_samples: int = 5,

        # Specific detector parameters
        volume_spike_threshold: float = 3.0,  # Standard deviations
        price_gap_threshold: float = 0.02,    # 2% price gap
        correlation_window: int = 20,
        correlation_threshold: float = 0.8
    ):
        """
        Initialize Advanced Anomaly Detector.

        Args:
            lookback_period: Historical periods for anomaly baseline
            detection_timeframes: List of timeframes for multi-scale detection
            anomaly_threshold: Threshold for anomaly classification (0-1)
            critical_threshold: Threshold for critical anomalies (0-1)
            early_warning_minutes: Minutes ahead for early warning alerts
            max_false_positive_rate: Maximum acceptable false positive rate
            enable_scaling: Whether to scale input features
            eps: DBSCAN epsilon parameter (auto-tuned if None)
            min_samples: DBSCAN minimum samples parameter
            volume_spike_threshold: Standard deviations for volume spike detection
            price_gap_threshold: Percentage threshold for price gap detection
            correlation_window: Window for correlation analysis
            correlation_threshold: Correlation break threshold
        """
        super().__init__(lookback_period, enable_scaling)

        self.detection_timeframes = detection_timeframes or ['1min', '5min', '1hour']
        self.anomaly_threshold = anomaly_threshold
        self.critical_threshold = critical_threshold
        self.early_warning_minutes = early_warning_minutes
        self.max_false_positive_rate = max_false_positive_rate

        # DBSCAN parameters
        self.eps = eps
        self.min_samples = min_samples

        # Specific detector parameters
        self.volume_spike_threshold = volume_spike_threshold
        self.price_gap_threshold = price_gap_threshold
        self.correlation_window = correlation_window
        self.correlation_threshold = correlation_threshold

        # Core detectors
        self.density_detector = None
        self.timeframe_detectors = {}

        # Scalers for different feature types
        self.price_scaler = RobustScaler() if enable_scaling else None
        self.volume_scaler = RobustScaler() if enable_scaling else None
        self.technical_scaler = RobustScaler() if enable_scaling else None

        # Anomaly tracking
        self.anomaly_history = []
        self.baseline_statistics = {}
        self.current_alerts = []

        # Performance tracking
        self.detection_accuracy = None
        self.false_positive_rate = None
        self.alert_lead_time = []

        self.model_params.update({
            'detection_timeframes': self.detection_timeframes,
            'anomaly_threshold': anomaly_threshold,
            'critical_threshold': critical_threshold,
            'volume_spike_threshold': volume_spike_threshold,
            'price_gap_threshold': price_gap_threshold
        })

        logger.info(
            "AnomalyDetector initialized",
            detection_timeframes=self.detection_timeframes,
            anomaly_threshold=anomaly_threshold,
            critical_threshold=critical_threshold
        )

    def _extract_anomaly_features(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Extract comprehensive features for anomaly detection.

        Args:
            data: Market data (OHLCV format preferred)

        Returns:
            Dictionary of feature arrays for different anomaly types
        """
        if isinstance(data, pd.DataFrame):
            if 'close' in data.columns:
                closes = data['close'].values
                highs = data.get('high', closes).values
                lows = data.get('low', closes).values
                volumes = data.get('volume', np.ones_like(closes)).values
                opens = data.get('open', closes).values
            else:
                closes = data.iloc[:, 0].values
                highs = lows = opens = closes
                volumes = np.ones_like(closes)
        else:
            data_array = np.asarray(data)
            closes = data_array[:, 0] if data_array.ndim > 1 else data_array
            highs = lows = opens = closes
            volumes = np.ones_like(closes)

        features = {}

        # 1. Price-based features
        returns = np.diff(closes) / closes[:-1]
        log_returns = np.diff(np.log(closes))

        # Price gaps (overnight gaps)
        price_gaps = (opens[1:] - closes[:-1]) / closes[:-1]

        # True ranges and volatility
        true_ranges = np.maximum(highs[1:] - lows[1:],
                                np.maximum(np.abs(highs[1:] - closes[:-1]),
                                          np.abs(lows[1:] - closes[:-1])))

        features['price'] = np.column_stack([
            returns,
            log_returns,
            price_gaps,
            true_ranges
        ])

        # 2. Volume-based features
        volume_returns = np.diff(volumes) / (volumes[:-1] + 1e-8)
        volume_ma = pd.Series(volumes).rolling(window=20).mean().values
        volume_relative = volumes / (volume_ma + 1e-8)

        features['volume'] = np.column_stack([
            volume_returns[1:],  # Align with price features
            volume_relative[1:]
        ])

        # 3. Technical indicator features
        # Rolling volatility
        rolling_vol = pd.Series(returns).rolling(window=10).std().fillna(0).values

        # Price momentum indicators
        momentum_5 = (closes[5:] - closes[:-5]) / closes[:-5]
        momentum_10 = (closes[10:] - closes[:-10]) / closes[:-10]

        # Align all arrays to same length
        min_length = min(len(rolling_vol), len(momentum_5), len(momentum_10))

        features['technical'] = np.column_stack([
            rolling_vol[-min_length:],
            momentum_5[-min_length:],
            momentum_10[-min_length:]
        ])

        # 4. Multi-timeframe features (simplified for now)
        # Calculate features at different scales
        features['multi_timeframe'] = self._calculate_multitimeframe_features(closes, volumes)

        return features

    def _calculate_multitimeframe_features(self, prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """Calculate features across multiple timeframes"""
        features = []

        # Different aggregation windows representing different timeframes
        windows = [1, 5, 15]  # Representing 1min, 5min, 15min in relative terms

        for window in windows:
            if len(prices) < window * 2:
                continue

            # Aggregate data
            aggregated_prices = []
            aggregated_volumes = []

            for i in range(window, len(prices), window):
                window_prices = prices[i-window:i]
                window_volumes = volumes[i-window:i]

                aggregated_prices.append(window_prices[-1])  # Close price
                aggregated_volumes.append(np.sum(window_volumes))  # Total volume

            if len(aggregated_prices) < 2:
                continue

            # Calculate returns and volatility for this timeframe
            agg_returns = np.diff(aggregated_prices) / np.array(aggregated_prices[:-1])
            agg_vol_changes = np.diff(aggregated_volumes) / (np.array(aggregated_volumes[:-1]) + 1e-8)

            # Take recent values
            recent_returns = agg_returns[-min(10, len(agg_returns)):]
            recent_vol_changes = agg_vol_changes[-min(10, len(agg_vol_changes)):]

            # Statistical features
            returns_mean = np.mean(recent_returns)
            returns_std = np.std(recent_returns)
            vol_mean = np.mean(recent_vol_changes)

            features.extend([returns_mean, returns_std, vol_mean])

        # Pad with zeros if insufficient data
        while len(features) < 9:  # 3 timeframes * 3 features each
            features.append(0.0)

        return np.array(features[:9])  # Ensure consistent size

    def _detect_volume_spikes(self, features: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect volume spikes using statistical thresholds.

        Args:
            features: Feature dictionary with volume data

        Returns:
            Tuple of (anomaly_mask, anomaly_scores)
        """
        volume_features = features['volume']
        if len(volume_features) == 0:
            return np.array([]), np.array([])

        volume_changes = volume_features[:, 0]  # Volume returns

        # Calculate rolling statistics
        window = min(20, len(volume_changes))
        if window < 5:
            return np.zeros(len(volume_changes), dtype=bool), np.zeros(len(volume_changes))

        rolling_mean = pd.Series(volume_changes).rolling(window=window).mean().fillna(0).values
        rolling_std = pd.Series(volume_changes).rolling(window=window).std().fillna(1).values

        # Z-score based spike detection
        z_scores = np.abs((volume_changes - rolling_mean) / (rolling_std + 1e-8))

        spike_mask = z_scores > self.volume_spike_threshold
        spike_scores = np.clip(z_scores / (self.volume_spike_threshold * 2), 0, 1)

        return spike_mask, spike_scores

    def _detect_price_gaps(self, features: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect significant price gaps.

        Args:
            features: Feature dictionary with price data

        Returns:
            Tuple of (anomaly_mask, anomaly_scores)
        """
        price_features = features['price']
        if len(price_features) < 3:
            return np.array([]), np.array([])

        price_gaps = price_features[:, 2]  # Price gaps column

        # Detect gaps exceeding threshold
        gap_mask = np.abs(price_gaps) > self.price_gap_threshold
        gap_scores = np.clip(np.abs(price_gaps) / self.price_gap_threshold, 0, 1)

        return gap_mask, gap_scores

    def _detect_correlation_breaks(self, features: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect correlation breakdowns between different market features.

        Args:
            features: Feature dictionary

        Returns:
            Tuple of (anomaly_mask, anomaly_scores)
        """
        if len(features['price']) < self.correlation_window:
            return np.array([]), np.array([])

        price_returns = features['price'][:, 0]  # Returns
        volume_changes = features['volume'][:, 0] if len(features['volume']) > 0 else np.zeros_like(price_returns)

        # Ensure same length
        min_length = min(len(price_returns), len(volume_changes))
        price_returns = price_returns[:min_length]
        volume_changes = volume_changes[:min_length]

        if min_length < self.correlation_window:
            return np.zeros(min_length, dtype=bool), np.zeros(min_length)

        # Rolling correlation
        correlations = []
        for i in range(self.correlation_window, min_length):
            window_price = price_returns[i-self.correlation_window:i]
            window_volume = volume_changes[i-self.correlation_window:i]

            # Calculate correlation
            corr = np.corrcoef(window_price, window_volume)[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0.0)

        correlations = np.array(correlations)

        # Detect correlation breaks
        baseline_correlation = np.median(correlations[:len(correlations)//2]) if len(correlations) > 10 else 0.0
        correlation_deviation = np.abs(correlations - baseline_correlation)

        break_threshold = (1 - self.correlation_threshold)
        break_mask = correlation_deviation > break_threshold
        break_scores = np.clip(correlation_deviation / break_threshold, 0, 1)

        # Pad to match original length
        full_break_mask = np.zeros(min_length, dtype=bool)
        full_break_scores = np.zeros(min_length)

        full_break_mask[self.correlation_window:] = break_mask
        full_break_scores[self.correlation_window:] = break_scores

        return full_break_mask, full_break_scores

    def _combine_anomaly_scores(self, detector_results: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """
        Combine anomaly scores from different detectors.

        Args:
            detector_results: Dictionary of (anomaly_mask, anomaly_scores) from each detector

        Returns:
            Combined anomaly analysis results
        """
        all_scores = []
        all_masks = []
        detector_contributions = {}

        # Get maximum length for alignment
        max_length = 0
        for detector_name, (mask, scores) in detector_results.items():
            if len(scores) > max_length:
                max_length = len(scores)

        if max_length == 0:
            return {
                'combined_scores': np.array([]),
                'combined_mask': np.array([]),
                'severity_levels': np.array([]),
                'anomaly_types': [],
                'detector_contributions': {}
            }

        # Align all detector results
        for detector_name, (mask, scores) in detector_results.items():
            if len(scores) == 0:
                continue

            # Pad shorter arrays
            aligned_scores = np.zeros(max_length)
            aligned_mask = np.zeros(max_length, dtype=bool)

            aligned_scores[:len(scores)] = scores
            aligned_mask[:len(mask)] = mask

            all_scores.append(aligned_scores)
            all_masks.append(aligned_mask)
            detector_contributions[detector_name] = aligned_scores

        if not all_scores:
            return {
                'combined_scores': np.array([]),
                'combined_mask': np.array([]),
                'severity_levels': np.array([]),
                'anomaly_types': [],
                'detector_contributions': {}
            }

        # Combine scores using weighted maximum
        all_scores = np.array(all_scores)
        combined_scores = np.max(all_scores, axis=0)

        # Any detector flagging an anomaly counts
        combined_mask = np.any(all_masks, axis=0)

        # Determine severity levels
        severity_levels = []
        for score in combined_scores:
            if score >= self.critical_threshold:
                severity_levels.append(AnomalySeverity.CRITICAL)
            elif score >= self.anomaly_threshold:
                severity_levels.append(AnomalySeverity.HIGH)
            elif score >= 0.3:
                severity_levels.append(AnomalySeverity.MEDIUM)
            else:
                severity_levels.append(AnomalySeverity.LOW)

        # Determine anomaly types for each point
        anomaly_types = []
        for i in range(max_length):
            types_for_point = []
            for detector_name, scores in detector_contributions.items():
                if scores[i] > self.anomaly_threshold:
                    if detector_name == 'volume_spikes':
                        types_for_point.append(AnomalyType.VOLUME_SPIKE)
                    elif detector_name == 'price_gaps':
                        types_for_point.append(AnomalyType.PRICE_GAP)
                    elif detector_name == 'correlation_breaks':
                        types_for_point.append(AnomalyType.CORRELATION_BREAK)
                    elif detector_name == 'density_outliers':
                        types_for_point.append(AnomalyType.DENSITY_OUTLIER)

            if len(types_for_point) > 1:
                anomaly_types.append(AnomalyType.MULTI_FACTOR)
            elif len(types_for_point) == 1:
                anomaly_types.append(types_for_point[0])
            else:
                anomaly_types.append(None)

        return {
            'combined_scores': combined_scores,
            'combined_mask': combined_mask,
            'severity_levels': severity_levels,
            'anomaly_types': anomaly_types,
            'detector_contributions': detector_contributions
        }

    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> 'AnomalyDetector':
        """
        Fit anomaly detector to historical market data.

        Args:
            data: Historical market data

        Returns:
            Self for method chaining
        """
        try:
            logger.info(
                "Starting anomaly detector training",
                data_points=len(data),
                detection_timeframes=self.detection_timeframes
            )

            # Extract features
            features = self._extract_anomaly_features(data)

            # Initialize and fit density-based detector
            self.density_detector = DensityBasedDetector(
                eps=self.eps,
                min_samples=self.min_samples,
                enable_scaling=False  # We handle scaling ourselves
            )

            # Create combined feature matrix for density detection
            combined_features = []
            for feature_type, feature_array in features.items():
                if len(feature_array) > 0:
                    if feature_array.ndim == 1:
                        combined_features.append(feature_array.reshape(-1, 1))
                    else:
                        combined_features.append(feature_array)

            if combined_features:
                # Find minimum length and align
                min_length = min(arr.shape[0] for arr in combined_features)
                aligned_features = []
                for arr in combined_features:
                    aligned_features.append(arr[:min_length])

                combined_matrix = np.hstack(aligned_features)

                # Scale features if enabled
                if self.enable_scaling:
                    # Scale different feature types separately for better results
                    if 'price' in features and len(features['price']) > 0:
                        if self.price_scaler is not None:
                            features['price'] = self.price_scaler.fit_transform(features['price'][:min_length])

                    if 'volume' in features and len(features['volume']) > 0:
                        if self.volume_scaler is not None:
                            features['volume'] = self.volume_scaler.fit_transform(features['volume'][:min_length])

                    if 'technical' in features and len(features['technical']) > 0:
                        if self.technical_scaler is not None:
                            features['technical'] = self.technical_scaler.fit_transform(features['technical'][:min_length])

                # Fit density detector
                self.density_detector.fit(combined_matrix)

            # Calculate baseline statistics for anomaly detection
            self._calculate_baseline_statistics(features)

            self.is_fitted = True

            logger.info(
                "Anomaly detector training completed",
                n_clusters=getattr(self.density_detector, 'n_clusters_', 0),
                n_noise_points=getattr(self.density_detector, 'n_noise_', 0)
            )

            return self

        except Exception as e:
            raise UnsupervisedModelError(f"Anomaly detector fitting failed: {str(e)}")

    def _calculate_baseline_statistics(self, features: Dict[str, np.ndarray]) -> None:
        """Calculate baseline statistics for anomaly thresholds"""
        self.baseline_statistics = {}

        for feature_type, feature_array in features.items():
            if len(feature_array) > 0:
                self.baseline_statistics[feature_type] = {
                    'mean': np.mean(feature_array, axis=0),
                    'std': np.std(feature_array, axis=0),
                    'median': np.median(feature_array, axis=0),
                    'q75': np.percentile(feature_array, 75, axis=0),
                    'q95': np.percentile(feature_array, 95, axis=0)
                }

    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """
        Detect anomalies in new market data.

        Args:
            data: Recent market data for anomaly detection

        Returns:
            Comprehensive anomaly detection results
        """
        self._validate_fitted()

        try:
            # Extract features
            features = self._extract_anomaly_features(data)

            # Run all anomaly detectors
            detector_results = {}

            # 1. Volume spike detection
            volume_mask, volume_scores = self._detect_volume_spikes(features)
            if len(volume_scores) > 0:
                detector_results['volume_spikes'] = (volume_mask, volume_scores)

            # 2. Price gap detection
            gap_mask, gap_scores = self._detect_price_gaps(features)
            if len(gap_scores) > 0:
                detector_results['price_gaps'] = (gap_mask, gap_scores)

            # 3. Correlation break detection
            corr_mask, corr_scores = self._detect_correlation_breaks(features)
            if len(corr_scores) > 0:
                detector_results['correlation_breaks'] = (corr_mask, corr_scores)

            # 4. Density-based outlier detection
            if self.density_detector is not None and self.density_detector.is_fitted:
                # Create combined feature matrix
                combined_features = []
                for feature_type, feature_array in features.items():
                    if len(feature_array) > 0:
                        if feature_array.ndim == 1:
                            combined_features.append(feature_array.reshape(-1, 1))
                        else:
                            combined_features.append(feature_array)

                if combined_features:
                    min_length = min(arr.shape[0] for arr in combined_features)
                    aligned_features = [arr[:min_length] for arr in combined_features]
                    combined_matrix = np.hstack(aligned_features)

                    density_scores = self.density_detector.get_anomaly_scores(combined_matrix)
                    density_mask = density_scores > self.anomaly_threshold
                    detector_results['density_outliers'] = (density_mask, density_scores)

            # Combine all results
            combined_results = self._combine_anomaly_scores(detector_results)

            # Generate alerts and warnings
            alerts = self._generate_alerts(combined_results)

            # Calculate early warning signals
            early_warnings = self._calculate_early_warnings(combined_results)

            result = {
                'anomaly_detected': np.any(combined_results['combined_mask']),
                'max_anomaly_score': np.max(combined_results['combined_scores']) if len(combined_results['combined_scores']) > 0 else 0.0,
                'anomaly_scores': combined_results['combined_scores'],
                'anomaly_mask': combined_results['combined_mask'],
                'severity_levels': [s.value if s else 'none' for s in combined_results['severity_levels']],
                'anomaly_types': [t.value if t else 'none' for t in combined_results['anomaly_types']],
                'detector_contributions': combined_results['detector_contributions'],
                'alerts': alerts,
                'early_warnings': early_warnings,
                'risk_assessment': self._assess_risk_level(combined_results),
                'detection_metadata': {
                    'detection_timestamp': pd.Timestamp.now().isoformat(),
                    'n_detectors_used': len(detector_results),
                    'detection_timeframes': self.detection_timeframes,
                    'data_points_analyzed': len(data)
                }
            }

            # Update anomaly history
            self._update_anomaly_history(result)

            logger.debug(
                "Anomaly detection completed",
                anomaly_detected=result['anomaly_detected'],
                max_score=result['max_anomaly_score'],
                n_alerts=len(alerts)
            )

            return result

        except Exception as e:
            raise UnsupervisedModelError(f"Anomaly detection failed: {str(e)}")

    def _generate_alerts(self, combined_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable alerts based on anomaly detection results"""
        alerts = []

        scores = combined_results['combined_scores']
        severity_levels = combined_results['severity_levels']
        anomaly_types = combined_results['anomaly_types']

        for i, (score, severity, anomaly_type) in enumerate(zip(scores, severity_levels, anomaly_types)):
            if score > self.anomaly_threshold:
                alert = {
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'alert_id': f"anomaly_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_{i}",
                    'anomaly_score': float(score),
                    'severity': severity.value if severity else 'unknown',
                    'anomaly_type': anomaly_type.value if anomaly_type else 'unknown',
                    'message': self._generate_alert_message(score, severity, anomaly_type),
                    'recommended_actions': self._get_recommended_actions(severity, anomaly_type),
                    'data_point_index': i
                }
                alerts.append(alert)

        return alerts

    def _generate_alert_message(self, score: float, severity: AnomalySeverity, anomaly_type: AnomalyType) -> str:
        """Generate human-readable alert message"""
        if severity == AnomalySeverity.CRITICAL:
            return f"CRITICAL ANOMALY DETECTED: {anomaly_type.value if anomaly_type else 'Unknown'} anomaly with score {score:.3f}. Potential black swan event."
        elif severity == AnomalySeverity.HIGH:
            return f"HIGH ANOMALY: {anomaly_type.value if anomaly_type else 'Unknown'} detected with score {score:.3f}. Immediate attention required."
        elif severity == AnomalySeverity.MEDIUM:
            return f"MEDIUM ANOMALY: {anomaly_type.value if anomaly_type else 'Unknown'} detected with score {score:.3f}. Monitor closely."
        else:
            return f"LOW ANOMALY: {anomaly_type.value if anomaly_type else 'Unknown'} detected with score {score:.3f}."

    def _get_recommended_actions(self, severity: AnomalySeverity, anomaly_type: AnomalyType) -> List[str]:
        """Get recommended actions based on anomaly characteristics"""
        actions = []

        if severity == AnomalySeverity.CRITICAL:
            actions.extend([
                "Immediately reduce position sizes",
                "Activate emergency risk protocols",
                "Consider market exit if trend continues",
                "Alert risk management team"
            ])
        elif severity == AnomalySeverity.HIGH:
            actions.extend([
                "Reduce position sizes by 50%",
                "Tighten stop losses",
                "Increase monitoring frequency",
                "Review hedging strategies"
            ])
        elif severity == AnomalySeverity.MEDIUM:
            actions.extend([
                "Monitor position risks closely",
                "Consider reducing leverage",
                "Review correlation exposures"
            ])

        if anomaly_type == AnomalyType.VOLUME_SPIKE:
            actions.append("Check for news or market events")
        elif anomaly_type == AnomalyType.PRICE_GAP:
            actions.append("Review overnight risk exposure")
        elif anomaly_type == AnomalyType.CORRELATION_BREAK:
            actions.append("Reassess portfolio correlation assumptions")

        return actions

    def _calculate_early_warnings(self, combined_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate early warning signals"""
        scores = combined_results['combined_scores']

        if len(scores) == 0:
            return {'early_warning_active': False}

        # Look for increasing anomaly trends
        recent_scores = scores[-min(10, len(scores)):]
        trend_slope = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0] if len(recent_scores) > 1 else 0

        # Early warning if anomaly scores are trending upward
        early_warning_active = (
            trend_slope > 0.01 and  # Positive trend
            np.mean(recent_scores) > 0.4  # Above warning threshold
        )

        return {
            'early_warning_active': early_warning_active,
            'trend_slope': float(trend_slope),
            'recent_average_score': float(np.mean(recent_scores)),
            'minutes_to_potential_anomaly': self.early_warning_minutes if early_warning_active else None
        }

    def _assess_risk_level(self, combined_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall risk level based on anomaly detection"""
        scores = combined_results['combined_scores']

        if len(scores) == 0:
            return {'risk_level': 'unknown', 'risk_score': 0.0}

        max_score = np.max(scores)
        avg_score = np.mean(scores)

        # Risk level determination
        if max_score >= self.critical_threshold:
            risk_level = 'critical'
        elif max_score >= self.anomaly_threshold:
            risk_level = 'high'
        elif avg_score > 0.3:
            risk_level = 'medium'
        else:
            risk_level = 'low'

        return {
            'risk_level': risk_level,
            'risk_score': float(max_score),
            'average_anomaly_score': float(avg_score),
            'recommendation': self._get_risk_recommendation(risk_level)
        }

    def _get_risk_recommendation(self, risk_level: str) -> str:
        """Get risk management recommendation"""
        recommendations = {
            'critical': 'IMMEDIATE ACTION REQUIRED: Reduce all positions and activate emergency protocols',
            'high': 'HIGH RISK: Reduce position sizes and increase monitoring',
            'medium': 'MODERATE RISK: Monitor closely and review risk limits',
            'low': 'LOW RISK: Continue normal operations with standard monitoring'
        }
        return recommendations.get(risk_level, 'Monitor market conditions')

    def update(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """
        Update anomaly detector with new market data.

        Args:
            data: New market data to incorporate
        """
        if not self.is_fitted:
            logger.warning("Anomaly detector not fitted, performing full fit")
            self.fit(data)
            return

        try:
            # Update density detector
            if self.density_detector is not None:
                self.density_detector.update(data)

            # Update baseline statistics with new data
            features = self._extract_anomaly_features(data)
            self._update_baseline_statistics(features)

            logger.debug("Anomaly detector updated incrementally")

        except Exception as e:
            logger.warning(f"Incremental update failed: {str(e)}, performing full refit")
            self.fit(data)

    def _update_baseline_statistics(self, new_features: Dict[str, np.ndarray]) -> None:
        """Update baseline statistics with new data"""
        decay_factor = 0.95  # Exponential decay for adaptation

        for feature_type, new_feature_array in new_features.items():
            if feature_type in self.baseline_statistics and len(new_feature_array) > 0:
                old_stats = self.baseline_statistics[feature_type]
                new_mean = np.mean(new_feature_array, axis=0)
                new_std = np.std(new_feature_array, axis=0)

                # Exponential moving average update
                self.baseline_statistics[feature_type]['mean'] = (
                    decay_factor * old_stats['mean'] + (1 - decay_factor) * new_mean
                )
                self.baseline_statistics[feature_type]['std'] = (
                    decay_factor * old_stats['std'] + (1 - decay_factor) * new_std
                )

    def _update_anomaly_history(self, detection_result: Dict[str, Any]) -> None:
        """Update anomaly detection history"""
        self.anomaly_history.append({
            'timestamp': pd.Timestamp.now(),
            'max_score': detection_result['max_anomaly_score'],
            'anomaly_detected': detection_result['anomaly_detected'],
            'n_alerts': len(detection_result['alerts']),
            'risk_level': detection_result['risk_assessment']['risk_level']
        })

        # Keep only recent history
        if len(self.anomaly_history) > 1000:
            self.anomaly_history = self.anomaly_history[-1000:]

    def get_detection_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of anomaly detection performance and status.

        Returns:
            Dictionary with detection summary
        """
        self._validate_fitted()

        summary = {
            'detection_parameters': {
                'anomaly_threshold': self.anomaly_threshold,
                'critical_threshold': self.critical_threshold,
                'detection_timeframes': self.detection_timeframes,
                'early_warning_minutes': self.early_warning_minutes
            },
            'detection_status': {
                'is_fitted': self.is_fitted,
                'n_clusters': getattr(self.density_detector, 'n_clusters_', 0) if self.density_detector else 0,
                'n_noise_points': getattr(self.density_detector, 'n_noise_', 0) if self.density_detector else 0
            },
            'performance_metrics': {
                'detection_accuracy': self.detection_accuracy,
                'false_positive_rate': self.false_positive_rate,
                'avg_alert_lead_time': np.mean(self.alert_lead_time) if self.alert_lead_time else None
            },
            'recent_activity': {
                'total_detections': len(self.anomaly_history),
                'recent_anomalies': len([h for h in self.anomaly_history if h['anomaly_detected']]),
                'current_alerts': len(self.current_alerts)
            }
        }

        return summary

    def get_risk_integration_data(self) -> Dict[str, Any]:
        """
        Get anomaly data formatted for risk management system integration.

        Returns:
            Dictionary compatible with risk management integration
        """
        if not self.is_fitted:
            return {
                'anomaly_system_active': False,
                'current_risk_multiplier': 1.0,
                'alert_level': 'none',
                'recommended_position_adjustment': 0.0
            }

        # Get recent anomaly activity
        recent_history = self.anomaly_history[-10:] if self.anomaly_history else []
        recent_anomaly_rate = len([h for h in recent_history if h['anomaly_detected']]) / max(len(recent_history), 1)

        # Calculate risk multiplier based on recent activity
        if recent_anomaly_rate > 0.5:  # More than 50% recent anomalies
            risk_multiplier = 1.5
            alert_level = 'high'
            position_adjustment = -0.3  # Reduce positions by 30%
        elif recent_anomaly_rate > 0.2:  # More than 20% recent anomalies
            risk_multiplier = 1.2
            alert_level = 'medium'
            position_adjustment = -0.1  # Reduce positions by 10%
        else:
            risk_multiplier = 1.0
            alert_level = 'low'
            position_adjustment = 0.0

        return {
            'anomaly_system_active': True,
            'current_risk_multiplier': risk_multiplier,
            'alert_level': alert_level,
            'recommended_position_adjustment': position_adjustment,
            'recent_anomaly_rate': recent_anomaly_rate,
            'active_alerts': len(self.current_alerts),
            'last_detection_time': max([h['timestamp'] for h in self.anomaly_history]) if self.anomaly_history else None
        }