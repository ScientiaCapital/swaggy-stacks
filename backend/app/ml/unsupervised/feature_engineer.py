"""
Automated feature engineering using dimensionality reduction techniques.

Provides intelligent feature selection and engineering using PCA analysis
to automatically reduce 50+ trading indicators to optimal feature sets.
"""

import warnings
from typing import Dict, Any, Optional, Union, List, Tuple

import numpy as np
import pandas as pd
import structlog
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_regression
from sklearn.preprocessing import StandardScaler

from .base import BaseUnsupervisedModel, UnsupervisedModelError
from .reduction import IncrementalPCAReducer

warnings.filterwarnings("ignore")
logger = structlog.get_logger()


class FeatureEngineer(BaseUnsupervisedModel):
    """
    Automated feature engineering using PCA and statistical analysis.

    Intelligently reduces high-dimensional trading indicators to optimal
    feature sets while preserving maximum information content.
    """

    def __init__(
        self,
        target_features: int = 10,
        variance_threshold: float = 0.95,
        correlation_threshold: float = 0.95,
        mutual_info_threshold: float = 0.1,
        pca_method: str = 'incremental',
        feature_selection_method: str = 'combined',
        lookback_period: int = 1000,
        enable_scaling: bool = True
    ):
        """
        Initialize Feature Engineer.

        Args:
            target_features: Target number of features to select
            variance_threshold: Minimum variance to retain in PCA
            correlation_threshold: Remove features with correlation above this
            mutual_info_threshold: Minimum mutual information to retain
            pca_method: 'incremental' or 'standard' PCA
            feature_selection_method: 'pca', 'statistical', or 'combined'
            lookback_period: Number of periods for analysis
            enable_scaling: Whether to scale input data
        """
        super().__init__(lookback_period, enable_scaling)

        self.target_features = target_features
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.mutual_info_threshold = mutual_info_threshold
        self.pca_method = pca_method
        self.feature_selection_method = feature_selection_method

        # Models and selectors
        self.pca_reducer = None
        self.feature_selector = None
        self.correlation_filter = None

        # Results
        self.selected_features_ = None
        self.feature_importance_ = None
        self.pca_components_ = None
        self.explained_variance_ratio_ = None
        self.feature_names_ = None

        self.model_params.update({
            'target_features': target_features,
            'variance_threshold': variance_threshold,
            'feature_selection_method': feature_selection_method
        })

        logger.info(
            "FeatureEngineer initialized",
            target_features=target_features,
            variance_threshold=variance_threshold,
            selection_method=feature_selection_method
        )

    def _remove_correlated_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove highly correlated features.

        Args:
            data: Input DataFrame with feature columns

        Returns:
            DataFrame with correlated features removed
        """
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        # Calculate correlation matrix
        corr_matrix = data.corr().abs()

        # Find pairs of highly correlated features
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > self.correlation_threshold:
                    # Keep the feature with higher variance
                    var_i = data.iloc[:, i].var()
                    var_j = data.iloc[:, j].var()
                    remove_idx = j if var_i > var_j else i
                    high_corr_pairs.append(remove_idx)

        # Remove duplicates and sort
        features_to_remove = sorted(list(set(high_corr_pairs)), reverse=True)

        # Remove highly correlated features
        filtered_data = data.drop(data.columns[features_to_remove], axis=1)

        logger.info(
            "Correlation filtering completed",
            original_features=len(data.columns),
            removed_features=len(features_to_remove),
            remaining_features=len(filtered_data.columns)
        )

        return filtered_data

    def _remove_low_variance_features(self, data: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        """
        Remove features with low variance.

        Args:
            data: Input DataFrame
            threshold: Minimum variance threshold

        Returns:
            DataFrame with low variance features removed
        """
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        # Calculate variance for each feature
        variances = data.var()
        low_var_features = variances[variances < threshold].index

        # Remove low variance features
        filtered_data = data.drop(low_var_features, axis=1)

        logger.info(
            "Low variance filtering completed",
            original_features=len(data.columns),
            removed_features=len(low_var_features),
            remaining_features=len(filtered_data.columns)
        )

        return filtered_data

    def _pca_feature_selection(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use PCA for feature selection and dimensionality reduction.

        Args:
            data: Input data

        Returns:
            Tuple of (transformed_data, feature_importance)
        """
        # Initialize PCA reducer
        if self.pca_method == 'incremental':
            self.pca_reducer = IncrementalPCAReducer(
                n_components=self.target_features,
                min_variance_ratio=self.variance_threshold,
                enable_scaling=False  # We handle scaling ourselves
            )
        else:
            from sklearn.decomposition import PCA
            self.pca_reducer = PCA(
                n_components=self.target_features,
                random_state=42
            )

        # Fit and transform
        if hasattr(self.pca_reducer, 'fit'):
            self.pca_reducer.fit(data)
            transformed_data = self.pca_reducer.predict(data)
        else:
            transformed_data = self.pca_reducer.fit_transform(data)

        # Get feature importance
        if hasattr(self.pca_reducer, 'get_feature_importance'):
            feature_importance = self.pca_reducer.get_feature_importance()
        elif hasattr(self.pca_reducer, 'components_'):
            # Calculate importance from components
            components = np.abs(self.pca_reducer.components_)
            if hasattr(self.pca_reducer, 'explained_variance_ratio_'):
                weights = self.pca_reducer.explained_variance_ratio_
                feature_importance = np.sum(components * weights[:, np.newaxis], axis=0)
            else:
                feature_importance = np.sum(components, axis=0)
            feature_importance = feature_importance / np.max(feature_importance)
        else:
            feature_importance = np.ones(data.shape[1]) / data.shape[1]

        # Store PCA results
        if hasattr(self.pca_reducer, 'explained_variance_ratio_'):
            self.explained_variance_ratio_ = self.pca_reducer.explained_variance_ratio_
        if hasattr(self.pca_reducer, 'components_'):
            self.pca_components_ = self.pca_reducer.components_

        return transformed_data, feature_importance

    def _statistical_feature_selection(self, data: np.ndarray, target: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use statistical methods for feature selection.

        Args:
            data: Input data
            target: Optional target variable for supervised selection

        Returns:
            Tuple of (selected_indices, feature_scores)
        """
        n_features = min(self.target_features, data.shape[1])

        if target is not None:
            # Supervised feature selection
            selector = SelectKBest(score_func=f_classif, k=n_features)
            selector.fit(data, target)
            selected_indices = selector.get_support(indices=True)
            feature_scores = selector.scores_
        else:
            # Unsupervised: use variance-based selection
            variances = np.var(data, axis=0)
            selected_indices = np.argsort(variances)[-n_features:]
            feature_scores = variances

        # Normalize scores
        if np.max(feature_scores) > 0:
            feature_scores = feature_scores / np.max(feature_scores)

        return selected_indices, feature_scores

    def _combined_feature_selection(self, data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Combine multiple feature selection methods.

        Args:
            data: Input data

        Returns:
            Tuple of (final_features, selection_results)
        """
        results = {}

        # 1. PCA-based selection
        pca_features, pca_importance = self._pca_feature_selection(data)
        results['pca_features'] = pca_features
        results['pca_importance'] = pca_importance

        # 2. Statistical selection (variance-based)
        stat_indices, stat_scores = self._statistical_feature_selection(data)
        results['statistical_indices'] = stat_indices
        results['statistical_scores'] = stat_scores

        # 3. Combine results
        # Weight PCA components by importance
        weighted_pca = pca_features * pca_importance[np.newaxis, :]

        # Create final feature set
        if len(pca_features[0]) >= self.target_features:
            # Use PCA features directly
            final_features = pca_features
            self.feature_importance_ = pca_importance
        else:
            # Supplement with statistical selection
            n_additional = self.target_features - len(pca_features[0])
            additional_indices = stat_indices[:n_additional]

            # Combine features
            final_features = np.column_stack([
                pca_features,
                data[:, additional_indices]
            ])

            # Combine importance scores
            combined_importance = np.concatenate([
                pca_importance,
                stat_scores[additional_indices]
            ])
            self.feature_importance_ = combined_importance

        results['final_features'] = final_features
        results['final_importance'] = self.feature_importance_

        return final_features, results

    def fit(self, data: Union[pd.DataFrame, np.ndarray], target: Optional[np.ndarray] = None) -> 'FeatureEngineer':
        """
        Fit feature engineer to training data.

        Args:
            data: Training data with features
            target: Optional target variable for supervised selection

        Returns:
            Self for method chaining
        """
        try:
            # Convert to DataFrame if needed
            if isinstance(data, np.ndarray):
                data = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(data.shape[1])])

            self.feature_names_ = list(data.columns)

            logger.info(
                "Starting feature engineering",
                original_features=len(data.columns),
                target_features=self.target_features,
                method=self.feature_selection_method
            )

            # Step 1: Remove low variance features
            data_filtered = self._remove_low_variance_features(data)

            # Step 2: Remove highly correlated features
            data_filtered = self._remove_correlated_features(data_filtered)

            # Step 3: Scale data if enabled
            processed_data = self._preprocess_data(data_filtered)

            # Step 4: Apply feature selection method
            if self.feature_selection_method == 'pca':
                final_features, pca_importance = self._pca_feature_selection(processed_data)
                self.feature_importance_ = pca_importance
                results = {'method': 'pca', 'importance': pca_importance}

            elif self.feature_selection_method == 'statistical':
                selected_indices, stat_scores = self._statistical_feature_selection(processed_data, target)
                final_features = processed_data[:, selected_indices]
                self.feature_importance_ = stat_scores[selected_indices]
                self.selected_features_ = selected_indices
                results = {'method': 'statistical', 'indices': selected_indices, 'scores': stat_scores}

            elif self.feature_selection_method == 'combined':
                final_features, results = self._combined_feature_selection(processed_data)

            else:
                raise UnsupervisedModelError(f"Unknown feature selection method: {self.feature_selection_method}")

            self.is_fitted = True

            # Log results
            final_variance = np.sum(self.explained_variance_ratio_) if self.explained_variance_ratio_ is not None else "N/A"

            logger.info(
                "Feature engineering completed",
                original_features=len(data.columns),
                final_features=final_features.shape[1],
                explained_variance=final_variance,
                method=self.feature_selection_method
            )

            return self

        except Exception as e:
            raise UnsupervisedModelError(f"Feature engineering failed: {str(e)}")

    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Transform new data using fitted feature selection.

        Args:
            data: Input data to transform

        Returns:
            Transformed data with selected features
        """
        self._validate_fitted()

        try:
            # Convert to DataFrame if needed
            if isinstance(data, np.ndarray):
                data = pd.DataFrame(data, columns=self.feature_names_ or [f'feature_{i}' for i in range(data.shape[1])])

            # Apply same preprocessing steps
            data_filtered = self._remove_low_variance_features(data)
            data_filtered = self._remove_correlated_features(data_filtered)
            processed_data = self._preprocess_data(data_filtered)

            # Apply feature selection
            if self.feature_selection_method == 'pca' or self.pca_reducer is not None:
                if hasattr(self.pca_reducer, 'predict'):
                    transformed_data = self.pca_reducer.predict(processed_data)
                else:
                    transformed_data = self.pca_reducer.transform(processed_data)

            elif self.selected_features_ is not None:
                transformed_data = processed_data[:, self.selected_features_]

            else:
                # Fallback: return first target_features
                transformed_data = processed_data[:, :self.target_features]

            logger.debug(
                "Feature transformation completed",
                original_shape=processed_data.shape,
                transformed_shape=transformed_data.shape
            )

            return transformed_data

        except Exception as e:
            raise UnsupervisedModelError(f"Feature transformation failed: {str(e)}")

    def update(self, data: Union[pd.DataFrame, np.ndarray], target: Optional[np.ndarray] = None) -> None:
        """
        Update feature selection with new data.

        Args:
            data: New data to incorporate
            target: Optional target variable
        """
        if self.pca_reducer is not None and hasattr(self.pca_reducer, 'update'):
            # Update PCA incrementally
            processed_data = self._preprocess_data(data)
            self.pca_reducer.update(processed_data)
            self.feature_importance_ = self.pca_reducer.get_feature_importance()

            logger.debug("Feature engineering updated incrementally")
        else:
            logger.warning(
                "Full refit required for feature engineering update",
                recommendation="Use incremental PCA for online updates"
            )
            # Full refit required
            self.fit(data, target)

    def get_feature_ranking(self) -> List[Tuple[int, float]]:
        """
        Get feature ranking by importance.

        Returns:
            List of (feature_index, importance_score) tuples sorted by importance
        """
        self._validate_fitted()

        if self.feature_importance_ is None:
            return []

        ranking = list(enumerate(self.feature_importance_))
        ranking.sort(key=lambda x: x[1], reverse=True)

        return ranking

    def get_selection_summary(self) -> Dict[str, Any]:
        """
        Get summary of feature selection results.

        Returns:
            Dictionary with selection summary
        """
        self._validate_fitted()

        summary = {
            'method': self.feature_selection_method,
            'target_features': self.target_features,
            'final_features': len(self.feature_importance_) if self.feature_importance_ is not None else 0,
            'variance_threshold': self.variance_threshold,
            'correlation_threshold': self.correlation_threshold
        }

        if self.explained_variance_ratio_ is not None:
            summary['explained_variance'] = np.sum(self.explained_variance_ratio_)
            summary['variance_per_component'] = self.explained_variance_ratio_.tolist()

        if self.feature_importance_ is not None:
            summary['top_features'] = self.get_feature_ranking()[:5]  # Top 5 features

        return summary