"""
Dimensionality reduction techniques for trading feature engineering.

Implements PCA, autoencoders, and t-SNE for compressing high-dimensional trading
indicators into meaningful lower-dimensional representations.
"""

import warnings
import os
from typing import Dict, Any, Optional, Union, List, Tuple

import numpy as np
import pandas as pd
import structlog
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import joblib

# Try to import TensorFlow/Keras, fallback gracefully if not available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    keras = None
    layers = None
    Model = None

from .base import BaseUnsupervisedModel, IncrementalLearner, UnsupervisedModelError

warnings.filterwarnings("ignore")
logger = structlog.get_logger()


class IncrementalPCAReducer(BaseUnsupervisedModel, IncrementalLearner):
    """
    Incremental PCA for streaming dimensionality reduction.

    Optimized for real-time trading where new data arrives continuously
    and full retraining is not feasible.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        batch_size: int = 100,
        min_variance_ratio: float = 0.95,
        whiten: bool = True,
        lookback_period: int = 1000,
        enable_scaling: bool = True
    ):
        """
        Initialize Incremental PCA reducer.

        Args:
            n_components: Number of components (auto-selected if None)
            batch_size: Batch size for incremental updates
            min_variance_ratio: Minimum explained variance to retain
            whiten: Whether to whiten the components
            lookback_period: Number of periods for analysis
            enable_scaling: Whether to scale input data
        """
        super().__init__(lookback_period, enable_scaling)

        self.n_components = n_components
        self.batch_size = batch_size
        self.min_variance_ratio = min_variance_ratio
        self.whiten = whiten
        self.learning_rate = 1.0  # For incremental learning interface

        self.model = None
        self.explained_variance_ratio_ = None
        self.cumulative_variance_ratio_ = None
        self.n_features_in_ = None
        self.n_samples_seen_ = 0

        self.model_params.update({
            'n_components': n_components,
            'batch_size': batch_size,
            'min_variance_ratio': min_variance_ratio,
            'whiten': whiten
        })

        logger.info(
            "IncrementalPCAReducer initialized",
            n_components=n_components,
            min_variance_ratio=min_variance_ratio,
            batch_size=batch_size
        )

    def _determine_optimal_components(self, n_features: int) -> int:
        """
        Determine optimal number of components based on variance ratio.

        Args:
            n_features: Number of input features

        Returns:
            Optimal number of components
        """
        if self.n_components is not None:
            return min(self.n_components, n_features)

        # Use a reasonable default based on feature count
        if n_features <= 10:
            return max(2, n_features // 2)
        elif n_features <= 50:
            return min(10, n_features // 3)
        else:
            return min(20, n_features // 4)

    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> 'IncrementalPCAReducer':
        """
        Fit incremental PCA to training data.

        Args:
            data: Training data

        Returns:
            Self for method chaining
        """
        try:
            processed_data = self._preprocess_data(data)
            self.n_features_in_ = processed_data.shape[1]

            # Determine optimal components
            optimal_components = self._determine_optimal_components(self.n_features_in_)

            # Initialize model
            self.model = IncrementalPCA(
                n_components=optimal_components,
                batch_size=self.batch_size,
                whiten=self.whiten
            )

            # Fit in batches
            n_samples = len(processed_data)
            for i in range(0, n_samples, self.batch_size):
                batch = processed_data[i:i + self.batch_size]
                self.model.partial_fit(batch)

            # Update tracking variables
            self.explained_variance_ratio_ = self.model.explained_variance_ratio_
            self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)
            self.n_samples_seen_ = n_samples
            self.is_fitted = True

            # Check if we need to adjust components for variance requirement
            if self.cumulative_variance_ratio_[-1] < self.min_variance_ratio:
                logger.warning(
                    "Variance ratio below threshold",
                    achieved_ratio=self.cumulative_variance_ratio_[-1],
                    target_ratio=self.min_variance_ratio,
                    suggestion="Consider increasing n_components"
                )

            logger.info(
                "IncrementalPCA fitted successfully",
                n_components=optimal_components,
                explained_variance=self.cumulative_variance_ratio_[-1],
                n_samples=n_samples,
                n_features=self.n_features_in_
            )

            return self

        except Exception as e:
            raise UnsupervisedModelError(f"PCA fitting failed: {str(e)}")

    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Transform data to reduced dimensions.

        Args:
            data: Input data to transform

        Returns:
            Transformed data in reduced dimensions
        """
        self._validate_fitted()

        try:
            processed_data = self._preprocess_data(data)
            transformed = self.model.transform(processed_data)

            logger.debug(
                "PCA transformation completed",
                original_shape=processed_data.shape,
                transformed_shape=transformed.shape
            )

            return transformed

        except Exception as e:
            raise UnsupervisedModelError(f"PCA transformation failed: {str(e)}")

    def update(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """
        Incrementally update PCA with new data.

        Args:
            data: New data to incorporate
        """
        self._validate_fitted()

        try:
            processed_data = self._preprocess_data(data)

            # Update in batches
            n_samples = len(processed_data)
            for i in range(0, n_samples, self.batch_size):
                batch = processed_data[i:i + self.batch_size]
                self.model.partial_fit(batch)

            # Update tracking
            self.explained_variance_ratio_ = self.model.explained_variance_ratio_
            self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)
            self.n_samples_seen_ += n_samples

            logger.debug(
                "PCA model updated",
                new_samples=n_samples,
                total_samples=self.n_samples_seen_,
                explained_variance=self.cumulative_variance_ratio_[-1]
            )

        except Exception as e:
            raise UnsupervisedModelError(f"PCA update failed: {str(e)}")

    def partial_fit(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """Alias for update method"""
        self.update(data)

    def get_learning_rate(self) -> float:
        """Get learning rate"""
        return self.learning_rate

    def set_learning_rate(self, learning_rate: float) -> None:
        """Set learning rate"""
        self.learning_rate = learning_rate

    def inverse_transform(self, transformed_data: np.ndarray) -> np.ndarray:
        """
        Transform data back to original space.

        Args:
            transformed_data: Data in reduced dimensions

        Returns:
            Data reconstructed in original dimensions
        """
        self._validate_fitted()

        try:
            return self.model.inverse_transform(transformed_data)
        except Exception as e:
            raise UnsupervisedModelError(f"PCA inverse transform failed: {str(e)}")

    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance based on component loadings.

        Returns:
            Feature importance scores
        """
        self._validate_fitted()

        try:
            # Calculate feature importance from components
            components = np.abs(self.model.components_)
            weighted_components = components * self.explained_variance_ratio_[:, np.newaxis]
            importance = np.sum(weighted_components, axis=0)

            # Normalize to [0, 1]
            if np.max(importance) > 0:
                importance = importance / np.max(importance)

            return importance

        except Exception as e:
            raise UnsupervisedModelError(f"Feature importance calculation failed: {str(e)}")


class AutoencoderReducer(BaseUnsupervisedModel):
    """
    Autoencoder-based dimensionality reduction for pattern compression.

    Uses neural networks to learn non-linear compressed representations
    of trading patterns for pattern memory storage.
    """

    def __init__(
        self,
        encoding_dim: Optional[int] = None,
        hidden_layers: List[int] = None,
        activation: str = 'relu',
        output_activation: str = 'linear',
        optimizer: str = 'adam',
        loss: str = 'mse',
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        patience: int = 10,
        lookback_period: int = 1000,
        enable_scaling: bool = True,
        model_save_path: Optional[str] = None
    ):
        """
        Initialize Autoencoder reducer.

        Args:
            encoding_dim: Dimension of encoded representation
            hidden_layers: List of hidden layer sizes
            activation: Activation function for hidden layers
            output_activation: Activation for output layer
            optimizer: Optimizer for training
            loss: Loss function
            epochs: Training epochs
            batch_size: Training batch size
            validation_split: Validation data fraction
            patience: Early stopping patience
            lookback_period: Number of periods for analysis
            enable_scaling: Whether to scale input data
            model_save_path: Path to save/load model
        """
        super().__init__(lookback_period, enable_scaling)

        if not TENSORFLOW_AVAILABLE:
            raise UnsupervisedModelError(
                "TensorFlow not available. Install with: pip install tensorflow"
            )

        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers or [128, 64]
        self.activation = activation
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.patience = patience
        self.model_save_path = model_save_path

        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.history = None
        self.reconstruction_error_ = None

        self.model_params.update({
            'encoding_dim': encoding_dim,
            'hidden_layers': hidden_layers,
            'activation': activation,
            'epochs': epochs,
            'batch_size': batch_size
        })

        logger.info(
            "AutoencoderReducer initialized",
            encoding_dim=encoding_dim,
            hidden_layers=hidden_layers,
            epochs=epochs
        )

    def _build_autoencoder(self, input_dim: int) -> None:
        """
        Build autoencoder architecture.

        Args:
            input_dim: Input dimension
        """
        # Determine encoding dimension
        if self.encoding_dim is None:
            self.encoding_dim = max(2, input_dim // 4)

        # Input layer
        input_layer = keras.Input(shape=(input_dim,))

        # Encoder
        encoded = input_layer
        for units in self.hidden_layers:
            encoded = layers.Dense(units, activation=self.activation)(encoded)

        # Encoding layer
        encoded = layers.Dense(self.encoding_dim, activation=self.activation, name='encoding')(encoded)

        # Decoder (mirror of encoder)
        decoded = encoded
        for units in reversed(self.hidden_layers):
            decoded = layers.Dense(units, activation=self.activation)(decoded)

        # Output layer
        decoded = layers.Dense(input_dim, activation=self.output_activation)(decoded)

        # Models
        self.autoencoder = Model(input_layer, decoded, name='autoencoder')
        self.encoder = Model(input_layer, encoded, name='encoder')

        # Decoder model (for generating from encoded space)
        encoded_input = keras.Input(shape=(self.encoding_dim,))
        decoder_layers = self.autoencoder.layers[-(len(self.hidden_layers) + 1):]
        decoded_output = encoded_input
        for layer in decoder_layers:
            decoded_output = layer(decoded_output)
        self.decoder = Model(encoded_input, decoded_output, name='decoder')

        # Compile
        self.autoencoder.compile(optimizer=self.optimizer, loss=self.loss, metrics=['mae'])

        logger.info(
            "Autoencoder architecture built",
            input_dim=input_dim,
            encoding_dim=self.encoding_dim,
            total_params=self.autoencoder.count_params()
        )

    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> 'AutoencoderReducer':
        """
        Fit autoencoder to training data.

        Args:
            data: Training data

        Returns:
            Self for method chaining
        """
        try:
            processed_data = self._preprocess_data(data)
            input_dim = processed_data.shape[1]

            # Build architecture
            self._build_autoencoder(input_dim)

            # Callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.patience,
                    restore_best_weights=True
                )
            ]

            if self.model_save_path:
                callbacks.append(
                    keras.callbacks.ModelCheckpoint(
                        self.model_save_path,
                        monitor='val_loss',
                        save_best_only=True
                    )
                )

            # Train
            self.history = self.autoencoder.fit(
                processed_data, processed_data,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                callbacks=callbacks,
                verbose=0
            )

            # Calculate reconstruction error
            reconstructed = self.autoencoder.predict(processed_data, verbose=0)
            self.reconstruction_error_ = np.mean(np.square(processed_data - reconstructed))

            self.is_fitted = True

            logger.info(
                "Autoencoder fitted successfully",
                encoding_dim=self.encoding_dim,
                reconstruction_error=self.reconstruction_error_,
                epochs_trained=len(self.history.history['loss']),
                final_loss=self.history.history['loss'][-1]
            )

            return self

        except Exception as e:
            raise UnsupervisedModelError(f"Autoencoder fitting failed: {str(e)}")

    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Encode data to reduced dimensions.

        Args:
            data: Input data to encode

        Returns:
            Encoded representations
        """
        self._validate_fitted()

        try:
            processed_data = self._preprocess_data(data)
            encoded = self.encoder.predict(processed_data, verbose=0)

            logger.debug(
                "Autoencoder encoding completed",
                original_shape=processed_data.shape,
                encoded_shape=encoded.shape
            )

            return encoded

        except Exception as e:
            raise UnsupervisedModelError(f"Autoencoder encoding failed: {str(e)}")

    def update(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """
        Update autoencoder with new data (requires retraining).

        Args:
            data: New data to incorporate
        """
        logger.warning(
            "Autoencoder requires full retraining for updates",
            recommendation="Consider incremental training techniques"
        )
        self.fit(data)

    def reconstruct(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Reconstruct data from input through encoder-decoder.

        Args:
            data: Input data to reconstruct

        Returns:
            Reconstructed data
        """
        self._validate_fitted()

        try:
            processed_data = self._preprocess_data(data)
            reconstructed = self.autoencoder.predict(processed_data, verbose=0)

            return reconstructed

        except Exception as e:
            raise UnsupervisedModelError(f"Autoencoder reconstruction failed: {str(e)}")

    def decode(self, encoded_data: np.ndarray) -> np.ndarray:
        """
        Decode from encoded space back to original dimensions.

        Args:
            encoded_data: Data in encoded space

        Returns:
            Decoded data
        """
        self._validate_fitted()

        try:
            return self.decoder.predict(encoded_data, verbose=0)
        except Exception as e:
            raise UnsupervisedModelError(f"Autoencoder decoding failed: {str(e)}")

    def save_model(self, filepath: str) -> None:
        """
        Save trained autoencoder model.

        Args:
            filepath: Path to save model
        """
        self._validate_fitted()

        try:
            self.autoencoder.save(filepath)
            logger.info("Autoencoder model saved", filepath=filepath)
        except Exception as e:
            raise UnsupervisedModelError(f"Model saving failed: {str(e)}")

    def load_model(self, filepath: str) -> None:
        """
        Load pre-trained autoencoder model.

        Args:
            filepath: Path to load model from
        """
        try:
            self.autoencoder = keras.models.load_model(filepath)

            # Extract encoder and decoder
            encoding_layer_name = 'encoding'
            if encoding_layer_name in [layer.name for layer in self.autoencoder.layers]:
                encoding_layer = self.autoencoder.get_layer(encoding_layer_name)
                self.encoder = Model(
                    self.autoencoder.input,
                    encoding_layer.output
                )

            self.is_fitted = True
            logger.info("Autoencoder model loaded", filepath=filepath)

        except Exception as e:
            raise UnsupervisedModelError(f"Model loading failed: {str(e)}")


class TSNEVisualizer(BaseUnsupervisedModel):
    """
    t-SNE for 2D visualization of high-dimensional trading states.

    Optimized for creating interpretable visualizations of market conditions
    and trading patterns. Not suitable for real-time processing.
    """

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        early_exaggeration: float = 12.0,
        learning_rate: float = 200.0,
        n_iter: int = 1000,
        metric: str = 'euclidean',
        init: str = 'random',
        random_state: int = 42,
        lookback_period: int = 1000,
        enable_scaling: bool = True
    ):
        """
        Initialize t-SNE visualizer.

        Args:
            n_components: Dimension of embedded space (usually 2)
            perplexity: Related to number of nearest neighbors
            early_exaggeration: Controls cluster tightness in early iterations
            learning_rate: Learning rate for optimization
            n_iter: Maximum iterations
            metric: Distance metric
            init: Initialization method
            random_state: Random seed
            lookback_period: Number of periods for analysis
            enable_scaling: Whether to scale input data
        """
        super().__init__(lookback_period, enable_scaling)

        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.metric = metric
        self.init = init
        self.random_state = random_state

        self.model = None
        self.embedding_ = None
        self.kl_divergence_ = None

        self.model_params.update({
            'n_components': n_components,
            'perplexity': perplexity,
            'learning_rate': learning_rate,
            'n_iter': n_iter
        })

        logger.info(
            "TSNEVisualizer initialized",
            n_components=n_components,
            perplexity=perplexity,
            n_iter=n_iter
        )

    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> 'TSNEVisualizer':
        """
        Fit t-SNE to data and create embedding.

        Args:
            data: Input data for visualization

        Returns:
            Self for method chaining
        """
        try:
            processed_data = self._preprocess_data(data)

            # Adjust perplexity if needed
            adjusted_perplexity = min(self.perplexity, (len(processed_data) - 1) / 3)

            if adjusted_perplexity != self.perplexity:
                logger.warning(
                    "Adjusted perplexity for dataset size",
                    original_perplexity=self.perplexity,
                    adjusted_perplexity=adjusted_perplexity,
                    n_samples=len(processed_data)
                )

            # Create and fit t-SNE
            self.model = TSNE(
                n_components=self.n_components,
                perplexity=adjusted_perplexity,
                early_exaggeration=self.early_exaggeration,
                learning_rate=self.learning_rate,
                n_iter=self.n_iter,
                metric=self.metric,
                init=self.init,
                random_state=self.random_state
            )

            self.embedding_ = self.model.fit_transform(processed_data)
            self.kl_divergence_ = self.model.kl_divergence_

            self.is_fitted = True

            logger.info(
                "t-SNE visualization completed",
                embedding_shape=self.embedding_.shape,
                kl_divergence=self.kl_divergence_,
                perplexity_used=adjusted_perplexity
            )

            return self

        except Exception as e:
            raise UnsupervisedModelError(f"t-SNE fitting failed: {str(e)}")

    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Get embedding for fitted data.

        Note: t-SNE doesn't support out-of-sample prediction.
        This returns the fitted embedding.

        Args:
            data: Input data (unused, returns fitted embedding)

        Returns:
            2D embedding coordinates
        """
        self._validate_fitted()

        logger.warning(
            "t-SNE doesn't support out-of-sample prediction",
            recommendation="Refit with combined data for new points"
        )

        return self.embedding_

    def update(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """
        Update visualization with new data (requires refitting).

        Args:
            data: New data to include in visualization
        """
        logger.warning(
            "t-SNE requires full refitting for updates",
            recommendation="Use for batch visualization only"
        )
        self.fit(data)

    def get_embedding(self) -> np.ndarray:
        """
        Get the 2D embedding coordinates.

        Returns:
            Embedding coordinates
        """
        self._validate_fitted()
        return self.embedding_

    def plot_embedding(
        self,
        labels: Optional[np.ndarray] = None,
        title: str = "t-SNE Visualization",
        figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """
        Plot the t-SNE embedding.

        Args:
            labels: Optional cluster labels for coloring
            title: Plot title
            figsize: Figure size
        """
        self._validate_fitted()

        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=figsize)

            if labels is not None:
                scatter = plt.scatter(
                    self.embedding_[:, 0],
                    self.embedding_[:, 1],
                    c=labels,
                    cmap='tab10',
                    alpha=0.7
                )
                plt.colorbar(scatter)
            else:
                plt.scatter(
                    self.embedding_[:, 0],
                    self.embedding_[:, 1],
                    alpha=0.7
                )

            plt.title(title)
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.grid(True, alpha=0.3)
            plt.show()

        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error("Plotting failed", error=str(e))