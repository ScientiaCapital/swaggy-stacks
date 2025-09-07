"""
Embedding Service Factory with lazy loading
Handles conditional imports and service selection based on configuration
"""

import logging
import os
from typing import Optional

from app.rag.services.embedding_base import EmbeddingServiceInterface
from app.rag.services.mock_embedding import MockEmbeddingService

logger = logging.getLogger(__name__)

# Global singleton instances
_embedding_service: Optional[EmbeddingServiceInterface] = None
_service_type: Optional[str] = None


class EmbeddingServiceFactory:
    """Factory for creating appropriate embedding services"""

    @staticmethod
    def create_service(
        service_type: str = "auto",
        model_name: Optional[str] = None,
        **kwargs
    ) -> EmbeddingServiceInterface:
        """
        Create embedding service based on type and availability
        
        Args:
            service_type: "local", "mock", "auto"
            model_name: Model name for local service
            **kwargs: Additional service parameters
        """
        if service_type == "auto":
            service_type = EmbeddingServiceFactory._detect_best_service()

        if service_type == "local":
            return EmbeddingServiceFactory._create_local_service(model_name, **kwargs)
        elif service_type == "mock":
            return EmbeddingServiceFactory._create_mock_service(model_name, **kwargs)
        else:
            raise ValueError(f"Unknown embedding service type: {service_type}")

    @staticmethod
    def _detect_best_service() -> str:
        """Detect the best available service based on environment"""
        # Check if ML features are explicitly disabled
        ml_enabled = os.getenv("ML_FEATURES_ENABLED", "false").lower() == "true"
        if not ml_enabled:
            logger.info("ML features disabled, using mock embedding service")
            return "mock"

        # Check if we're in test environment
        if os.getenv("TESTING") == "true" or "pytest" in os.environ.get("_", ""):
            logger.info("Test environment detected, using mock embedding service")
            return "mock"

        # Try to detect if ML dependencies are available
        try:
            import torch
            import sentence_transformers
            logger.info("ML dependencies available, using local embedding service")
            return "local"
        except ImportError as e:
            logger.warning(f"ML dependencies not available ({e}), falling back to mock service")
            return "mock"

    @staticmethod
    def _create_local_service(model_name: Optional[str] = None, **kwargs) -> EmbeddingServiceInterface:
        """Create local embedding service with lazy import"""
        try:
            # Lazy import to avoid requiring torch/transformers at module level
            from app.rag.services.local_embedding import LocalEmbeddingService
            
            model_name = model_name or os.getenv(
                "EMBEDDING_MODEL", 
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            use_mps = os.getenv("USE_MPS_DEVICE", "true").lower() == "true"
            
            logger.info(f"Creating LocalEmbeddingService with model: {model_name}")
            return LocalEmbeddingService(
                model_name=model_name,
                use_mps=use_mps,
                **kwargs
            )
        except ImportError as e:
            logger.error(f"Cannot create local embedding service: {e}")
            logger.info("Falling back to mock embedding service")
            return EmbeddingServiceFactory._create_mock_service(model_name, **kwargs)
        except Exception as e:
            logger.error(f"Error creating local embedding service: {e}")
            logger.info("Falling back to mock embedding service")
            return EmbeddingServiceFactory._create_mock_service(model_name, **kwargs)

    @staticmethod
    def _create_mock_service(model_name: Optional[str] = None, **kwargs) -> EmbeddingServiceInterface:
        """Create mock embedding service"""
        model_name = model_name or "mock-embedding-service"
        logger.info(f"Creating MockEmbeddingService: {model_name}")
        return MockEmbeddingService(model_name=model_name, **kwargs)


async def get_embedding_service(force_recreate: bool = False) -> EmbeddingServiceInterface:
    """
    Get singleton embedding service instance
    
    Args:
        force_recreate: Force recreation of service (useful for config changes)
    """
    global _embedding_service, _service_type
    
    # Determine service type from environment
    current_service_type = os.getenv("EMBEDDING_SERVICE_TYPE", "auto")
    
    # Create new service if needed
    if _embedding_service is None or force_recreate or _service_type != current_service_type:
        logger.info(f"Creating embedding service (type: {current_service_type})")
        
        # Create the service
        _embedding_service = EmbeddingServiceFactory.create_service(
            service_type=current_service_type
        )
        _service_type = current_service_type
        
        # Initialize the service
        await _embedding_service.initialize()
        
        logger.info(f"Embedding service initialized successfully")
    
    return _embedding_service


def reset_embedding_service():
    """Reset singleton service (useful for testing)"""
    global _embedding_service, _service_type
    _embedding_service = None
    _service_type = None
    logger.info("Embedding service singleton reset")


# Convenience functions for specific service types
async def get_local_embedding_service(**kwargs) -> EmbeddingServiceInterface:
    """Get local embedding service specifically"""
    service = EmbeddingServiceFactory.create_service("local", **kwargs)
    await service.initialize()
    return service


async def get_mock_embedding_service(**kwargs) -> EmbeddingServiceInterface:
    """Get mock embedding service specifically"""
    service = EmbeddingServiceFactory.create_service("mock", **kwargs)
    await service.initialize()
    return service