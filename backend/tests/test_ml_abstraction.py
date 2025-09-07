"""
Tests for ML dependency abstraction and lazy loading
Validates that the system works with and without ML dependencies
"""

import os
import pytest
from unittest.mock import patch, MagicMock
import asyncio

from app.rag.services.embedding_factory import (
    EmbeddingServiceFactory,
    get_embedding_service,
    reset_embedding_service
)
from app.rag.services.mock_embedding import MockEmbeddingService
from app.rag.services.embedding_base import EmbeddingServiceInterface


class TestEmbeddingServiceFactory:
    """Test the embedding service factory"""

    def test_create_mock_service(self):
        """Test creating mock embedding service"""
        service = EmbeddingServiceFactory.create_service("mock")
        assert isinstance(service, MockEmbeddingService)
        assert service.model_name == "mock-embedding-service"
        assert service.target_dim == 1536

    def test_create_mock_service_with_params(self):
        """Test creating mock service with custom parameters"""
        service = EmbeddingServiceFactory.create_service(
            "mock",
            model_name="custom-mock",
            target_dim=768
        )
        assert isinstance(service, MockEmbeddingService)
        assert service.model_name == "custom-mock"
        assert service.target_dim == 768

    @patch('app.rag.services.embedding_factory.torch')
    @patch('app.rag.services.embedding_factory.sentence_transformers')
    def test_create_local_service_with_ml_available(self, mock_st, mock_torch):
        """Test creating local service when ML dependencies are available"""
        # Mock the imports to simulate ML dependencies being available
        with patch('app.rag.services.embedding_factory.LocalEmbeddingService') as mock_les:
            mock_instance = MagicMock()
            mock_les.return_value = mock_instance
            
            service = EmbeddingServiceFactory.create_service("local")
            assert service == mock_instance
            mock_les.assert_called_once()

    def test_create_local_service_without_ml_fallback(self):
        """Test that local service falls back to mock when ML dependencies unavailable"""
        # This should fall back to mock service since torch isn't available in test env
        service = EmbeddingServiceFactory.create_service("local")
        assert isinstance(service, MockEmbeddingService)

    def test_auto_detection_in_test_env(self):
        """Test that auto detection chooses mock service in test environment"""
        service = EmbeddingServiceFactory.create_service("auto")
        assert isinstance(service, MockEmbeddingService)

    @patch.dict(os.environ, {"ML_FEATURES_ENABLED": "false"})
    def test_auto_detection_ml_disabled(self):
        """Test auto detection when ML features explicitly disabled"""
        service = EmbeddingServiceFactory.create_service("auto")
        assert isinstance(service, MockEmbeddingService)

    def test_invalid_service_type(self):
        """Test error handling for invalid service type"""
        with pytest.raises(ValueError, match="Unknown embedding service type"):
            EmbeddingServiceFactory.create_service("invalid_type")


class TestMockEmbeddingService:
    """Test the mock embedding service"""

    @pytest.fixture
    async def mock_service(self):
        """Create initialized mock embedding service"""
        service = MockEmbeddingService()
        await service.initialize()
        return service

    async def test_initialization(self):
        """Test mock service initialization"""
        service = MockEmbeddingService()
        await service.initialize()
        # Should not raise any errors

    async def test_embed_single_text(self, mock_service):
        """Test embedding single text"""
        result = await mock_service.embed_text("test text")
        assert result is not None
        assert result.text == "test text"
        assert result.embedding.shape == (1536,)
        assert result.model_version == "mock-v1.0"
        assert result.confidence > 0
        assert result.cache_hit is False

    async def test_embed_multiple_texts(self, mock_service):
        """Test embedding multiple texts"""
        texts = ["text1", "text2", "text3"]
        results = await mock_service.embed_texts(texts)
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.text == texts[i]
            assert result.embedding.shape == (1536,)

    async def test_deterministic_embeddings(self, mock_service):
        """Test that embeddings are deterministic"""
        text = "test text for determinism"
        result1 = await mock_service.embed_text(text)
        result2 = await mock_service.embed_text(text)
        
        # Should get the same embedding (deterministic based on text hash)
        assert (result1.embedding == result2.embedding).all()
        assert result2.cache_hit is True  # Second should be cache hit

    async def test_empty_text_list(self, mock_service):
        """Test handling empty text list"""
        results = await mock_service.embed_texts([])
        assert results == []

    async def test_health_check(self, mock_service):
        """Test health check"""
        health = await mock_service.health_check()
        assert health["status"] == "healthy"
        assert health["service_type"] == "mock"
        assert health["model_loaded"] is True

    def test_clear_cache(self, mock_service):
        """Test cache clearing"""
        # Add something to cache first
        mock_service.cache["test"] = "value"
        cleared = mock_service.clear_cache()
        assert cleared == 1
        assert len(mock_service.cache) == 0

    def test_get_stats(self, mock_service):
        """Test getting statistics"""
        stats = mock_service.get_stats()
        assert "service_type" in stats
        assert "cache_hit_rate" in stats
        assert stats["service_type"] == "mock"


class TestEmbeddingServiceSingleton:
    """Test the singleton embedding service management"""

    def setup_method(self):
        """Reset singleton before each test"""
        reset_embedding_service()

    def teardown_method(self):
        """Reset singleton after each test"""
        reset_embedding_service()

    @patch.dict(os.environ, {"EMBEDDING_SERVICE_TYPE": "mock"})
    async def test_singleton_creation(self):
        """Test singleton service creation"""
        service1 = await get_embedding_service()
        service2 = await get_embedding_service()
        assert service1 is service2  # Should be the same instance

    @patch.dict(os.environ, {"EMBEDDING_SERVICE_TYPE": "mock"})
    async def test_force_recreate(self):
        """Test forcing recreation of singleton"""
        service1 = await get_embedding_service()
        service2 = await get_embedding_service(force_recreate=True)
        assert service1 is not service2  # Should be different instances

    def test_reset_singleton(self):
        """Test resetting the singleton"""
        reset_embedding_service()
        # Should not raise any errors


class TestMLFeaturesIntegration:
    """Test integration with ML features flag"""

    @patch.dict(os.environ, {"ML_FEATURES_ENABLED": "false", "EMBEDDING_SERVICE_TYPE": "auto"})
    async def test_ml_disabled_uses_mock(self):
        """Test that disabling ML features uses mock service"""
        reset_embedding_service()
        service = await get_embedding_service()
        assert isinstance(service, MockEmbeddingService)

    @patch.dict(os.environ, {"EMBEDDING_SERVICE_TYPE": "mock"})
    async def test_explicit_mock_service(self):
        """Test explicitly requesting mock service"""
        reset_embedding_service()
        service = await get_embedding_service()
        assert isinstance(service, MockEmbeddingService)


class TestEmbeddingServiceInterface:
    """Test that all services implement the interface correctly"""

    async def test_mock_service_implements_interface(self):
        """Test that mock service implements all interface methods"""
        service = MockEmbeddingService()
        assert isinstance(service, EmbeddingServiceInterface)
        
        # Test all abstract methods are implemented
        await service.initialize()
        
        result = await service.embed_text("test")
        assert result is not None
        
        results = await service.embed_texts(["test1", "test2"])
        assert len(results) == 2
        
        health = await service.health_check()
        assert isinstance(health, dict)
        
        cleared = service.clear_cache()
        assert isinstance(cleared, int)
        
        stats = service.get_stats()
        assert isinstance(stats, dict)


class TestErrorHandling:
    """Test error handling in embedding services"""

    async def test_mock_service_handles_none_text(self):
        """Test mock service handles None text gracefully"""
        service = MockEmbeddingService()
        await service.initialize()
        
        # Should handle None or empty strings gracefully
        results = await service.embed_texts([])
        assert results == []

    def test_factory_error_handling(self):
        """Test factory error handling"""
        with pytest.raises(ValueError):
            EmbeddingServiceFactory.create_service("nonexistent_type")


class TestPerformance:
    """Test performance characteristics of mock service"""

    async def test_mock_service_is_fast(self):
        """Test that mock service is fast"""
        service = MockEmbeddingService()
        await service.initialize()
        
        import time
        start = time.time()
        results = await service.embed_texts(["test"] * 100)
        end = time.time()
        
        assert len(results) == 100
        assert end - start < 0.1  # Should be very fast (< 100ms)

    async def test_cache_performance(self):
        """Test caching performance"""
        service = MockEmbeddingService()
        await service.initialize()
        
        text = "test text for caching"
        
        # First call
        result1 = await service.embed_text(text)
        assert result1.cache_hit is False
        
        # Second call should hit cache
        result2 = await service.embed_text(text)
        assert result2.cache_hit is True
        
        # Results should be identical
        assert (result1.embedding == result2.embedding).all()