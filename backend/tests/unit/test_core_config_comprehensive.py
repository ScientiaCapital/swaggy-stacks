"""
Comprehensive tests for core configuration settings
"""

import pytest
import os
from unittest.mock import patch
from app.core.config import Settings, get_settings


class TestSettingsComprehensive:
    """Comprehensive test coverage for Settings class"""

    def test_all_default_values(self):
        """Test all default configuration values"""
        settings = Settings()

        # Project settings
        assert settings.PROJECT_NAME == "Swaggy Stacks - Advanced Markov Trading System"
        assert settings.VERSION == "1.0.0"
        assert settings.API_V1_STR == "/api/v1"

        # Security settings
        assert settings.ACCESS_TOKEN_EXPIRE_MINUTES == 60 * 24 * 8  # 8 days

        # Host settings
        assert "localhost" in settings.ALLOWED_HOSTS
        assert "127.0.0.1" in settings.ALLOWED_HOSTS
        assert "0.0.0.0" in settings.ALLOWED_HOSTS

        # Database defaults
        assert settings.POSTGRES_SERVER == "localhost"
        assert settings.POSTGRES_USER == "postgres"
        assert settings.POSTGRES_PASSWORD == "password"
        assert settings.POSTGRES_DB == "trading_system"
        assert settings.POSTGRES_PORT == "5432"

        # Redis defaults
        assert settings.REDIS_URL == "redis://localhost:6379"

        # Alpaca defaults
        assert settings.ALPACA_API_KEY == ""
        assert settings.ALPACA_SECRET_KEY == ""
        assert settings.ALPACA_BASE_URL == "https://paper-api.alpaca.markets"
        assert settings.ALPACA_DATA_URL == "https://data.alpaca.markets"

    def test_database_url_property(self):
        """Test DATABASE_URL property construction"""
        settings = Settings()
        expected_url = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_SERVER}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
        assert settings.DATABASE_URL == expected_url

    def test_database_url_with_custom_values(self):
        """Test DATABASE_URL with custom database settings"""
        with patch.dict(os.environ, {
            "POSTGRES_SERVER": "custom-server",
            "POSTGRES_USER": "custom-user",
            "POSTGRES_PASSWORD": "custom-pass",
            "POSTGRES_DB": "custom-db",
            "POSTGRES_PORT": "5433"
        }):
            settings = Settings()
            expected_url = "postgresql://custom-user:custom-pass@custom-server:5433/custom-db"
            assert settings.DATABASE_URL == expected_url

    def test_trading_settings_defaults(self):
        """Test trading-specific default settings"""
        settings = Settings()

        assert settings.DEFAULT_POSITION_SIZE == 1000.0
        assert settings.MAX_POSITION_SIZE == 10000.0
        assert settings.MAX_DAILY_LOSS == 500.0
        assert settings.RISK_FREE_RATE == 0.02

    def test_environment_variable_overrides(self):
        """Test environment variable overrides for all settings"""
        env_overrides = {
            "SECRET_KEY": "test-secret",
            "POSTGRES_SERVER": "test-server",
            "POSTGRES_USER": "test-user",
            "POSTGRES_PASSWORD": "test-pass",
            "POSTGRES_DB": "test-db",
            "POSTGRES_PORT": "9999",
            "REDIS_URL": "redis://test-redis:6380",
            "ALPACA_API_KEY": "test-api-key",
            "ALPACA_SECRET_KEY": "test-secret-key",
            "ALPACA_BASE_URL": "https://test-api.alpaca.markets",
            "ALPACA_DATA_URL": "https://test-data.alpaca.markets"
        }

        with patch.dict(os.environ, env_overrides):
            settings = Settings()

            assert settings.SECRET_KEY == "test-secret"
            assert settings.POSTGRES_SERVER == "test-server"
            assert settings.POSTGRES_USER == "test-user"
            assert settings.POSTGRES_PASSWORD == "test-pass"
            assert settings.POSTGRES_DB == "test-db"
            assert settings.POSTGRES_PORT == "9999"
            assert settings.REDIS_URL == "redis://test-redis:6380"
            assert settings.ALPACA_API_KEY == "test-api-key"
            assert settings.ALPACA_SECRET_KEY == "test-secret-key"
            assert settings.ALPACA_BASE_URL == "https://test-api.alpaca.markets"
            assert settings.ALPACA_DATA_URL == "https://test-data.alpaca.markets"

    def test_cors_origins_default(self):
        """Test default CORS origins"""
        settings = Settings()

        assert isinstance(settings.BACKEND_CORS_ORIGINS, list)
        # Should contain localhost variations
        cors_origins = [str(origin) for origin in settings.BACKEND_CORS_ORIGINS]
        expected_origins = ["http://localhost:3000", "http://localhost:8080", "http://localhost:8000"]

        for expected in expected_origins:
            assert expected in cors_origins

    def test_alpaca_configuration_production_vs_paper(self):
        """Test Alpaca configuration differences"""
        # Test paper trading (default)
        settings = Settings()
        assert "paper-api" in settings.ALPACA_BASE_URL

        # Test production URL override
        with patch.dict(os.environ, {"ALPACA_BASE_URL": "https://api.alpaca.markets"}):
            settings = Settings()
            assert settings.ALPACA_BASE_URL == "https://api.alpaca.markets"
            assert "paper-api" not in settings.ALPACA_BASE_URL

    def test_settings_type_validation(self):
        """Test that settings maintain correct types"""
        settings = Settings()

        # String types
        assert isinstance(settings.PROJECT_NAME, str)
        assert isinstance(settings.VERSION, str)
        assert isinstance(settings.SECRET_KEY, str)
        assert isinstance(settings.POSTGRES_SERVER, str)
        assert isinstance(settings.REDIS_URL, str)

        # Integer types
        assert isinstance(settings.ACCESS_TOKEN_EXPIRE_MINUTES, int)

        # Float types
        assert isinstance(settings.DEFAULT_POSITION_SIZE, float)
        assert isinstance(settings.MAX_POSITION_SIZE, float)
        assert isinstance(settings.MAX_DAILY_LOSS, float)
        assert isinstance(settings.RISK_FREE_RATE, float)

        # List types
        assert isinstance(settings.ALLOWED_HOSTS, list)
        assert isinstance(settings.BACKEND_CORS_ORIGINS, list)

    def test_sensitive_data_handling(self):
        """Test that sensitive data is properly handled"""
        settings = Settings()

        # API keys should default to empty strings, not None
        assert settings.ALPACA_API_KEY == ""
        assert settings.ALPACA_SECRET_KEY == ""

        # With actual values
        with patch.dict(os.environ, {
            "ALPACA_API_KEY": "sensitive-key",
            "ALPACA_SECRET_KEY": "sensitive-secret"
        }):
            settings = Settings()
            assert settings.ALPACA_API_KEY == "sensitive-key"
            assert settings.ALPACA_SECRET_KEY == "sensitive-secret"

    def test_get_settings_caching(self):
        """Test get_settings function caching behavior"""
        # First call
        settings1 = get_settings()

        # Second call should return same instance (cached)
        settings2 = get_settings()

        assert settings1 is settings2
        assert id(settings1) == id(settings2)

    def test_configuration_completeness(self):
        """Test that all required configuration is present"""
        settings = Settings()

        # Verify critical configuration exists
        required_attrs = [
            'PROJECT_NAME', 'VERSION', 'API_V1_STR', 'SECRET_KEY',
            'DATABASE_URL', 'REDIS_URL', 'ALPACA_BASE_URL', 'ALPACA_DATA_URL',
            'DEFAULT_POSITION_SIZE', 'MAX_POSITION_SIZE', 'MAX_DAILY_LOSS'
        ]

        for attr in required_attrs:
            assert hasattr(settings, attr), f"Missing required configuration: {attr}"
            value = getattr(settings, attr)
            assert value is not None, f"Configuration {attr} should not be None"