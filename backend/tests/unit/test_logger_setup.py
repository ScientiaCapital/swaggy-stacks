"""
Test logger setup module
"""

import pytest
import structlog
from unittest.mock import patch, MagicMock
from app.core.logger_setup import LoggerSetup


class TestLoggerSetup:
    """Test LoggerSetup functionality"""

    def setup_method(self):
        """Reset logger setup before each test"""
        LoggerSetup.reset()

    def test_initial_state(self):
        """Test initial state of LoggerSetup"""
        assert not LoggerSetup.is_configured()
        assert len(LoggerSetup._loggers) == 0

    def test_configure_logging_basic(self):
        """Test basic logging configuration"""
        LoggerSetup.configure_logging()

        assert LoggerSetup.is_configured()
        # Should not reconfigure if already configured
        with patch('structlog.configure') as mock_configure:
            LoggerSetup.configure_logging()
            mock_configure.assert_not_called()

    def test_configure_logging_with_params(self):
        """Test logging configuration with parameters"""
        with patch('structlog.configure') as mock_configure:
            LoggerSetup.configure_logging(
                log_level="DEBUG",
                log_format="console",
                enable_colors=False
            )

            mock_configure.assert_called_once()
            # Check that processors were configured
            call_args = mock_configure.call_args
            assert 'processors' in call_args[1]
            assert 'wrapper_class' in call_args[1]

    def test_get_logger_with_name(self):
        """Test getting logger with explicit name"""
        logger = LoggerSetup.get_logger("test_logger")

        assert logger is not None
        # Logger might be BoundLoggerLazyProxy initially
        assert hasattr(logger, 'info')  # Check it has logger methods
        assert LoggerSetup.is_configured()

    def test_get_logger_auto_name(self):
        """Test getting logger with auto-detected name"""
        logger = LoggerSetup.get_logger()

        assert logger is not None
        # Should use the calling module name (test module)
        assert "test_logger_setup" in LoggerSetup._loggers

    def test_logger_caching(self):
        """Test that loggers are cached"""
        logger1 = LoggerSetup.get_logger("test_cache")
        logger2 = LoggerSetup.get_logger("test_cache")

        # Should return the same instance
        assert logger1 is logger2
        assert len(LoggerSetup._loggers) == 1

    def test_multiple_loggers(self):
        """Test creating multiple different loggers"""
        logger1 = LoggerSetup.get_logger("logger1")
        logger2 = LoggerSetup.get_logger("logger2")

        assert logger1 is not logger2
        assert len(LoggerSetup._loggers) == 2
        assert "logger1" in LoggerSetup._loggers
        assert "logger2" in LoggerSetup._loggers

    def test_reset(self):
        """Test resetting logger setup"""
        # Configure and create some loggers
        LoggerSetup.configure_logging()
        LoggerSetup.get_logger("test")

        assert LoggerSetup.is_configured()
        assert len(LoggerSetup._loggers) > 0

        # Reset
        LoggerSetup.reset()

        assert not LoggerSetup.is_configured()
        assert len(LoggerSetup._loggers) == 0

    def test_json_format(self):
        """Test JSON format configuration"""
        with patch('structlog.configure') as mock_configure:
            LoggerSetup.configure_logging(log_format="json")

            call_args = mock_configure.call_args
            processors = call_args[1]['processors']

            # Should contain JSONRenderer
            renderer_types = [type(p).__name__ for p in processors]
            assert 'JSONRenderer' in renderer_types

    def test_console_format(self):
        """Test console format configuration"""
        with patch('structlog.configure') as mock_configure:
            LoggerSetup.configure_logging(log_format="console", enable_colors=True)

            call_args = mock_configure.call_args
            processors = call_args[1]['processors']

            # Should contain ConsoleRenderer
            renderer_types = [type(p).__name__ for p in processors]
            assert 'ConsoleRenderer' in renderer_types

    @patch('app.core.logger_setup.settings')
    def test_log_level_from_settings(self, mock_settings):
        """Test log level from settings"""
        mock_settings.LOG_LEVEL = "DEBUG"

        with patch('logging.basicConfig') as mock_basic_config:
            LoggerSetup.configure_logging()

            # Should use DEBUG level from settings
            mock_basic_config.assert_called_once()
            call_args = mock_basic_config.call_args
            import logging
            assert call_args[1]['level'] == logging.DEBUG

    def test_log_level_parameter_override(self):
        """Test log level parameter overrides settings"""
        with patch('logging.basicConfig') as mock_basic_config:
            LoggerSetup.configure_logging(log_level="ERROR")

            call_args = mock_basic_config.call_args
            import logging
            assert call_args[1]['level'] == logging.ERROR

    def test_invalid_log_level(self):
        """Test invalid log level defaults to INFO"""
        with patch('logging.basicConfig') as mock_basic_config:
            LoggerSetup.configure_logging(log_level="INVALID")

            call_args = mock_basic_config.call_args
            import logging
            assert call_args[1]['level'] == logging.INFO