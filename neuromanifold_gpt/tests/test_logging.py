"""Tests for unified logging module with rich-formatted output."""

import os
from io import StringIO

import pytest


class TestRichLogger:
    """Test suite for the RichLogger class."""

    def test_logger_creation(self):
        """Test that logger can be created with a name."""
        from neuromanifold_gpt.utils.logging import RichLogger

        logger = RichLogger("test_module")
        assert logger.name == "test_module"
        assert logger._logger is not None

    def test_logger_info_method(self):
        """Test info() logging method."""
        from neuromanifold_gpt.utils.logging import RichLogger

        logger = RichLogger("test")
        # Should not raise exception
        logger.info("Test info message")

    def test_logger_warning_method(self):
        """Test warning() logging method."""
        from neuromanifold_gpt.utils.logging import RichLogger

        logger = RichLogger("test")
        # Should not raise exception
        logger.warning("Test warning message")

    def test_logger_error_method(self):
        """Test error() logging method."""
        from neuromanifold_gpt.utils.logging import RichLogger

        logger = RichLogger("test")
        # Should not raise exception
        logger.error("Test error message")

    def test_logger_debug_method(self):
        """Test debug() logging method."""
        from neuromanifold_gpt.utils.logging import RichLogger

        logger = RichLogger("test")
        # Should not raise exception
        logger.debug("Test debug message")

    def test_logger_metric_method(self):
        """Test metric() logging method with value and unit."""
        from neuromanifold_gpt.utils.logging import RichLogger

        logger = RichLogger("test")
        # Should not raise exception
        logger.metric("accuracy", 0.95, unit="%")
        logger.metric("loss", 0.42)

    def test_logger_metric_formatting(self):
        """Test metric() formats float values correctly."""
        from neuromanifold_gpt.utils.logging import RichLogger

        logger = RichLogger("test")
        # Should format to 4 decimal places
        logger.metric("value", 3.14159265)

    def test_logger_metric_integer_value(self):
        """Test metric() handles integer values."""
        from neuromanifold_gpt.utils.logging import RichLogger

        logger = RichLogger("test")
        logger.metric("count", 42)

    def test_logger_progress_method(self):
        """Test progress() logging method."""
        from neuromanifold_gpt.utils.logging import RichLogger

        logger = RichLogger("test")
        # Should not raise exception
        logger.progress("Training", 50, 100)

    def test_logger_progress_percentage_calculation(self):
        """Test progress() calculates percentage correctly."""
        from neuromanifold_gpt.utils.logging import RichLogger

        logger = RichLogger("test")
        # Should calculate 50% for 50/100
        logger.progress("Task", 50, 100)
        # Should calculate 33.3% for 1/3
        logger.progress("Task", 1, 3)

    def test_logger_progress_zero_total(self):
        """Test progress() handles zero total gracefully."""
        from neuromanifold_gpt.utils.logging import RichLogger

        logger = RichLogger("test")
        # Should not divide by zero
        logger.progress("Task", 0, 0)

    def test_logger_section_method(self):
        """Test section() logging method."""
        from neuromanifold_gpt.utils.logging import RichLogger

        logger = RichLogger("test")
        # Should not raise exception
        logger.section("Test Section")

    def test_logger_bind_method(self):
        """Test bind() method for structured logging."""
        from neuromanifold_gpt.utils.logging import RichLogger

        logger = RichLogger("test")
        bound_logger = logger.bind(request_id="123", user="test_user")
        # Should return self for chaining
        assert isinstance(bound_logger, RichLogger)
        assert bound_logger is logger

    def test_logger_with_kwargs(self):
        """Test logging methods accept additional kwargs."""
        from neuromanifold_gpt.utils.logging import RichLogger

        logger = RichLogger("test")
        # Should not raise exception with extra context
        logger.info("Message", extra_field="value")
        logger.warning("Message", count=42)
        logger.error("Message", context={"key": "value"})


class TestGetLogger:
    """Test suite for the get_logger() function."""

    def test_get_logger_returns_rich_logger(self):
        """Test get_logger() returns RichLogger instance."""
        from neuromanifold_gpt.utils.logging import RichLogger, get_logger

        logger = get_logger("test_module")
        assert isinstance(logger, RichLogger)

    def test_get_logger_sets_name(self):
        """Test get_logger() sets logger name correctly."""
        from neuromanifold_gpt.utils.logging import get_logger

        logger = get_logger("my_module")
        assert logger.name == "my_module"

    def test_get_logger_multiple_instances(self):
        """Test get_logger() creates independent instances."""
        from neuromanifold_gpt.utils.logging import get_logger

        logger1 = get_logger("module1")
        logger2 = get_logger("module2")

        assert logger1.name == "module1"
        assert logger2.name == "module2"
        # Should be different instances
        assert logger1 is not logger2


class TestConfigureLogging:
    """Test suite for the configure_logging() function."""

    def test_configure_logging_default(self):
        """Test configure_logging() with defaults."""
        from neuromanifold_gpt.utils.logging import configure_logging

        # Should not raise exception
        configure_logging()

    def test_configure_logging_custom_level(self):
        """Test configure_logging() with custom level."""
        from neuromanifold_gpt.utils.logging import configure_logging

        # Should accept standard log levels
        configure_logging(level="DEBUG")
        configure_logging(level="INFO")
        configure_logging(level="WARNING")
        configure_logging(level="ERROR")

    def test_configure_logging_custom_format(self):
        """Test configure_logging() with custom format."""
        from neuromanifold_gpt.utils.logging import configure_logging

        custom_format = "{time} | {level} | {message}"
        configure_logging(format=custom_format)

    def test_configure_logging_custom_theme(self):
        """Test configure_logging() with custom theme."""
        from neuromanifold_gpt.utils.logging import configure_logging

        custom_theme = {
            "metric": "bold magenta",
            "progress": "cyan",
        }
        configure_logging(theme=custom_theme)

    def test_configure_logging_env_var_level(self, monkeypatch):
        """Test configure_logging() respects LOG_LEVEL environment variable."""
        from neuromanifold_gpt.utils.logging import configure_logging

        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        # Should use LOG_LEVEL from environment
        configure_logging()

    def test_configure_logging_env_var_format(self, monkeypatch):
        """Test configure_logging() respects LOG_FORMAT environment variable."""
        from neuromanifold_gpt.utils.logging import configure_logging

        custom_format = "{level} | {message}"
        monkeypatch.setenv("LOG_FORMAT", custom_format)
        # Should use LOG_FORMAT from environment
        configure_logging()

    def test_configure_logging_parameter_overrides_env(self, monkeypatch):
        """Test configure_logging() parameters override environment variables."""
        from neuromanifold_gpt.utils.logging import configure_logging

        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        # Parameter should take precedence
        configure_logging(level="ERROR")


class TestLoggingIntegration:
    """Integration tests for logging module."""

    def test_logger_workflow(self):
        """Test typical logging workflow."""
        from neuromanifold_gpt.utils.logging import configure_logging, get_logger

        # Configure logging
        configure_logging(level="INFO")

        # Get logger
        logger = get_logger(__name__)

        # Use various logging methods
        logger.info("Starting process")
        logger.section("Initialization")
        logger.metric("initial_value", 0.0)
        logger.progress("Setup", 1, 3)
        logger.warning("This is a warning")
        logger.progress("Setup", 2, 3)
        logger.metric("accuracy", 0.95, unit="%")
        logger.progress("Setup", 3, 3)
        logger.section("Complete")

    def test_logger_with_context(self):
        """Test logger with bound context."""
        from neuromanifold_gpt.utils.logging import get_logger

        logger = get_logger("test")
        # Bind context
        logger = logger.bind(experiment="exp_001", batch=32)
        # Log with context
        logger.info("Training started")
        logger.metric("loss", 0.42)

    def test_multiple_loggers_coexist(self):
        """Test multiple logger instances work independently."""
        from neuromanifold_gpt.utils.logging import get_logger

        logger1 = get_logger("module1")
        logger2 = get_logger("module2")
        logger3 = get_logger("module3")

        # All should work independently
        logger1.info("Message from module1")
        logger2.info("Message from module2")
        logger3.info("Message from module3")

    def test_configure_multiple_times(self):
        """Test configure_logging() can be called multiple times."""
        from neuromanifold_gpt.utils.logging import configure_logging, get_logger

        # Initial configuration
        configure_logging(level="INFO")
        logger = get_logger("test")
        logger.info("First config")

        # Reconfigure
        configure_logging(level="DEBUG")
        logger.debug("Second config")

        # Reconfigure again
        configure_logging(level="WARNING")
        logger.warning("Third config")


class TestLoggingEdgeCases:
    """Test edge cases and error handling."""

    def test_logger_empty_name(self):
        """Test logger with empty name."""
        from neuromanifold_gpt.utils.logging import get_logger

        logger = get_logger("")
        assert logger.name == ""
        logger.info("Message")

    def test_logger_special_characters_in_name(self):
        """Test logger name with special characters."""
        from neuromanifold_gpt.utils.logging import get_logger

        logger = get_logger("my.module-name_v2")
        assert logger.name == "my.module-name_v2"
        logger.info("Message")

    def test_metric_with_empty_unit(self):
        """Test metric() with empty unit string."""
        from neuromanifold_gpt.utils.logging import get_logger

        logger = get_logger("test")
        logger.metric("value", 42, unit="")

    def test_progress_with_negative_values(self):
        """Test progress() with negative values."""
        from neuromanifold_gpt.utils.logging import get_logger

        logger = get_logger("test")
        # Should handle gracefully
        logger.progress("Task", -1, 100)

    def test_progress_current_exceeds_total(self):
        """Test progress() when current exceeds total."""
        from neuromanifold_gpt.utils.logging import get_logger

        logger = get_logger("test")
        # Should calculate >100%
        logger.progress("Task", 150, 100)

    def test_section_with_unicode(self):
        """Test section() with unicode characters."""
        from neuromanifold_gpt.utils.logging import get_logger

        logger = get_logger("test")
        logger.section("Test Section ✓ 🎯 αβγ")

    def test_message_with_unicode(self):
        """Test logging messages with unicode characters."""
        from neuromanifold_gpt.utils.logging import get_logger

        logger = get_logger("test")
        logger.info("Message with emoji 🚀 and symbols ∞ ≈")
        logger.metric("accuracy", 0.95, unit="✓")
