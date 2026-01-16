"""Tests for structured error handling with rich-formatted panels."""

import pytest
from io import StringIO
from unittest.mock import patch


class TestNeuroManifoldError:
    """Test suite for the base NeuroManifoldError class."""

    def test_base_error_with_all_fields(self):
        """Test base error with problem, cause, recovery, and context."""
        from neuromanifold_gpt.errors import NeuroManifoldError

        with patch("neuromanifold_gpt.errors.console.print"):
            error = NeuroManifoldError(
                problem="Test problem",
                cause="Test cause",
                recovery="Test recovery",
                context="Test context",
            )

        assert error.problem == "Test problem"
        assert error.cause == "Test cause"
        assert error.recovery == "Test recovery"
        assert error.context == "Test context"
        assert "Problem:" in error.message
        assert "Cause:" in error.message
        assert "Recovery:" in error.message
        assert "Context:" in error.message

    def test_base_error_with_only_problem(self):
        """Test base error with only required problem field."""
        from neuromanifold_gpt.errors import NeuroManifoldError

        with patch("neuromanifold_gpt.errors.console.print"):
            error = NeuroManifoldError(problem="Test problem")

        assert error.problem == "Test problem"
        assert error.cause is None
        assert error.recovery is None
        assert error.context is None
        assert "Problem:" in error.message
        assert "Cause:" not in error.message
        assert "Recovery:" not in error.message
        assert "Context:" not in error.message

    def test_base_error_with_partial_fields(self):
        """Test base error with problem and recovery, but no cause or context."""
        from neuromanifold_gpt.errors import NeuroManifoldError

        with patch("neuromanifold_gpt.errors.console.print"):
            error = NeuroManifoldError(
                problem="Test problem", recovery="Test recovery"
            )

        assert error.problem == "Test problem"
        assert error.cause is None
        assert error.recovery == "Test recovery"
        assert error.context is None
        assert "Problem:" in error.message
        assert "Recovery:" in error.message
        assert "Cause:" not in error.message
        assert "Context:" not in error.message

    def test_base_error_is_exception(self):
        """Test that NeuroManifoldError is a proper Exception."""
        from neuromanifold_gpt.errors import NeuroManifoldError

        with patch("neuromanifold_gpt.errors.console.print"):
            error = NeuroManifoldError(problem="Test")

        assert isinstance(error, Exception)
        assert str(error) == "Test"

    def test_base_error_can_be_raised(self):
        """Test that NeuroManifoldError can be raised and caught."""
        from neuromanifold_gpt.errors import NeuroManifoldError

        with pytest.raises(NeuroManifoldError) as exc_info:
            with patch("neuromanifold_gpt.errors.console.print"):
                raise NeuroManifoldError(problem="Test problem")

        assert exc_info.value.problem == "Test problem"


class TestConfigurationError:
    """Test suite for ConfigurationError class."""

    def test_configuration_error_inherits_from_base(self):
        """Test that ConfigurationError inherits from NeuroManifoldError."""
        from neuromanifold_gpt.errors import ConfigurationError, NeuroManifoldError

        with patch("neuromanifold_gpt.errors.console.print"):
            error = ConfigurationError(problem="Test")

        assert isinstance(error, NeuroManifoldError)
        assert isinstance(error, Exception)

    def test_configuration_error_with_all_fields(self):
        """Test ConfigurationError with all fields."""
        from neuromanifold_gpt.errors import ConfigurationError

        with patch("neuromanifold_gpt.errors.console.print"):
            error = ConfigurationError(
                problem="n_embd must be divisible by n_heads",
                cause="n_embd=100 is not divisible by n_heads=8",
                recovery="Use n_embd=128 (or any multiple of 8) with n_heads=8",
                context="Current config: n_embd=100, n_heads=8",
            )

        assert error.problem == "n_embd must be divisible by n_heads"
        assert "n_embd=100" in error.cause
        assert "n_embd=128" in error.recovery
        assert "Current config" in error.context

    def test_configuration_error_can_be_raised(self):
        """Test that ConfigurationError can be raised and caught."""
        from neuromanifold_gpt.errors import ConfigurationError

        with pytest.raises(ConfigurationError) as exc_info:
            with patch("neuromanifold_gpt.errors.console.print"):
                raise ConfigurationError(
                    problem="Invalid configuration",
                    cause="Test cause",
                    recovery="Test recovery",
                )

        assert exc_info.value.problem == "Invalid configuration"
        assert exc_info.value.cause == "Test cause"


class TestModelError:
    """Test suite for ModelError class."""

    def test_model_error_inherits_from_base(self):
        """Test that ModelError inherits from NeuroManifoldError."""
        from neuromanifold_gpt.errors import ModelError, NeuroManifoldError

        with patch("neuromanifold_gpt.errors.console.print"):
            error = ModelError(problem="Test")

        assert isinstance(error, NeuroManifoldError)
        assert isinstance(error, Exception)

    def test_model_error_with_model_loading_scenario(self):
        """Test ModelError with typical model loading scenario."""
        from neuromanifold_gpt.errors import ModelError

        with patch("neuromanifold_gpt.errors.console.print"):
            error = ModelError(
                problem="Cannot load pretrained model 'invalid-model'",
                cause="Model type must be one of: gpt2, gpt2-medium, gpt2-large, gpt2-xl",
                recovery="Specify a valid model type or use --init_from=scratch",
            )

        assert "Cannot load" in error.problem
        assert "gpt2" in error.cause
        assert "--init_from=scratch" in error.recovery

    def test_model_error_can_be_raised(self):
        """Test that ModelError can be raised and caught."""
        from neuromanifold_gpt.errors import ModelError

        with pytest.raises(ModelError) as exc_info:
            with patch("neuromanifold_gpt.errors.console.print"):
                raise ModelError(
                    problem="Model architecture mismatch",
                    cause="Expected 124M parameters but checkpoint has 355M",
                    recovery="Use a compatible checkpoint or adjust config",
                )

        assert "architecture mismatch" in exc_info.value.problem


class TestValidationError:
    """Test suite for ValidationError class."""

    def test_validation_error_inherits_from_base(self):
        """Test that ValidationError inherits from NeuroManifoldError."""
        from neuromanifold_gpt.errors import ValidationError, NeuroManifoldError

        with patch("neuromanifold_gpt.errors.console.print"):
            error = ValidationError(problem="Test")

        assert isinstance(error, NeuroManifoldError)
        assert isinstance(error, Exception)

    def test_validation_error_with_argument_validation_scenario(self):
        """Test ValidationError with typical argument validation scenario."""
        from neuromanifold_gpt.errors import ValidationError

        with patch("neuromanifold_gpt.errors.console.print"):
            error = ValidationError(
                problem="Invalid argument format",
                cause="Arguments cannot start with '--' when using config file syntax",
                recovery="Use either: python script.py config_name OR python script.py --flag=value",
            )

        assert "Invalid argument" in error.problem
        assert "cannot start with '--'" in error.cause
        assert "python script.py" in error.recovery

    def test_validation_error_can_be_raised(self):
        """Test that ValidationError can be raised and caught."""
        from neuromanifold_gpt.errors import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            with patch("neuromanifold_gpt.errors.console.print"):
                raise ValidationError(
                    problem="Input validation failed",
                    cause="Value out of range",
                    recovery="Provide a value between 0 and 100",
                )

        assert "validation failed" in exc_info.value.problem


class TestRuntimeError:
    """Test suite for RuntimeError class."""

    def test_runtime_error_inherits_from_base(self):
        """Test that RuntimeError inherits from NeuroManifoldError."""
        from neuromanifold_gpt.errors import RuntimeError, NeuroManifoldError

        with patch("neuromanifold_gpt.errors.console.print"):
            error = RuntimeError(problem="Test")

        assert isinstance(error, NeuroManifoldError)
        assert isinstance(error, Exception)

    def test_runtime_error_with_feature_not_enabled_scenario(self):
        """Test RuntimeError with typical feature not enabled scenario."""
        from neuromanifold_gpt.errors import RuntimeError

        with patch("neuromanifold_gpt.errors.console.print"):
            error = RuntimeError(
                problem="Imagination module not available",
                cause="Model was initialized without imagination support",
                recovery="Set use_imagination=True in config to enable this feature",
            )

        assert "Imagination module" in error.problem
        assert "without imagination support" in error.cause
        assert "use_imagination=True" in error.recovery

    def test_runtime_error_can_be_raised(self):
        """Test that RuntimeError can be raised and caught."""
        from neuromanifold_gpt.errors import RuntimeError

        with pytest.raises(RuntimeError) as exc_info:
            with patch("neuromanifold_gpt.errors.console.print"):
                raise RuntimeError(
                    problem="DAG planner not available",
                    cause="Model was initialized without DAG planner support",
                    recovery="Set use_dag_planner=True in config to enable this feature",
                )

        assert "DAG planner" in exc_info.value.problem


class TestErrorFormatting:
    """Test suite for error message formatting."""

    def test_error_message_contains_rich_formatting(self):
        """Test that error messages contain rich markup for formatting."""
        from neuromanifold_gpt.errors import ConfigurationError

        with patch("neuromanifold_gpt.errors.console.print"):
            error = ConfigurationError(
                problem="Test problem",
                cause="Test cause",
                recovery="Test recovery",
            )

        # Check for rich markup tags
        assert "[bold red]" in error.message
        assert "[bold yellow]" in error.message
        assert "[bold green]" in error.message

    def test_error_display_calls_console_print(self):
        """Test that error initialization calls console.print for display."""
        from neuromanifold_gpt.errors import ConfigurationError

        with patch("neuromanifold_gpt.errors.console.print") as mock_print:
            error = ConfigurationError(problem="Test")

        # Verify that console.print was called once
        assert mock_print.call_count == 1

        # Verify that a Panel object was passed to print
        call_args = mock_print.call_args[0]
        from rich.panel import Panel

        assert isinstance(call_args[0], Panel)

    def test_error_panel_has_correct_title(self):
        """Test that error panel displays the correct class name as title."""
        from neuromanifold_gpt.errors import ConfigurationError

        with patch("neuromanifold_gpt.errors.console.print") as mock_print:
            error = ConfigurationError(problem="Test")

        call_args = mock_print.call_args[0]
        panel = call_args[0]

        # Check that the panel title contains the error class name
        assert "ConfigurationError" in panel.title

    def test_all_error_types_format_consistently(self):
        """Test that all error types format messages consistently."""
        from neuromanifold_gpt.errors import (
            ConfigurationError,
            ModelError,
            ValidationError,
            RuntimeError,
        )

        error_classes = [
            ConfigurationError,
            ModelError,
            ValidationError,
            RuntimeError,
        ]

        for error_cls in error_classes:
            with patch("neuromanifold_gpt.errors.console.print"):
                error = error_cls(
                    problem="Test problem",
                    cause="Test cause",
                    recovery="Test recovery",
                    context="Test context",
                )

            # All errors should have the same structure
            assert "Problem:" in error.message
            assert "Cause:" in error.message
            assert "Recovery:" in error.message
            assert "Context:" in error.message
            assert "[bold red]" in error.message
            assert "[bold yellow]" in error.message
            assert "[bold green]" in error.message
            assert "[bold blue]" in error.message


class TestErrorUsagePatterns:
    """Test suite for common error usage patterns."""

    def test_config_validation_error_pattern(self):
        """Test typical config validation error pattern."""
        from neuromanifold_gpt.errors import ConfigurationError

        with patch("neuromanifold_gpt.errors.console.print"):
            with pytest.raises(ConfigurationError) as exc_info:
                n_embd = 100
                n_heads = 8
                if n_embd % n_heads != 0:
                    raise ConfigurationError(
                        problem="n_embd must be divisible by n_heads",
                        cause=f"n_embd={n_embd} is not divisible by n_heads={n_heads}",
                        recovery=f"Set n_embd to a multiple of {n_heads} (e.g., {(n_embd // n_heads + 1) * n_heads})",
                    )

        assert exc_info.value.problem == "n_embd must be divisible by n_heads"
        assert "100" in exc_info.value.cause
        assert "104" in exc_info.value.recovery

    def test_model_loading_error_pattern(self):
        """Test typical model loading error pattern."""
        from neuromanifold_gpt.errors import ModelError

        with patch("neuromanifold_gpt.errors.console.print"):
            with pytest.raises(ModelError) as exc_info:
                model_type = "invalid-model"
                valid_models = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
                if model_type not in valid_models:
                    raise ModelError(
                        problem=f"Cannot load pretrained model '{model_type}'",
                        cause=f"Model type must be one of: {', '.join(valid_models)}",
                        recovery="Specify a valid model type or use --init_from=scratch",
                    )

        assert "invalid-model" in exc_info.value.problem
        assert "gpt2" in exc_info.value.cause

    def test_argument_validation_error_pattern(self):
        """Test typical argument validation error pattern."""
        from neuromanifold_gpt.errors import ValidationError

        with patch("neuromanifold_gpt.errors.console.print"):
            with pytest.raises(ValidationError) as exc_info:
                arg = "--config"
                if arg.startswith("--"):
                    raise ValidationError(
                        problem="Invalid argument format",
                        cause="Arguments cannot start with '--' when using config file syntax",
                        recovery="Use either: python script.py config_name OR python script.py --flag=value",
                    )

        assert "Invalid argument" in exc_info.value.problem
        assert "--" in exc_info.value.cause

    def test_feature_disabled_error_pattern(self):
        """Test typical feature disabled runtime error pattern."""
        from neuromanifold_gpt.errors import RuntimeError

        with patch("neuromanifold_gpt.errors.console.print"):
            with pytest.raises(RuntimeError) as exc_info:
                use_imagination = False
                if not use_imagination:
                    raise RuntimeError(
                        problem="Imagination module not available",
                        cause="Model was initialized without imagination support",
                        recovery="Set use_imagination=True in config to enable this feature",
                    )

        assert "Imagination" in exc_info.value.problem
        assert "use_imagination=True" in exc_info.value.recovery
