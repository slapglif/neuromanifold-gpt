#!/usr/bin/env python3
"""Test config loader functionality.

This test suite verifies the type-safe configuration loader that replaces
the unsafe exec(open()) pattern. Tests cover:
- Loading default configurations
- Loading from preset modules
- Applying CLI overrides with type validation
- Error handling for invalid keys and type mismatches

Note: Imports are done inside test methods to avoid torch dependency at module level.
"""

import sys
import os
import importlib.util
from pathlib import Path

import pytest

# Get the project root directory
_test_dir = Path(__file__).parent
_project_root = _test_dir.parent.parent


def _load_module_direct(module_name, file_path):
    """Load a module directly from file path without triggering package imports."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Load required modules at module level (before pytest collection)
# This bypasses the neuromanifold_gpt package __init__.py which requires torch
if 'neuromanifold_gpt.errors' not in sys.modules:
    _load_module_direct(
        'neuromanifold_gpt.errors',
        _project_root / 'neuromanifold_gpt' / 'errors.py'
    )
if 'neuromanifold_gpt.config.training' not in sys.modules:
    _load_module_direct(
        'neuromanifold_gpt.config.training',
        _project_root / 'neuromanifold_gpt' / 'config' / 'training.py'
    )
if 'neuromanifold_gpt.config.loader' not in sys.modules:
    _load_module_direct(
        'neuromanifold_gpt.config.loader',
        _project_root / 'neuromanifold_gpt' / 'config' / 'loader.py'
    )

# Get module references for use in tests
training = sys.modules['neuromanifold_gpt.config.training']
loader = sys.modules['neuromanifold_gpt.config.loader']
errors = sys.modules['neuromanifold_gpt.errors']


class TestDefaultConfigs:
    """Test loading default configurations without any overrides."""

    def test_training_config_defaults(self):
        """Test loading default TrainingConfig."""
        
        config = loader.load_config(training.TrainingConfig, [], show_help=False)

        assert config.batch_size == 64
        assert config.learning_rate == 1e-3
        assert config.n_layer == 6
        assert config.n_head == 6
        assert config.n_embd == 384
        assert config.dataset == "shakespeare_char"
        assert config.model_type == "neuromanifold"

    def test_sampling_config_defaults(self):
        """Test loading default SamplingConfig."""
        
        config = loader.load_config(training.SamplingConfig, [], show_help=False)

        assert config.init_from == 'resume'
        assert config.num_samples == 10
        assert config.max_new_tokens == 500
        assert config.temperature == 0.8
        assert config.top_k == 200
        assert config.device == 'cuda'

    def test_eval_config_defaults(self):
        """Test loading default EvalConfig."""
        
        config = loader.load_config(training.EvalConfig, [], show_help=False)

        assert config.out_dir == 'out'
        assert config.benchmark == 'lambada'
        assert config.device == 'cuda'
        assert config.dtype == 'bfloat16'
        assert config.seed == 1337

    def test_bench_config_defaults(self):
        """Test loading default BenchConfig."""
        
        config = loader.load_config(training.BenchConfig, [], show_help=False)

        assert config.batch_size == 12
        assert config.block_size == 1024
        assert config.n_layer == 12
        assert config.n_head == 12
        assert config.n_embd == 768
        assert config.learning_rate == 1e-4


class TestCLIOverrides:
    """Test applying CLI overrides to configuration."""

    def test_single_override(self):
        """Test overriding a single config value."""
        
        config = loader.load_config(
            training.TrainingConfig,
            ['--batch_size=32'],
            show_help=False
        )
        assert config.batch_size == 32
        # Other values should remain default
        assert config.learning_rate == 1e-3

    def test_multiple_overrides(self):
        """Test overriding multiple config values."""
        
        config = loader.load_config(
            training.TrainingConfig,
            ['--batch_size=32', '--learning_rate=0.001', '--n_layer=8'],
            show_help=False
        )
        assert config.batch_size == 32
        assert config.learning_rate == 0.001
        assert config.n_layer == 8

    def test_float_override(self):
        """Test overriding float values."""
        
        config = loader.load_config(
            training.SamplingConfig,
            ['--temperature=0.9'],
            show_help=False
        )
        assert config.temperature == 0.9

    def test_string_override(self):
        """Test overriding string values."""
        
        config = loader.load_config(
            training.TrainingConfig,
            ['--dataset=openwebtext', '--out_dir=out-test'],
            show_help=False
        )
        assert config.dataset == 'openwebtext'
        assert config.out_dir == 'out-test'

    def test_boolean_override(self):
        """Test overriding boolean values."""
        
        config = loader.load_config(
            training.TrainingConfig,
            ['--use_kan=False', '--wandb_log=True'],
            show_help=False
        )
        assert config.use_kan == False
        assert config.wandb_log == True


class TestPresetLoading:
    """Test loading configuration from preset modules."""

    def test_load_nano_preset_directly(self):
        """Test loading the nano preset configuration directly."""
        # Load the preset module directly to avoid full package dependencies
        if 'neuromanifold_gpt.config.presets.nano' not in sys.modules:
            _load_module_direct(
                'neuromanifold_gpt.config.presets.nano',
                _project_root / 'neuromanifold_gpt' / 'config' / 'presets' / 'nano.py'
            )

        preset = sys.modules['neuromanifold_gpt.config.presets.nano']
        config = preset.get_nano_config()

        # Verify nano preset values
        assert config.block_size == 256
        assert config.n_layer == 4
        assert config.n_heads == 4
        assert config.n_embd == 128
        assert config.manifold_dim == 32

    def test_preset_function_returns_correct_type(self):
        """Test that preset functions return the correct config type."""
        if 'neuromanifold_gpt.config.base' not in sys.modules:
            _load_module_direct(
                'neuromanifold_gpt.config.base',
                _project_root / 'neuromanifold_gpt' / 'config' / 'base.py'
            )
        if 'neuromanifold_gpt.config.presets.nano' not in sys.modules:
            _load_module_direct(
                'neuromanifold_gpt.config.presets.nano',
                _project_root / 'neuromanifold_gpt' / 'config' / 'presets' / 'nano.py'
            )

        preset = sys.modules['neuromanifold_gpt.config.presets.nano']
        base = sys.modules['neuromanifold_gpt.config.base']

        config = preset.get_nano_config()
        assert isinstance(config, base.NeuroManifoldConfig)


class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_invalid_key_error(self):
        """Test that invalid config keys raise ValidationError."""
        
        with pytest.raises(errors.ValidationError) as exc_info:
            loader.load_config(
                training.TrainingConfig,
                ['--invalid_key=123'],
                show_help=False
            )
        assert "Unknown config key" in str(exc_info.value)

    def test_type_mismatch_error(self):
        """Test that type mismatches raise ValidationError."""
        
        with pytest.raises(errors.ValidationError) as exc_info:
            loader.load_config(
                training.TrainingConfig,
                ['--batch_size=not_a_number'],
                show_help=False
            )
        assert "type mismatch" in str(exc_info.value).lower()

    def test_invalid_override_format(self):
        """Test that improperly formatted overrides raise ValidationError."""

        with pytest.raises(errors.ValidationError) as exc_info:
            loader.load_config(
                training.TrainingConfig,
                ['batch_size=32'],  # Missing '--' prefix
                show_help=False
            )
        assert "override argument format" in str(exc_info.value).lower()

    def test_invalid_module_format(self):
        """Test that module names with '--' prefix raise ValidationError."""

        with pytest.raises(errors.ValidationError) as exc_info:
            loader.load_config(
                training.TrainingConfig,
                ['--config.nano'],  # Module name shouldn't start with '--'
                show_help=False
            )
        assert "module argument" in str(exc_info.value).lower()

    def test_nonexistent_module(self):
        """Test that loading non-existent modules raises ValidationError."""
        
        with pytest.raises(errors.ValidationError) as exc_info:
            loader.load_config(
                training.TrainingConfig,
                ['neuromanifold_gpt.config.presets.nonexistent'],
                show_help=False
            )
        assert "Cannot load config module" in str(exc_info.value)


class TestTypeValidation:
    """Test type validation for different data types."""

    def test_int_validation(self):
        """Test integer type validation."""
        
        # Valid int
        config = loader.load_config(
            training.TrainingConfig,
            ['--batch_size=32', '--n_layer=12'],
            show_help=False
        )
        assert config.batch_size == 32
        assert config.n_layer == 12

        # Invalid int (float)
        with pytest.raises(errors.ValidationError):
            loader.load_config(
                training.TrainingConfig,
                ['--batch_size=32.5'],
                show_help=False
            )

    def test_float_validation(self):
        """Test float type validation."""
        
        # Valid float
        config = loader.load_config(
            training.TrainingConfig,
            ['--learning_rate=0.001', '--dropout=0.1'],
            show_help=False
        )
        assert config.learning_rate == 0.001
        assert config.dropout == 0.1

        # Invalid float (string should fail)
        with pytest.raises(errors.ValidationError):
            loader.load_config(
                training.TrainingConfig,
                ['--learning_rate=invalid'],
                show_help=False
            )

    def test_string_validation(self):
        """Test string type validation."""
        
        # Valid string
        config = loader.load_config(
            training.TrainingConfig,
            ['--dataset=openwebtext', '--model_type=gpt'],
            show_help=False
        )
        assert config.dataset == 'openwebtext'
        assert config.model_type == 'gpt'

    def test_boolean_validation(self):
        """Test boolean type validation."""
        
        # Valid bool
        config = loader.load_config(
            training.TrainingConfig,
            ['--use_kan=True', '--wandb_log=False'],
            show_help=False
        )
        assert config.use_kan == True
        assert config.wandb_log == False

        # Invalid bool (string that's not 'True' or 'False')
        with pytest.raises(errors.ValidationError):
            loader.load_config(
                training.TrainingConfig,
                ['--use_kan=yes'],
                show_help=False
            )


class TestMultipleConfigClasses:
    """Test loading different config classes."""

    def test_all_config_classes_loadable(self):
        """Test that all config classes can be loaded."""
        

        # TrainingConfig
        train_config = loader.load_config(training.TrainingConfig, [], show_help=False)
        assert isinstance(train_config, training.TrainingConfig)

        # SamplingConfig
        sample_config = loader.load_config(training.SamplingConfig, [], show_help=False)
        assert isinstance(sample_config, training.SamplingConfig)

        # EvalConfig
        eval_config = loader.load_config(training.EvalConfig, [], show_help=False)
        assert isinstance(eval_config, training.EvalConfig)

        # BenchConfig
        bench_config = loader.load_config(training.BenchConfig, [], show_help=False)
        assert isinstance(bench_config, training.BenchConfig)

    def test_config_specific_fields(self):
        """Test that each config has its specific fields."""
        

        # TrainingConfig specific
        train_config = loader.load_config(training.TrainingConfig, [], show_help=False)
        assert hasattr(train_config, 'learning_rate')
        assert hasattr(train_config, 'max_iters')

        # SamplingConfig specific
        sample_config = loader.load_config(training.SamplingConfig, [], show_help=False)
        assert hasattr(sample_config, 'temperature')
        assert hasattr(sample_config, 'num_samples')

        # EvalConfig specific
        eval_config = loader.load_config(training.EvalConfig, [], show_help=False)
        assert hasattr(eval_config, 'benchmark')

        # BenchConfig specific
        bench_config = loader.load_config(training.BenchConfig, [], show_help=False)
        assert hasattr(bench_config, 'profiler_wait')
        assert hasattr(bench_config, 'burnin_steps')


class TestCompatibilityLayer:
    """Test the backward compatibility shim (compat.py)."""

    def test_apply_config_overrides_basic(self):
        """Test basic config override functionality from compat layer."""
        # Load compat module
        if 'neuromanifold_gpt.config.compat' not in sys.modules:
            _load_module_direct(
                'neuromanifold_gpt.config.compat',
                _project_root / 'neuromanifold_gpt' / 'config' / 'compat.py'
            )

        compat = sys.modules['neuromanifold_gpt.config.compat']

        # Create a config dict
        config_dict = {
            'batch_size': 16,
            'learning_rate': 1e-3,
            'n_layer': 6
        }

        # Apply overrides
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = compat.apply_config_overrides(
                config_dict,
                ['--batch_size=32', '--learning_rate=0.001'],
                show_deprecation_warning=True
            )

            # Check that deprecation warning was issued
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()

        # Check overrides were applied
        assert result['batch_size'] == 32
        assert result['learning_rate'] == 0.001
        assert result['n_layer'] == 6  # Unchanged

    def test_apply_config_overrides_type_validation(self):
        """Test that compat layer validates types correctly."""
        if 'neuromanifold_gpt.config.compat' not in sys.modules:
            _load_module_direct(
                'neuromanifold_gpt.config.compat',
                _project_root / 'neuromanifold_gpt' / 'config' / 'compat.py'
            )

        compat = sys.modules['neuromanifold_gpt.config.compat']

        config_dict = {'batch_size': 16, 'learning_rate': 1e-3}

        # Type mismatch should raise ValidationError
        with pytest.raises(errors.ValidationError) as exc_info:
            compat.apply_config_overrides(
                config_dict,
                ['--batch_size=not_a_number'],
                show_deprecation_warning=False
            )
        assert "type mismatch" in str(exc_info.value).lower()

    def test_apply_config_overrides_unknown_key(self):
        """Test that compat layer rejects unknown config keys."""
        if 'neuromanifold_gpt.config.compat' not in sys.modules:
            _load_module_direct(
                'neuromanifold_gpt.config.compat',
                _project_root / 'neuromanifold_gpt' / 'config' / 'compat.py'
            )

        compat = sys.modules['neuromanifold_gpt.config.compat']

        config_dict = {'batch_size': 16}

        # Unknown key should raise ValidationError
        with pytest.raises(errors.ValidationError) as exc_info:
            compat.apply_config_overrides(
                config_dict,
                ['--unknown_key=123'],
                show_deprecation_warning=False
            )
        assert "unknown config key" in str(exc_info.value).lower()

    def test_apply_config_overrides_invalid_format(self):
        """Test that compat layer rejects invalid override formats."""
        if 'neuromanifold_gpt.config.compat' not in sys.modules:
            _load_module_direct(
                'neuromanifold_gpt.config.compat',
                _project_root / 'neuromanifold_gpt' / 'config' / 'compat.py'
            )

        compat = sys.modules['neuromanifold_gpt.config.compat']

        config_dict = {'batch_size': 16}

        # Missing '=' should raise ValidationError
        with pytest.raises(errors.ValidationError) as exc_info:
            compat.apply_config_overrides(
                config_dict,
                ['--invalid_format'],
                show_deprecation_warning=False
            )
        assert "format" in str(exc_info.value).lower()

    def test_apply_config_overrides_numeric_coercion(self):
        """Test that compat layer allows int to float coercion."""
        if 'neuromanifold_gpt.config.compat' not in sys.modules:
            _load_module_direct(
                'neuromanifold_gpt.config.compat',
                _project_root / 'neuromanifold_gpt' / 'config' / 'compat.py'
            )

        compat = sys.modules['neuromanifold_gpt.config.compat']

        config_dict = {'learning_rate': 1e-3}

        # Int should be coerced to float for float fields
        result = compat.apply_config_overrides(
            config_dict,
            ['--learning_rate=1'],
            show_deprecation_warning=False
        )
        assert result['learning_rate'] == 1.0
        assert isinstance(result['learning_rate'], float)


class TestBackwardCompatibility:
    """Test backward compatibility with old config system."""

    def test_config_accessible_via_dot_notation(self):
        """Test that config values are accessible via dot notation."""

        config = loader.load_config(training.TrainingConfig, [], show_help=False)

        # All values should be accessible via dot notation
        _ = config.batch_size
        _ = config.learning_rate
        _ = config.n_layer
        _ = config.dataset
        _ = config.model_type

        # No AttributeError should be raised

    def test_config_values_match_defaults(self):
        """Test that default config values match expected defaults."""

        config = loader.load_config(training.TrainingConfig, [], show_help=False)

        # Training defaults
        assert config.max_iters == 5000
        assert config.eval_interval == 250
        assert config.log_interval == 10

        # Model defaults
        assert config.n_layer == 6
        assert config.n_head == 6
        assert config.n_embd == 384

        # Optimization defaults
        assert config.learning_rate == 1e-3
        assert config.weight_decay == 0.1

    def test_cli_override_compatibility_with_equals_format(self):
        """Test that CLI overrides work with --key=value format."""
        config = loader.load_config(
            training.TrainingConfig,
            ['--batch_size=32', '--learning_rate=0.001'],
            show_help=False
        )
        assert config.batch_size == 32
        assert config.learning_rate == 0.001

    def test_error_messages_are_helpful(self):
        """Test that error messages provide helpful guidance."""
        # Unknown key error should suggest available keys
        with pytest.raises(errors.ValidationError) as exc_info:
            loader.load_config(
                training.TrainingConfig,
                ['--invalid_key=123'],
                show_help=False
            )
        error_msg = str(exc_info.value)
        assert "unknown" in error_msg.lower() or "not found" in error_msg.lower()

    def test_multiple_config_types_support(self):
        """Test that all config types from training module are supported."""
        # This ensures backward compatibility with all existing config classes
        configs = [
            (training.TrainingConfig, 'batch_size', 64),
            (training.SamplingConfig, 'num_samples', 10),
            (training.EvalConfig, 'benchmark', 'lambada'),
            (training.BenchConfig, 'batch_size', 12),
        ]

        for config_class, field_name, expected_value in configs:
            config = loader.load_config(config_class, [], show_help=False)
            assert hasattr(config, field_name)
            assert getattr(config, field_name) == expected_value


# Note: Run with: pytest neuromanifold_gpt/tests/test_config_loader.py -v
