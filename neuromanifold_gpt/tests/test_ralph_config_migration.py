"""Tests for Ralph config migration equivalence.

This test suite verifies that the new composition-based Ralph configs
(neuromanifold_gpt/config/ralph_configs/iterations.py) are exactly equivalent
to the old ralph_iter*.py files (config/ralph_iter*.py).

For each of the 73 Ralph iterations:
1. Load the old config file and extract parameter values
2. Load the new config from iterations.py
3. Compare all parameter values for equality
4. Report any discrepancies

This ensures the migration preserved all experimental configurations.
"""

import ast
import importlib.util
import re
from pathlib import Path
from typing import Any

import pytest


def get_repo_root() -> Path:
    """Find repository root by looking for config/ directory."""
    # Start from the test file location and resolve to absolute path
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "config").is_dir() and (current / "neuromanifold_gpt").is_dir():
            return current
        current = current.parent
    raise RuntimeError("Could not find repository root (looking for config/ and neuromanifold_gpt/ directories)")


def find_ralph_iter_files() -> list[tuple[int, Path]]:
    """Find all ralph_iter*.py files in config/ directory.

    Returns:
        List of (iteration_number, file_path) tuples, sorted by iteration.
    """
    repo_root = get_repo_root()
    config_dir = repo_root / "config"

    files = []
    pattern = re.compile(r'ralph_iter(\d+)\.py$')

    for file_path in config_dir.glob('ralph_iter*.py'):
        match = pattern.search(file_path.name)
        if match:
            iteration = int(match.group(1))
            files.append((iteration, file_path))

    return sorted(files)


def parse_old_config(file_path: Path) -> dict[str, Any]:
    """Parse an old ralph_iter*.py file and extract config values.

    Args:
        file_path: Path to ralph_iter*.py file

    Returns:
        Dictionary mapping parameter names to values
    """
    with open(file_path, 'r') as f:
        content = f.read()

    # Parse the Python file into an AST
    tree = ast.parse(content, filename=str(file_path))

    config_values = {}

    # Extract all module-level assignments
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            # Get the variable name
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                var_name = node.targets[0].id

                # Get the value
                try:
                    value = ast.literal_eval(node.value)
                    config_values[var_name] = value
                except (ValueError, TypeError):
                    # Skip non-literal values (shouldn't happen in these configs)
                    pass

    return config_values


def load_new_config(iteration: int) -> object:
    """Load a new config from iterations.py.

    Args:
        iteration: Ralph iteration number

    Returns:
        RalphBaseConfig instance from the new system

    Raises:
        ImportError: If config cannot be loaded
        AttributeError: If iteration function doesn't exist
    """
    # Import modules directly to avoid torch dependency in neuromanifold_gpt.__init__
    import sys
    repo_root = get_repo_root()

    # Create a fake neuromanifold_gpt.config.ralph_base module to satisfy imports
    # This prevents the full package from being loaded
    if 'neuromanifold_gpt' not in sys.modules:
        sys.modules['neuromanifold_gpt'] = type(sys)('neuromanifold_gpt')
    if 'neuromanifold_gpt.config' not in sys.modules:
        sys.modules['neuromanifold_gpt.config'] = type(sys)('neuromanifold_gpt.config')

    # Load ralph_base module
    ralph_base_path = repo_root / "neuromanifold_gpt" / "config" / "ralph_base.py"
    spec = importlib.util.spec_from_file_location("neuromanifold_gpt.config.ralph_base", ralph_base_path)
    ralph_base = importlib.util.module_from_spec(spec)
    sys.modules['neuromanifold_gpt.config.ralph_base'] = ralph_base
    spec.loader.exec_module(ralph_base)

    # Load ralph_builder module (it will now find ralph_base via sys.modules)
    ralph_builder_path = repo_root / "neuromanifold_gpt" / "config" / "ralph_builder.py"
    spec = importlib.util.spec_from_file_location("neuromanifold_gpt.config.ralph_builder", ralph_builder_path)
    ralph_builder = importlib.util.module_from_spec(spec)
    sys.modules['neuromanifold_gpt.config.ralph_builder'] = ralph_builder
    spec.loader.exec_module(ralph_builder)

    # Load iterations module
    iterations_path = repo_root / "neuromanifold_gpt" / "config" / "ralph_configs" / "iterations.py"
    spec = importlib.util.spec_from_file_location("iterations", iterations_path)
    iterations = importlib.util.module_from_spec(spec)
    sys.modules['iterations'] = iterations
    spec.loader.exec_module(iterations)

    # Get the iteration function
    func_name = f'ralph_iter{iteration}'
    if not hasattr(iterations, func_name):
        raise AttributeError(f"Function {func_name} not found in iterations module")

    # Call the function to get the config
    func = getattr(iterations, func_name)
    return func()


def config_to_dict(config: object) -> dict[str, Any]:
    """Convert a RalphBaseConfig dataclass to a dictionary.

    Args:
        config: RalphBaseConfig instance

    Returns:
        Dictionary of all config parameters
    """
    from dataclasses import fields

    return {field.name: getattr(config, field.name) for field in fields(config)}


class TestRalphConfigMigration:
    """Test suite for Ralph config migration equivalence."""

    def test_all_old_configs_found(self):
        """Test that we can find ralph_iter*.py files."""
        files = find_ralph_iter_files()
        assert len(files) == 73, f"Expected 73 ralph_iter files, found {len(files)}"

    def test_all_new_configs_exist(self):
        """Test that all iteration functions exist in iterations.py."""
        repo_root = get_repo_root()
        iterations_path = repo_root / "neuromanifold_gpt" / "config" / "ralph_configs" / "iterations.py"

        # Read the iterations.py file and check for function definitions
        with open(iterations_path, 'r') as f:
            content = f.read()

        files = find_ralph_iter_files()
        missing = []

        for iteration, _ in files:
            func_name = f'ralph_iter{iteration}'
            # Check if function is defined in the file
            if f'def {func_name}()' not in content:
                missing.append(iteration)

        assert not missing, f"Missing iteration functions: {missing}"

    @pytest.mark.parametrize("iteration,file_path", find_ralph_iter_files())
    def test_config_equivalence(self, iteration: int, file_path: Path):
        """Test that old and new configs are equivalent.

        Args:
            iteration: Ralph iteration number
            file_path: Path to old config file
        """
        # Parse old config
        old_config_dict = parse_old_config(file_path)

        # Load new config
        new_config = load_new_config(iteration)
        new_config_dict = config_to_dict(new_config)

        # Compare all parameters from old config
        discrepancies = []
        for param_name, old_value in old_config_dict.items():
            if param_name not in new_config_dict:
                discrepancies.append(
                    f"Parameter '{param_name}' exists in old config but not in new"
                )
                continue

            new_value = new_config_dict[param_name]

            # Compare values (handle floating point comparison)
            if isinstance(old_value, float) and isinstance(new_value, float):
                if abs(old_value - new_value) > 1e-9:
                    discrepancies.append(
                        f"Parameter '{param_name}': old={old_value}, new={new_value}"
                    )
            elif old_value != new_value:
                discrepancies.append(
                    f"Parameter '{param_name}': old={old_value}, new={new_value}"
                )

        # Report all discrepancies if any exist
        if discrepancies:
            error_msg = f"Config discrepancies for iteration {iteration}:\n"
            error_msg += "\n".join(f"  - {d}" for d in discrepancies)
            pytest.fail(error_msg)

    def test_config_completeness(self):
        """Test that new configs have all expected parameters."""
        # Load iteration 1 as a representative example
        config = load_new_config(1)
        config_dict = config_to_dict(config)

        # Expected core parameters (from RalphBaseConfig)
        expected_params = {
            # Data
            'dataset', 'batch_size', 'block_size', 'num_workers',
            # Model
            'model_type', 'n_layer', 'n_head', 'n_embd', 'dropout', 'bias',
            # NeuroManifold features
            'use_sdr', 'use_kan', 'kan_type', 'kan_num_centers',
            'use_mhc', 'use_full_mhc', 'mhc_n_streams',
            # FHN
            'fhn_threshold', 'fhn_tau', 'n_fhn_steps',
            'use_fhn_imex', 'use_fhn_partitioning', 'use_fhn_fused',
            # Speed optimizations
            'skip_manifold_spectral',
            # Training
            'max_iters', 'gradient_accumulation_steps', 'learning_rate',
            'min_lr', 'weight_decay', 'warmup_iters', 'lr_decay_iters',
            'grad_clip', 'early_stopping_patience',
            # Eval/logging
            'eval_interval', 'log_interval', 'eval_iters', 'sample_interval',
            # Output
            'out_dir', 'save_checkpoints',
            # Hardware
            'devices', 'precision', 'compile_model',
            # Logging
            'wandb_log',
        }

        missing = expected_params - set(config_dict.keys())
        assert not missing, f"Missing expected parameters: {missing}"

    def test_all_iterations_loadable(self):
        """Test that all iteration configs can be loaded without errors."""
        files = find_ralph_iter_files()
        errors = []

        for iteration, _ in files:
            try:
                config = load_new_config(iteration)
                # Basic sanity checks
                assert config.dataset == "shakespeare_char"
                assert config.n_embd % config.n_head == 0  # Valid architecture
            except Exception as e:
                errors.append(f"Iteration {iteration}: {e}")

        assert not errors, f"Failed to load configs:\n" + "\n".join(errors)

    def test_sample_iterations_spot_check(self):
        """Spot check a few key iterations for correct values."""
        # Iteration 1 - tiny config
        config1 = load_new_config(1)
        assert config1.n_layer == 2
        assert config1.n_embd == 256
        assert config1.n_head == 4
        assert config1.batch_size == 32
        assert config1.max_iters == 500
        assert config1.use_mhc is True

        # Iteration 10 - check it differs from iteration 1
        config10 = load_new_config(10)
        # Should have different parameters (verify migration captured variations)
        assert (config10.n_layer, config10.n_embd, config10.batch_size) != \
               (config1.n_layer, config1.n_embd, config1.batch_size)
