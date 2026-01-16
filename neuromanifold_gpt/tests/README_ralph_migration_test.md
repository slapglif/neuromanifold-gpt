# Ralph Config Migration Test

## Overview

The `test_ralph_config_migration.py` test suite verifies that the new composition-based Ralph configs are equivalent to the old ralph_iter*.py files.

## Running the Tests

```bash
pytest neuromanifold_gpt/tests/test_ralph_config_migration.py -v
```

The `pytest.ini` configuration file automatically uses `--import-mode=importlib` to avoid loading the full neuromanifold_gpt package (which requires torch).

## Test Coverage

- **test_all_old_configs_found**: Verifies all 73 ralph_iter*.py files are found
- **test_all_new_configs_exist**: Verifies all iteration functions exist in iterations.py
- **test_config_equivalence**: Parametrized test comparing each old vs new config (73 tests)
- **test_config_completeness**: Verifies new configs have all expected parameters
- **test_all_iterations_loadable**: Tests that all configs can be loaded without errors
- **test_sample_iterations_spot_check**: Spot checks key iterations for correctness

## Known Issues (Migration Bugs)

The test suite identified 2 configuration discrepancies:

### Iteration 28
**Missing parameters in RalphBaseConfig:**
- `manifold_dim` (set to 8 in iter28, default should be 64)
- `n_eigenvectors` (set to 4 in iter28, default should be 32)

### Iteration 54
**Missing parameter in RalphBaseConfig:**
- `seed` (set to 42 in iter54, no default in base config)

These parameters exist in the old ralph_iter files but were not included in the RalphBaseConfig dataclass. They need to be added to fix the migration.

## Test Results

As of the last run:
- **Total tests**: 78
- **Passed**: 76
- **Failed**: 2 (iterations 28 and 54 due to missing parameters)

The test suite is working correctly - it successfully identified real migration bugs that need to be fixed.
