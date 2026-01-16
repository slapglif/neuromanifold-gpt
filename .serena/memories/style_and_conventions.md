# Style and Conventions for NeuroManifoldGPT

## General Principles
- **Simplicity and Readability**: Follow the nanoGPT philosophy of "teeth over education". Code should be plain and readable.
- **Maintainability**: Use the refactored configuration system to avoid code duplication.
- **Type Safety**: Use typed dataclasses for configurations (`NeuroManifoldConfig`, `TrainConfig`).

## Naming Conventions
- **Modules/Packages**: `snake_case` (e.g., `neuromanifold_gpt`).
- **Classes**: `PascalCase` (e.g., `NeuroManifoldGPT`, `MemmapDataset`).
- **Functions/Variables**: `snake_case` (e.g., `get_logger`, `n_embd`).
- **Constants**: `UPPER_SNAKE_CASE`.

## Configuration System
- Use `RalphConfigBuilder` for composing experiment configurations.
- Avoid creating new top-level config files in `config/` if possible; instead, add iterations to the Ralph Loop registry in `neuromanifold_gpt/config/ralph_configs/`.

## Logging
- Use the unified logging module: `from neuromanifold_gpt.utils.logging import get_logger`.
- Prefer `logger.metric()` for reporting performance numbers.
- Use `logger.section()` to mark major execution phases.
- Use `logger.progress()` for tracking long-running tasks.

## Imports
- Use absolute imports for `neuromanifold_gpt` package.
- Organize imports in sections: standard library, external packages, local modules.
