#!/usr/bin/env python3
"""Migration script to extract deltas from ralph_iter config files.

This script:
1. Parses all ralph_iter*.py files in config/
2. Extracts configuration parameter values by executing them
3. Compares against RalphBaseConfig defaults to identify deltas
4. Generates builder-based code for the new config system
5. Preserves comments and iteration metadata

Usage:
    # Dry run to see what will be migrated
    python scripts/migrate_ralph_configs.py --dry-run

    # Generate migration output to file
    python scripts/migrate_ralph_configs.py --output neuromanifold_gpt/config/ralph_configs/iterations.py

    # Analyze specific iteration
    python scripts/migrate_ralph_configs.py --iteration 1 --verbose
"""

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Any


def extract_comment_header(file_path: Path) -> str:
    """Extract the comment header from a ralph_iter file.

    Returns the first block of comment lines (starting with #) at the top
    of the file, preserving iteration goals and metadata.

    Args:
        file_path: Path to ralph_iter*.py file

    Returns:
        Comment header as a string, or empty string if none found
    """
    header_lines = []
    with open(file_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith('#'):
                header_lines.append(stripped)
            elif not stripped:  # Allow blank lines in header
                continue
            else:  # Stop at first non-comment, non-blank line
                break
    return '\n'.join(header_lines)


def parse_config_file(file_path: Path) -> dict[str, Any]:
    """Parse a ralph_iter config file and extract parameter values.

    Uses AST parsing to extract simple assignments (name = value).
    Supports basic Python literals: strings, numbers, booleans, None.
    Skips import statements and complex expressions.

    Args:
        file_path: Path to ralph_iter*.py file

    Returns:
        Dictionary mapping parameter names to values
    """
    config_values = {}

    with open(file_path) as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"Warning: Failed to parse {file_path}: {e}", file=sys.stderr)
        return config_values

    for node in ast.walk(tree):
        # Look for simple assignments: name = value
        if isinstance(node, ast.Assign):
            # Only handle single target assignments
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id

                # Extract value if it's a simple literal
                value = _extract_literal(node.value)
                if value is not None:
                    config_values[name] = value

    return config_values


def _extract_literal(node: ast.expr) -> Any:
    """Extract a Python literal value from an AST node.

    Supports: strings, numbers, booleans, None.
    Returns None for unsupported expressions.

    Args:
        node: AST expression node

    Returns:
        Python literal value, or None if not extractable
    """
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.Num):  # Python < 3.8 compatibility
        return node.n
    elif isinstance(node, ast.Str):  # Python < 3.8 compatibility
        return node.s
    elif isinstance(node, ast.NameConstant):  # Python < 3.8 compatibility
        return node.value
    elif isinstance(node, (ast.List, ast.Tuple)):
        # Support simple lists/tuples of literals
        values = [_extract_literal(elt) for elt in node.elts]
        if all(v is not None for v in values):
            return type(node)(values)
    return None


def get_base_config_defaults() -> dict[str, Any]:
    """Get default values from RalphBaseConfig.

    Returns:
        Dictionary mapping parameter names to default values
    """
    # Import here to avoid dependency issues during dry-run
    try:
        from neuromanifold_gpt.config.ralph_base import RalphBaseConfig
        base = RalphBaseConfig()

        # Extract all fields and their values
        defaults = {}
        for field_name in base.__dataclass_fields__:
            defaults[field_name] = getattr(base, field_name)

        return defaults
    except ImportError as e:
        print(f"Warning: Could not import RalphBaseConfig: {e}", file=sys.stderr)
        print("Using fallback defaults", file=sys.stderr)

        # Fallback: hardcoded defaults matching RalphBaseConfig
        return {
            'dataset': 'shakespeare_char',
            'batch_size': 64,
            'block_size': 128,
            'num_workers': 4,
            'model_type': 'neuromanifold',
            'n_layer': 6,
            'n_head': 6,
            'n_embd': 384,
            'dropout': 0.1,
            'bias': False,
            'use_sdr': False,
            'use_kan': False,
            'kan_type': 'faster',
            'kan_num_centers': 2,
            'use_mhc': False,
            'use_full_mhc': False,
            'mhc_n_streams': 2,
            'fhn_threshold': 0.5,
            'fhn_tau': 12.5,
            'n_fhn_steps': 0,
            'use_fhn_imex': True,
            'use_fhn_partitioning': False,
            'use_fhn_fused': False,
            'skip_manifold_spectral': False,
            'max_iters': 1000,
            'gradient_accumulation_steps': 1,
            'learning_rate': 2e-3,
            'min_lr': 1e-4,
            'weight_decay': 0.1,
            'warmup_iters': 50,
            'lr_decay_iters': 1000,
            'grad_clip': 1.0,
            'early_stopping_patience': 0,
            'eval_interval': 100,
            'log_interval': 50,
            'eval_iters': 10,
            'sample_interval': 0,
            'out_dir': 'out-ralph',
            'save_checkpoints': False,
            'devices': 1,
            'precision': 'bf16-mixed',
            'compile_model': False,
            'wandb_log': False,
        }


def compute_deltas(config_values: dict[str, Any], base_defaults: dict[str, Any]) -> dict[str, Any]:
    """Compute delta between config values and base defaults.

    Only includes parameters that differ from the base configuration.

    Args:
        config_values: Parsed config parameter values
        base_defaults: RalphBaseConfig default values

    Returns:
        Dictionary of parameters that differ from defaults
    """
    deltas = {}

    for name, value in config_values.items():
        # Skip if this isn't a recognized config parameter
        if name not in base_defaults:
            continue

        # Include if different from default
        if value != base_defaults[name]:
            deltas[name] = value

    return deltas


def format_delta_value(value: Any) -> str:
    """Format a parameter value for Python code generation.

    Args:
        value: Parameter value to format

    Returns:
        Python code string representation
    """
    if isinstance(value, str):
        return f'"{value}"'
    elif isinstance(value, bool):
        return str(value)
    elif isinstance(value, (int, float)):
        return str(value)
    elif value is None:
        return 'None'
    elif isinstance(value, (list, tuple)):
        return str(value)
    else:
        return repr(value)


def generate_builder_code(iteration: int, deltas: dict[str, Any], comment_header: str) -> str:
    """Generate builder-based configuration code for an iteration.

    Args:
        iteration: Ralph iteration number
        deltas: Dictionary of parameter overrides
        comment_header: Original comment header from config file

    Returns:
        Python code defining the configuration function
    """
    lines = []

    # Add comment header if present (before function definition)
    if comment_header:
        # Split multiline headers and add each line
        for line in comment_header.split('\n'):
            lines.append(line)
        lines.append('')

    # Generate function definition
    lines.append(f'def ralph_iter{iteration}() -> RalphBaseConfig:')
    lines.append(f'    """Ralph iteration {iteration} configuration."""')

    if deltas:
        lines.append('    return RalphConfigBuilder().with_overrides(')

        # Format delta parameters
        delta_items = sorted(deltas.items())
        for i, (name, value) in enumerate(delta_items):
            formatted_value = format_delta_value(value)
            comma = ',' if i < len(delta_items) - 1 else ''
            lines.append(f'        {name}={formatted_value}{comma}')

        lines.append('    ).build()')
    else:
        lines.append('    # No deltas - uses base config defaults')
        lines.append('    return RalphConfigBuilder().build()')

    lines.append('')
    lines.append('')

    return '\n'.join(lines)


def find_ralph_iter_files(config_dir: Path) -> list[tuple[int, Path]]:
    """Find all ralph_iter*.py files and extract iteration numbers.

    Args:
        config_dir: Directory containing config files

    Returns:
        List of (iteration_number, file_path) tuples, sorted by iteration
    """
    files = []
    pattern = re.compile(r'ralph_iter(\d+)\.py$')

    for file_path in config_dir.glob('ralph_iter*.py'):
        match = pattern.search(file_path.name)
        if match:
            iteration = int(match.group(1))
            files.append((iteration, file_path))

    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description='Migrate ralph_iter config files to composition pattern'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Report what would be migrated without generating output'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output file path for generated iterations.py'
    )
    parser.add_argument(
        '--iteration',
        type=int,
        help='Process only a specific iteration (for testing)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed delta information'
    )
    parser.add_argument(
        '--config-dir',
        type=Path,
        default=Path('config'),
        help='Directory containing ralph_iter files (default: config/)'
    )

    args = parser.parse_args()

    # Find all ralph_iter files
    ralph_files = find_ralph_iter_files(args.config_dir)

    if not ralph_files:
        print(f"Error: No ralph_iter files found in {args.config_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(ralph_files)} ralph_iter files")

    if args.dry_run:
        # Just list what would be migrated
        for iteration, file_path in ralph_files:
            print(f"  ralph_iter{iteration}: {file_path.name}")
        return

    # Filter to specific iteration if requested
    if args.iteration is not None:
        ralph_files = [(it, path) for it, path in ralph_files if it == args.iteration]
        if not ralph_files:
            print(f"Error: ralph_iter{args.iteration} not found", file=sys.stderr)
            sys.exit(1)

    # Get base config defaults
    base_defaults = get_base_config_defaults()

    # Process each file
    iterations_data = []
    for iteration, file_path in ralph_files:
        if args.verbose:
            print(f"\nProcessing ralph_iter{iteration}...")

        # Extract comment header
        comment_header = extract_comment_header(file_path)

        # Parse config values
        config_values = parse_config_file(file_path)

        # Compute deltas
        deltas = compute_deltas(config_values, base_defaults)

        if args.verbose:
            print(f"  Found {len(deltas)} deltas from base config:")
            for name, value in sorted(deltas.items()):
                base_val = base_defaults.get(name, '<not in base>')
                print(f"    {name}: {base_val} -> {value}")

        iterations_data.append((iteration, deltas, comment_header))

    # Generate output
    if args.output:
        generate_iterations_file(args.output, iterations_data)
        print(f"\nGenerated {args.output}")
    elif args.iteration is None:
        # Print to stdout if no output file specified and not single iteration
        output_lines = generate_iterations_content(iterations_data)
        print('\n'.join(output_lines))


def generate_iterations_content(iterations_data: list[tuple[int, dict[str, Any], str]]) -> list[str]:
    """Generate the content for iterations.py module.

    Args:
        iterations_data: List of (iteration, deltas, comment_header) tuples

    Returns:
        List of lines for the output file
    """
    lines = [
        '"""Ralph Loop iteration configurations using composition pattern.',
        '',
        'This module defines all Ralph Loop experimental configurations using the',
        'RalphConfigBuilder pattern. Each iteration is a function returning a',
        'RalphBaseConfig with specific parameter overrides.',
        '',
        'Generated by scripts/migrate_ralph_configs.py',
        '"""',
        '',
        'from neuromanifold_gpt.config.ralph_base import RalphBaseConfig',
        'from neuromanifold_gpt.config.ralph_builder import RalphConfigBuilder',
        '',
        '',
    ]

    # Generate function for each iteration
    for iteration, deltas, comment_header in iterations_data:
        builder_code = generate_builder_code(iteration, deltas, comment_header)
        lines.append(builder_code)

    return lines


def generate_iterations_file(output_path: Path, iterations_data: list[tuple[int, dict[str, Any], str]]) -> None:
    """Write the generated iterations.py file.

    Args:
        output_path: Path where iterations.py should be written
        iterations_data: List of (iteration, deltas, comment_header) tuples
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = generate_iterations_content(iterations_data)

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
        f.write('\n')


if __name__ == '__main__':
    main()
