"""
Config preset discovery utilities for NeuroManifoldGPT.

Provides utilities to discover and list available config presets from
config/presets/ directory with rich table UI showing preset descriptions,
model size estimates, and key settings.
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, NamedTuple

from rich.console import Console
from rich.table import Table

console = Console()


class PresetInfo(NamedTuple):
    """Information about a config preset."""
    name: str
    description: str
    n_layer: Optional[int]
    n_head: Optional[int]
    n_embd: Optional[int]
    max_iters: Optional[int]
    file_path: str


def _extract_description(file_path: Path) -> str:
    """Extract description from header comments in preset file.

    Args:
        file_path: Path to preset file

    Returns:
        First non-empty comment line from file header
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Extract first comment line (skip shebang if present)
        for line in lines:
            line = line.strip()
            if line.startswith('#') and not line.startswith('#!'):
                # Remove leading # and whitespace
                description = line.lstrip('#').strip()
                if description:
                    return description

        return "No description available"
    except Exception:
        return "Error reading description"


def _extract_key_settings(file_path: Path) -> Dict[str, Optional[int]]:
    """Extract key configuration settings from preset file.

    Args:
        file_path: Path to preset file

    Returns:
        Dictionary with keys: n_layer, n_head, n_embd, max_iters
        Values are None if not found in file.
    """
    settings = {
        'n_layer': None,
        'n_head': None,
        'n_embd': None,
        'max_iters': None,
        'use_nano_config': False
    }

    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Extract each setting using regex
        for key in settings.keys():
            if key == 'use_nano_config':
                # Special handling for boolean
                match = re.search(rf'{key}\s*=\s*(True|False)', content)
                if match:
                    settings[key] = match.group(1) == 'True'
            else:
                match = re.search(rf'{key}\s*=\s*(\d+)', content)
                if match:
                    settings[key] = int(match.group(1))

        # If use_nano_config is True, apply nano defaults
        if settings['use_nano_config']:
            if settings['n_layer'] is None:
                settings['n_layer'] = 4
            if settings['n_head'] is None:
                settings['n_head'] = 4
            if settings['n_embd'] is None:
                settings['n_embd'] = 128

    except Exception:
        pass

    return settings


def _estimate_model_size(n_layer: Optional[int], n_embd: Optional[int]) -> str:
    """Estimate model parameter count based on architecture.

    Args:
        n_layer: Number of transformer layers
        n_embd: Embedding dimension

    Returns:
        Human-readable parameter count estimate (e.g., "~124M")
    """
    if n_layer is None or n_embd is None:
        return "N/A"

    # Rough parameter estimate for transformer:
    # params ≈ 12 * n_layer * n_embd^2 (for attention, FFN, etc)
    params = 12 * n_layer * (n_embd ** 2)

    if params < 1_000_000:
        return f"~{params // 1000}K"
    elif params < 1_000_000_000:
        return f"~{params // 1_000_000}M"
    else:
        return f"~{params / 1_000_000_000:.1f}B"


def _estimate_training_time(max_iters: Optional[int]) -> str:
    """Estimate training time based on max_iters.

    Args:
        max_iters: Maximum training iterations

    Returns:
        Human-readable time estimate (e.g., "~8h", "~2d")
    """
    if max_iters is None:
        return "N/A"

    # Rough estimates assuming ~1 iter/sec on modern GPU
    # This varies widely based on model size and hardware
    hours = max_iters / 3600

    if hours < 1:
        return f"~{int(hours * 60)}m"
    elif hours < 24:
        return f"~{int(hours)}h"
    else:
        days = hours / 24
        return f"~{int(days)}d"


def _scan_presets(directory: str) -> List[PresetInfo]:
    """Scan directory for preset configuration files.

    Args:
        directory: Directory path to scan for presets

    Returns:
        List of PresetInfo objects for each preset found
    """
    presets = []
    dir_path = Path(directory)

    if not dir_path.exists():
        return presets

    # Find all .py files in presets directory
    for preset_file in sorted(dir_path.glob("*.py")):
        if preset_file.is_file() and preset_file.name != "__init__.py":
            # Extract preset name (filename without .py)
            name = preset_file.stem

            # Extract description and settings
            description = _extract_description(preset_file)
            settings = _extract_key_settings(preset_file)

            presets.append(PresetInfo(
                name=name,
                description=description,
                n_layer=settings['n_layer'],
                n_head=settings['n_head'],
                n_embd=settings['n_embd'],
                max_iters=settings['max_iters'],
                file_path=str(preset_file)
            ))

    return presets


def list_presets(directory: str = "neuromanifold_gpt/config/presets", show_table: bool = True) -> List[str]:
    """List all available config presets with details.

    Args:
        directory: Directory containing preset files
        show_table: If True, display rich table of presets

    Returns:
        List of preset names found
    """
    presets = _scan_presets(directory)

    if not presets:
        if show_table:
            console.print(f"[yellow]No presets found in {directory}[/yellow]")
        return []

    # Display rich table if requested
    if show_table:
        table = Table(title=f"Available Config Presets")
        table.add_column("Preset Name", style="cyan", width=18)
        table.add_column("Description", style="white", width=40)
        table.add_column("Model Size", justify="right", style="yellow", width=12)
        table.add_column("Training Time", justify="right", style="blue", width=14)
        table.add_column("Key Settings", style="green", width=20)

        for preset in presets:
            # Format key settings
            if preset.n_layer and preset.n_head and preset.n_embd:
                key_settings = f"{preset.n_layer}L/{preset.n_head}H/{preset.n_embd}E"
            else:
                key_settings = "N/A"

            # Estimate sizes
            model_size = _estimate_model_size(preset.n_layer, preset.n_embd)
            training_time = _estimate_training_time(preset.max_iters)

            table.add_row(
                preset.name,
                preset.description,
                model_size,
                training_time,
                key_settings
            )

        console.print(table)

    return [preset.name for preset in presets]


def get_preset_info(preset_name: str, directory: str = "neuromanifold_gpt/config/presets") -> Optional[Dict]:
    """Get detailed information about a specific preset.

    Args:
        preset_name: Name of the preset (without .py extension)
        directory: Directory containing preset files

    Returns:
        Dictionary with preset information, or None if preset not found
    """
    preset_file = Path(directory) / f"{preset_name}.py"

    if not preset_file.exists():
        return None

    description = _extract_description(preset_file)
    settings = _extract_key_settings(preset_file)

    return {
        'name': preset_name,
        'description': description,
        'file_path': str(preset_file),
        'n_layer': settings['n_layer'],
        'n_head': settings['n_head'],
        'n_embd': settings['n_embd'],
        'max_iters': settings['max_iters'],
        'model_size': _estimate_model_size(settings['n_layer'], settings['n_embd']),
        'training_time': _estimate_training_time(settings['max_iters'])
    }
