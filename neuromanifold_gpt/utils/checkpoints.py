"""
Checkpoint selection utilities for NeuroManifoldGPT.

Provides interactive checkpoint selection with rich table UI showing
validation loss, file age, and size.
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt

console = Console()


def _format_age(timestamp: float) -> str:
    """Format file age as human-readable string."""
    age_seconds = datetime.now().timestamp() - timestamp

    if age_seconds < 60:
        return f"{int(age_seconds)}s ago"
    elif age_seconds < 3600:
        return f"{int(age_seconds / 60)}m ago"
    elif age_seconds < 86400:
        return f"{int(age_seconds / 3600)}h ago"
    else:
        return f"{int(age_seconds / 86400)}d ago"


def _format_size(size_bytes: int) -> str:
    """Format file size as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f}MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f}GB"


def _extract_val_loss(filename: str) -> Optional[float]:
    """Extract validation loss from checkpoint filename.

    Supports patterns like:
    - ckpt-000123-1.2345.ckpt (Lightning format)
    - ckpt-{step}-{val/loss}.ckpt
    - checkpoint_loss_1.2345.pt
    """
    # Try Lightning format: ckpt-{step}-{val_loss}.ckpt
    match = re.search(r'ckpt-\d+-(\d+\.\d+)\.ckpt', filename)
    if match:
        return float(match.group(1))

    # Try alternative format: loss_1.2345
    match = re.search(r'loss[_-](\d+\.\d+)', filename, re.IGNORECASE)
    if match:
        return float(match.group(1))

    # Try format with val in name: val_1.2345 or val-1.2345
    match = re.search(r'val[_-](\d+\.\d+)', filename, re.IGNORECASE)
    if match:
        return float(match.group(1))

    return None


def _scan_checkpoints(directory: str) -> List[Tuple[str, Optional[float], float, int]]:
    """Scan directory for checkpoint files.

    Returns:
        List of tuples: (filename, val_loss, age_timestamp, size_bytes)
    """
    checkpoints = []
    dir_path = Path(directory)

    if not dir_path.exists():
        return checkpoints

    # Find all checkpoint files
    patterns = ['*.pt', '*.ckpt', '*.pth']
    for pattern in patterns:
        for ckpt_file in dir_path.glob(pattern):
            if ckpt_file.is_file():
                stat = ckpt_file.stat()
                val_loss = _extract_val_loss(ckpt_file.name)
                checkpoints.append((
                    ckpt_file.name,
                    val_loss,
                    stat.st_mtime,
                    stat.st_size
                ))

    return checkpoints


def select_checkpoint(
    directory: str = "out",
    auto_select_best: bool = False,
    show_table: bool = True
) -> Optional[str]:
    """Interactively select a checkpoint file from a directory.

    Args:
        directory: Directory to scan for checkpoints
        auto_select_best: If True, automatically select checkpoint with lowest val_loss
        show_table: If True, display rich table of available checkpoints

    Returns:
        Full path to selected checkpoint, or None if no checkpoints found
    """
    checkpoints = _scan_checkpoints(directory)

    if not checkpoints:
        console.print(f"[yellow]No checkpoints found in {directory}[/yellow]")
        return None

    # Sort by val_loss (None values last), then by age (newest first)
    checkpoints.sort(key=lambda x: (
        x[1] if x[1] is not None else float('inf'),  # val_loss
        -x[2]  # negative age for newest first
    ))

    # Auto-select best checkpoint if requested
    if auto_select_best:
        best_ckpt = checkpoints[0]
        ckpt_path = os.path.join(directory, best_ckpt[0])
        if show_table:
            console.print(f"[green]Auto-selected best checkpoint:[/green] {best_ckpt[0]}")
        return ckpt_path

    # Display table
    if show_table:
        table = Table(title=f"Available Checkpoints in {directory}")
        table.add_column("#", style="cyan", width=4)
        table.add_column("Filename", style="white")
        table.add_column("Val Loss", justify="right", style="yellow")
        table.add_column("Age", justify="right", style="blue")
        table.add_column("Size", justify="right", style="green")

        for idx, (filename, val_loss, mtime, size) in enumerate(checkpoints, 1):
            loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
            age_str = _format_age(mtime)
            size_str = _format_size(size)

            table.add_row(
                str(idx),
                filename,
                loss_str,
                age_str,
                size_str
            )

        console.print(table)

    # Prompt for selection
    if len(checkpoints) == 1:
        console.print(f"[green]Using only available checkpoint:[/green] {checkpoints[0][0]}")
        return os.path.join(directory, checkpoints[0][0])

    # Interactive selection
    while True:
        choice = Prompt.ask(
            f"\nSelect checkpoint [1-{len(checkpoints)}]",
            default="1"
        )

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(checkpoints):
                selected = checkpoints[idx]
                ckpt_path = os.path.join(directory, selected[0])
                console.print(f"[green]Selected:[/green] {selected[0]}")
                return ckpt_path
            else:
                console.print(f"[red]Invalid choice. Please enter 1-{len(checkpoints)}[/red]")
        except ValueError:
            console.print(f"[red]Invalid input. Please enter a number[/red]")


def find_best_checkpoint(directory: str = "out") -> Optional[str]:
    """Find checkpoint with lowest validation loss.

    Args:
        directory: Directory to scan for checkpoints

    Returns:
        Full path to best checkpoint, or None if no checkpoints found
    """
    return select_checkpoint(directory, auto_select_best=True, show_table=False)


def list_checkpoints(directory: str = "out") -> List[str]:
    """List all checkpoint files in directory.

    Args:
        directory: Directory to scan for checkpoints

    Returns:
        List of checkpoint filenames (not full paths)
    """
    checkpoints = _scan_checkpoints(directory)
    return [ckpt[0] for ckpt in checkpoints]
