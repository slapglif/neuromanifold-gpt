"""
Checkpoint selection utilities for NeuroManifoldGPT.

Provides interactive checkpoint selection with rich table UI showing
validation loss, file age, and size.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

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
    match = re.search(r"ckpt-\d+-(\d+\.\d+)\.ckpt", filename)
    if match:
        return float(match.group(1))

    # Try alternative format: loss_1.2345
    match = re.search(r"loss[_-](\d+\.\d+)", filename, re.IGNORECASE)
    if match:
        return float(match.group(1))

    # Try format with val in name: val_1.2345 or val-1.2345
    match = re.search(r"val[_-](\d+\.\d+)", filename, re.IGNORECASE)
    if match:
        return float(match.group(1))

    return None


def _extract_training_step(filename: str) -> Optional[int]:
    """Extract training step from checkpoint filename.

    Supports patterns like:
    - ckpt-000123-1.2345.ckpt (Lightning format)
    - ckpt-{step}-{val_loss}.ckpt
    - checkpoint_step_12345.pt
    - iter_12345.pt
    """
    # Try Lightning format: ckpt-{step}-{val_loss}.ckpt
    match = re.search(r"ckpt-(\d+)-\d+\.\d+\.ckpt", filename)
    if match:
        return int(match.group(1))

    # Try step in name: step_12345 or step-12345
    match = re.search(r"step[_-](\d+)", filename, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Try iter in name: iter_12345 or iter-12345
    match = re.search(r"iter[_-](\d+)", filename, re.IGNORECASE)
    if match:
        return int(match.group(1))

    return None


def _scan_checkpoints(directory: str) -> List[Tuple[str, Optional[float], float, int]]:
    """Scan directory for checkpoint files.

    Supports both unified and separated checkpoint formats:
    - Unified: Single .pt/.ckpt/.pth file
    - Separated: Pairs of -model.pt and -optimizer.pt files

    For separated checkpoints, the -model.pt file is returned and size includes
    both model and optimizer files (if present).

    Returns:
        List of tuples: (filename, val_loss, age_timestamp, size_bytes)
    """
    checkpoints = []
    dir_path = Path(directory)

    if not dir_path.exists():
        return checkpoints

    seen_separated = set()  # Track separated checkpoints we've already added

    # Find all checkpoint files
    patterns = ["*.pt", "*.ckpt", "*.pth"]
    for pattern in patterns:
        for ckpt_file in dir_path.glob(pattern):
            if not ckpt_file.is_file():
                continue

            filename = ckpt_file.name

            # Skip optimizer files - they'll be handled with their model files
            if filename.endswith("-optimizer.pt"):
                continue

            # Handle separated checkpoint format
            if filename.endswith("-model.pt"):
                # Get the base name (without -model.pt)
                base_name = filename[:-9]  # Remove '-model.pt'

                # Skip if already processed
                if base_name in seen_separated:
                    continue
                seen_separated.add(base_name)

                # Get model file stats
                model_stat = ckpt_file.stat()
                total_size = model_stat.st_size
                mtime = model_stat.st_mtime

                # Check for corresponding optimizer file and add its size
                optimizer_file = dir_path / f"{base_name}-optimizer.pt"
                if optimizer_file.exists():
                    optimizer_stat = optimizer_file.stat()
                    total_size += optimizer_stat.st_size

                val_loss = _extract_val_loss(filename)
                checkpoints.append(
                    (filename, val_loss, mtime, total_size)  # Return the model filename
                )

            # Handle unified checkpoint format
            else:
                # Skip if this looks like it's part of a separated checkpoint
                base_name = filename.rsplit(".", 1)[0]
                model_file = dir_path / f"{base_name}-model.pt"
                if model_file.exists():
                    # This is a base file that has separated versions, skip it
                    continue

                stat = ckpt_file.stat()
                val_loss = _extract_val_loss(filename)
                checkpoints.append((filename, val_loss, stat.st_mtime, stat.st_size))

    return checkpoints


def select_checkpoint(
    directory: str = "out", auto_select_best: bool = False, show_table: bool = True
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
    checkpoints.sort(
        key=lambda x: (
            x[1] if x[1] is not None else float("inf"),  # val_loss
            -x[2],  # negative age for newest first
        )
    )

    # Auto-select best checkpoint if requested
    if auto_select_best:
        best_ckpt = checkpoints[0]
        ckpt_path = os.path.join(directory, best_ckpt[0])
        if show_table:
            console.print(
                f"[green]Auto-selected best checkpoint:[/green] {best_ckpt[0]}"
            )
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

            table.add_row(str(idx), filename, loss_str, age_str, size_str)

        console.print(table)

    # Prompt for selection
    if len(checkpoints) == 1:
        console.print(
            f"[green]Using only available checkpoint:[/green] {checkpoints[0][0]}"
        )
        return os.path.join(directory, checkpoints[0][0])

    # Interactive selection
    while True:
        choice = Prompt.ask(f"\nSelect checkpoint [1-{len(checkpoints)}]", default="1")

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(checkpoints):
                selected = checkpoints[idx]
                ckpt_path = os.path.join(directory, selected[0])
                console.print(f"[green]Selected:[/green] {selected[0]}")
                return ckpt_path
            else:
                console.print(
                    f"[red]Invalid choice. Please enter 1-{len(checkpoints)}[/red]"
                )
        except ValueError:
            console.print("[red]Invalid input. Please enter a number[/red]")


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


def export_checkpoints_metadata(
    directory: str = "out", output_file: str = "checkpoints_metadata.json"
) -> None:
    """Export checkpoint metadata to a JSON file.

    Scans the directory for checkpoint files and exports their metadata including
    validation loss, training step, timestamp, age, and file size information.

    Args:
        directory: Directory to scan for checkpoints (default: "out")
        output_file: Path to output JSON file (default: "checkpoints_metadata.json")

    Raises:
        ValueError: If directory does not exist
        IOError: If unable to write output file

    Example:
        >>> export_checkpoints_metadata("out", "checkpoints.json")
        >>> # Creates checkpoints.json with metadata for all checkpoints in out/
    """
    dir_path = Path(directory)

    # Validate directory exists
    if not dir_path.exists():
        raise ValueError(f"Directory does not exist: {directory}")

    if not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    # Scan for checkpoints
    checkpoints_data = _scan_checkpoints(directory)

    # Build checkpoint metadata list
    checkpoints_list = []
    for filename, val_loss, mtime, size_bytes in checkpoints_data:
        # Extract training step if present
        training_step = _extract_training_step(filename)

        # Format age as human-readable string
        age = _format_age(mtime)

        checkpoint_info = {
            "filename": filename,
            "val_loss": val_loss,  # None will be serialized as null in JSON
            "training_step": training_step,  # None will be serialized as null in JSON
            "timestamp": mtime,
            "age": age,
            "size_bytes": size_bytes,
            "size_mb": size_bytes / 1e6,
        }
        checkpoints_list.append(checkpoint_info)

    # Build output structure with metadata header
    output_data = {
        "metadata": {
            "exported_at": datetime.now().isoformat(),
            "directory": str(dir_path.absolute()),
            "checkpoint_count": len(checkpoints_list),
        },
        "checkpoints": checkpoints_list,
    }

    # Write to JSON file
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

    except IOError as e:
        raise IOError(f"Failed to write output file {output_file}: {e}")
