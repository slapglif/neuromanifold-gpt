"""
Checkpoint scanner utility with progress indicators.

Provides utilities to scan directories for PyTorch checkpoint files
with progress feedback using rich.progress.
"""
import os
from pathlib import Path
from typing import List, Optional, Tuple

from .progress import create_progress_bar


def scan_checkpoints(
    directory: str,
    pattern: str = "*.pt",
    recursive: bool = True,
    show_progress: bool = True,
) -> List[Tuple[str, int]]:
    """
    Scan a directory for checkpoint files with optional progress indicator.

    Args:
        directory: Directory path to scan
        pattern: File pattern to match (default: "*.pt")
        recursive: Whether to scan subdirectories recursively (default: True)
        show_progress: Whether to show progress bar during scanning (default: True)

    Returns:
        List of tuples containing (filepath, filesize_bytes) for each checkpoint found,
        sorted by modification time (newest first)

    Example:
        >>> checkpoints = scan_checkpoints("./out", pattern="ckpt*.pt")
        >>> for path, size in checkpoints:
        ...     print(f"{path}: {size / 1e6:.1f} MB")
    """
    directory_path = Path(directory)

    if not directory_path.exists():
        raise ValueError(f"Directory does not exist: {directory}")

    if not directory_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    # First pass: count files to scan for progress bar
    if recursive:
        all_files = list(directory_path.rglob("*"))
    else:
        all_files = list(directory_path.glob("*"))

    total_files = len([f for f in all_files if f.is_file()])

    # Second pass: scan files and filter by pattern
    checkpoints = []

    if show_progress and total_files > 0:
        progress = create_progress_bar("Scanning checkpoints", total=total_files)
        with progress:
            task_id = progress.add_task("Scanning", total=total_files)

            for file_path in all_files:
                if file_path.is_file():
                    if file_path.match(pattern):
                        size = file_path.stat().st_size
                        checkpoints.append((str(file_path), size))
                    progress.update(task_id, advance=1)
    else:
        # No progress bar - just scan
        for file_path in all_files:
            if file_path.is_file() and file_path.match(pattern):
                size = file_path.stat().st_size
                checkpoints.append((str(file_path), size))

    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda x: os.path.getmtime(x[0]), reverse=True)

    return checkpoints


def find_latest_checkpoint(
    directory: str,
    pattern: str = "*.pt",
    show_progress: bool = False,
) -> Optional[str]:
    """
    Find the most recent checkpoint file in a directory.

    Args:
        directory: Directory path to scan
        pattern: File pattern to match (default: "*.pt")
        show_progress: Whether to show progress bar during scanning (default: False)

    Returns:
        Path to the most recent checkpoint file, or None if no checkpoints found

    Example:
        >>> latest = find_latest_checkpoint("./out")
        >>> if latest:
        ...     checkpoint = torch.load(latest)
    """
    checkpoints = scan_checkpoints(
        directory, pattern, recursive=True, show_progress=show_progress
    )

    if not checkpoints:
        return None

    # Already sorted by modification time, newest first
    return checkpoints[0][0]


def get_checkpoint_info(checkpoint_path: str) -> dict:
    """
    Get metadata information about a checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Dictionary containing checkpoint metadata:
            - path: Full path to checkpoint
            - size_bytes: File size in bytes
            - size_mb: File size in megabytes
            - modified: Last modification timestamp

    Example:
        >>> info = get_checkpoint_info("out/ckpt.pt")
        >>> print(f"Size: {info['size_mb']:.1f} MB")
    """
    path = Path(checkpoint_path)

    if not path.exists():
        raise ValueError(f"Checkpoint file does not exist: {checkpoint_path}")

    stat = path.stat()

    return {
        "path": str(path.absolute()),
        "size_bytes": stat.st_size,
        "size_mb": stat.st_size / 1e6,
        "modified": stat.st_mtime,
    }
