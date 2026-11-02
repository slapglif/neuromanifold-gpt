"""
Progress indicators for long-running operations.

Provides reusable progress bar utilities using rich.progress for:
- Checkpoint loading operations (torch.load and load_state_dict)
- Evaluation loops with ETA
- File scanning operations
"""
from contextlib import contextmanager
from typing import Optional

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def create_progress_bar(description: str = "Processing", total: Optional[int] = None):
    """
    Create a rich progress bar with standard columns.

    Args:
        description: Description text to show before the progress bar
        total: Total number of items (if None, shows a spinner instead of a bar)

    Returns:
        Progress instance configured with standard columns

    Example:
        >>> with create_progress_bar("Loading checkpoint", total=100) as progress:
        ...     task_id = progress.add_task("loading", total=100)
        ...     for i in range(100):
        ...         progress.update(task_id, advance=1)
    """
    if total is None:
        # Indeterminate progress - use spinner
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
        )
    else:
        # Determinate progress - use bar with ETA
        return Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
        )


@contextmanager
def checkpoint_progress(description: str = "Loading checkpoint"):
    """
    Context manager for showing progress during checkpoint operations.

    This is specifically designed for torch.load() and load_state_dict()
    operations which don't provide native progress callbacks.

    Args:
        description: Description of the checkpoint operation

    Yields:
        None (shows spinner during the context)

    Example:
        >>> with checkpoint_progress("Loading model checkpoint"):
        ...     checkpoint = torch.load('ckpt.pt')
        ...     model.load_state_dict(checkpoint['model'])
    """
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
    )

    with progress:
        task_id = progress.add_task(description, total=None)
        try:
            yield progress
        finally:
            progress.update(task_id, completed=True)


def progress_bar(
    iterable, description: str = "Processing", total: Optional[int] = None
):
    """
    Wrap an iterable with a progress bar.

    Args:
        iterable: The iterable to wrap
        description: Description text to show
        total: Total number of items (if None, tries to get len(iterable))

    Yields:
        Items from the iterable

    Example:
        >>> for item in progress_bar(range(100), "Processing items"):
        ...     process(item)
    """
    if total is None:
        try:
            total = len(iterable)
        except (TypeError, AttributeError):
            # Iterable doesn't support len(), use indeterminate progress
            pass

    progress = create_progress_bar(description, total)

    with progress:
        task_id = progress.add_task(description, total=total)
        for item in iterable:
            yield item
            progress.update(task_id, advance=1)
