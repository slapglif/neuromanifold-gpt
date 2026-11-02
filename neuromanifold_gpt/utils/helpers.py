"""
Helper utility functions.

Provides basic utility functions for value checking and defaults:
- exists(): Check if a value is not None
- default(): Return value or default if None
"""
from typing import Optional, TypeVar

T = TypeVar("T")


def exists(v) -> bool:
    """Check if a value is not None.

    Args:
        v: Value to check

    Returns:
        True if v is not None, False otherwise

    Example:
        >>> exists(1)
        True
        >>> exists(None)
        False
        >>> exists(0)
        True
        >>> exists("")
        True
    """
    return v is not None


def default(v: Optional[T], d: T) -> T:
    """Return value v if it exists, otherwise return default d.

    Args:
        v: Value to check
        d: Default value to return if v is None

    Returns:
        v if v is not None, else d

    Example:
        >>> default(None, 5)
        5
        >>> default(10, 5)
        10
        >>> default(0, 5)
        0
        >>> default("", "default")
        ''
    """
    return v if exists(v) else d
