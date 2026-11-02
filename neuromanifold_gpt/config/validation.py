"""Dependency validation for NeuroManifoldGPT.

This module provides runtime validation of dependencies to detect version
incompatibilities early and provide helpful error messages. It checks:
- PyTorch >= 2.0
- Lightning >= 2.0
- Other required packages (einops, scipy, loguru, rich)
- CUDA availability and compatibility
- Known incompatibility patterns

The validation runs at startup before heavy imports to fail fast with
clear guidance for resolving issues.
"""

import sys
from importlib.metadata import PackageNotFoundError, version
from typing import List, Optional, Tuple

# Minimum required versions from pyproject.toml
REQUIRED_VERSIONS = {
    "torch": "2.0",
    "lightning": "2.0",
    "einops": "0.7",
    "scipy": "1.10",
    "loguru": "0.7",
    "rich": "13.0",
}


def _parse_version(version_str: str) -> Tuple[int, ...]:
    """Parse version string into tuple of integers for comparison.

    Args:
        version_str: Version string like "2.0.1" or "2.0.1+cu118"

    Returns:
        Tuple of version numbers, e.g., (2, 0, 1)
    """
    # Strip build metadata (e.g., +cu118)
    version_str = version_str.split("+")[0]
    # Parse numeric components
    try:
        return tuple(int(x) for x in version_str.split(".")[:3])
    except (ValueError, AttributeError):
        # Fall back to 0 if parsing fails
        return (0,)


def _check_package_version(
    package_name: str, min_version: str
) -> Tuple[bool, Optional[str], Optional[str]]:
    """Check if a package meets the minimum version requirement.

    Args:
        package_name: Name of the package to check
        min_version: Minimum required version string

    Returns:
        Tuple of (is_valid, installed_version, error_message)
        - is_valid: True if requirement is met
        - installed_version: Version string of installed package, None if not found
        - error_message: Error message if requirement not met, None otherwise
    """
    try:
        installed_version = version(package_name)
    except PackageNotFoundError:
        return False, None, f"{package_name} is not installed"

    installed_tuple = _parse_version(installed_version)
    required_tuple = _parse_version(min_version)

    if installed_tuple >= required_tuple:
        return True, installed_version, None
    else:
        return (
            False,
            installed_version,
            f"{package_name}>={min_version} required, but {installed_version} is installed",
        )


def _check_cuda_availability() -> Tuple[bool, Optional[str]]:
    """Check if CUDA is available and functional.

    Returns:
        Tuple of (has_cuda, message)
        - has_cuda: True if CUDA is available
        - message: Warning or info message about CUDA status
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False, (
                "CUDA is not available. Training will run on CPU (very slow). "
                "For GPU training, ensure CUDA-enabled PyTorch is installed."
            )

        # Check CUDA version compatibility
        cuda_version = torch.version.cuda
        if cuda_version:
            return (
                True,
                f"CUDA {cuda_version} is available with {torch.cuda.device_count()} GPU(s)",
            )
        else:
            return False, "PyTorch reports CUDA available but version unknown"

    except ImportError:
        return False, "Cannot import torch to check CUDA availability"


def _check_known_incompatibilities() -> List[str]:
    """Check for known incompatibility patterns between dependencies.

    Returns:
        List of warning messages for known issues
    """
    warnings_list = []

    try:
        import torch

        torch_version = _parse_version(torch.__version__)

        # Check PyTorch 2.0 specific issues
        if torch_version >= (2, 0, 0) and torch_version < (2, 1, 0):
            try:
                lightning_ver = version("lightning")
                lightning_tuple = _parse_version(lightning_ver)
                if lightning_tuple < (2, 0, 2):
                    warnings_list.append(
                        "PyTorch 2.0.x with Lightning <2.0.2 may have compilation issues. "
                        "Consider upgrading Lightning to >=2.0.2"
                    )
            except PackageNotFoundError:
                pass

    except ImportError:
        pass

    return warnings_list


def validate_dependencies(verbose: bool = True) -> bool:
    """Validate all dependencies meet version requirements.

    Checks that all required packages are installed with compatible versions,
    CUDA is available if needed, and no known incompatibilities exist.
    Prints informative messages about any issues found.

    Args:
        verbose: If True, print detailed information about checks

    Returns:
        True if all requirements are met, False if critical issues found

    Raises:
        SystemExit: If critical dependencies are missing or incompatible
    """
    all_valid = True
    errors = []
    warnings_list = []

    # Check each required package
    for package, min_version in REQUIRED_VERSIONS.items():
        is_valid, installed_ver, error_msg = _check_package_version(
            package, min_version
        )

        if not is_valid:
            all_valid = False
            errors.append(error_msg)
        elif verbose and installed_ver:
            print(f"✓ {package} {installed_ver} (>={min_version} required)")

    # Check CUDA availability (warning only, not an error)
    has_cuda, cuda_msg = _check_cuda_availability()
    if verbose:
        if has_cuda:
            print(f"✓ {cuda_msg}")
        else:
            print(f"⚠ {cuda_msg}")
            warnings_list.append(cuda_msg)

    # Check for known incompatibilities
    known_issues = _check_known_incompatibilities()
    warnings_list.extend(known_issues)

    # Report results
    if errors:
        print("\n❌ Dependency validation failed:\n", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        print("\nTo install missing dependencies, run:", file=sys.stderr)
        print("  pip install -e .", file=sys.stderr)
        sys.exit(1)

    if warnings_list and verbose:
        print("\n⚠️  Warnings:", file=sys.stderr)
        for warning in warnings_list:
            print(f"  - {warning}", file=sys.stderr)

    if verbose and all_valid and not warnings_list:
        print("\n✓ All dependencies validated successfully!")

    return all_valid


if __name__ == "__main__":
    """Run validation when module is executed directly."""
    validate_dependencies(verbose=True)
