"""Export discovered architectures to Python config files.

This module provides functions to export NAS-discovered architectures to
reusable Python configuration files that can be used for training.

Key features:
- Export ArchitectureConfig to Python config files
- Include performance metrics and metadata in comments
- Generate human-readable, well-documented configs
- Create output directory structure automatically

Example:
    >>> from neuromanifold_gpt.nas.search_space import SearchSpace
    >>> from neuromanifold_gpt.nas.export import export_config
    >>>
    >>> search_space = SearchSpace()
    >>> arch = search_space.sample()
    >>> export_config(
    ...     architecture=arch,
    ...     output_path="config/nas_discovered/arch_001.py",
    ...     description="Top architecture from random search"
    ... )
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from loguru import logger

from neuromanifold_gpt.nas.search_space import ArchitectureConfig


def export_config(
    architecture: ArchitectureConfig,
    output_path: str,
    description: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
    training_params: Optional[Dict[str, Any]] = None,
) -> None:
    """Export an ArchitectureConfig to a Python config file.

    Creates a Python configuration file that can be imported and used for training.
    The generated file follows the pattern of existing config files in the codebase.

    Args:
        architecture: ArchitectureConfig to export
        output_path: Path to save the config file (e.g., "config/nas_discovered/arch_001.py")
        description: Human-readable description of the architecture (optional)
        metrics: Performance metrics dictionary (e.g., {"perplexity": 10.5, "loss": 2.3})
        training_params: Additional training parameters to include (e.g., learning_rate, batch_size)

    Example:
        >>> export_config(
        ...     architecture=arch,
        ...     output_path="config/nas_discovered/best_arch.py",
        ...     description="Best architecture from evolutionary search",
        ...     metrics={"perplexity": 10.5, "params": 12.3e6}
        ... )
    """
    # Validate architecture
    is_valid, error = architecture.validate()
    if not is_valid:
        raise ValueError(f"Invalid architecture: {error}")

    # Create output directory if needed
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate config content
    config_lines = []

    # Header comment
    config_lines.append("# Neural Architecture Search - Discovered Configuration")
    if description:
        config_lines.append(f"# {description}")
    config_lines.append(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Add architecture ID if available
    if architecture.architecture_id:
        config_lines.append(f"# Architecture ID: {architecture.architecture_id}")

    # Add performance metrics if provided
    if metrics:
        config_lines.append("#")
        config_lines.append("# Performance Metrics:")
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                config_lines.append(f"#   {key}: {value:.4f}")
            elif isinstance(value, int):
                config_lines.append(f"#   {key}: {value:,}")
            else:
                config_lines.append(f"#   {key}: {value}")

    config_lines.append("")

    # Model architecture section
    config_lines.append("# Model Architecture")
    config_lines.append(f"n_layer = {architecture.n_layer}")
    config_lines.append(f"n_embd = {architecture.n_embd}")
    config_lines.append(f"n_heads = {architecture.n_heads}")
    config_lines.append(f"dropout = {architecture.dropout}")
    config_lines.append("")

    # Attention configuration section
    config_lines.append("# Attention Configuration")
    config_lines.append(f"attention_type = \"{architecture.attention_type}\"")
    config_lines.append(f"use_qk_norm = {architecture.use_qk_norm}")
    config_lines.append("")

    # Component choices section
    config_lines.append("# Component Choices")
    config_lines.append(f"use_mhc = {architecture.use_mhc}  # Manifold-Constrained Hyper-Connections")
    config_lines.append(f"use_mla = {architecture.use_mla}  # Multi-Head Latent Attention")
    config_lines.append(f"use_moe = {architecture.use_moe}  # Mixture of Experts")
    config_lines.append(f"use_kan = {architecture.use_kan}  # Kolmogorov-Arnold Networks")
    config_lines.append("")

    # KAN configuration (if enabled)
    if architecture.use_kan:
        config_lines.append("# KAN Configuration")
        config_lines.append(f"kan_type = \"{architecture.kan_type}\"")
        config_lines.append(f"kan_num_centers = {architecture.kan_num_centers}")
        config_lines.append("")

    # FHN dynamics (if using FHN-based attention)
    if architecture.attention_type in ["fhn", "kaufmann"]:
        config_lines.append("# FHN Dynamics")
        config_lines.append(f"fhn_threshold = {architecture.fhn_threshold}")
        config_lines.append(f"fhn_tau = {architecture.fhn_tau}")
        config_lines.append(f"use_fhn_parallel = {architecture.use_fhn_parallel}")
        config_lines.append("")

    # Manifold projection section
    config_lines.append("# Manifold Projection")
    config_lines.append(f"manifold_dim = {architecture.manifold_dim}")
    config_lines.append(f"n_eigenvectors = {architecture.n_eigenvectors}")
    config_lines.append(f"use_multiscale_manifold = {architecture.use_multiscale_manifold}")
    config_lines.append("")

    # Add training parameters if provided
    if training_params:
        config_lines.append("# Training Parameters")
        for key, value in sorted(training_params.items()):
            if isinstance(value, str):
                config_lines.append(f"{key} = \"{value}\"")
            elif isinstance(value, bool):
                config_lines.append(f"{key} = {value}")
            elif isinstance(value, float):
                config_lines.append(f"{key} = {value}")
            elif isinstance(value, int):
                config_lines.append(f"{key} = {value}")
            else:
                config_lines.append(f"{key} = {repr(value)}")
        config_lines.append("")

    # Model type specification
    config_lines.append("# Model Type")
    config_lines.append("model_type = 'neuromanifold'")
    config_lines.append("")

    # Write to file
    config_content = "\n".join(config_lines)
    output_path.write_text(config_content)

    logger.info(f"Exported architecture to {output_path}")


def export_architecture_to_dict(
    architecture: ArchitectureConfig,
    metrics: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Export architecture and metrics to a dictionary.

    This is useful for JSON serialization or programmatic access.

    Args:
        architecture: ArchitectureConfig to export
        metrics: Optional performance metrics to include

    Returns:
        Dictionary containing architecture configuration and metadata

    Example:
        >>> config_dict = export_architecture_to_dict(arch, metrics={"perplexity": 10.5})
        >>> import json
        >>> with open("config.json", "w") as f:
        ...     json.dump(config_dict, f, indent=2)
    """
    export_dict = {
        "architecture": architecture.to_dict(),
        "exported_at": datetime.now().isoformat(),
    }

    if metrics:
        export_dict["metrics"] = metrics

    return export_dict


def generate_config_summary(
    architecture: ArchitectureConfig,
    metrics: Optional[Dict[str, Any]] = None
) -> str:
    """Generate a human-readable summary of an architecture configuration.

    Args:
        architecture: ArchitectureConfig to summarize
        metrics: Optional performance metrics to include

    Returns:
        Multi-line string with formatted architecture summary

    Example:
        >>> summary = generate_config_summary(arch, metrics={"perplexity": 10.5})
        >>> print(summary)
    """
    lines = []

    # Header
    lines.append("=" * 60)
    lines.append("Architecture Summary")
    if architecture.architecture_id:
        lines.append(f"ID: {architecture.architecture_id}")
    lines.append("=" * 60)
    lines.append("")

    # Model size
    lines.append("Model Size:")
    lines.append(f"  Layers: {architecture.n_layer}")
    lines.append(f"  Embedding Dimension: {architecture.n_embd}")
    lines.append(f"  Attention Heads: {architecture.n_heads}")
    lines.append("")

    # Attention
    lines.append("Attention:")
    lines.append(f"  Type: {architecture.attention_type}")
    lines.append(f"  QK Normalization: {architecture.use_qk_norm}")
    lines.append("")

    # Components
    lines.append("Components:")
    components = []
    if architecture.use_mhc:
        components.append("MHC (Manifold Hyper-Connections)")
    if architecture.use_mla:
        components.append("MLA (Multi-Head Latent Attention)")
    if architecture.use_moe:
        components.append("MoE (Mixture of Experts)")
    if architecture.use_kan:
        components.append(f"KAN ({architecture.kan_type})")

    if components:
        for comp in components:
            lines.append(f"  - {comp}")
    else:
        lines.append("  - Standard components only")
    lines.append("")

    # Manifold
    lines.append("Manifold Projection:")
    lines.append(f"  Dimension: {architecture.manifold_dim}")
    lines.append(f"  Eigenvectors: {architecture.n_eigenvectors}")
    lines.append(f"  Multi-scale: {architecture.use_multiscale_manifold}")
    lines.append("")

    # Performance metrics
    if metrics:
        lines.append("Performance Metrics:")
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            elif isinstance(value, int):
                lines.append(f"  {key}: {value:,}")
            else:
                lines.append(f"  {key}: {value}")
        lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)
