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

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
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


def export_to_json(
    architecture: ArchitectureConfig,
    output_path: str,
    description: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
    training_params: Optional[Dict[str, Any]] = None,
) -> None:
    """Export an ArchitectureConfig to a JSON file.

    Creates a JSON file containing the complete architecture configuration,
    metrics, and metadata. This format is ideal for programmatic processing,
    experiment tracking, and integration with other tools.

    Args:
        architecture: ArchitectureConfig to export
        output_path: Path to save the JSON file (e.g., "results/architectures/arch_001.json")
        description: Human-readable description of the architecture (optional)
        metrics: Performance metrics dictionary (e.g., {"perplexity": 10.5, "loss": 2.3})
        training_params: Additional training parameters to include (e.g., learning_rate, batch_size)

    Example:
        >>> export_to_json(
        ...     architecture=arch,
        ...     output_path="results/nas_discovered/best_arch.json",
        ...     description="Best architecture from evolutionary search",
        ...     metrics={"perplexity": 10.5, "loss": 2.3, "params": 12.3e6}
        ... )
    """
    # Validate architecture
    is_valid, error = architecture.validate()
    if not is_valid:
        raise ValueError(f"Invalid architecture: {error}")

    # Create output directory if needed
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Build export dictionary
    export_dict = {
        "architecture": architecture.to_dict(),
        "metadata": {
            "exported_at": datetime.now().isoformat(),
            "architecture_id": architecture.architecture_id,
        }
    }

    # Add description if provided
    if description:
        export_dict["metadata"]["description"] = description

    # Add metrics if provided
    if metrics:
        export_dict["metrics"] = metrics

    # Add training parameters if provided
    if training_params:
        export_dict["training_params"] = training_params

    # Write to JSON file with pretty formatting
    with open(output_path_obj, "w") as f:
        json.dump(export_dict, f, indent=2)

    logger.info(f"Exported architecture to JSON: {output_path}")


def generate_summary_report(
    architectures: List[ArchitectureConfig],
    metrics_list: Optional[List[Dict[str, Any]]] = None,
    output_path: Optional[str] = None,
    top_k: int = 10,
) -> str:
    """Generate a comprehensive summary report for multiple architectures.

    Creates a detailed report summarizing the results of a NAS experiment,
    including statistics about the search space, best architectures, and
    performance distributions.

    Args:
        architectures: List of ArchitectureConfig objects to summarize
        metrics_list: Optional list of metrics dicts, one per architecture
        output_path: Optional path to save the report (e.g., "results/summary_report.txt")
        top_k: Number of top architectures to include in detail (default: 10)

    Returns:
        Multi-line string with formatted summary report

    Example:
        >>> report = generate_summary_report(
        ...     architectures=all_architectures,
        ...     metrics_list=all_metrics,
        ...     output_path="results/nas_summary.txt",
        ...     top_k=5
        ... )
        >>> print(report)
    """
    lines = []

    # Header
    lines.append("=" * 80)
    lines.append("NEURAL ARCHITECTURE SEARCH - SUMMARY REPORT")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total Architectures Evaluated: {len(architectures)}")
    lines.append("")

    # Overall statistics
    lines.append("=" * 80)
    lines.append("SEARCH SPACE EXPLORATION")
    lines.append("=" * 80)
    lines.append("")

    # Component usage statistics
    if architectures:
        lines.append("Component Usage:")
        mhc_count = sum(1 for arch in architectures if arch.use_mhc)
        mla_count = sum(1 for arch in architectures if arch.use_mla)
        moe_count = sum(1 for arch in architectures if arch.use_moe)
        kan_count = sum(1 for arch in architectures if arch.use_kan)

        lines.append(f"  MHC (Manifold Hyper-Connections): {mhc_count}/{len(architectures)} ({100*mhc_count/len(architectures):.1f}%)")
        lines.append(f"  MLA (Multi-Head Latent Attention): {mla_count}/{len(architectures)} ({100*mla_count/len(architectures):.1f}%)")
        lines.append(f"  MoE (Mixture of Experts): {moe_count}/{len(architectures)} ({100*moe_count/len(architectures):.1f}%)")
        lines.append(f"  KAN (Kolmogorov-Arnold Networks): {kan_count}/{len(architectures)} ({100*kan_count/len(architectures):.1f}%)")
        lines.append("")

        # Attention type distribution
        lines.append("Attention Type Distribution:")
        attention_types = {}
        for arch in architectures:
            attention_types[arch.attention_type] = attention_types.get(arch.attention_type, 0) + 1
        for att_type, count in sorted(attention_types.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  {att_type}: {count}/{len(architectures)} ({100*count/len(architectures):.1f}%)")
        lines.append("")

        # Model size statistics
        n_layers = [arch.n_layer for arch in architectures]
        n_embds = [arch.n_embd for arch in architectures]
        lines.append("Model Size Statistics:")
        lines.append(f"  Layers - Min: {min(n_layers)}, Max: {max(n_layers)}, Avg: {sum(n_layers)/len(n_layers):.1f}")
        lines.append(f"  Embedding Dim - Min: {min(n_embds)}, Max: {max(n_embds)}, Avg: {sum(n_embds)/len(n_embds):.1f}")
        lines.append("")

    # Performance metrics summary (if provided)
    if metrics_list and len(metrics_list) == len(architectures):
        lines.append("=" * 80)
        lines.append("PERFORMANCE SUMMARY")
        lines.append("=" * 80)
        lines.append("")

        # Collect all metric keys
        all_metric_keys = set()
        for metrics in metrics_list:
            if metrics:
                all_metric_keys.update(metrics.keys())

        # Compute statistics for each metric
        for metric_key in sorted(all_metric_keys):
            values = [m[metric_key] for m in metrics_list if m and metric_key in m and isinstance(m[metric_key], (int, float))]
            if values:
                lines.append(f"{metric_key}:")
                lines.append(f"  Best: {min(values):.4f}")
                lines.append(f"  Worst: {max(values):.4f}")
                lines.append(f"  Mean: {sum(values)/len(values):.4f}")
                lines.append("")

    # Top architectures (if metrics provided)
    if metrics_list and len(metrics_list) == len(architectures):
        lines.append("=" * 80)
        lines.append(f"TOP {min(top_k, len(architectures))} ARCHITECTURES")
        lines.append("=" * 80)
        lines.append("")

        # Sort by primary metric (assume first numeric metric or "loss" or "perplexity")
        def get_sort_key(idx):
            metrics = metrics_list[idx]
            if not metrics:
                return float('inf')
            # Prefer common metrics
            for key in ["loss", "perplexity", "validation_loss", "val_loss"]:
                if key in metrics and isinstance(metrics[key], (int, float)):
                    return metrics[key]
            # Otherwise use first numeric metric
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    return value
            return float('inf')

        # Get indices sorted by performance
        sorted_indices = sorted(range(len(architectures)), key=get_sort_key)

        # Show top K architectures
        for rank, idx in enumerate(sorted_indices[:top_k], 1):
            arch = architectures[idx]
            metrics = metrics_list[idx] if metrics_list else None

            lines.append(f"Rank #{rank}")
            lines.append("-" * 40)

            if arch.architecture_id:
                lines.append(f"Architecture ID: {arch.architecture_id}")

            lines.append(f"Configuration: {arch.n_layer} layers, {arch.n_embd} dim, {arch.n_heads} heads")
            lines.append(f"Attention: {arch.attention_type}")

            components = []
            if arch.use_mhc:
                components.append("MHC")
            if arch.use_mla:
                components.append("MLA")
            if arch.use_moe:
                components.append("MoE")
            if arch.use_kan:
                components.append(f"KAN-{arch.kan_type}")
            lines.append(f"Components: {', '.join(components) if components else 'Standard'}")

            if metrics:
                lines.append("Metrics:")
                for key, value in sorted(metrics.items()):
                    if isinstance(value, float):
                        lines.append(f"  {key}: {value:.4f}")
                    elif isinstance(value, int):
                        lines.append(f"  {key}: {value:,}")
                    else:
                        lines.append(f"  {key}: {value}")

            lines.append("")

    # Footer
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    report_text = "\n".join(lines)

    # Save to file if path provided
    if output_path:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        output_path_obj.write_text(report_text)
        logger.info(f"Generated summary report: {output_path}")

    return report_text
