#!/usr/bin/env python3
"""
Attention pattern visualization utilities for NeuroManifold GPT.

Creates heatmap visualizations of attention patterns for debugging and analysis:
- Single attention head visualization
- Multi-head attention grids
- Comparison between standard and NeuroManifold attention
- Time-series attention pattern tracking

Usage:
    from neuromanifold_gpt.utils.attention_viz import visualize_attention_pattern

    # Visualize single attention matrix
    visualize_attention_pattern(
        attention_weights,  # Shape: (seq_len, seq_len) or (heads, seq_len, seq_len)
        output_path='attention.png',
        title='Attention Pattern'
    )

    # Visualize multi-head attention
    visualize_attention_pattern(
        multi_head_weights,  # Shape: (num_heads, seq_len, seq_len)
        output_path='multihead_attention.png',
        title='Multi-Head Attention Patterns'
    )

    # Compare two attention patterns
    compare_attention_patterns(
        standard_attn,
        neuromanifold_attn,
        output_path='attention_comparison.png'
    )
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def visualize_attention_pattern(
    attention_weights: Union[torch.Tensor, np.ndarray],
    output_path: Union[str, Path],
    title: str = "Attention Pattern",
    figsize: Optional[Tuple[int, int]] = None,
    dpi: int = 300,
    cmap: str = 'viridis',
    show_values: bool = False,
    vmin: float = 0.0,
    vmax: float = 1.0
) -> None:
    """Visualize attention pattern(s) as heatmap(s).

    Args:
        attention_weights: Attention weights tensor
            - Shape: (seq_len, seq_len) for single attention matrix
            - Shape: (num_heads, seq_len, seq_len) for multi-head attention
        output_path: Path to save the visualization
        title: Plot title
        figsize: Figure size (width, height). Auto-calculated if None
        dpi: Output resolution (default: 300)
        cmap: Matplotlib colormap name (default: 'viridis')
        show_values: Whether to annotate cells with values (default: False)
        vmin: Minimum value for colormap normalization
        vmax: Maximum value for colormap normalization
    """
    if not MATPLOTLIB_AVAILABLE:
        print("WARNING: matplotlib not available, skipping attention visualization")
        return

    # Convert to numpy if needed
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Handle different input shapes
    if attention_weights.ndim == 2:
        # Single attention matrix
        _plot_single_attention(
            attention_weights,
            output_path,
            title,
            figsize,
            dpi,
            cmap,
            show_values,
            vmin,
            vmax
        )
    elif attention_weights.ndim == 3:
        # Multi-head attention
        _plot_multihead_attention(
            attention_weights,
            output_path,
            title,
            figsize,
            dpi,
            cmap,
            show_values,
            vmin,
            vmax
        )
    else:
        raise ValueError(
            f"Expected attention_weights with shape (seq_len, seq_len) or "
            f"(num_heads, seq_len, seq_len), got shape {attention_weights.shape}"
        )

    print(f"  Saved attention visualization: {output_path}")


def _plot_single_attention(
    attention_matrix: np.ndarray,
    output_path: Path,
    title: str,
    figsize: Optional[Tuple[int, int]],
    dpi: int,
    cmap: str,
    show_values: bool,
    vmin: float,
    vmax: float
) -> None:
    """Plot single attention matrix as heatmap."""
    seq_len = attention_matrix.shape[0]

    # Auto-calculate figure size based on sequence length
    if figsize is None:
        size = min(max(seq_len / 10, 6), 12)
        figsize = (size, size)

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(
        attention_matrix,
        cmap=cmap,
        aspect='auto',
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest'
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', fontsize=11)

    # Configure axes
    ax.set_xlabel('Key Position', fontsize=12)
    ax.set_ylabel('Query Position', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    # Add grid
    ax.set_xticks(np.arange(seq_len))
    ax.set_yticks(np.arange(seq_len))
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.2)

    # Optionally annotate cells with values
    if show_values and seq_len <= 20:  # Only for small matrices
        for i in range(seq_len):
            for j in range(seq_len):
                text = ax.text(
                    j, i, f'{attention_matrix[i, j]:.2f}',
                    ha='center', va='center',
                    color='white' if attention_matrix[i, j] > 0.5 else 'black',
                    fontsize=8
                )

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def _plot_multihead_attention(
    attention_weights: np.ndarray,
    output_path: Path,
    title: str,
    figsize: Optional[Tuple[int, int]],
    dpi: int,
    cmap: str,
    show_values: bool,
    vmin: float,
    vmax: float
) -> None:
    """Plot multi-head attention as grid of heatmaps."""
    num_heads, seq_len, _ = attention_weights.shape

    # Calculate grid layout
    ncols = min(4, num_heads)  # Max 4 columns
    nrows = (num_heads + ncols - 1) // ncols

    # Auto-calculate figure size
    if figsize is None:
        figsize = (ncols * 4, nrows * 4)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Handle case where axes is not an array
    if num_heads == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Plot each head
    for head_idx in range(num_heads):
        ax = axes[head_idx]
        im = ax.imshow(
            attention_weights[head_idx],
            cmap=cmap,
            aspect='auto',
            vmin=vmin,
            vmax=vmax,
            interpolation='nearest'
        )

        ax.set_title(f'Head {head_idx + 1}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Key', fontsize=9)
        ax.set_ylabel('Query', fontsize=9)

        # Reduce tick density for readability
        if seq_len > 20:
            step = max(seq_len // 10, 1)
            ax.set_xticks(np.arange(0, seq_len, step))
            ax.set_yticks(np.arange(0, seq_len, step))

    # Hide unused subplots
    for idx in range(num_heads, len(axes)):
        axes[idx].axis('off')

    # Add shared colorbar
    fig.colorbar(im, ax=axes, label='Attention Weight', fraction=0.046, pad=0.04)

    # Add super title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.00)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def compare_attention_patterns(
    attention_a: Union[torch.Tensor, np.ndarray],
    attention_b: Union[torch.Tensor, np.ndarray],
    output_path: Union[str, Path],
    label_a: str = "Standard Attention",
    label_b: str = "NeuroManifold Attention",
    title: str = "Attention Pattern Comparison",
    figsize: Tuple[int, int] = (14, 6),
    dpi: int = 300,
    cmap: str = 'viridis'
) -> None:
    """Compare two attention patterns side by side.

    Args:
        attention_a: First attention pattern (seq_len, seq_len)
        attention_b: Second attention pattern (seq_len, seq_len)
        output_path: Path to save the comparison visualization
        label_a: Label for first attention pattern
        label_b: Label for second attention pattern
        title: Overall plot title
        figsize: Figure size (width, height)
        dpi: Output resolution
        cmap: Matplotlib colormap name
    """
    if not MATPLOTLIB_AVAILABLE:
        print("WARNING: matplotlib not available, skipping attention comparison")
        return

    # Convert to numpy if needed
    if isinstance(attention_a, torch.Tensor):
        attention_a = attention_a.detach().cpu().numpy()
    if isinstance(attention_b, torch.Tensor):
        attention_b = attention_b.detach().cpu().numpy()

    # Handle multi-head by taking mean across heads
    if attention_a.ndim == 3:
        attention_a = attention_a.mean(axis=0)
    if attention_b.ndim == 3:
        attention_b = attention_b.mean(axis=0)

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create side-by-side comparison
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    # Common colormap range
    vmin = min(attention_a.min(), attention_b.min())
    vmax = max(attention_a.max(), attention_b.max())

    # Plot first attention pattern
    im1 = ax1.imshow(attention_a, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    ax1.set_title(label_a, fontsize=12, fontweight='bold')
    ax1.set_xlabel('Key Position', fontsize=10)
    ax1.set_ylabel('Query Position', fontsize=10)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Plot second attention pattern
    im2 = ax2.imshow(attention_b, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    ax2.set_title(label_b, fontsize=12, fontweight='bold')
    ax2.set_xlabel('Key Position', fontsize=10)
    ax2.set_ylabel('Query Position', fontsize=10)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Plot difference
    diff = attention_b - attention_a
    im3 = ax3.imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
    ax3.set_title('Difference (B - A)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Key Position', fontsize=10)
    ax3.set_ylabel('Query Position', fontsize=10)
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label('Weight Difference', fontsize=10)

    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.00)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  Saved attention comparison: {output_path}")


def plot_attention_entropy(
    attention_weights: Union[torch.Tensor, np.ndarray],
    output_path: Union[str, Path],
    title: str = "Attention Entropy Over Time",
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 300
) -> None:
    """Plot attention entropy to detect attention collapse or dispersion.

    Args:
        attention_weights: Attention weights
            - Shape: (seq_len, seq_len) or (num_heads, seq_len, seq_len)
        output_path: Path to save the plot
        title: Plot title
        figsize: Figure size
        dpi: Output resolution
    """
    if not MATPLOTLIB_AVAILABLE:
        print("WARNING: matplotlib not available, skipping entropy plot")
        return

    # Convert to numpy if needed
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate entropy for each query position
    # H(p) = -sum(p * log(p))
    epsilon = 1e-10  # Avoid log(0)

    if attention_weights.ndim == 2:
        # Single attention matrix
        entropies = -np.sum(
            attention_weights * np.log(attention_weights + epsilon),
            axis=1
        )

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(entropies, linewidth=2, color='#3498db')
        ax.set_xlabel('Query Position', fontsize=12)
        ax.set_ylabel('Attention Entropy', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)

    elif attention_weights.ndim == 3:
        # Multi-head attention
        num_heads = attention_weights.shape[0]

        fig, ax = plt.subplots(figsize=figsize)

        for head_idx in range(num_heads):
            entropies = -np.sum(
                attention_weights[head_idx] * np.log(attention_weights[head_idx] + epsilon),
                axis=1
            )
            ax.plot(entropies, linewidth=1.5, alpha=0.7, label=f'Head {head_idx + 1}')

        ax.set_xlabel('Query Position', fontsize=12)
        ax.set_ylabel('Attention Entropy', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.grid(alpha=0.3)

    else:
        raise ValueError(
            f"Expected attention_weights with 2 or 3 dimensions, got {attention_weights.ndim}"
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  Saved attention entropy plot: {output_path}")
