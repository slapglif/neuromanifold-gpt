#!/usr/bin/env python3
"""
Visualization tools for NeuroManifold attention benchmark results.

Creates comparison charts for:
- Perplexity comparison
- Sample quality metrics
- Speed vs sequence length
- Memory usage vs batch size
- Memory scaling analysis

Usage:
    from neuromanifold_gpt.benchmarks.visualize import plot_results

    # From JSON file
    plot_results('benchmark_results.json', output_dir='plots')

    # From results dict
    results = run_all_benchmarks()
    plot_results(results, output_dir='plots')
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Union

try:
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def plot_perplexity_comparison(quality_results: Dict, output_path: Path):
    """Plot perplexity comparison between standard and NeuroManifold attention.

    Args:
        quality_results: Dict with 'perplexity' key containing standard/neuromanifold results
        output_path: Path to save the plot
    """
    if not MATPLOTLIB_AVAILABLE:
        print("WARNING: matplotlib not available, skipping perplexity plot")
        return

    if not quality_results or "perplexity" not in quality_results:
        print("WARNING: No perplexity results found, skipping plot")
        return

    perplexity_data = quality_results["perplexity"]

    # Extract data
    models = ["Standard", "NeuroManifold"]
    perplexities = [
        perplexity_data["standard"]["perplexity"],
        perplexity_data["neuromanifold"]["perplexity"],
    ]
    val_losses = [
        perplexity_data["standard"]["val_loss"],
        perplexity_data["neuromanifold"]["val_loss"],
    ]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Perplexity comparison
    bars1 = ax1.bar(models, perplexities, color=["#3498db", "#e74c3c"])
    ax1.set_ylabel("Perplexity (lower is better)", fontsize=12)
    ax1.set_title("Perplexity Comparison", fontsize=14, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    # Plot 2: Validation loss comparison
    bars2 = ax2.bar(models, val_losses, color=["#3498db", "#e74c3c"])
    ax2.set_ylabel("Validation Loss (lower is better)", fontsize=12)
    ax2.set_title("Validation Loss Comparison", fontsize=14, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved perplexity plot: {output_path}")


def plot_sample_quality(quality_results: Dict, output_path: Path):
    """Plot sample quality metrics (diversity).

    Args:
        quality_results: Dict with 'sample_diversity' key
        output_path: Path to save the plot
    """
    if not MATPLOTLIB_AVAILABLE:
        print("WARNING: matplotlib not available, skipping sample quality plot")
        return

    if not quality_results or "sample_diversity" not in quality_results:
        print("WARNING: No sample diversity results found, skipping plot")
        return

    diversity_data = quality_results["sample_diversity"]

    # Extract data
    metrics = ["Unique\nUnigrams", "Unique\nBigrams", "Unique\nTrigrams"]
    standard_scores = [
        diversity_data["standard"]["unique_unigrams"],
        diversity_data["standard"]["unique_bigrams"],
        diversity_data["standard"]["unique_trigrams"],
    ]
    neuromanifold_scores = [
        diversity_data["neuromanifold"]["unique_unigrams"],
        diversity_data["neuromanifold"]["unique_bigrams"],
        diversity_data["neuromanifold"]["unique_trigrams"],
    ]

    # Create grouped bar chart
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(
        x - width / 2, standard_scores, width, label="Standard", color="#3498db"
    )
    bars2 = ax.bar(
        x + width / 2,
        neuromanifold_scores,
        width,
        label="NeuroManifold",
        color="#e74c3c",
    )

    ax.set_ylabel("Diversity Score (higher is better)", fontsize=12)
    ax.set_title("Sample Quality: Diversity Metrics", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved sample quality plot: {output_path}")


def plot_speed_summary(speed_results: Dict, output_path: Path):
    """Plot speed benchmark summary.

    Args:
        speed_results: Dict with speed benchmark status
        output_path: Path to save the plot
    """
    if not MATPLOTLIB_AVAILABLE:
        print("WARNING: matplotlib not available, skipping speed plot")
        return

    # Note: Current speed benchmark doesn't return structured data
    # This is a placeholder for when we have detailed speed results
    if not speed_results or "status" not in speed_results:
        print("WARNING: No speed results found, skipping plot")
        return

    # Create a simple status indicator plot
    fig, ax = plt.subplots(figsize=(8, 4))
    status = speed_results.get("status", "unknown")
    note = speed_results.get("note", "No details available")

    ax.text(
        0.5,
        0.6,
        f"Speed Benchmark: {status.upper()}",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="green" if status == "completed" else "red",
    )
    ax.text(
        0.5, 0.4, note, ha="center", va="center", fontsize=11, style="italic", wrap=True
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved speed summary plot: {output_path}")


def plot_memory_summary(memory_results: Dict, output_path: Path):
    """Plot memory benchmark summary.

    Args:
        memory_results: Dict with memory benchmark status
        output_path: Path to save the plot
    """
    if not MATPLOTLIB_AVAILABLE:
        print("WARNING: matplotlib not available, skipping memory plot")
        return

    # Note: Current memory benchmark doesn't return structured data
    # This is a placeholder for when we have detailed memory results
    if not memory_results or "status" not in memory_results:
        print("WARNING: No memory results found, skipping plot")
        return

    # Create a simple status indicator plot
    fig, ax = plt.subplots(figsize=(8, 4))
    status = memory_results.get("status", "unknown")
    note = memory_results.get("note", "No details available")

    ax.text(
        0.5,
        0.6,
        f"Memory Benchmark: {status.upper()}",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="green" if status == "completed" else "red",
    )
    ax.text(
        0.5, 0.4, note, ha="center", va="center", fontsize=11, style="italic", wrap=True
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved memory summary plot: {output_path}")


def plot_results(
    results: Union[str, Path, Dict], output_dir: Union[str, Path] = "benchmark_plots"
):
    """Create all visualization plots from benchmark results.

    Args:
        results: Either a path to JSON results file or a results dictionary
        output_dir: Directory to save plots (default: 'benchmark_plots')

    Returns:
        Dict mapping plot names to output paths
    """
    if not MATPLOTLIB_AVAILABLE:
        print("ERROR: matplotlib is required for visualization")
        print("Install with: pip install matplotlib")
        return {}

    # Load results if path provided
    if isinstance(results, (str, Path)):
        results_path = Path(results)
        if not results_path.exists():
            print(f"ERROR: Results file not found: {results_path}")
            return {}

        with open(results_path, "r") as f:
            results_data = json.load(f)
    else:
        results_data = results

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Generating Benchmark Visualizations")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print()

    plots = {}

    # Plot perplexity comparison
    if "quality" in results_data and results_data["quality"]:
        quality_results = results_data["quality"]

        # Perplexity plot
        perplexity_path = output_dir / "perplexity_comparison.png"
        plot_perplexity_comparison(quality_results, perplexity_path)
        plots["perplexity"] = perplexity_path

        # Sample quality plot
        sample_quality_path = output_dir / "sample_quality.png"
        plot_sample_quality(quality_results, sample_quality_path)
        plots["sample_quality"] = sample_quality_path

    # Plot speed summary
    if "speed" in results_data and results_data["speed"]:
        speed_path = output_dir / "speed_summary.png"
        plot_speed_summary(results_data["speed"], speed_path)
        plots["speed"] = speed_path

    # Plot memory summary
    if "memory" in results_data and results_data["memory"]:
        memory_path = output_dir / "memory_summary.png"
        plot_memory_summary(results_data["memory"], memory_path)
        plots["memory"] = memory_path

    print()
    print("=" * 80)
    print(f"Generated {len(plots)} visualization(s)")
    print("=" * 80)

    return plots


def main():
    """Command-line interface for visualization."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations from NeuroManifold benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate plots from JSON results
  python neuromanifold_gpt/benchmarks/visualize.py benchmark_results.json

  # Specify output directory
  python neuromanifold_gpt/benchmarks/visualize.py benchmark_results.json --output plots

  # Check if matplotlib is available
  python -c "from neuromanifold_gpt.benchmarks.visualize import plot_results; print('OK')"
        """,
    )

    parser.add_argument(
        "results",
        type=str,
        help="Path to JSON results file (e.g., benchmark_results.json)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="benchmark_plots",
        help="Output directory for plots (default: benchmark_plots)",
    )

    args = parser.parse_args()

    if not MATPLOTLIB_AVAILABLE:
        print("ERROR: matplotlib is required for visualization")
        print("Install with: pip install matplotlib")
        return 1

    try:
        plots = plot_results(args.results, args.output)

        if plots:
            print("\nGenerated plots:")
            for name, path in plots.items():
                print(f"  {name}: {path}")
            return 0
        else:
            print("\nNo plots generated (check warnings above)")
            return 1

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
