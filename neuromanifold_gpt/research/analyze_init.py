"""Weight initialization analysis script.

Analyzes weight distributions for different initialization strategies.
Reports statistics by layer type and helps verify proper initialization.
Includes optional visualization of weight distributions.
"""

import argparse
from pathlib import Path

import torch

from neuromanifold_gpt.config.base import NeuroManifoldConfig
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT

try:
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def analyze_layer_weights(model, strategy, collect_distributions=False):
    """Analyze weight statistics grouped by layer type.

    Args:
        model: The model to analyze
        strategy: Initialization strategy name
        collect_distributions: If True, collect full weight values for visualization

    Returns:
        Dictionary with statistics and optionally weight distributions
    """
    stats = {
        "embeddings": [],
        "linear": [],
        "lm_head": [],
        "residual_proj": [],
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        weight_data = param.data
        param_stats = {
            "name": name,
            "shape": tuple(weight_data.shape),
            "mean": weight_data.mean().item(),
            "std": weight_data.std().item(),
            "min": weight_data.min().item(),
            "max": weight_data.max().item(),
        }

        # Optionally collect weight values for distribution plotting
        if collect_distributions:
            param_stats["values"] = weight_data.detach().cpu().flatten().numpy()

        # Categorize by layer type
        if "embedding" in name.lower() or "token_embed" in name:
            stats["embeddings"].append(param_stats)
        elif "lm_head" in name:
            stats["lm_head"].append(param_stats)
        elif "c_proj" in name or "out_proj" in name:
            stats["residual_proj"].append(param_stats)
        elif "weight" in name and isinstance(param, torch.Tensor) and param.ndim >= 2:
            stats["linear"].append(param_stats)

    return stats


def print_stats_table(stats_dict, category_name):
    """Print statistics table for a category of layers."""
    if not stats_dict:
        return

    print(f"\n{category_name}:")
    print("-" * 80)
    print(f"{'Layer':<40} {'Shape':<20} {'Mean':>10} {'Std':>10}")
    print("-" * 80)

    for stat in stats_dict:
        shape_str = f"{stat['shape']}"
        print(
            f"{stat['name']:<40} {shape_str:<20} {stat['mean']:>10.6f} {stat['std']:>10.6f}"
        )

    # Summary statistics
    means = [s["mean"] for s in stats_dict]
    stds = [s["std"] for s in stats_dict]
    print("-" * 80)
    print(
        f"{'SUMMARY':<40} {'':20} {sum(means)/len(means):>10.6f} {sum(stds)/len(stds):>10.6f}"
    )


def plot_weight_distributions(stats, strategy, output_dir):
    """Create histogram plots of weight distributions by layer type.

    Args:
        stats: Statistics dictionary from analyze_layer_weights
        strategy: Initialization strategy name
        output_dir: Directory to save plots
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping visualization")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create figure with subplots for each category
    categories = ["embeddings", "linear", "residual_proj", "lm_head"]
    num_plots = sum(1 for cat in categories if stats[cat])

    if num_plots == 0:
        print("Warning: No layers to visualize")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Weight Distributions - {strategy.upper()}", fontsize=16)
    axes = axes.flatten()

    plot_idx = 0
    for category in categories:
        if not stats[category]:
            continue

        ax = axes[plot_idx]
        plot_idx += 1

        # Collect all weights for this category
        all_weights = []
        for layer_stats in stats[category]:
            if "values" in layer_stats:
                all_weights.extend(layer_stats["values"])

        if not all_weights:
            continue

        # Create histogram
        all_weights = np.array(all_weights)
        ax.hist(all_weights, bins=100, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.5, label="Zero")

        # Add statistics
        mean_val = np.mean(all_weights)
        std_val = np.std(all_weights)
        ax.axvline(
            mean_val,
            color="green",
            linestyle="-",
            linewidth=2,
            alpha=0.7,
            label=f"Mean: {mean_val:.4f}",
        )
        ax.axvline(
            mean_val + std_val,
            color="orange",
            linestyle="--",
            linewidth=1.5,
            alpha=0.6,
            label=f"±1σ: {std_val:.4f}",
        )
        ax.axvline(
            mean_val - std_val, color="orange", linestyle="--", linewidth=1.5, alpha=0.6
        )

        ax.set_title(
            f'{category.replace("_", " ").title()} ({len(stats[category])} layers)'
        )
        ax.set_xlabel("Weight Value")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    # Save plot
    plot_file = output_path / f"weight_dist_{strategy}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nVisualization saved to: {plot_file}")


def plot_strategy_comparison(all_stats, strategies, output_dir):
    """Create comparison plots across different initialization strategies.

    Args:
        all_stats: Dictionary mapping strategy names to stats
        strategies: List of strategy names
        output_dir: Directory to save plots
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create comparison plot for each layer category
    categories = ["embeddings", "linear", "residual_proj", "lm_head"]

    for category in categories:
        # Check if any strategy has this category
        has_data = any(
            all_stats[strat][category] for strat in strategies if strat in all_stats
        )
        if not has_data:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))

        for strat in strategies:
            if strat not in all_stats or not all_stats[strat][category]:
                continue

            # Collect all weights for this category and strategy
            all_weights = []
            for layer_stats in all_stats[strat][category]:
                if "values" in layer_stats:
                    all_weights.extend(layer_stats["values"])

            if all_weights:
                all_weights = np.array(all_weights)
                ax.hist(
                    all_weights,
                    bins=80,
                    alpha=0.4,
                    label=strat.upper(),
                    edgecolor="black",
                    linewidth=0.3,
                )

        ax.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_title(f'Strategy Comparison - {category.replace("_", " ").title()}')
        ax.set_xlabel("Weight Value")
        ax.set_ylabel("Count (log scale)")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_file = output_path / f"comparison_{category}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Comparison plot saved to: {plot_file}")


def analyze_initialization(
    strategy,
    n_layer,
    n_embd,
    vocab_size=1000,
    block_size=256,
    mup_base_width=128,
    visualize=False,
    output_dir="./init_plots",
):
    """Analyze weight initialization for a given configuration.

    Args:
        strategy: Initialization strategy name
        n_layer: Number of transformer layers
        n_embd: Embedding dimension
        vocab_size: Vocabulary size
        block_size: Context length
        mup_base_width: Base width for muP scaling
        visualize: If True, create and save distribution plots
        output_dir: Directory to save plots

    Returns:
        Statistics dictionary
    """
    print(f"\n{'='*80}")
    print("Weight Statistics")
    print(f"{'='*80}")
    print(f"Strategy: {strategy}")
    print(f"Layers: {n_layer}, Embedding Dim: {n_embd}, Vocab: {vocab_size}")

    # Create config
    cfg = NeuroManifoldConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_heads=max(2, n_embd // 64),  # Ensure n_embd divisible by n_heads
        n_embd=n_embd,
        init_strategy=strategy,
        mup_base_width=mup_base_width,
        use_sdr=False,  # Use dense embeddings for simpler analysis
    )

    # Create model
    print(f"\nInitializing model with strategy '{strategy}'...")
    model = NeuroManifoldGPT(cfg)

    # Analyze weights
    stats = analyze_layer_weights(model, strategy, collect_distributions=visualize)

    # Print results
    print_stats_table(stats["embeddings"], "Embedding Layers")
    print_stats_table(stats["linear"], "Linear Layers")
    print_stats_table(stats["residual_proj"], "Residual Projection Layers")
    print_stats_table(stats["lm_head"], "Language Model Head")

    # Activation scale estimates (rough heuristic)
    print(f"\n{'='*80}")
    print("Activation Scale Estimates")
    print(f"{'='*80}")

    # Estimate forward pass variance propagation
    if stats["embeddings"]:
        emb_std = sum(s["std"] for s in stats["embeddings"]) / len(stats["embeddings"])
        print(f"Expected embedding magnitude: ~{emb_std:.6f}")

    if stats["linear"]:
        linear_std = sum(s["std"] for s in stats["linear"]) / len(stats["linear"])
        # Rough estimate: each layer multiplies by std, accumulates over sqrt(fan_in)
        fan_in_est = n_embd
        activation_scale = linear_std * (fan_in_est**0.5)
        print(f"Expected activation scale per layer: ~{activation_scale:.6f}")
        print(
            f"Expected activation scale after {n_layer} layers: ~{activation_scale * (n_layer ** 0.5):.6f}"
        )

    # Gradient scale predictions
    print("\nGradient Scale Predictions:")
    if strategy == "mup":
        print("  muP: Gradients should be O(1) independent of width")
        print(f"  Width ratio (base/current): {mup_base_width / n_embd:.4f}")
    elif strategy == "gpt2_scaled":
        print(
            f"  GPT-2 scaled: Residual gradients scaled by 1/sqrt(2*{n_layer}) = {1.0/((2*n_layer)**0.5):.6f}"
        )
    else:
        print("  Standard initialization - gradients may grow/shrink with depth")

    print(f"\n{'='*80}\n")

    # Create visualizations if requested
    if visualize:
        plot_weight_distributions(stats, strategy, output_dir)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Analyze weight initialization strategies"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="deepseek",
        choices=["deepseek", "gpt2", "gpt2_scaled", "mup"],
        help="Initialization strategy to analyze",
    )
    parser.add_argument(
        "--n-layer", type=int, default=6, help="Number of transformer layers"
    )
    parser.add_argument("--n-embd", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--vocab-size", type=int, default=1000, help="Vocabulary size")
    parser.add_argument(
        "--block-size", type=int, default=256, help="Block size / context length"
    )
    parser.add_argument(
        "--mup-base-width", type=int, default=128, help="Base width for muP scaling"
    )
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Compare all initialization strategies",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create and save weight distribution plots",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./init_plots",
        help="Directory to save visualization plots",
    )

    args = parser.parse_args()

    if args.compare_all:
        print("\n" + "=" * 80)
        print("Comparing all strategies")
        print("=" * 80)

        strategies = ["deepseek", "gpt2", "gpt2_scaled", "mup"]
        all_stats = {}

        for strategy in strategies:
            stats = analyze_initialization(
                strategy=strategy,
                n_layer=args.n_layer,
                n_embd=args.n_embd,
                vocab_size=args.vocab_size,
                block_size=args.block_size,
                mup_base_width=args.mup_base_width,
                visualize=args.visualize,
                output_dir=args.output_dir,
            )
            if args.visualize:
                all_stats[strategy] = stats

        # Create comparison plots if visualizing
        if args.visualize and all_stats:
            print(f"\n{'='*80}")
            print("Creating strategy comparison plots...")
            print(f"{'='*80}")
            plot_strategy_comparison(all_stats, strategies, args.output_dir)
    else:
        analyze_initialization(
            strategy=args.strategy,
            n_layer=args.n_layer,
            n_embd=args.n_embd,
            vocab_size=args.vocab_size,
            block_size=args.block_size,
            mup_base_width=args.mup_base_width,
            visualize=args.visualize,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
