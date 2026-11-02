#!/usr/bin/env python3
"""
Visualization tools for hyperparameter optimization results.

Creates visualization charts for:
- Optimization history (objective value over trials)
- Parameter importances
- Parameter relationships
- Trial distributions

Usage:
    from neuromanifold_gpt.hpo.visualize import plot_optimization_history, plot_param_importances

    # After running HPO
    hpo = OptunaHPO(config)
    study = hpo.optimize()

    # Create visualizations
    plot_optimization_history(study, 'optimization_history.png')
    plot_param_importances(study, 'param_importances.png')
"""

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

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def plot_optimization_history(
    study: "optuna.Study", output_path: Union[str, Path], show_best: bool = True
):
    """Plot optimization history showing objective value over trials.

    Creates a line plot showing:
    - Objective value for each trial
    - Best value achieved so far (running minimum/maximum)
    - Trial states (completed, pruned, failed)

    Args:
        study: Optuna Study instance with completed trials
        output_path: Path to save the plot
        show_best: Whether to show the best value line (default: True)
    """
    if not MATPLOTLIB_AVAILABLE:
        print("WARNING: matplotlib not available, skipping optimization history plot")
        return

    if not OPTUNA_AVAILABLE:
        print("WARNING: optuna not available, skipping optimization history plot")
        return

    if not study or len(study.trials) == 0:
        print("WARNING: No trials found in study, skipping plot")
        return

    output_path = Path(output_path)

    # Extract trial data
    trial_numbers = []
    trial_values = []
    trial_states = []

    for trial in study.trials:
        trial_numbers.append(trial.number)
        trial_states.append(trial.state)

        # Get value if trial completed
        if trial.state == optuna.trial.TrialState.COMPLETE:
            trial_values.append(trial.value)
        else:
            trial_values.append(None)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot completed trials
    completed_mask = np.array(
        [s == optuna.trial.TrialState.COMPLETE for s in trial_states]
    )
    if np.any(completed_mask):
        completed_nums = np.array(trial_numbers)[completed_mask]
        completed_vals = np.array([v for v in trial_values if v is not None])

        ax.scatter(
            completed_nums,
            completed_vals,
            color="#3498db",
            s=50,
            alpha=0.6,
            label="Completed",
            zorder=3,
        )
        ax.plot(
            completed_nums,
            completed_vals,
            color="#3498db",
            alpha=0.3,
            linewidth=1,
            zorder=2,
        )

    # Plot pruned trials (if any)
    pruned_mask = np.array([s == optuna.trial.TrialState.PRUNED for s in trial_states])
    if np.any(pruned_mask):
        pruned_nums = np.array(trial_numbers)[pruned_mask]
        # Use a placeholder value at the bottom of the plot for pruned trials
        if np.any(completed_mask):
            completed_vals_arr = np.array([v for v in trial_values if v is not None])
            pruned_y = (
                np.min(completed_vals_arr)
                if study.direction == optuna.study.StudyDirection.MINIMIZE
                else np.max(completed_vals_arr)
            )
        else:
            pruned_y = 0
        ax.scatter(
            pruned_nums,
            [pruned_y] * len(pruned_nums),
            color="#95a5a6",
            s=30,
            alpha=0.5,
            marker="x",
            label="Pruned",
            zorder=3,
        )

    # Plot failed trials (if any)
    failed_mask = np.array([s == optuna.trial.TrialState.FAIL for s in trial_states])
    if np.any(failed_mask):
        failed_nums = np.array(trial_numbers)[failed_mask]
        # Use a placeholder value at the top of the plot for failed trials
        if np.any(completed_mask):
            completed_vals_arr = np.array([v for v in trial_values if v is not None])
            failed_y = (
                np.max(completed_vals_arr)
                if study.direction == optuna.study.StudyDirection.MINIMIZE
                else np.min(completed_vals_arr)
            )
        else:
            failed_y = 0
        ax.scatter(
            failed_nums,
            [failed_y] * len(failed_nums),
            color="#e74c3c",
            s=30,
            alpha=0.5,
            marker="x",
            label="Failed",
            zorder=3,
        )

    # Plot best value line
    if show_best and np.any(completed_mask):
        completed_vals_arr = np.array([v for v in trial_values if v is not None])
        completed_nums_arr = np.array(trial_numbers)[completed_mask]

        if study.direction == optuna.study.StudyDirection.MINIMIZE:
            best_values = np.minimum.accumulate(completed_vals_arr)
        else:
            best_values = np.maximum.accumulate(completed_vals_arr)

        ax.plot(
            completed_nums_arr,
            best_values,
            color="#2ecc71",
            linewidth=2,
            label="Best Value",
            zorder=4,
        )

        # Mark the best trial
        best_idx = (
            np.argmin(best_values)
            if study.direction == optuna.study.StudyDirection.MINIMIZE
            else np.argmax(best_values)
        )
        ax.scatter(
            [completed_nums_arr[best_idx]],
            [best_values[best_idx]],
            color="#2ecc71",
            s=200,
            marker="*",
            zorder=5,
            edgecolors="black",
            linewidths=1.5,
        )

    # Formatting
    ax.set_xlabel("Trial Number", fontsize=12)
    ax.set_ylabel("Objective Value", fontsize=12)

    direction_text = (
        "minimize"
        if study.direction == optuna.study.StudyDirection.MINIMIZE
        else "maximize"
    )
    ax.set_title(
        f"Optimization History (direction: {direction_text})",
        fontsize=14,
        fontweight="bold",
    )

    ax.grid(alpha=0.3, zorder=1)
    ax.legend(fontsize=10, loc="best")

    # Add study statistics as text
    n_completed = sum(1 for s in trial_states if s == optuna.trial.TrialState.COMPLETE)
    n_pruned = sum(1 for s in trial_states if s == optuna.trial.TrialState.PRUNED)
    n_failed = sum(1 for s in trial_states if s == optuna.trial.TrialState.FAIL)

    stats_text = f"Trials: {len(trial_numbers)} (✓ {n_completed}"
    if n_pruned > 0:
        stats_text += f", ✂ {n_pruned}"
    if n_failed > 0:
        stats_text += f", ✗ {n_failed}"
    stats_text += ")"

    if n_completed > 0:
        stats_text += (
            f"\nBest: {study.best_value:.4f} (trial {study.best_trial.number})"
        )

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved optimization history plot: {output_path}")


def plot_param_importances(
    study: "optuna.Study", output_path: Union[str, Path], top_n: int = 10
):
    """Plot parameter importance using fANOVA.

    Creates a horizontal bar chart showing which hyperparameters had the
    greatest impact on the objective value.

    Args:
        study: Optuna Study instance with completed trials
        output_path: Path to save the plot
        top_n: Number of top parameters to display (default: 10)
    """
    if not MATPLOTLIB_AVAILABLE:
        print("WARNING: matplotlib not available, skipping param importances plot")
        return

    if not OPTUNA_AVAILABLE:
        print("WARNING: optuna not available, skipping param importances plot")
        return

    if not study or len(study.trials) == 0:
        print("WARNING: No trials found in study, skipping plot")
        return

    # Need at least 2 completed trials for importance calculation
    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if len(completed_trials) < 2:
        print(
            f"WARNING: Need at least 2 completed trials for importance calculation (found {len(completed_trials)}), skipping plot"
        )
        return

    output_path = Path(output_path)

    try:
        # Calculate parameter importances using fANOVA
        # This requires optuna.importance module
        from optuna.importance import FanovaImportanceEvaluator

        evaluator = FanovaImportanceEvaluator()
        importances = evaluator.evaluate(study)

        if not importances:
            print("WARNING: No parameter importances calculated, skipping plot")
            return

        # Sort by importance and take top N
        sorted_importances = sorted(
            importances.items(), key=lambda x: x[1], reverse=True
        )
        if len(sorted_importances) > top_n:
            sorted_importances = sorted_importances[:top_n]

        param_names = [item[0] for item in sorted_importances]
        importance_values = [item[1] for item in sorted_importances]

        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, max(6, len(param_names) * 0.4)))

        y_pos = np.arange(len(param_names))
        bars = ax.barh(y_pos, importance_values, color="#3498db", alpha=0.7)

        # Color the most important parameter differently
        if bars:
            bars[0].set_color("#e74c3c")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(param_names)
        ax.invert_yaxis()  # Highest importance at top
        ax.set_xlabel("Importance", fontsize=12)
        ax.set_title(
            "Hyperparameter Importances (fANOVA)", fontsize=14, fontweight="bold"
        )
        ax.grid(axis="x", alpha=0.3)

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, importance_values)):
            ax.text(
                val,
                bar.get_y() + bar.get_height() / 2,
                f" {val:.3f}",
                va="center",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved parameter importances plot: {output_path}")

    except ImportError:
        print("WARNING: optuna.importance module not available, using fallback method")
        # Fallback: simple variance-based importance
        _plot_param_importances_fallback(study, output_path, top_n)
    except Exception as e:
        print(f"WARNING: Failed to calculate parameter importances: {e}")
        print("Using fallback method...")
        _plot_param_importances_fallback(study, output_path, top_n)


def _plot_param_importances_fallback(
    study: "optuna.Study", output_path: Path, top_n: int = 10
):
    """Fallback method for parameter importance using simple variance analysis.

    Args:
        study: Optuna Study instance with completed trials
        output_path: Path to save the plot
        top_n: Number of top parameters to display
    """
    # Get completed trials
    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    if len(completed_trials) < 2:
        print("WARNING: Need at least 2 completed trials, skipping plot")
        return

    # Calculate variance of objective value for each parameter value
    param_variances = {}

    # Get all parameter names
    all_params = set()
    for trial in completed_trials:
        all_params.update(trial.params.keys())

    # For each parameter, calculate how much the objective varies with its values
    for param_name in all_params:
        # Group trials by parameter value
        param_groups = {}
        for trial in completed_trials:
            if param_name in trial.params:
                param_val = trial.params[param_name]
                # Convert to string for grouping
                param_val_str = str(param_val)
                if param_val_str not in param_groups:
                    param_groups[param_val_str] = []
                param_groups[param_val_str].append(trial.value)

        # Calculate variance between groups
        if len(param_groups) > 1:
            group_means = [np.mean(vals) for vals in param_groups.values()]
            np.mean([v for vals in param_groups.values() for v in vals])
            # Between-group variance
            variance = np.var(group_means)
            param_variances[param_name] = variance
        else:
            param_variances[param_name] = 0.0

    if not param_variances:
        print("WARNING: No parameter variances calculated, skipping plot")
        return

    # Normalize to sum to 1
    total_variance = sum(param_variances.values())
    if total_variance > 0:
        param_importances = {k: v / total_variance for k, v in param_variances.items()}
    else:
        param_importances = param_variances

    # Sort and take top N
    sorted_importances = sorted(
        param_importances.items(), key=lambda x: x[1], reverse=True
    )
    if len(sorted_importances) > top_n:
        sorted_importances = sorted_importances[:top_n]

    param_names = [item[0] for item in sorted_importances]
    importance_values = [item[1] for item in sorted_importances]

    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, max(6, len(param_names) * 0.4)))

    y_pos = np.arange(len(param_names))
    bars = ax.barh(y_pos, importance_values, color="#3498db", alpha=0.7)

    # Color the most important parameter differently
    if bars:
        bars[0].set_color("#e74c3c")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(param_names)
    ax.invert_yaxis()  # Highest importance at top
    ax.set_xlabel("Relative Importance", fontsize=12)
    ax.set_title(
        "Hyperparameter Importances (Variance-based)", fontsize=14, fontweight="bold"
    )
    ax.grid(axis="x", alpha=0.3)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, importance_values)):
        ax.text(
            val,
            bar.get_y() + bar.get_height() / 2,
            f" {val:.3f}",
            va="center",
            fontsize=9,
        )

    # Add note about method
    ax.text(
        0.98,
        0.02,
        "Note: Using variance-based importance\n(fANOVA not available)",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        horizontalalignment="right",
        style="italic",
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.3),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved parameter importances plot (fallback): {output_path}")


def plot_all_visualizations(
    study: "optuna.Study", output_dir: Union[str, Path] = "hpo_plots"
) -> Dict[str, Path]:
    """Create all HPO visualization plots.

    Args:
        study: Optuna Study instance with completed trials
        output_dir: Directory to save plots (default: 'hpo_plots')

    Returns:
        Dict mapping plot names to output paths
    """
    if not MATPLOTLIB_AVAILABLE:
        print("ERROR: matplotlib is required for visualization")
        print("Install with: pip install matplotlib")
        return {}

    if not OPTUNA_AVAILABLE:
        print("ERROR: optuna is required for visualization")
        print("Install with: pip install optuna")
        return {}

    if not study or len(study.trials) == 0:
        print("ERROR: Study has no trials to visualize")
        return {}

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Generating HPO Visualizations")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print()

    plots = {}

    # Plot optimization history
    history_path = output_dir / "optimization_history.png"
    plot_optimization_history(study, history_path)
    plots["optimization_history"] = history_path

    # Plot parameter importances
    importances_path = output_dir / "param_importances.png"
    plot_param_importances(study, importances_path)
    plots["param_importances"] = importances_path

    print()
    print("=" * 80)
    print(f"Generated {len(plots)} visualization(s)")
    print("=" * 80)

    return plots


def main():
    """Command-line interface for HPO visualization."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate visualizations from Optuna HPO study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load study from database and create plots
  python neuromanifold_gpt/hpo/visualize.py --storage sqlite:///hpo_study.db --name my-study

  # Specify output directory
  python neuromanifold_gpt/hpo/visualize.py --storage sqlite:///hpo_study.db --name my-study --output plots
        """,
    )

    parser.add_argument(
        "--storage",
        type=str,
        required=True,
        help="Optuna storage URL (e.g., sqlite:///study.db)",
    )

    parser.add_argument("--name", type=str, required=True, help="Study name")

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="hpo_plots",
        help="Output directory for plots (default: hpo_plots)",
    )

    args = parser.parse_args()

    if not MATPLOTLIB_AVAILABLE:
        print("ERROR: matplotlib is required for visualization")
        print("Install with: pip install matplotlib")
        return 1

    if not OPTUNA_AVAILABLE:
        print("ERROR: optuna is required for visualization")
        print("Install with: pip install optuna")
        return 1

    try:
        # Load study from storage
        study = optuna.load_study(study_name=args.name, storage=args.storage)

        print(f"Loaded study: {args.name}")
        print(f"Trials: {len(study.trials)}")

        # Generate plots
        plots = plot_all_visualizations(study, args.output)

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
