#!/usr/bin/env python3
"""Export best architectures discovered by NAS.

This script demonstrates how to export NAS-discovered architectures to various
formats for reuse in training. It can load search results and export top
architectures as Python config files, JSON files, or generate summary reports.

The script will:
1. Load NAS search results
2. Extract top-k architectures by performance
3. Export as Python config files
4. Optionally export as JSON
5. Generate summary reports

Usage:
    # Export top 5 architectures from search results
    python examples/nas_export_best.py nas_results/search_results.json

    # Export top 10 architectures with custom output directory
    python examples/nas_export_best.py nas_results/search_results.json --top-k 10 --output exported_configs/

    # Export with JSON format and summary report
    python examples/nas_export_best.py nas_results/search_results.json --format json --summary

Examples:
    # Basic export of top 3 architectures
    python examples/nas_export_best.py nas_results/search_results.json --top-k 3

    # Export all formats with detailed summaries
    python examples/nas_export_best.py nas_results/search_results.json \\
        --format all --summary --output my_configs/

    # Filter by performance threshold
    python examples/nas_export_best.py nas_results/search_results.json \\
        --max-perplexity 15.0 --top-k 5
"""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_dependencies():
    """Load dependencies after argument parsing."""
    # Configure logging
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, format="{time:HH:mm:ss} | {level} | {message}")

    from neuromanifold_gpt.nas.search_space import ArchitectureConfig
    from neuromanifold_gpt.nas.export import (
        export_config,
        export_to_json,
        generate_config_summary,
        generate_summary_report,
    )

    return (logger, ArchitectureConfig, export_config, export_to_json,
            generate_config_summary, generate_summary_report)


def load_search_results(results_path: str) -> Dict[str, Any]:
    """Load NAS search results from JSON file.

    Args:
        results_path: Path to search_results.json

    Returns:
        Dictionary containing search results

    Raises:
        FileNotFoundError: If results file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    results_path = Path(results_path)

    if not results_path.exists():
        raise FileNotFoundError(f"Search results not found: {results_path}")

    logger.info(f"Loading search results from {results_path}...")

    with open(results_path, "r") as f:
        results = json.load(f)

    logger.info(f"Loaded {len(results.get('architectures', []))} architectures")

    return results


def parse_architecture_results(
    results: Dict[str, Any]
) -> List[Tuple[ArchitectureConfig, Dict[str, Any]]]:
    """Parse search results into ArchitectureConfig objects with metrics.

    Args:
        results: Search results dictionary

    Returns:
        List of (architecture, metrics) tuples
    """
    architectures = results.get("architectures", [])
    eval_results = results.get("evaluation_results", [])

    if len(architectures) != len(eval_results):
        logger.warning(
            f"Mismatch: {len(architectures)} architectures but {len(eval_results)} results"
        )

    parsed = []
    for i, arch_dict in enumerate(architectures):
        # Reconstruct ArchitectureConfig
        arch = ArchitectureConfig(**arch_dict)

        # Get corresponding metrics
        metrics = eval_results[i] if i < len(eval_results) else {}

        parsed.append((arch, metrics))

    return parsed


def filter_by_performance(
    arch_results: List[Tuple[ArchitectureConfig, Dict[str, Any]]],
    max_perplexity: float = None,
    max_loss: float = None,
    min_params: int = None,
    max_params: int = None,
) -> List[Tuple[ArchitectureConfig, Dict[str, Any]]]:
    """Filter architectures by performance criteria.

    Args:
        arch_results: List of (architecture, metrics) tuples
        max_perplexity: Maximum allowed perplexity (optional)
        max_loss: Maximum allowed loss (optional)
        min_params: Minimum parameter count (optional)
        max_params: Maximum parameter count (optional)

    Returns:
        Filtered list of (architecture, metrics) tuples
    """
    filtered = []

    for arch, metrics in arch_results:
        # Check perplexity
        if max_perplexity is not None:
            ppl = metrics.get("perplexity", float("inf"))
            if ppl > max_perplexity:
                continue

        # Check loss
        if max_loss is not None:
            loss = metrics.get("final_loss", float("inf"))
            if loss > max_loss:
                continue

        # Check parameter count
        if min_params is not None or max_params is not None:
            n_params = metrics.get("n_params", 0)
            if min_params is not None and n_params < min_params:
                continue
            if max_params is not None and n_params > max_params:
                continue

        filtered.append((arch, metrics))

    return filtered


def sort_by_performance(
    arch_results: List[Tuple[ArchitectureConfig, Dict[str, Any]]],
    metric: str = "perplexity",
) -> List[Tuple[ArchitectureConfig, Dict[str, Any]]]:
    """Sort architectures by performance metric.

    Args:
        arch_results: List of (architecture, metrics) tuples
        metric: Metric to sort by (default: "perplexity")

    Returns:
        Sorted list (best first)
    """
    def get_sort_key(item):
        arch, metrics = item
        value = metrics.get(metric, float("inf"))
        return value

    return sorted(arch_results, key=get_sort_key)


def export_architectures(
    arch_results: List[Tuple[ArchitectureConfig, Dict[str, Any]]],
    output_dir: Path,
    format: str = "python",
    prefix: str = "nas_discovered",
):
    """Export architectures to files.

    Args:
        arch_results: List of (architecture, metrics) tuples
        output_dir: Output directory
        format: Export format ("python", "json", or "all")
        prefix: Filename prefix
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for rank, (arch, metrics) in enumerate(arch_results, 1):
        filename_base = f"{prefix}_rank{rank}"

        # Build description
        ppl = metrics.get("perplexity", "N/A")
        loss = metrics.get("final_loss", "N/A")
        n_params = metrics.get("n_params", "N/A")

        if isinstance(ppl, (int, float)):
            ppl_str = f"{ppl:.2f}"
        else:
            ppl_str = str(ppl)

        if isinstance(loss, (int, float)):
            loss_str = f"{loss:.4f}"
        else:
            loss_str = str(loss)

        if isinstance(n_params, int):
            params_str = f"{n_params:,}"
        else:
            params_str = str(n_params)

        description = (
            f"NAS-discovered architecture (rank {rank})\n"
            f"Perplexity: {ppl_str}, Loss: {loss_str}, Parameters: {params_str}"
        )

        # Export Python config
        if format in ["python", "all"]:
            config_file = output_dir / f"{filename_base}.py"
            export_config(
                architecture=arch,
                output_path=str(config_file),
                description=description,
                metrics=metrics,
            )
            logger.info(f"Exported rank {rank} to {config_file}")

        # Export JSON
        if format in ["json", "all"]:
            json_file = output_dir / f"{filename_base}.json"
            export_to_json(
                architecture=arch,
                output_path=str(json_file),
                description=description,
                metrics=metrics,
            )
            logger.info(f"Exported rank {rank} to {json_file}")


def generate_report(
    arch_results: List[Tuple[ArchitectureConfig, Dict[str, Any]]],
    output_path: Path,
    search_metadata: Dict[str, Any],
):
    """Generate a summary report of exported architectures.

    Args:
        arch_results: List of (architecture, metrics) tuples
        output_path: Path to save report
        search_metadata: Metadata from search results
    """
    lines = []

    # Header
    lines.append("# Neural Architecture Search - Export Report")
    lines.append("")
    lines.append(f"**Search Strategy:** {search_metadata.get('strategy_name', 'Unknown')}")
    lines.append(f"**Total Evaluations:** {search_metadata.get('n_evaluations', 'Unknown')}")
    lines.append(f"**Search Time:** {search_metadata.get('search_time', 'Unknown')}s")
    lines.append(f"**Exported Architectures:** {len(arch_results)}")
    lines.append("")

    # Top architectures
    lines.append("## Exported Architectures")
    lines.append("")

    for rank, (arch, metrics) in enumerate(arch_results, 1):
        lines.append(f"### Rank {rank}")
        lines.append("")

        # Generate detailed summary
        summary = generate_config_summary(arch, metrics)
        lines.append("```")
        lines.append(summary)
        lines.append("```")
        lines.append("")

    # Write report
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Generated summary report at {output_path}")


def main():
    # Parse arguments first (handles --help without loading dependencies)
    parser = argparse.ArgumentParser(
        description="Export best architectures discovered by NAS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input/output
    parser.add_argument(
        "results",
        type=str,
        help="Path to search_results.json from NAS",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="exported_configs",
        help="Output directory for exported configs (default: exported_configs)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top architectures to export (default: 5)",
    )

    # Export format
    parser.add_argument(
        "--format",
        type=str,
        default="python",
        choices=["python", "json", "all"],
        help="Export format: 'python', 'json', or 'all' (default: python)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="nas_discovered",
        help="Filename prefix for exported configs (default: nas_discovered)",
    )

    # Filtering
    parser.add_argument(
        "--max-perplexity",
        type=float,
        default=None,
        help="Only export architectures with perplexity <= this value (optional)",
    )
    parser.add_argument(
        "--max-loss",
        type=float,
        default=None,
        help="Only export architectures with loss <= this value (optional)",
    )
    parser.add_argument(
        "--min-params",
        type=int,
        default=None,
        help="Only export architectures with parameters >= this value (optional)",
    )
    parser.add_argument(
        "--max-params",
        type=int,
        default=None,
        help="Only export architectures with parameters <= this value (optional)",
    )

    # Sorting
    parser.add_argument(
        "--sort-by",
        type=str,
        default="perplexity",
        help="Metric to sort by (default: perplexity)",
    )

    # Reports
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Generate summary report (markdown)",
    )

    args = parser.parse_args()

    # Load dependencies after argument parsing
    (logger, ArchitectureConfig, export_config, export_to_json,
     generate_config_summary, generate_summary_report) = load_dependencies()

    logger.info("=" * 60)
    logger.info("NAS Architecture Exporter")
    logger.info("=" * 60)
    logger.info(f"Results file: {args.results}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Format: {args.format}")
    logger.info(f"Top-K: {args.top_k}")
    logger.info("")

    try:
        # Load search results
        results = load_search_results(args.results)

        # Parse architectures and metrics
        arch_results = parse_architecture_results(results)
        logger.info(f"Parsed {len(arch_results)} architectures")

        # Apply filters
        if any([args.max_perplexity, args.max_loss, args.min_params, args.max_params]):
            logger.info("Applying performance filters...")
            arch_results = filter_by_performance(
                arch_results,
                max_perplexity=args.max_perplexity,
                max_loss=args.max_loss,
                min_params=args.min_params,
                max_params=args.max_params,
            )
            logger.info(f"Filtered to {len(arch_results)} architectures")

        # Sort by performance
        arch_results = sort_by_performance(arch_results, metric=args.sort_by)
        logger.info(f"Sorted by {args.sort_by}")

        # Take top-k
        arch_results = arch_results[: args.top_k]
        logger.info(f"Exporting top {len(arch_results)} architectures")
        logger.info("")

        # Export architectures
        output_dir = Path(args.output)
        export_architectures(
            arch_results=arch_results,
            output_dir=output_dir,
            format=args.format,
            prefix=args.prefix,
        )

        # Generate summary report if requested
        if args.summary:
            report_path = output_dir / "export_summary.md"
            search_metadata = {
                "strategy_name": results.get("strategy_name", "Unknown"),
                "n_evaluations": results.get("n_evaluations", "Unknown"),
                "search_time": results.get("search_time", "Unknown"),
            }
            generate_report(arch_results, report_path, search_metadata)

        logger.info("")
        logger.info("=" * 60)
        logger.info("Export complete!")
        logger.info("=" * 60)
        logger.info(f"Exported {len(arch_results)} architectures to {output_dir}")
        logger.info("")
        logger.info("Next steps:")
        logger.info(f"  1. Review exported configs in {output_dir}/")
        logger.info(f"  2. Train best config: python train.py config={output_dir}/{args.prefix}_rank1.py")
        logger.info("")

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in results file: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
