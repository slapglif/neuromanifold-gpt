#!/usr/bin/env python3
"""Neural Architecture Search for NeuroManifoldGPT.

This script demonstrates how to use the NAS framework to discover optimal
architectures for your dataset. It supports multiple search strategies
(random, evolutionary) and provides flexible compute budget control.

The script will:
1. Load your dataset
2. Define the architecture search space
3. Run the selected search strategy
4. Export top architectures as config files
5. Generate a summary report

Usage:
    # Quick random search (20 evaluations)
    python examples/nas_search.py --strategy random --budget 20

    # Evolutionary search with time limit
    python examples/nas_search.py --strategy evolutionary --max-time 3600

    # Custom dataset and output directory
    python examples/nas_search.py --data data/custom.txt --output nas_results/

    # Resume from checkpoint
    python examples/nas_search.py --resume nas_results/checkpoint.json

Examples:
    # Fast test (5 architectures, 100 training iterations each)
    python examples/nas_search.py --strategy random --budget 5 --iters 100

    # Full search with evolutionary algorithm
    python examples/nas_search.py --strategy evolutionary --budget 100 \\
        --population 20 --max-time 7200

    # Search with early stopping
    python examples/nas_search.py --strategy random --budget 50 \\
        --target-ppl 15.0 --patience 10
"""

import argparse
import sys
import os
from pathlib import Path
import json
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_dependencies():
    """Load dependencies after argument parsing."""
    # Configure logging
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, format="{time:HH:mm:ss} | {level} | {message}")

    import torch
    import torch.nn.functional as F

    from neuromanifold_gpt.nas.search_space import SearchSpace
    from neuromanifold_gpt.nas.evaluator import ArchitectureEvaluator, ComputeBudget
    from neuromanifold_gpt.nas.strategies.random_search import RandomSearch
    from neuromanifold_gpt.nas.strategies.evolutionary import EvolutionarySearch
    from neuromanifold_gpt.nas.export import export_config, generate_summary_report

    return (logger, torch, SearchSpace, ArchitectureEvaluator, ComputeBudget,
            RandomSearch, EvolutionarySearch, export_config, generate_summary_report)


def load_dataset(data_path: str):
    """Load text dataset and create vocabulary.

    Args:
        data_path: Path to text file

    Returns:
        Tuple of (data tensor, vocab_size, char_to_idx, idx_to_char)
    """
    logger.info(f"Loading dataset from {data_path}...")

    # Try different possible data locations
    possible_paths = [
        Path(data_path),
        Path("neuromanifold_gpt/data") / data_path,
        Path("data") / data_path,
    ]

    data_file = None
    for p in possible_paths:
        if p.exists():
            data_file = p
            break

    if data_file is None:
        raise FileNotFoundError(
            f"Could not find data file: {data_path}\n"
            f"Tried: {[str(p) for p in possible_paths]}"
        )

    # Load text
    with open(data_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Build character vocabulary
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    # Encode text
    data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)

    logger.info(f"Loaded {len(text):,} characters, vocab size: {vocab_size}")

    return data, vocab_size, char_to_idx, idx_to_char


def create_searcher(args, search_space, evaluator, budget):
    """Create the appropriate search strategy.

    Args:
        args: Command-line arguments
        search_space: SearchSpace instance
        evaluator: ArchitectureEvaluator instance
        budget: ComputeBudget instance

    Returns:
        Searcher instance
    """
    if args.strategy == "random":
        logger.info("Using Random Search strategy")
        return RandomSearch(
            search_space=search_space,
            evaluator=evaluator,
            budget=budget,
            seed=args.seed,
        )
    elif args.strategy == "evolutionary":
        logger.info(f"Using Evolutionary Search strategy (pop={args.population})")
        return EvolutionarySearch(
            search_space=search_space,
            evaluator=evaluator,
            budget=budget,
            population_size=args.population,
            tournament_size=args.tournament_size,
            mutation_rate=args.mutation_rate,
            elitism_ratio=args.elitism,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")


def export_results(search_result, output_dir: Path, top_k: int = 5):
    """Export search results and top architectures.

    Args:
        search_result: SearchResult from the search
        output_dir: Directory to save results
        top_k: Number of top architectures to export
    """
    logger.info(f"\nExporting results to {output_dir}...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full search results as JSON
    results_file = output_dir / "search_results.json"
    with open(results_file, "w") as f:
        json.dump(search_result.to_dict(), f, indent=2)
    logger.info(f"Saved search results to {results_file}")

    # Export top-k architectures as config files
    top_archs = search_result.get_top_k(top_k)

    configs_dir = output_dir / "configs"
    configs_dir.mkdir(exist_ok=True)

    for rank, (arch, result) in enumerate(top_archs, 1):
        config_name = f"nas_discovered_rank{rank}"
        config_file = configs_dir / f"{config_name}.py"

        export_config(
            architecture=arch,
            output_path=config_file,
            config_name=config_name,
            description=(
                f"NAS-discovered architecture (rank {rank}/{top_k})\n"
                f"Perplexity: {result.perplexity:.2f}, "
                f"Loss: {result.final_loss:.4f}, "
                f"Parameters: {result.n_params:,}"
            ),
        )
        logger.info(f"Exported rank {rank} config to {config_file}")

    # Generate summary report
    report_file = output_dir / "search_summary.md"
    summary_lines = [
        "# Neural Architecture Search Results",
        "",
        f"**Strategy:** {search_result.strategy_name}",
        f"**Search Time:** {search_result.search_time:.1f}s",
        f"**Evaluations:** {search_result.n_evaluations}",
        f"**Search Space Size:** {search_result.search_space_size:,}",
        "",
        "## Top Architectures",
        "",
    ]

    for rank, (arch, result) in enumerate(top_archs, 1):
        summary = generate_summary_report(arch, result)
        summary_lines.extend([
            f"### Rank {rank}",
            "",
            summary,
            "",
        ])

    with open(report_file, "w") as f:
        f.write("\n".join(summary_lines))
    logger.info(f"Generated summary report at {report_file}")


def main():
    # Parse arguments first (handles --help without loading dependencies)
    parser = argparse.ArgumentParser(
        description="Neural Architecture Search for NeuroManifoldGPT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Search configuration
    parser.add_argument(
        "--strategy",
        type=str,
        default="random",
        choices=["random", "evolutionary"],
        help="Search strategy to use (default: random)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=20,
        help="Maximum number of architectures to evaluate (default: 20)",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=None,
        help="Maximum search time in seconds (default: no limit)",
    )
    parser.add_argument(
        "--target-ppl",
        type=float,
        default=None,
        help="Stop when perplexity reaches this target (default: no target)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience (evaluations without improvement) (default: no patience)",
    )

    # Evolutionary strategy specific
    parser.add_argument(
        "--population",
        type=int,
        default=20,
        help="Population size for evolutionary search (default: 20)",
    )
    parser.add_argument(
        "--tournament-size",
        type=int,
        default=3,
        help="Tournament size for parent selection (default: 3)",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.3,
        help="Mutation rate for evolutionary search (default: 0.3)",
    )
    parser.add_argument(
        "--elitism",
        type=float,
        default=0.1,
        help="Elitism ratio (fraction of population to preserve) (default: 0.1)",
    )

    # Training configuration
    parser.add_argument(
        "--iters",
        type=int,
        default=200,
        help="Training iterations per architecture evaluation (default: 200)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=256,
        help="Context length for training (default: 256)",
    )

    # Data and output
    parser.add_argument(
        "--data",
        type=str,
        default="shakespeare.txt",
        help="Path to training data (default: shakespeare.txt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="nas_results",
        help="Output directory for results (default: nas_results)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top architectures to export (default: 5)",
    )

    # Checkpoint and resume
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint file (default: start fresh)",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10,
        help="Save checkpoint every N evaluations (default: 10)",
    )

    # Device and misc
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: 'cuda', 'cpu', or 'auto' (default: auto)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    # Load dependencies after argument parsing (allows --help to work without dependencies)
    (logger, torch, SearchSpace, ArchitectureEvaluator, ComputeBudget,
     RandomSearch, EvolutionarySearch, export_config, generate_summary_report) = load_dependencies()

    # Auto-detect device
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    logger.info("="*60)
    logger.info("Neural Architecture Search for NeuroManifoldGPT")
    logger.info("="*60)
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Budget: {args.budget} evaluations")
    if args.max_time:
        logger.info(f"Max time: {args.max_time}s")
    logger.info("")

    # Handle resume
    if args.resume:
        logger.info(f"Resume functionality not yet implemented")
        logger.info(f"Checkpoint: {args.resume}")
        # TODO: Implement checkpoint loading
        sys.exit(1)

    # Load dataset
    data, vocab_size, char_to_idx, idx_to_char = load_dataset(args.data)

    # Create search space
    search_space = SearchSpace()
    search_space_size = search_space.size()
    logger.info(f"Search space size: {search_space_size:,} possible architectures")
    logger.info("")

    # Create evaluator
    evaluator = ArchitectureEvaluator(
        vocab_size=vocab_size,
        block_size=args.block_size,
        device=args.device,
    )

    # Create compute budget
    budget = ComputeBudget(
        max_evaluations=args.budget,
        max_time_seconds=args.max_time,
        min_perplexity_target=args.target_ppl,
        patience=args.patience,
    )

    # Create searcher
    searcher = create_searcher(args, search_space, evaluator, budget)

    # Run search
    logger.info("Starting architecture search...")
    logger.info("")

    search_start = time.time()
    search_result = searcher.search(
        data=data,
        n_iters=args.iters,
        batch_size=args.batch_size,
    )
    search_time = time.time() - search_start

    # Display results
    logger.info("")
    logger.info("="*60)
    logger.info("SEARCH COMPLETE")
    logger.info("="*60)
    logger.info(f"Total time: {search_time:.1f}s")
    logger.info(f"Evaluations: {search_result.n_evaluations}")
    logger.info(f"Coverage: {search_result.n_evaluations / search_space_size * 100:.4f}%")
    logger.info("")

    # Show top architectures
    logger.info("Top Architectures:")
    logger.info("-"*60)
    top_archs = search_result.get_top_k(args.top_k)

    for rank, (arch, result) in enumerate(top_archs, 1):
        logger.info(
            f"{rank}. PPL={result.perplexity:.2f} | "
            f"Loss={result.final_loss:.4f} | "
            f"Params={result.n_params:,} | "
            f"Speed={result.time_per_iter_ms:.1f}ms/iter"
        )
        logger.info(f"   Attention={arch.attention_type}, MHC={arch.use_mhc}, "
                   f"KAN={arch.use_kan}, Layers={arch.n_layer}")

    logger.info("")

    # Export results
    output_dir = Path(args.output)
    export_results(search_result, output_dir, top_k=args.top_k)

    logger.info("")
    logger.info("="*60)
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*60)
    logger.info("")
    logger.info("Next steps:")
    logger.info(f"  1. Review summary: {output_dir / 'search_summary.md'}")
    logger.info(f"  2. Train best config: python train.py config={output_dir / 'configs/nas_discovered_rank1.py'}")
    logger.info("")


if __name__ == "__main__":
    main()
