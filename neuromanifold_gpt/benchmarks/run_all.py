#!/usr/bin/env python3
"""
Master benchmark runner for NeuroManifold attention mechanisms.

Runs all benchmark suites and aggregates results:
- Quality benchmarks (perplexity, sample quality)
- Speed benchmarks (forward/backward timing, throughput)
- Memory benchmarks (peak memory usage, scaling)

Usage:
    python neuromanifold_gpt/benchmarks/run_all.py --output results.json
    python neuromanifold_gpt/benchmarks/run_all.py --quick-test
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path


def run_all_benchmarks(quick_test: bool = False, output_file: str = None, dataset: str = "shakespeare_char"):
    """Run all benchmark suites and collect results.

    Args:
        quick_test: If True, run with reduced iterations for quick validation
        output_file: Optional path to save results as JSON
        dataset: Dataset name (directory under data/)

    Returns:
        Dict containing all benchmark results
    """
    print("=" * 80)
    print("NeuroManifold Attention Benchmark Suite")
    print("=" * 80)
    print(f"Mode: {'Quick Test' if quick_test else 'Full Benchmark'}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "quick_test": quick_test,
        },
        "quality": None,
        "speed": None,
        "memory": None,
    }

    start_time = time.time()

    # Run quality benchmarks
    print("\n" + "=" * 80)
    print("PHASE 1: Quality Benchmarks")
    print("=" * 80 + "\n")
    try:
        from neuromanifold_gpt.benchmarks.attention_quality import (
            benchmark_quality,
            benchmark_sample_quality
        )

        # Perplexity benchmark
        quality_results = benchmark_quality(
            dataset=dataset,
            eval_iters=10 if quick_test else 200,
            batch_size=4 if quick_test else 12,
            block_size=256 if quick_test else 1024,
            device="cuda",
            dtype="bfloat16",
            verbose=True
        )

        # Sample quality benchmark
        sample_results = benchmark_sample_quality(
            dataset=dataset,
            num_samples=5 if quick_test else 10,
            max_new_tokens=50 if quick_test else 100,
            temperature=0.8,
            top_k=200,
            device="cuda",
            dtype="bfloat16",
            verbose=True
        )

        results["quality"] = {
            "perplexity": quality_results,
            "sample_diversity": sample_results,
        }

        print("✓ Quality benchmarks completed")

    except Exception as e:
        print(f"✗ Quality benchmarks failed: {e}")
        results["quality"] = {"error": str(e)}

    # Run speed benchmarks
    print("\n" + "=" * 80)
    print("PHASE 2: Speed Benchmarks")
    print("=" * 80 + "\n")
    try:
        from neuromanifold_gpt.benchmarks.attention_speed import benchmark_speed

        # Speed benchmark doesn't return results, it prints them
        # We'll capture that it ran successfully
        benchmark_speed(quick_test=quick_test)

        results["speed"] = {
            "status": "completed",
            "note": "See console output for detailed timing results"
        }

        print("✓ Speed benchmarks completed")

    except Exception as e:
        print(f"✗ Speed benchmarks failed: {e}")
        results["speed"] = {"error": str(e)}

    # Run memory benchmarks
    print("\n" + "=" * 80)
    print("PHASE 3: Memory Benchmarks")
    print("=" * 80 + "\n")
    try:
        from neuromanifold_gpt.benchmarks.attention_memory import benchmark_memory

        # Memory benchmark doesn't return results, it prints them
        # We'll capture that it ran successfully
        benchmark_memory(quick_test=quick_test)

        results["memory"] = {
            "status": "completed",
            "note": "See console output for detailed memory usage results"
        }

        print("✓ Memory benchmarks completed")

    except Exception as e:
        print(f"✗ Memory benchmarks failed: {e}")
        results["memory"] = {"error": str(e)}

    # Calculate total time
    total_time = time.time() - start_time
    results["metadata"]["total_time_seconds"] = total_time

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print()

    # Print quality results if available
    if results["quality"] and "perplexity" in results["quality"]:
        perplexity_data = results["quality"]["perplexity"]
        print("Quality Results:")
        print(f"  Standard perplexity:      {perplexity_data['standard']['perplexity']:.2f}")
        print(f"  NeuroManifold perplexity: {perplexity_data['neuromanifold']['perplexity']:.2f}")
        improvement = (
            (perplexity_data["standard"]["perplexity"] - perplexity_data["neuromanifold"]["perplexity"])
            / perplexity_data["standard"]["perplexity"] * 100
        )
        print(f"  Improvement:              {improvement:.1f}%")
        print()

    # Save results to JSON if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {output_file}")
        print()

    print("=" * 80)
    print("All benchmarks completed!")
    print("=" * 80)

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run all NeuroManifold attention benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark suite
  python neuromanifold_gpt/benchmarks/run_all.py --output benchmark_results.json

  # Quick test (reduced iterations)
  python neuromanifold_gpt/benchmarks/run_all.py --quick-test

  # Just verify the benchmarks work
  python neuromanifold_gpt/benchmarks/run_all.py --quick-test --output test_results.json
        """
    )

    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with reduced iterations"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Save results to JSON file (e.g., benchmark_results.json)"
    )

    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="shakespeare_char",
        help="Dataset name (default: shakespeare_char)"
    )

    args = parser.parse_args()

    try:
        results = run_all_benchmarks(
            quick_test=args.quick_test,
            output_file=args.output,
            dataset=args.dataset
        )

        # Exit with success
        return 0

    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        return 130

    except Exception as e:
        print(f"\n\nFatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
