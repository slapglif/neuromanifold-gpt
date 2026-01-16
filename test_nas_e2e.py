#!/usr/bin/env python3
"""End-to-end verification test for Neural Architecture Search.

This script verifies the complete NAS workflow:
1. Run random search with 5 evaluations on Shakespeare dataset
2. Verify search completes and produces results
3. Export top-3 architectures
4. Verify exported configs can instantiate models
5. Train one exported config for 100 iterations
6. Verify training completes without errors
"""

import sys
import os
from pathlib import Path
import shutil
import tempfile
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")


def test_nas_search():
    """Test 1: Run random search with 5 evaluations."""
    logger.info("=" * 70)
    logger.info("TEST 1: Running random search with 5 evaluations")
    logger.info("=" * 70)

    from neuromanifold_gpt.nas import SearchSpace, ArchitectureEvaluator
    from neuromanifold_gpt.nas.strategies.random_search import RandomSearch
    from neuromanifold_gpt.nas.evaluator import ComputeBudget
    import torch
    import numpy as np

    # Create output directory
    output_dir = Path("./test_nas_output")
    output_dir.mkdir(exist_ok=True)

    # Load Shakespeare dataset
    logger.info("Loading Shakespeare dataset...")
    data_dir = Path("data/shakespeare_char")

    if not data_dir.exists():
        logger.error(f"Shakespeare dataset not found at {data_dir}")
        return False

    train_data = np.memmap(data_dir / 'train.bin', dtype=np.uint16, mode='r')
    val_data = np.memmap(data_dir / 'val.bin', dtype=np.uint16, mode='r')

    # Load metadata
    meta_path = data_dir / 'meta.pkl'
    if meta_path.exists():
        import pickle
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        vocab_size = meta['vocab_size']
        logger.info(f"Vocabulary size: {vocab_size}")
    else:
        vocab_size = 65  # Default for char-level
        logger.warning(f"Meta file not found, using default vocab_size: {vocab_size}")

    # Initialize search components
    logger.info("Initializing search space and evaluator...")
    search_space = SearchSpace(vocab_size=vocab_size)
    evaluator = ArchitectureEvaluator(
        train_data=train_data,
        val_data=val_data,
        vocab_size=vocab_size,
        max_iters=100,  # Quick evaluation
        batch_size=32,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Create compute budget
    budget = ComputeBudget(
        max_evaluations=5,
        max_time_seconds=600,  # 10 minutes max
    )

    # Run random search
    logger.info("Running random search...")
    searcher = RandomSearch(search_space=search_space, seed=42)
    results = searcher.search(evaluator=evaluator, budget=budget)

    # Verify results
    if not results:
        logger.error("Search returned empty results!")
        return False

    if len(results.architectures) == 0:
        logger.error("No architectures evaluated!")
        return False

    logger.info(f"✓ Search completed successfully with {len(results.architectures)} architectures")
    logger.info(f"  Best perplexity: {results.best_by_perplexity.perplexity:.2f}")

    # Save results
    results_file = output_dir / "nas_results.json"
    results.save(results_file)
    logger.info(f"✓ Results saved to {results_file}")

    return results, output_dir, vocab_size


def test_export_architectures(results, output_dir, vocab_size):
    """Test 2: Export top-3 architectures."""
    logger.info("=" * 70)
    logger.info("TEST 2: Exporting top-3 architectures")
    logger.info("=" * 70)

    from neuromanifold_gpt.nas.export import export_config, export_to_json, generate_summary_report

    # Get top 3 architectures
    top_archs = results.get_top_k(k=3, metric='perplexity')

    if len(top_archs) < 3:
        logger.warning(f"Only {len(top_archs)} architectures available (expected 3)")
        if len(top_archs) == 0:
            logger.error("No architectures to export!")
            return False

    # Export each architecture
    exported_configs = []
    for i, (arch, eval_result) in enumerate(top_archs[:3]):
        logger.info(f"\nExporting architecture {i+1}/3...")
        logger.info(f"  Perplexity: {eval_result.perplexity:.2f}, Loss: {eval_result.loss:.4f}")

        # Export to Python config
        config_file = output_dir / f"discovered_arch_{i+1}.py"
        export_config(arch, config_file, eval_result)
        exported_configs.append(config_file)
        logger.info(f"  ✓ Python config: {config_file}")

        # Export to JSON
        json_file = output_dir / f"discovered_arch_{i+1}.json"
        export_to_json(arch, json_file, eval_result)
        logger.info(f"  ✓ JSON config: {json_file}")

    # Generate summary report
    report_file = output_dir / "nas_summary.md"
    generate_summary_report(top_archs[:3], report_file, results.search_stats)
    logger.info(f"\n✓ Summary report: {report_file}")

    return exported_configs


def test_instantiate_models(exported_configs, vocab_size):
    """Test 3: Verify exported configs can instantiate models."""
    logger.info("=" * 70)
    logger.info("TEST 3: Instantiating models from exported configs")
    logger.info("=" * 70)

    import importlib.util
    from neuromanifold_gpt.model import NeuroManifoldGPT

    for i, config_file in enumerate(exported_configs):
        logger.info(f"\nLoading config {i+1}: {config_file.name}")

        # Load the Python config file
        spec = importlib.util.spec_from_file_location(f"discovered_config_{i}", config_file)
        if spec is None or spec.loader is None:
            logger.error(f"Failed to load config file: {config_file}")
            return False

        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)

        # Get the config
        if not hasattr(config_module, 'config'):
            logger.error(f"Config file missing 'config' variable: {config_file}")
            return False

        config = config_module.config
        logger.info(f"  Config loaded: {config.n_layer} layers, {config.n_embd} embd, {config.n_head} heads")

        # Try to instantiate the model
        try:
            model = NeuroManifoldGPT(config)
            param_count = sum(p.numel() for p in model.parameters())
            logger.info(f"  ✓ Model instantiated successfully with {param_count:,} parameters")
        except Exception as e:
            logger.error(f"Failed to instantiate model: {e}")
            return False

    logger.info("\n✓ All models instantiated successfully")
    return True


def test_train_exported_config(exported_configs, vocab_size):
    """Test 4: Train one exported config for 100 iterations."""
    logger.info("=" * 70)
    logger.info("TEST 4: Training exported config for 100 iterations")
    logger.info("=" * 70)

    import importlib.util
    from neuromanifold_gpt.model import NeuroManifoldGPT
    import torch
    import numpy as np
    from pathlib import Path

    # Use first exported config
    config_file = exported_configs[0]
    logger.info(f"Training config: {config_file.name}")

    # Load config
    spec = importlib.util.spec_from_file_location("train_config", config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.config

    # Load Shakespeare dataset
    data_dir = Path("data/shakespeare_char")
    train_data = np.memmap(data_dir / 'train.bin', dtype=np.uint16, mode='r')

    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    model = NeuroManifoldGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Training loop
    logger.info("Training for 100 iterations...")
    model.train()

    batch_size = 32
    block_size = config.block_size

    for iter_num in range(100):
        # Get batch
        ix = torch.randint(len(train_data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(train_data[i:i+block_size].astype(np.int64)) for i in ix]).to(device)
        y = torch.stack([torch.from_numpy(train_data[i+1:i+1+block_size].astype(np.int64)) for i in ix]).to(device)

        # Forward pass
        try:
            logits, loss = model(x, y)

            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if (iter_num + 1) % 20 == 0:
                logger.info(f"  Iteration {iter_num + 1}/100, Loss: {loss.item():.4f}")

        except Exception as e:
            logger.error(f"Training failed at iteration {iter_num}: {e}")
            import traceback
            traceback.print_exc()
            return False

    logger.info("✓ Training completed successfully for 100 iterations")
    return True


def cleanup(output_dir):
    """Clean up test output directory."""
    if output_dir.exists():
        logger.info(f"\nCleaning up test output: {output_dir}")
        shutil.rmtree(output_dir)


def main():
    """Run all end-to-end verification tests."""
    logger.info("\n" + "=" * 70)
    logger.info("NEURAL ARCHITECTURE SEARCH - END-TO-END VERIFICATION")
    logger.info("=" * 70 + "\n")

    try:
        # Test 1: Run search
        result = test_nas_search()
        if not result:
            logger.error("❌ TEST 1 FAILED: Search did not complete")
            return False

        results, output_dir, vocab_size = result

        # Test 2: Export architectures
        exported_configs = test_export_architectures(results, output_dir, vocab_size)
        if not exported_configs:
            logger.error("❌ TEST 2 FAILED: Export failed")
            cleanup(output_dir)
            return False

        # Test 3: Instantiate models
        if not test_instantiate_models(exported_configs, vocab_size):
            logger.error("❌ TEST 3 FAILED: Model instantiation failed")
            cleanup(output_dir)
            return False

        # Test 4: Train exported config
        if not test_train_exported_config(exported_configs, vocab_size):
            logger.error("❌ TEST 4 FAILED: Training failed")
            cleanup(output_dir)
            return False

        # All tests passed
        logger.info("\n" + "=" * 70)
        logger.info("✅ ALL TESTS PASSED - END-TO-END VERIFICATION SUCCESSFUL")
        logger.info("=" * 70)

        logger.info(f"\nTest outputs saved to: {output_dir}")
        logger.info("You can inspect the exported architectures and results.")

        return True

    except Exception as e:
        logger.error(f"❌ VERIFICATION FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
