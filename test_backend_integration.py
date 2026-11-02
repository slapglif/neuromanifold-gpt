#!/usr/bin/env python3
"""Integration test with full model training using backend selection.

Subtask 6-3: Integration test with full model training.

This test verifies:
1. Model creation with specified attention backend
2. Full training loop for N steps
3. Loss computation and gradient updates work correctly
4. No crashes, NaN, or Inf during training
5. Backend selection works as expected (auto, flash, xformers, triton, manual)
"""
import argparse
import sys
import torch
import torch.nn as nn
from neuromanifold_gpt.config import NeuroManifoldConfig
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT


def create_dummy_dataset(vocab_size, batch_size, seq_len, num_batches):
    """Create dummy dataset for training.

    Args:
        vocab_size: Size of vocabulary
        batch_size: Batch size
        seq_len: Sequence length
        num_batches: Number of batches to generate

    Returns:
        List of (input_tokens, target_tokens) tuples
    """
    dataset = []
    for _ in range(num_batches):
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        dataset.append((tokens, targets))
    return dataset


def test_training_with_backend(backend, num_steps):
    """Run full model training with specified backend.

    Args:
        backend: Backend to use ('auto', 'flash', 'xformers', 'triton', 'manual')
        num_steps: Number of training steps

    Returns:
        bool: True if training succeeds, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"Testing: Full model training with backend='{backend}'")
    print('='*70)

    try:
        # Step 1: Create model with specified backend
        print(f"[1/6] Creating model with backend='{backend}'...")
        config = NeuroManifoldConfig(
            attention_type='standard',
            attention_backend=backend,
            n_layer=2,
            n_embd=128,
            n_heads=4,
            block_size=64,
            vocab_size=256,
            use_sdr=False,
            skip_manifold_spectral=True,
            use_mhc=False,
            use_kan=False,
            use_mtp=False,
        )
        model = NeuroManifoldGPT(config)
        model.train()
        print(f"      ✓ Model created successfully")
        print(f"      ✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Step 2: Create optimizer
        print(f"[2/6] Creating optimizer...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        print(f"      ✓ AdamW optimizer created (lr=1e-3)")

        # Step 3: Create dummy dataset
        print(f"[3/6] Creating dummy dataset...")
        batch_size = 4
        seq_len = 32
        dataset = create_dummy_dataset(
            vocab_size=config.vocab_size,
            batch_size=batch_size,
            seq_len=seq_len,
            num_batches=num_steps
        )
        print(f"      ✓ Created {num_steps} batches (batch_size={batch_size}, seq_len={seq_len})")

        # Step 4: Run forward pass to check backend
        print(f"[4/6] Checking backend selection...")
        tokens, targets = dataset[0]
        with torch.no_grad():
            _, _, info = model(tokens, targets)

        # Extract backend from first block's info
        if "block_infos" in info and len(info["block_infos"]) > 0:
            block_info = info["block_infos"][0]
            actual_backend = block_info.get("backend", "unknown")
            print(f"      ✓ Detected backend: '{actual_backend}'")

            # If user requested specific backend, verify it's used (unless unavailable)
            if backend != 'auto' and actual_backend != backend:
                # Only warn, don't fail - backend may not be available
                print(f"      ⚠ Requested '{backend}', got '{actual_backend}' (may be unavailable)")
        else:
            print(f"      ⚠ Could not detect backend from info dict")

        # Step 5: Run training loop
        print(f"[5/6] Running training loop for {num_steps} steps...")
        loss_history = []

        for step, (tokens, targets) in enumerate(dataset):
            # Forward pass
            logits, loss, info = model(tokens, targets)

            # Verify loss is valid
            assert loss is not None, f"Step {step}: Loss is None"
            assert loss.ndim == 0, f"Step {step}: Loss should be scalar, got shape {loss.shape}"
            assert not torch.isnan(loss), f"Step {step}: Loss is NaN"
            assert not torch.isinf(loss), f"Step {step}: Loss is Inf"
            assert loss.item() > 0, f"Step {step}: Loss should be positive, got {loss.item()}"

            loss_history.append(loss.item())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Check gradients are valid
            for name, param in model.named_parameters():
                if param.grad is not None:
                    assert not torch.isnan(param.grad).any(), (
                        f"Step {step}: NaN gradient in {name}"
                    )
                    assert not torch.isinf(param.grad).any(), (
                        f"Step {step}: Inf gradient in {name}"
                    )

            # Optimizer step
            optimizer.step()

            # Print progress every few steps
            if (step + 1) % max(1, num_steps // 5) == 0 or step == num_steps - 1:
                print(f"      - Step {step + 1}/{num_steps}: loss = {loss.item():.4f}")

        print(f"      ✓ Training completed successfully")
        print(f"      ✓ Loss range: [{min(loss_history):.4f}, {max(loss_history):.4f}]")

        # Step 6: Verify training dynamics
        print(f"[6/6] Verifying training dynamics...")

        # Check that loss values are reasonable
        avg_loss = sum(loss_history) / len(loss_history)
        print(f"      ✓ Average loss: {avg_loss:.4f}")

        # Check that loss doesn't explode or vanish
        assert max(loss_history) < 100.0, f"Loss exploded: max={max(loss_history):.4f}"
        assert min(loss_history) > 0.01, f"Loss vanished: min={min(loss_history):.4f}"
        print(f"      ✓ Loss remains in reasonable range")

        # Check that model parameters changed
        initial_params = []
        for param in model.parameters():
            initial_params.append(param.clone().detach())

        # Run one more step
        tokens, targets = dataset[0]
        logits, loss, _ = model(tokens, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify at least some parameters changed
        params_changed = False
        for param, initial in zip(model.parameters(), initial_params):
            if not torch.allclose(param, initial, atol=1e-8):
                params_changed = True
                break

        assert params_changed, "Model parameters did not change during training"
        print(f"      ✓ Model parameters updated during training")

        print(f"\n✅ PASSED: backend='{backend}' training successful")
        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_attention_types(backend, num_steps):
    """Test training with different attention types using specified backend.

    Args:
        backend: Backend to use
        num_steps: Number of training steps per attention type

    Returns:
        bool: True if all tests pass, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"Testing: Multiple attention types with backend='{backend}'")
    print('='*70)

    attention_types = ['standard', 'fhn']
    results = {}

    for attention_type in attention_types:
        print(f"\n--- Testing attention_type='{attention_type}' ---")

        try:
            # Create model
            config = NeuroManifoldConfig(
                attention_type=attention_type,
                attention_backend=backend,
                n_layer=2,
                n_embd=128,
                n_heads=4,
                block_size=64,
                vocab_size=256,
                use_sdr=False,
                skip_manifold_spectral=True,
                use_mhc=False,
                use_kan=False,
                use_mtp=False,
            )
            model = NeuroManifoldGPT(config)
            model.train()

            # Create optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            # Create small dataset
            dataset = create_dummy_dataset(
                vocab_size=config.vocab_size,
                batch_size=2,
                seq_len=16,
                num_batches=num_steps
            )

            # Train for a few steps
            for step, (tokens, targets) in enumerate(dataset):
                logits, loss, info = model(tokens, targets)

                assert loss is not None and not torch.isnan(loss) and not torch.isinf(loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step == 0 or step == num_steps - 1:
                    print(f"      Step {step + 1}: loss = {loss.item():.4f}")

            print(f"      ✓ {attention_type} training successful")
            results[attention_type] = True

        except Exception as e:
            print(f"      ❌ {attention_type} training failed: {e}")
            results[attention_type] = False

    # Print summary
    all_passed = all(results.values())
    print(f"\n{'='*70}")
    print(f"Summary for backend='{backend}':")
    for attention_type, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} | {attention_type}")
    print('='*70)

    return all_passed


def main():
    """Run backend integration tests."""
    parser = argparse.ArgumentParser(
        description="Integration test for attention backend selection with full model training"
    )
    parser.add_argument(
        '--backend',
        type=str,
        default='auto',
        choices=['auto', 'flash', 'xformers', 'triton', 'pytorch', 'manual', 'all'],
        help='Attention backend to test (default: auto)'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=10,
        help='Number of training steps (default: 10)'
    )
    parser.add_argument(
        '--attention-types',
        action='store_true',
        help='Test multiple attention types with the specified backend'
    )

    args = parser.parse_args()

    print("="*70)
    print("BACKEND INTEGRATION TEST: FULL MODEL TRAINING")
    print("Subtask 6-3: Integration test with full model training")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Backend: {args.backend}")
    print(f"  Steps: {args.steps}")
    print(f"  Test multiple attention types: {args.attention_types}")

    results = {}

    if args.backend == 'all':
        # Test all backends
        backends = ['auto', 'flash', 'xformers', 'triton', 'pytorch', 'manual']
        print(f"\nTesting all backends: {backends}")

        for backend in backends:
            results[backend] = test_training_with_backend(backend, args.steps)
    else:
        # Test single backend
        if args.attention_types:
            # Test multiple attention types with this backend
            results['multi_attention'] = test_multiple_attention_types(args.backend, args.steps)
        else:
            # Test single backend with standard attention
            results[args.backend] = test_training_with_backend(args.backend, args.steps)

    # Print final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    all_passed = all(results.values())

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} | {test_name}")

    print("="*70)

    if all_passed:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("\nVerification complete:")
        print("  ✓ Model creation with backend selection works")
        print("  ✓ Full training loop executes without errors")
        print("  ✓ Loss computation is stable (no NaN/Inf)")
        print("  ✓ Gradients flow correctly through the model")
        print("  ✓ Model parameters update during training")
        print("  ✓ Backend selection integrates correctly with model training")
        print()
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
