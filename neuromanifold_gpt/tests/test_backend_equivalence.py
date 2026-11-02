#!/usr/bin/env python3
"""Test numerical equivalence across attention backends.

Tests that flash, xformers, triton, and manual backends produce
numerically equivalent outputs (within floating point tolerance).
"""
import pytest
import torch
import torch.nn as nn

from neuromanifold_gpt.model.attention import (
    FHNAttention,
    StandardAttention,
)

# Tolerance for numerical comparisons (accounting for different precision in backends)
ATOL = 1e-4  # Absolute tolerance
RTOL = 1e-3  # Relative tolerance


def force_backend(attn_module, backend_name):
    """Force an attention module to use a specific backend for testing.

    Args:
        attn_module: Attention module (StandardAttention or FHNAttention)
        backend_name: Backend to force ('flash', 'xformers', 'manual')
    """
    if backend_name == "flash":
        attn_module.flash = True
        attn_module.xformers = False
    elif backend_name == "xformers":
        attn_module.flash = False
        attn_module.xformers = True
    elif backend_name == "manual":
        attn_module.flash = False
        attn_module.xformers = False
    else:
        raise ValueError(f"Unknown backend: {backend_name}")


def get_available_backends():
    """Detect which backends are available on current system.

    Returns:
        List of available backend names
    """
    backends = ["manual"]  # Manual always available

    # Check Flash Attention (PyTorch >= 2.0)
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        backends.append("flash")

    # Check xformers
    try:
        from neuromanifold_gpt.model.attention.memory_efficient import (
            XFORMERS_AVAILABLE,
        )

        if XFORMERS_AVAILABLE:
            backends.append("xformers")
    except ImportError:
        pass

    return backends


@pytest.fixture
def sample_input():
    """Create sample input for attention tests."""
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 16
    embed_dim = 384

    x = torch.randn(batch_size, seq_len, embed_dim)
    return x


@pytest.fixture
def sample_spectral_basis():
    """Create sample spectral basis for FHN tests."""
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 16
    spectral_dim = 32

    return torch.randn(batch_size, seq_len, spectral_dim)


def test_standard_attention_backend_equivalence(sample_input):
    """Test that StandardAttention produces similar outputs across backends."""
    available_backends = get_available_backends()

    if len(available_backends) < 2:
        pytest.skip(f"Need at least 2 backends, only have: {available_backends}")

    print(f"\nTesting StandardAttention equivalence across: {available_backends}")

    # Create attention module with fixed seed
    torch.manual_seed(42)
    embed_dim = 384
    n_heads = 8

    outputs = {}
    infos = {}

    for backend in available_backends:
        # Create fresh attention module for each backend
        torch.manual_seed(42)  # Same initialization
        attn = StandardAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            dropout=0.0,  # No dropout for deterministic comparison
            bias=True,
            block_size=1024,
        )
        attn.eval()  # Evaluation mode for deterministic behavior

        # Force specific backend
        force_backend(attn, backend)

        # Run forward pass with fixed seed
        with torch.no_grad():
            y, info = attn(sample_input)

        outputs[backend] = y
        infos[backend] = info

        print(
            f"  {backend:10s}: shape={y.shape}, mean={y.mean():.6f}, std={y.std():.6f}"
        )
        assert info["backend"] == backend, f"Expected {backend}, got {info['backend']}"

    # Compare outputs between backends
    backend_pairs = []
    for i, backend1 in enumerate(available_backends):
        for backend2 in available_backends[i + 1 :]:
            backend_pairs.append((backend1, backend2))

    for backend1, backend2 in backend_pairs:
        out1 = outputs[backend1]
        out2 = outputs[backend2]

        # Check shapes match
        assert out1.shape == out2.shape, f"{backend1} and {backend2} shapes differ"

        # Check numerical equivalence
        max_diff = (out1 - out2).abs().max().item()
        mean_diff = (out1 - out2).abs().mean().item()

        print(
            f"  {backend1} vs {backend2}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}"
        )

        assert torch.allclose(out1, out2, atol=ATOL, rtol=RTOL), (
            f"{backend1} and {backend2} outputs differ: "
            f"max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}"
        )

    print("  ✓ All backends produce numerically equivalent outputs")


def test_fhn_attention_fast_path_equivalence(sample_input, sample_spectral_basis):
    """Test FHNAttention fast path (n_fhn_steps=0) equivalence across backends."""
    available_backends = get_available_backends()

    if len(available_backends) < 2:
        pytest.skip(f"Need at least 2 backends, only have: {available_backends}")

    print(
        f"\nTesting FHNAttention (fast path) equivalence across: {available_backends}"
    )

    # Create attention module with fixed seed
    torch.manual_seed(42)
    embed_dim = 384
    n_heads = 8

    outputs = {}
    infos = {}

    for backend in available_backends:
        # Create fresh attention module for each backend
        torch.manual_seed(42)  # Same initialization
        attn = FHNAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_fhn_steps=0,  # Fast path: no FHN dynamics
            dropout=0.0,
            use_flash_fhn_fusion=True,
        )
        attn.eval()

        # Force specific backend
        force_backend(attn, backend)

        # Run forward pass
        with torch.no_grad():
            y, info = attn(sample_input, sample_spectral_basis)

        outputs[backend] = y
        infos[backend] = info

        print(
            f"  {backend:10s}: shape={y.shape}, mean={y.mean():.6f}, std={y.std():.6f}"
        )
        assert info["backend"] == backend, f"Expected {backend}, got {info['backend']}"
        assert info["fhn_state"] == 0.0, "Fast path should have zero FHN state"

    # Compare outputs between backends
    backend_pairs = []
    for i, backend1 in enumerate(available_backends):
        for backend2 in available_backends[i + 1 :]:
            backend_pairs.append((backend1, backend2))

    for backend1, backend2 in backend_pairs:
        out1 = outputs[backend1]
        out2 = outputs[backend2]

        # Check shapes match
        assert out1.shape == out2.shape, f"{backend1} and {backend2} shapes differ"

        # Check numerical equivalence
        max_diff = (out1 - out2).abs().max().item()
        mean_diff = (out1 - out2).abs().mean().item()

        print(
            f"  {backend1} vs {backend2}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}"
        )

        assert torch.allclose(out1, out2, atol=ATOL, rtol=RTOL), (
            f"{backend1} and {backend2} outputs differ: "
            f"max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}"
        )

    print("  ✓ All backends produce numerically equivalent outputs")


def test_standard_attention_gradient_equivalence(sample_input):
    """Test that gradients are equivalent across backends."""
    available_backends = get_available_backends()

    if len(available_backends) < 2:
        pytest.skip(f"Need at least 2 backends, only have: {available_backends}")

    print(
        f"\nTesting StandardAttention gradient equivalence across: {available_backends}"
    )

    embed_dim = 384
    n_heads = 8

    gradients = {}

    for backend in available_backends:
        # Create fresh attention module
        torch.manual_seed(42)
        attn = StandardAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            dropout=0.0,
            bias=True,
            block_size=1024,
        )
        attn.train()  # Training mode for gradients

        # Force specific backend
        force_backend(attn, backend)

        # Create input with gradients
        x = sample_input.clone().detach().requires_grad_(True)

        # Forward pass
        y, info = attn(x)

        # Backward pass
        loss = y.sum()
        loss.backward()

        gradients[backend] = x.grad.clone()

        print(
            f"  {backend:10s}: grad mean={x.grad.mean():.6f}, grad std={x.grad.std():.6f}"
        )

    # Compare gradients between backends
    backend_pairs = []
    for i, backend1 in enumerate(available_backends):
        for backend2 in available_backends[i + 1 :]:
            backend_pairs.append((backend1, backend2))

    for backend1, backend2 in backend_pairs:
        grad1 = gradients[backend1]
        grad2 = gradients[backend2]

        # Check shapes match
        assert (
            grad1.shape == grad2.shape
        ), f"{backend1} and {backend2} gradient shapes differ"

        # Check numerical equivalence
        max_diff = (grad1 - grad2).abs().max().item()
        mean_diff = (grad1 - grad2).abs().mean().item()

        print(
            f"  {backend1} vs {backend2}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}"
        )

        # Use slightly higher tolerance for gradients due to numerical precision
        assert torch.allclose(grad1, grad2, atol=ATOL * 10, rtol=RTOL * 10), (
            f"{backend1} and {backend2} gradients differ: "
            f"max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}"
        )

    print("  ✓ All backends produce numerically equivalent gradients")


def test_backend_equivalence_different_sequence_lengths():
    """Test equivalence across backends for different sequence lengths."""
    available_backends = get_available_backends()

    if len(available_backends) < 2:
        pytest.skip(f"Need at least 2 backends, only have: {available_backends}")

    print("\nTesting backend equivalence across sequence lengths")

    embed_dim = 384
    n_heads = 8
    batch_size = 2
    seq_lengths = [8, 16, 32, 64]

    for seq_len in seq_lengths:
        print(f"\n  Sequence length: {seq_len}")

        # Create input
        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_len, embed_dim)

        outputs = {}

        for backend in available_backends:
            # Create attention module
            torch.manual_seed(42)
            attn = StandardAttention(
                embed_dim=embed_dim,
                n_heads=n_heads,
                dropout=0.0,
                block_size=1024,
            )
            attn.eval()

            # Force backend
            force_backend(attn, backend)

            # Forward pass
            with torch.no_grad():
                y, info = attn(x)

            outputs[backend] = y

        # Compare outputs
        backend_pairs = []
        for i, backend1 in enumerate(available_backends):
            for backend2 in available_backends[i + 1 :]:
                backend_pairs.append((backend1, backend2))

        for backend1, backend2 in backend_pairs:
            out1 = outputs[backend1]
            out2 = outputs[backend2]

            max_diff = (out1 - out2).abs().max().item()
            mean_diff = (out1 - out2).abs().mean().item()

            assert torch.allclose(out1, out2, atol=ATOL, rtol=RTOL), (
                f"seq_len={seq_len}: {backend1} and {backend2} differ: "
                f"max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}"
            )

        print(f"    ✓ All backends equivalent for seq_len={seq_len}")


def test_backend_equivalence_different_batch_sizes():
    """Test equivalence across backends for different batch sizes."""
    available_backends = get_available_backends()

    if len(available_backends) < 2:
        pytest.skip(f"Need at least 2 backends, only have: {available_backends}")

    print("\nTesting backend equivalence across batch sizes")

    embed_dim = 384
    n_heads = 8
    seq_len = 16
    batch_sizes = [1, 2, 4, 8]

    for batch_size in batch_sizes:
        print(f"\n  Batch size: {batch_size}")

        # Create input
        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_len, embed_dim)

        outputs = {}

        for backend in available_backends:
            # Create attention module
            torch.manual_seed(42)
            attn = StandardAttention(
                embed_dim=embed_dim,
                n_heads=n_heads,
                dropout=0.0,
                block_size=1024,
            )
            attn.eval()

            # Force backend
            force_backend(attn, backend)

            # Forward pass
            with torch.no_grad():
                y, info = attn(x)

            outputs[backend] = y

        # Compare outputs
        backend_pairs = []
        for i, backend1 in enumerate(available_backends):
            for backend2 in available_backends[i + 1 :]:
                backend_pairs.append((backend1, backend2))

        for backend1, backend2 in backend_pairs:
            out1 = outputs[backend1]
            out2 = outputs[backend2]

            max_diff = (out1 - out2).abs().max().item()
            mean_diff = (out1 - out2).abs().mean().item()

            assert torch.allclose(out1, out2, atol=ATOL, rtol=RTOL), (
                f"batch_size={batch_size}: {backend1} and {backend2} differ: "
                f"max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}"
            )

        print(f"    ✓ All backends equivalent for batch_size={batch_size}")


def test_backend_info_dict_consistency():
    """Test that info dict contains consistent information across backends."""
    available_backends = get_available_backends()

    if len(available_backends) < 2:
        pytest.skip(f"Need at least 2 backends, only have: {available_backends}")

    print("\nTesting info dict consistency across backends")

    torch.manual_seed(42)
    x = torch.randn(2, 16, 384)

    embed_dim = 384
    n_heads = 8

    infos = {}

    for backend in available_backends:
        torch.manual_seed(42)
        attn = StandardAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            dropout=0.0,
            block_size=1024,
        )
        attn.eval()

        force_backend(attn, backend)

        with torch.no_grad():
            y, info = attn(x)

        infos[backend] = info

        # Check required keys
        assert "attention_type" in info
        assert "backend" in info
        assert info["attention_type"] == "standard"
        assert info["backend"] == backend

        print(f"  {backend:10s}: keys={list(info.keys())}")

    print("  ✓ All backends provide consistent info dict structure")


def main():
    """Run all backend equivalence tests."""
    print("=" * 70)
    print("Backend Equivalence Tests")
    print("=" * 70)

    available_backends = get_available_backends()
    print(f"\nAvailable backends: {available_backends}")

    if len(available_backends) < 2:
        print("\n⚠ WARNING: Only 1 backend available, skipping equivalence tests")
        print("  Install xformers or use PyTorch >= 2.0 for multi-backend testing")
        return

    # Create fixtures
    torch.manual_seed(42)
    sample_input = torch.randn(2, 16, 384)
    sample_spectral_basis = torch.randn(2, 16, 32)

    try:
        print("\n" + "-" * 70)
        test_standard_attention_backend_equivalence(sample_input)

        print("\n" + "-" * 70)
        test_fhn_attention_fast_path_equivalence(sample_input, sample_spectral_basis)

        print("\n" + "-" * 70)
        test_standard_attention_gradient_equivalence(sample_input)

        print("\n" + "-" * 70)
        test_backend_equivalence_different_sequence_lengths()

        print("\n" + "-" * 70)
        test_backend_equivalence_different_batch_sizes()

        print("\n" + "-" * 70)
        test_backend_info_dict_consistency()

        print("\n" + "=" * 70)
        print("All Backend Equivalence Tests Passed! ✓")
        print("=" * 70)
        print("\nSummary:")
        print(f"  • Tested {len(available_backends)} backends: {available_backends}")
        print("  • Forward pass outputs are numerically equivalent")
        print("  • Gradient computations are numerically equivalent")
        print("  • Equivalence holds across different sequence lengths")
        print("  • Equivalence holds across different batch sizes")
        print("  • Info dicts are consistent across backends")
        print()
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
