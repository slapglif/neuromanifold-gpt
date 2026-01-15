#!/usr/bin/env python3
"""Standalone test for chunked spectral attention - bypasses package imports."""
import sys
import importlib.util
import torch

# Load spectral module directly without triggering package __init__
spec = importlib.util.spec_from_file_location(
    "spectral",
    "./neuromanifold_gpt/model/spectral.py"
)
spectral_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spectral_module)

FastSpectralAttention = spectral_module.FastSpectralAttention


def _non_chunked_causal_cumsum(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Non-chunked reference implementation for numerical comparison."""
    kv_prod = k.unsqueeze(-1) * v.unsqueeze(-2)
    kv_causal = torch.cumsum(kv_prod, dim=2)
    return kv_causal


def test_numerical_equivalence():
    """Test that chunked and non-chunked produce equivalent results."""
    print("Running numerical equivalence tests...")

    test_cases = [
        (64, 128, "chunk_size=64, T=128 (exactly 2 chunks)"),
        (128, 256, "chunk_size=128, T=256 (exactly 2 chunks)"),
        (256, 512, "chunk_size=256, T=512 (exactly 2 chunks)"),
        (64, 63, "chunk_size=64, T=63 (single partial chunk)"),
        (128, 257, "chunk_size=128, T=257 (2 full + 1 partial)"),
        (64, 1, "chunk_size=64, T=1 (single timestep)"),
    ]

    passed = 0
    failed = 0

    for chunk_size, T, description in test_cases:
        attn = FastSpectralAttention(embed_dim=128, n_eigenvectors=16, n_heads=4, chunk_size=chunk_size)

        B = 2
        k = torch.randn(B, attn.n_heads, T, attn.n_eig)
        v = torch.randn(B, attn.n_heads, T, attn.n_eig)

        chunked_output = attn._chunked_causal_cumsum(k, v)
        reference_output = _non_chunked_causal_cumsum(k, v)

        if torch.allclose(chunked_output, reference_output, atol=1e-5, rtol=1e-5):
            print(f"  ✓ {description}")
            passed += 1
        else:
            max_diff = (chunked_output - reference_output).abs().max()
            print(f"  ✗ {description} - Max diff: {max_diff}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def test_causality():
    """Test that causality is preserved across chunk boundaries."""
    print("\nTesting causality preservation...")

    attn = FastSpectralAttention(embed_dim=128, n_eigenvectors=16, n_heads=4, chunk_size=64)

    B, T = 2, 128
    k = torch.randn(B, attn.n_heads, T, attn.n_eig)
    v = torch.randn(B, attn.n_heads, T, attn.n_eig)

    kv_causal = attn._chunked_causal_cumsum(k, v)

    # Test at chunk boundaries and within chunks
    test_timesteps = [0, 31, 63, 64, 127]
    passed = 0
    failed = 0

    for t in test_timesteps:
        # Manually compute cumsum up to time t
        expected = torch.zeros(B, attn.n_heads, attn.n_eig, attn.n_eig, device=k.device)
        for i in range(t + 1):
            expected += k[:, :, i:i+1, :].transpose(-1, -2) @ v[:, :, i:i+1, :]

        actual = kv_causal[:, :, t, :, :]

        if torch.allclose(actual, expected.squeeze(2), atol=1e-5, rtol=1e-5):
            print(f"  ✓ Causality preserved at t={t}")
            passed += 1
        else:
            max_diff = (actual - expected.squeeze(2)).abs().max()
            print(f"  ✗ Causality violated at t={t}, max diff: {max_diff}")
            failed += 1

    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


def test_forward_pass():
    """Test full forward pass with different chunk sizes."""
    print("\nTesting full forward pass equivalence...")

    # Create two attention modules with different chunk sizes
    attn_small = FastSpectralAttention(embed_dim=128, n_eigenvectors=16, n_heads=4, chunk_size=64)
    attn_large = FastSpectralAttention(embed_dim=128, n_eigenvectors=16, n_heads=4, chunk_size=256)

    # Copy weights
    attn_large.load_state_dict(attn_small.state_dict())

    B, T, C = 2, 256, 128
    x = torch.randn(B, T, C)
    spectral_basis = torch.randn(B, T, 16)
    spectral_freqs = torch.randn(B, 16)

    with torch.no_grad():
        out_small = attn_small(x, spectral_basis, spectral_freqs)
        out_large = attn_large(x, spectral_basis, spectral_freqs)

    if torch.allclose(out_small, out_large, atol=1e-5, rtol=1e-5):
        print("  ✓ Forward pass outputs match")
        return True
    else:
        max_diff = (out_small - out_large).abs().max()
        print(f"  ✗ Forward pass outputs differ, max diff: {max_diff}")
        return False


if __name__ == "__main__":
    print("="*60)
    print("Chunked Spectral Attention Tests")
    print("="*60)

    all_passed = True

    all_passed &= test_numerical_equivalence()
    all_passed &= test_causality()
    all_passed &= test_forward_pass()

    print("\n" + "="*60)
    if all_passed:
        print("All tests PASSED ✓")
        sys.exit(0)
    else:
        print("Some tests FAILED ✗")
        sys.exit(1)
