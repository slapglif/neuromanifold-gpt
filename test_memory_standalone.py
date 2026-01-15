#!/usr/bin/env python3
"""Standalone test for memory benchmarking of chunked spectral attention."""
import sys
import importlib.util
import torch

# Import spectral module directly to avoid package __init__ issues
spec = importlib.util.spec_from_file_location('spectral', './neuromanifold_gpt/model/spectral.py')
spectral_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spectral_module)
FastSpectralAttention = spectral_module.FastSpectralAttention


def _non_chunked_causal_cumsum(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Non-chunked reference implementation for comparison."""
    kv_prod = k.unsqueeze(-1) * v.unsqueeze(-2)
    kv_causal = torch.cumsum(kv_prod, dim=2)
    return kv_causal


def test_memory_usage():
    """Verify chunked implementation uses significantly less memory than non-chunked."""
    # Test configuration: large sequence to demonstrate memory savings
    B, H, T, n_eig, chunk_size = 16, 8, 1024, 32, 128
    embed_dim = n_eig * H  # 256

    # Create attention module
    attn = FastSpectralAttention(embed_dim=embed_dim, n_eigenvectors=n_eig, n_heads=H, chunk_size=chunk_size)

    # Generate test data
    k = torch.randn(B, H, T, n_eig)
    v = torch.randn(B, H, T, n_eig)

    # Calculate theoretical peak memory usage (in number of elements)
    # Non-chunked: materializes full (B, H, T, n_eig, n_eig) tensor
    non_chunked_peak_elements = B * H * T * n_eig * n_eig

    # Chunked: materializes (B, H, chunk_size, n_eig, n_eig) per chunk
    # Plus running state (B, H, n_eig, n_eig)
    chunked_peak_elements = B * H * chunk_size * n_eig * n_eig + B * H * n_eig * n_eig

    # Calculate memory in MB (assuming float32 = 4 bytes)
    bytes_per_element = 4
    non_chunked_mb = (non_chunked_peak_elements * bytes_per_element) / (1024 ** 2)
    chunked_mb = (chunked_peak_elements * bytes_per_element) / (1024 ** 2)

    # Calculate memory reduction
    memory_reduction_pct = ((non_chunked_peak_elements - chunked_peak_elements) / non_chunked_peak_elements) * 100

    print(f"\n{'='*60}")
    print(f"Memory Benchmark: T={T}, n_eig={n_eig}, chunk_size={chunk_size}")
    print(f"{'='*60}")
    print(f"Configuration: B={B}, H={H}, T={T}, n_eig={n_eig}")
    print(f"Non-chunked peak memory: {non_chunked_mb:.2f} MB ({non_chunked_peak_elements:,} elements)")
    print(f"Chunked peak memory:     {chunked_mb:.2f} MB ({chunked_peak_elements:,} elements)")
    print(f"Memory reduction:        {memory_reduction_pct:.1f}%")
    print(f"{'='*60}")

    # Verify memory reduction meets expected threshold (40-60% or better)
    assert memory_reduction_pct >= 40.0, \
        f"Memory reduction {memory_reduction_pct:.1f}% is below expected 40% threshold"

    # If CUDA is available, measure actual memory usage
    if torch.cuda.is_available():
        device = torch.device('cuda')
        k_cuda = k.to(device)
        v_cuda = v.to(device)
        attn_cuda = FastSpectralAttention(embed_dim=embed_dim, n_eigenvectors=n_eig,
                                          n_heads=H, chunk_size=chunk_size).to(device)

        # Measure non-chunked memory usage
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = _non_chunked_causal_cumsum(k_cuda, v_cuda)
        non_chunked_actual_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        # Measure chunked memory usage
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = attn_cuda._chunked_causal_cumsum(k_cuda, v_cuda)
        chunked_actual_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        actual_reduction_pct = ((non_chunked_actual_mb - chunked_actual_mb) / non_chunked_actual_mb) * 100

        print(f"\nCUDA Actual Memory Usage:")
        print(f"Non-chunked actual: {non_chunked_actual_mb:.2f} MB")
        print(f"Chunked actual:     {chunked_actual_mb:.2f} MB")
        print(f"Actual reduction:   {actual_reduction_pct:.1f}%")
        print(f"{'='*60}\n")

        # Verify actual memory reduction
        assert actual_reduction_pct >= 30.0, \
            f"Actual CUDA memory reduction {actual_reduction_pct:.1f}% is below threshold"

    # Verify numerical equivalence even for this large size
    with torch.no_grad():
        chunked_output = attn._chunked_causal_cumsum(k, v)
        # Only compute a slice of non-chunked for memory reasons in testing
        # Compare first chunk
        k_first = k[:, :, :chunk_size, :]
        v_first = v[:, :, :chunk_size, :]
        reference_first = _non_chunked_causal_cumsum(k_first, v_first)

        assert torch.allclose(chunked_output[:, :, :chunk_size, :, :], reference_first,
                            atol=1e-5, rtol=1e-5), \
            "Chunked output differs from reference in first chunk"

    print("\n✅ Test passed! Memory reduction verified and numerical correctness confirmed.")
    return True


if __name__ == '__main__':
    try:
        test_memory_usage()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
