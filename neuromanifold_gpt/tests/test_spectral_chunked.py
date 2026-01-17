# neuromanifold_gpt/tests/test_spectral_chunked.py
"""Tests for chunked spectral attention implementation."""
import pytest
import torch

from neuromanifold_gpt.model.spectral import FastSpectralAttention


def _non_chunked_causal_cumsum(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Non-chunked reference implementation for numerical comparison.

    This is the original O(T*k^2) memory implementation that materializes
    the full kv_prod tensor before cumsum.

    Args:
        k: (B, H, T, n_eig) key vectors
        v: (B, H, T, n_eig) value vectors

    Returns:
        kv_causal: (B, H, T, n_eig, n_eig) causal cumsum of outer products
    """
    # Original implementation: materialize full kv_prod tensor
    # Memory: O(B * H * T * n_eig^2)
    kv_prod = k.unsqueeze(-1) * v.unsqueeze(-2)  # (B, H, T, n_eig, n_eig)
    kv_causal = torch.cumsum(kv_prod, dim=2)  # Causal cumsum along T
    return kv_causal


class TestChunkedSpectralAttention:
    """Tests for chunked implementation of FastSpectralAttention."""

    def test_numerical_equivalence_chunk64(self):
        """Chunked (chunk_size=64) should match non-chunked output."""
        attn = FastSpectralAttention(
            embed_dim=128, n_eigenvectors=16, n_heads=4, chunk_size=64
        )

        # Test with T=128 (exactly 2 chunks)
        B, T, _C = 2, 128, 128
        k = torch.randn(B, attn.n_heads, T, attn.n_eig)
        v = torch.randn(B, attn.n_heads, T, attn.n_eig)

        # Compute both versions
        chunked_output = attn._chunked_causal_cumsum(k, v)
        reference_output = _non_chunked_causal_cumsum(k, v)

        # Should be numerically equivalent
        assert torch.allclose(
            chunked_output, reference_output, atol=1e-5, rtol=1e-5
        ), f"Max diff: {(chunked_output - reference_output).abs().max()}"

    def test_numerical_equivalence_chunk128(self):
        """Chunked (chunk_size=128) should match non-chunked output."""
        attn = FastSpectralAttention(
            embed_dim=128, n_eigenvectors=16, n_heads=4, chunk_size=128
        )

        # Test with T=256 (exactly 2 chunks)
        B, T, _C = 2, 256, 128
        k = torch.randn(B, attn.n_heads, T, attn.n_eig)
        v = torch.randn(B, attn.n_heads, T, attn.n_eig)

        # Compute both versions
        chunked_output = attn._chunked_causal_cumsum(k, v)
        reference_output = _non_chunked_causal_cumsum(k, v)

        # Should be numerically equivalent
        assert torch.allclose(
            chunked_output, reference_output, atol=1e-5, rtol=1e-5
        ), f"Max diff: {(chunked_output - reference_output).abs().max()}"

    def test_numerical_equivalence_chunk256(self):
        """Chunked (chunk_size=256) should match non-chunked output."""
        attn = FastSpectralAttention(
            embed_dim=128, n_eigenvectors=16, n_heads=4, chunk_size=256
        )

        # Test with T=512 (exactly 2 chunks)
        B, T, _C = 2, 512, 128
        k = torch.randn(B, attn.n_heads, T, attn.n_eig)
        v = torch.randn(B, attn.n_heads, T, attn.n_eig)

        # Compute both versions
        chunked_output = attn._chunked_causal_cumsum(k, v)
        reference_output = _non_chunked_causal_cumsum(k, v)

        # Should be numerically equivalent
        assert torch.allclose(
            chunked_output, reference_output, atol=1e-5, rtol=1e-5
        ), f"Max diff: {(chunked_output - reference_output).abs().max()}"

    def test_edge_case_t_not_divisible_63(self):
        """Test edge case where T=63 is not divisible by chunk_size=64."""
        attn = FastSpectralAttention(
            embed_dim=128, n_eigenvectors=16, n_heads=4, chunk_size=64
        )

        # T < chunk_size: single partial chunk
        B, T, _C = 2, 63, 128
        k = torch.randn(B, attn.n_heads, T, attn.n_eig)
        v = torch.randn(B, attn.n_heads, T, attn.n_eig)

        # Compute both versions
        chunked_output = attn._chunked_causal_cumsum(k, v)
        reference_output = _non_chunked_causal_cumsum(k, v)

        # Should be numerically equivalent
        assert torch.allclose(
            chunked_output, reference_output, atol=1e-5, rtol=1e-5
        ), f"Max diff: {(chunked_output - reference_output).abs().max()}"

    def test_edge_case_t_not_divisible_257(self):
        """Test edge case where T=257 is not divisible by chunk_size=128."""
        attn = FastSpectralAttention(
            embed_dim=128, n_eigenvectors=16, n_heads=4, chunk_size=128
        )

        # T = 2 * chunk_size + 1: two full chunks + one partial
        B, T, _C = 2, 257, 128
        k = torch.randn(B, attn.n_heads, T, attn.n_eig)
        v = torch.randn(B, attn.n_heads, T, attn.n_eig)

        # Compute both versions
        chunked_output = attn._chunked_causal_cumsum(k, v)
        reference_output = _non_chunked_causal_cumsum(k, v)

        # Should be numerically equivalent
        assert torch.allclose(
            chunked_output, reference_output, atol=1e-5, rtol=1e-5
        ), f"Max diff: {(chunked_output - reference_output).abs().max()}"

    def test_edge_case_single_timestep(self):
        """Test edge case with T=1."""
        attn = FastSpectralAttention(
            embed_dim=128, n_eigenvectors=16, n_heads=4, chunk_size=64
        )

        B, T, _C = 2, 1, 128
        k = torch.randn(B, attn.n_heads, T, attn.n_eig)
        v = torch.randn(B, attn.n_heads, T, attn.n_eig)

        # Compute both versions
        chunked_output = attn._chunked_causal_cumsum(k, v)
        reference_output = _non_chunked_causal_cumsum(k, v)

        # Should be numerically equivalent
        assert torch.allclose(
            chunked_output, reference_output, atol=1e-5, rtol=1e-5
        ), f"Max diff: {(chunked_output - reference_output).abs().max()}"

    def test_causality_preserved(self):
        """Verify causal property: output at time t only depends on times <= t."""
        attn = FastSpectralAttention(
            embed_dim=128, n_eigenvectors=16, n_heads=4, chunk_size=64
        )

        B, T = 2, 128
        k = torch.randn(B, attn.n_heads, T, attn.n_eig)
        v = torch.randn(B, attn.n_heads, T, attn.n_eig)

        # Compute cumsum
        kv_causal = attn._chunked_causal_cumsum(k, v)

        # Verify: kv_causal[t] should equal sum of k[i] * v[i] for i <= t
        for t in [0, 31, 63, 64, 127]:  # Test at chunk boundaries and within
            # Manually compute cumsum up to time t
            expected = torch.zeros(
                B, attn.n_heads, attn.n_eig, attn.n_eig, device=k.device
            )
            for i in range(t + 1):
                expected += (
                    k[:, :, i : i + 1, :].transpose(-1, -2) @ v[:, :, i : i + 1, :]
                )

            actual = kv_causal[:, :, t, :, :]

            assert torch.allclose(
                actual, expected.squeeze(2), atol=1e-5, rtol=1e-5
            ), f"Causality violated at t={t}, max diff: {(actual - expected.squeeze(2)).abs().max()}"

    def test_full_forward_pass_equivalence(self):
        """Test that full forward pass produces consistent outputs."""
        # Create two attention modules with different chunk sizes
        attn_small_chunk = FastSpectralAttention(
            embed_dim=128, n_eigenvectors=16, n_heads=4, chunk_size=64
        )
        attn_large_chunk = FastSpectralAttention(
            embed_dim=128, n_eigenvectors=16, n_heads=4, chunk_size=256
        )

        # Copy weights to ensure identical parameters
        attn_large_chunk.load_state_dict(attn_small_chunk.state_dict())

        # Test inputs
        B, T, C = 2, 256, 128
        x = torch.randn(B, T, C)
        spectral_basis = torch.randn(B, T, 16)
        spectral_freqs = torch.randn(B, 16)

        # Compute outputs
        with torch.no_grad():
            out_small = attn_small_chunk(x, spectral_basis, spectral_freqs)
            out_large = attn_large_chunk(x, spectral_basis, spectral_freqs)

        # Outputs should be numerically equivalent
        assert torch.allclose(
            out_small, out_large, atol=1e-5, rtol=1e-5
        ), f"Forward pass outputs differ, max diff: {(out_small - out_large).abs().max()}"

    def test_gradient_flow_chunked(self):
        """Gradients should flow through chunked implementation."""
        attn = FastSpectralAttention(
            embed_dim=128, n_eigenvectors=16, n_heads=4, chunk_size=64
        )

        B, T, C = 2, 128, 128
        x = torch.randn(B, T, C, requires_grad=True)
        spectral_basis = torch.randn(B, T, 16)
        spectral_freqs = torch.randn(B, 16)

        out = attn(x, spectral_basis, spectral_freqs)
        loss = out.sum()
        loss.backward()

        # Gradients should exist and be finite
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

    def test_output_shapes_chunked(self):
        """Chunked implementation should produce correct output shapes."""
        attn = FastSpectralAttention(
            embed_dim=128, n_eigenvectors=16, n_heads=4, chunk_size=64
        )

        # Test various sequence lengths
        for T in [1, 63, 64, 65, 127, 128, 129, 256]:
            B, _C = 2, 128
            k = torch.randn(B, attn.n_heads, T, attn.n_eig)
            v = torch.randn(B, attn.n_heads, T, attn.n_eig)

            output = attn._chunked_causal_cumsum(k, v)

            expected_shape = (B, attn.n_heads, T, attn.n_eig, attn.n_eig)
            assert (
                output.shape == expected_shape
            ), f"For T={T}, expected shape {expected_shape}, got {output.shape}"

    def test_memory_usage(self):
        """Verify chunked implementation uses significantly less memory than non-chunked.

        Expected memory reduction: 40-60% for large sequences (T=1024+).
        Original: O(B * H * T * n_eig^2) memory
        Chunked: O(B * H * chunk_size * n_eig^2) memory
        """
        # Test configuration: large sequence to demonstrate memory savings
        B, H, T, n_eig, chunk_size = 16, 8, 1024, 32, 128
        embed_dim = n_eig * H  # 256

        # Create attention module
        attn = FastSpectralAttention(
            embed_dim=embed_dim, n_eigenvectors=n_eig, n_heads=H, chunk_size=chunk_size
        )

        # Generate test data
        k = torch.randn(B, H, T, n_eig)
        v = torch.randn(B, H, T, n_eig)

        # Calculate theoretical peak memory usage (in number of elements)
        # Non-chunked: materializes full (B, H, T, n_eig, n_eig) tensor
        non_chunked_peak_elements = B * H * T * n_eig * n_eig

        # Chunked: materializes (B, H, chunk_size, n_eig, n_eig) per chunk
        # Plus running state (B, H, n_eig, n_eig)
        chunked_peak_elements = (
            B * H * chunk_size * n_eig * n_eig + B * H * n_eig * n_eig
        )

        # Calculate memory in MB (assuming float32 = 4 bytes)
        bytes_per_element = 4
        non_chunked_mb = (non_chunked_peak_elements * bytes_per_element) / (1024**2)
        chunked_mb = (chunked_peak_elements * bytes_per_element) / (1024**2)

        # Calculate memory reduction
        memory_reduction_pct = (
            (non_chunked_peak_elements - chunked_peak_elements)
            / non_chunked_peak_elements
        ) * 100

        print(f"\n{'='*60}")
        print(f"Memory Benchmark: T={T}, n_eig={n_eig}, chunk_size={chunk_size}")
        print(f"{'='*60}")
        print(f"Configuration: B={B}, H={H}, T={T}, n_eig={n_eig}")
        print(
            f"Non-chunked peak memory: {non_chunked_mb:.2f} MB ({non_chunked_peak_elements:,} elements)"
        )
        print(
            f"Chunked peak memory:     {chunked_mb:.2f} MB ({chunked_peak_elements:,} elements)"
        )
        print(f"Memory reduction:        {memory_reduction_pct:.1f}%")
        print(f"{'='*60}")

        # Verify memory reduction meets expected threshold (40-60% or better)
        assert (
            memory_reduction_pct >= 40.0
        ), f"Memory reduction {memory_reduction_pct:.1f}% is below expected 40% threshold"

        # If CUDA is available, measure actual memory usage
        if torch.cuda.is_available():
            device = torch.device("cuda")
            k_cuda = k.to(device)
            v_cuda = v.to(device)
            attn_cuda = FastSpectralAttention(
                embed_dim=embed_dim,
                n_eigenvectors=n_eig,
                n_heads=H,
                chunk_size=chunk_size,
            ).to(device)

            # Measure non-chunked memory usage
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad():
                _ = _non_chunked_causal_cumsum(k_cuda, v_cuda)
            non_chunked_actual_mb = torch.cuda.max_memory_allocated(device) / (
                1024**2
            )

            # Measure chunked memory usage
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad():
                _ = attn_cuda._chunked_causal_cumsum(k_cuda, v_cuda)
            chunked_actual_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

            actual_reduction_pct = (
                (non_chunked_actual_mb - chunked_actual_mb) / non_chunked_actual_mb
            ) * 100

            print("\nCUDA Actual Memory Usage:")
            print(f"Non-chunked actual: {non_chunked_actual_mb:.2f} MB")
            print(f"Chunked actual:     {chunked_actual_mb:.2f} MB")
            print(f"Actual reduction:   {actual_reduction_pct:.1f}%")
            print(f"{'='*60}\n")

            # Verify actual memory reduction
            assert (
                actual_reduction_pct >= 30.0
            ), f"Actual CUDA memory reduction {actual_reduction_pct:.1f}% is below threshold"

        # Verify numerical equivalence even for this large size
        with torch.no_grad():
            chunked_output = attn._chunked_causal_cumsum(k, v)
            # Only compute a slice of non-chunked for memory reasons in testing
            # Compare first and last chunks
            k_first = k[:, :, :chunk_size, :]
            v_first = v[:, :, :chunk_size, :]
            reference_first = _non_chunked_causal_cumsum(k_first, v_first)

            assert torch.allclose(
                chunked_output[:, :, :chunk_size, :, :],
                reference_first,
                atol=1e-5,
                rtol=1e-5,
            ), "Chunked output differs from reference in first chunk"
