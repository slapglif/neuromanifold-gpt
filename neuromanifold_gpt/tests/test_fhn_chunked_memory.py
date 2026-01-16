# neuromanifold_gpt/tests/test_fhn_chunked_memory.py
"""Tests for chunked FHN attention implementation with memory efficiency."""
import pytest
import torch
from neuromanifold_gpt.model.attention.fhn import FHNAttention


def _non_chunked_fhn_attention(
    attn: FHNAttention,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Non-chunked reference implementation for numerical comparison.

    This is the standard O(T²) memory implementation that materializes
    the full attention matrix.

    Args:
        attn: FHNAttention module instance
        q: (B, H, T, D) query vectors
        k: (B, H, T, D) key vectors
        v: (B, H, T, D) value vectors

    Returns:
        out: (B, H, T, D) attention output
        fhn_state_mean: Scalar mean FHN state
    """
    B, H, T, D = q.shape

    # Compute full attention weights
    attn_weights = torch.einsum("b h t d, b h s d -> b h t s", q, k)
    attn_weights = attn_weights / (D ** 0.5)

    # Causal mask
    causal_mask = torch.triu(
        torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1
    )
    attn_weights = attn_weights.masked_fill(
        causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
    )

    # Softmax
    attn_probs = torch.softmax(attn_weights, dim=-1)
    attn_probs = attn.dropout(attn_probs)

    # FHN dynamics if configured
    if attn.n_fhn_steps > 0:
        attn_energy = attn_probs.sum(dim=-1)
        fhn_out, fhn_state = attn.fhn(
            attn_energy.unsqueeze(-1).expand(-1, -1, -1, D),
            n_steps=attn.n_fhn_steps,
        )
        fhn_gate = torch.sigmoid(fhn_out.mean(dim=-1)).unsqueeze(-1)
        attn_probs = attn_probs * (0.5 + 0.5 * fhn_gate)
        attn_probs = attn_probs / (attn_probs.sum(dim=-1, keepdim=True) + 1e-8)
        fhn_state_mean = fhn_state.mean()
    else:
        fhn_state_mean = torch.tensor(0.0, device=q.device)

    # Apply attention to values
    out = torch.einsum("b h t s, b h s d -> b h t d", attn_probs, v)

    return out, fhn_state_mean


class TestChunkedFHNAttention:
    """Tests for chunked implementation of FHNAttention."""

    def test_numerical_equivalence_chunk256(self):
        """Chunked (chunk_size=256) should match non-chunked output."""
        attn = FHNAttention(
            embed_dim=128,
            n_heads=4,
            n_fhn_steps=2,
            chunk_size=256,
            dropout=0.0,  # Disable dropout for deterministic comparison
        )
        attn.eval()

        # Test with T=512 (exactly 2 chunks)
        B, T = 2, 512
        q = torch.randn(B, attn.n_heads, T, attn.head_dim)
        k = torch.randn(B, attn.n_heads, T, attn.head_dim)
        v = torch.randn(B, attn.n_heads, T, attn.head_dim)

        # Compute both versions
        with torch.no_grad():
            chunked_output, chunked_fhn = attn._chunked_fhn_attention(q, k, v, chunk_size=256)
            reference_output, reference_fhn = _non_chunked_fhn_attention(attn, q, k, v)

        # Should be numerically equivalent
        assert torch.allclose(chunked_output, reference_output, atol=1e-5, rtol=1e-5), \
            f"Max diff: {(chunked_output - reference_output).abs().max()}"

    def test_numerical_equivalence_chunk512(self):
        """Chunked (chunk_size=512) should match non-chunked output."""
        attn = FHNAttention(
            embed_dim=128,
            n_heads=4,
            n_fhn_steps=2,
            chunk_size=512,
            dropout=0.0,
        )
        attn.eval()

        # Test with T=1024 (exactly 2 chunks)
        B, T = 2, 1024
        q = torch.randn(B, attn.n_heads, T, attn.head_dim)
        k = torch.randn(B, attn.n_heads, T, attn.head_dim)
        v = torch.randn(B, attn.n_heads, T, attn.head_dim)

        # Compute both versions
        with torch.no_grad():
            chunked_output, chunked_fhn = attn._chunked_fhn_attention(q, k, v, chunk_size=512)
            reference_output, reference_fhn = _non_chunked_fhn_attention(attn, q, k, v)

        # Should be numerically equivalent
        assert torch.allclose(chunked_output, reference_output, atol=1e-5, rtol=1e-5), \
            f"Max diff: {(chunked_output - reference_output).abs().max()}"

    def test_numerical_equivalence_chunk1024(self):
        """Chunked (chunk_size=1024) should match non-chunked output."""
        attn = FHNAttention(
            embed_dim=128,
            n_heads=4,
            n_fhn_steps=2,
            chunk_size=1024,
            dropout=0.0,
        )
        attn.eval()

        # Test with T=2048 (exactly 2 chunks)
        B, T = 2, 2048
        q = torch.randn(B, attn.n_heads, T, attn.head_dim)
        k = torch.randn(B, attn.n_heads, T, attn.head_dim)
        v = torch.randn(B, attn.n_heads, T, attn.head_dim)

        # Compute both versions
        with torch.no_grad():
            chunked_output, chunked_fhn = attn._chunked_fhn_attention(q, k, v, chunk_size=1024)
            reference_output, reference_fhn = _non_chunked_fhn_attention(attn, q, k, v)

        # Should be numerically equivalent
        assert torch.allclose(chunked_output, reference_output, atol=1e-5, rtol=1e-5), \
            f"Max diff: {(chunked_output - reference_output).abs().max()}"

    def test_edge_case_t_not_divisible_255(self):
        """Test edge case where T=255 is not divisible by chunk_size=256."""
        attn = FHNAttention(
            embed_dim=128,
            n_heads=4,
            n_fhn_steps=2,
            chunk_size=256,
            dropout=0.0,
        )
        attn.eval()

        # T < chunk_size: single partial chunk
        B, T = 2, 255
        q = torch.randn(B, attn.n_heads, T, attn.head_dim)
        k = torch.randn(B, attn.n_heads, T, attn.head_dim)
        v = torch.randn(B, attn.n_heads, T, attn.head_dim)

        # Compute both versions
        with torch.no_grad():
            chunked_output, chunked_fhn = attn._chunked_fhn_attention(q, k, v, chunk_size=256)
            reference_output, reference_fhn = _non_chunked_fhn_attention(attn, q, k, v)

        # Should be numerically equivalent
        assert torch.allclose(chunked_output, reference_output, atol=1e-5, rtol=1e-5), \
            f"Max diff: {(chunked_output - reference_output).abs().max()}"

    def test_edge_case_t_not_divisible_513(self):
        """Test edge case where T=513 is not divisible by chunk_size=256."""
        attn = FHNAttention(
            embed_dim=128,
            n_heads=4,
            n_fhn_steps=2,
            chunk_size=256,
            dropout=0.0,
        )
        attn.eval()

        # T = 2 * chunk_size + 1: two full chunks + one partial
        B, T = 2, 513
        q = torch.randn(B, attn.n_heads, T, attn.head_dim)
        k = torch.randn(B, attn.n_heads, T, attn.head_dim)
        v = torch.randn(B, attn.n_heads, T, attn.head_dim)

        # Compute both versions
        with torch.no_grad():
            chunked_output, chunked_fhn = attn._chunked_fhn_attention(q, k, v, chunk_size=256)
            reference_output, reference_fhn = _non_chunked_fhn_attention(attn, q, k, v)

        # Should be numerically equivalent
        assert torch.allclose(chunked_output, reference_output, atol=1e-5, rtol=1e-5), \
            f"Max diff: {(chunked_output - reference_output).abs().max()}"

    def test_edge_case_single_timestep(self):
        """Test edge case with T=1."""
        attn = FHNAttention(
            embed_dim=128,
            n_heads=4,
            n_fhn_steps=2,
            chunk_size=256,
            dropout=0.0,
        )
        attn.eval()

        B, T = 2, 1
        q = torch.randn(B, attn.n_heads, T, attn.head_dim)
        k = torch.randn(B, attn.n_heads, T, attn.head_dim)
        v = torch.randn(B, attn.n_heads, T, attn.head_dim)

        # Compute both versions
        with torch.no_grad():
            chunked_output, chunked_fhn = attn._chunked_fhn_attention(q, k, v, chunk_size=256)
            reference_output, reference_fhn = _non_chunked_fhn_attention(attn, q, k, v)

        # Should be numerically equivalent
        assert torch.allclose(chunked_output, reference_output, atol=1e-5, rtol=1e-5), \
            f"Max diff: {(chunked_output - reference_output).abs().max()}"

    def test_causality_preserved(self):
        """Verify causal property: output at time t only depends on times <= t."""
        attn = FHNAttention(
            embed_dim=128,
            n_heads=4,
            n_fhn_steps=2,
            chunk_size=256,
            dropout=0.0,
        )
        attn.eval()

        B, T = 2, 512
        q = torch.randn(B, attn.n_heads, T, attn.head_dim)
        k = torch.randn(B, attn.n_heads, T, attn.head_dim)
        v = torch.randn(B, attn.n_heads, T, attn.head_dim)

        # Compute full output
        with torch.no_grad():
            full_output, _ = attn._chunked_fhn_attention(q, k, v, chunk_size=256)

        # Test at chunk boundaries and within chunks
        for t in [0, 127, 255, 256, 383, 511]:
            # Truncate inputs at time t
            q_trunc = q[:, :, :t+1, :]
            k_trunc = k[:, :, :t+1, :]
            v_trunc = v[:, :, :t+1, :]

            # Compute output with truncated inputs
            with torch.no_grad():
                trunc_output, _ = attn._chunked_fhn_attention(q_trunc, k_trunc, v_trunc, chunk_size=256)

            # Output at time t should match
            assert torch.allclose(full_output[:, :, t, :], trunc_output[:, :, t, :], atol=1e-5, rtol=1e-5), \
                f"Causality violated at t={t}, max diff: {(full_output[:, :, t, :] - trunc_output[:, :, t, :]).abs().max()}"

    def test_full_forward_pass_equivalence(self):
        """Test that full forward pass produces consistent outputs."""
        # Create two attention modules with different chunk sizes
        attn_small_chunk = FHNAttention(
            embed_dim=128,
            n_heads=4,
            n_fhn_steps=2,
            chunk_size=256,
            dropout=0.0,
            pos_emb_type="learned",
        )
        attn_large_chunk = FHNAttention(
            embed_dim=128,
            n_heads=4,
            n_fhn_steps=2,
            chunk_size=512,
            dropout=0.0,
            pos_emb_type="learned",
        )

        # Copy weights to ensure identical parameters
        attn_large_chunk.load_state_dict(attn_small_chunk.state_dict(), strict=False)

        # Test inputs
        B, T, C = 2, 1024, 128
        x = torch.randn(B, T, C)
        spectral_basis = torch.randn(B, T, 32)

        # Compute outputs
        with torch.no_grad():
            out_small, info_small = attn_small_chunk(x, spectral_basis)
            out_large, info_large = attn_large_chunk(x, spectral_basis)

        # Outputs should be numerically equivalent
        assert torch.allclose(out_small, out_large, atol=1e-4, rtol=1e-4), \
            f"Forward pass outputs differ, max diff: {(out_small - out_large).abs().max()}"

    def test_gradient_flow_chunked(self):
        """Gradients should flow through chunked implementation."""
        attn = FHNAttention(
            embed_dim=128,
            n_heads=4,
            n_fhn_steps=2,
            chunk_size=256,
            dropout=0.0,
        )

        B, T, C = 2, 512, 128
        x = torch.randn(B, T, C, requires_grad=True)
        spectral_basis = torch.randn(B, T, 32)

        out, _ = attn(x, spectral_basis)
        loss = out.sum()
        loss.backward()

        # Gradients should exist and be finite
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()
        assert x.grad.abs().sum() > 0

    def test_output_shapes_chunked(self):
        """Chunked implementation should produce correct output shapes."""
        attn = FHNAttention(
            embed_dim=128,
            n_heads=4,
            n_fhn_steps=2,
            chunk_size=256,
        )

        for T in [255, 256, 512, 1024, 2048]:
            B, C = 2, 128
            x = torch.randn(B, T, C)
            spectral_basis = torch.randn(B, T, 32)

            out, info = attn(x, spectral_basis)

            assert out.shape == (B, T, C)
            assert 'fhn_state' in info
            assert 'pulse_widths' in info

    def test_various_sequence_lengths(self):
        """Chunked path should handle various sequence lengths."""
        attn = FHNAttention(
            embed_dim=128,
            n_heads=4,
            n_fhn_steps=2,
            chunk_size=512,
        )

        for T in [256, 512, 1024, 2048]:
            B, C = 2, 128
            x = torch.randn(B, T, C)
            spectral_basis = torch.randn(B, T, 32)

            out, info = attn(x, spectral_basis)

            assert out.shape == (B, T, C)
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()

    def test_various_batch_sizes(self):
        """Chunked path should handle various batch sizes."""
        attn = FHNAttention(
            embed_dim=128,
            n_heads=4,
            n_fhn_steps=2,
            chunk_size=256,
        )

        T, C = 512, 128
        for B in [1, 2, 4, 8]:
            x = torch.randn(B, T, C)
            spectral_basis = torch.randn(B, T, 32)

            out, info = attn(x, spectral_basis)

            assert out.shape == (B, T, C)

    def test_various_head_counts(self):
        """Chunked path should handle various head configurations."""
        T, C = 512, 128
        B = 2

        for n_heads in [2, 4, 8]:
            attn = FHNAttention(
                embed_dim=128,
                n_heads=n_heads,
                n_fhn_steps=2,
                chunk_size=256,
            )

            x = torch.randn(B, T, C)
            spectral_basis = torch.randn(B, T, 32)

            out, info = attn(x, spectral_basis)

            assert out.shape == (B, T, C)

    def test_fhn_state_computed_chunked(self):
        """FHN state should be computed in chunked mode when n_fhn_steps > 0."""
        attn = FHNAttention(
            embed_dim=128,
            n_heads=4,
            n_fhn_steps=2,
            chunk_size=256,
        )

        B, T, C = 2, 512, 128
        x = torch.randn(B, T, C)
        spectral_basis = torch.randn(B, T, 32)

        _, info = attn(x, spectral_basis)

        # FHN state should be non-zero
        assert info['fhn_state'] is not None
        assert info['fhn_state'] != 0.0

    def test_no_fhn_steps_uses_flash_path(self):
        """When n_fhn_steps=0, should use Flash Attention fast path."""
        attn = FHNAttention(
            embed_dim=128,
            n_heads=4,
            n_fhn_steps=0,
            chunk_size=256,
            pos_emb_type="learned",  # No ALiBi
        )

        # Even with long sequence, should use flash path (no chunking needed)
        B, T, C = 2, 1024, 128
        x = torch.randn(B, T, C)
        spectral_basis = torch.randn(B, T, 32)

        _, info = attn(x, spectral_basis)

        # FHN state should be zero
        assert float(info['fhn_state']) == 0.0
        # Attention probs not computed in flash path
        assert info['attn_probs'] is None

    def test_chunked_path_triggered_for_long_sequences(self):
        """Chunked path should be used when T > chunk_size."""
        attn = FHNAttention(
            embed_dim=128,
            n_heads=4,
            n_fhn_steps=2,
            chunk_size=256,
        )

        # Short sequence: should use standard path
        B, T_short, C = 2, 200, 128
        x_short = torch.randn(B, T_short, C)
        spectral_basis_short = torch.randn(B, T_short, 32)

        out_short, info_short = attn(x_short, spectral_basis_short)
        assert out_short.shape == (B, T_short, C)

        # Long sequence: should use chunked path
        T_long = 512
        x_long = torch.randn(B, T_long, C)
        spectral_basis_long = torch.randn(B, T_long, 32)

        out_long, info_long = attn(x_long, spectral_basis_long)
        assert out_long.shape == (B, T_long, C)

        # Both should produce valid outputs
        assert not torch.isnan(out_short).any()
        assert not torch.isnan(out_long).any()

    def test_deterministic_with_fixed_seed(self):
        """Chunked path should be deterministic with fixed seed."""
        def run_with_seed(seed=42):
            torch.manual_seed(seed)
            attn = FHNAttention(
                embed_dim=128,
                n_heads=4,
                n_fhn_steps=2,
                chunk_size=256,
                dropout=0.0,
            )
            attn.eval()

            torch.manual_seed(seed + 1000)
            x = torch.randn(2, 512, 128)
            spectral_basis = torch.randn(2, 512, 32)

            with torch.no_grad():
                out, _ = attn(x, spectral_basis)
            return out

        out1 = run_with_seed(42)
        out2 = run_with_seed(42)

        assert torch.allclose(out1, out2, atol=1e-6), \
            "Outputs should be identical with same seed"

    def test_fhn_steps_zero_vs_nonzero(self):
        """Test behavior difference between n_fhn_steps=0 and n_fhn_steps>0."""
        B, T, C = 2, 512, 128
        x = torch.randn(B, T, C)
        spectral_basis = torch.randn(B, T, 32)

        attn_no_fhn = FHNAttention(
            embed_dim=128,
            n_heads=4,
            n_fhn_steps=0,
            chunk_size=256,
            pos_emb_type="learned",
        )

        attn_with_fhn = FHNAttention(
            embed_dim=128,
            n_heads=4,
            n_fhn_steps=2,
            chunk_size=256,
            pos_emb_type="learned",
        )

        # Copy weights
        attn_with_fhn.load_state_dict(attn_no_fhn.state_dict(), strict=False)

        with torch.no_grad():
            out_no_fhn, info_no_fhn = attn_no_fhn(x, spectral_basis)
            out_with_fhn, info_with_fhn = attn_with_fhn(x, spectral_basis)

        # Outputs should differ (FHN modulation changes attention)
        assert not torch.allclose(out_no_fhn, out_with_fhn, atol=1e-3), \
            "Outputs should differ when FHN modulation is applied"

        # FHN state should be zero vs non-zero
        assert float(info_no_fhn['fhn_state']) == 0.0
        assert float(info_with_fhn['fhn_state']) != 0.0

    def test_extreme_sequence_length_4096(self):
        """Test chunked attention with very long sequence (T=4096)."""
        attn = FHNAttention(
            embed_dim=128,
            n_heads=4,
            n_fhn_steps=2,
            chunk_size=512,
        )

        B, T, C = 1, 4096, 128
        x = torch.randn(B, T, C)
        spectral_basis = torch.randn(B, T, 32)

        out, info = attn(x, spectral_basis)

        assert out.shape == (B, T, C)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_different_chunk_sizes_consistency(self):
        """Different chunk sizes should produce similar outputs."""
        B, T, C = 2, 1024, 128
        x = torch.randn(B, T, C)
        spectral_basis = torch.randn(B, T, 32)

        attn_256 = FHNAttention(
            embed_dim=128,
            n_heads=4,
            n_fhn_steps=2,
            chunk_size=256,
            dropout=0.0,
        )

        attn_512 = FHNAttention(
            embed_dim=128,
            n_heads=4,
            n_fhn_steps=2,
            chunk_size=512,
            dropout=0.0,
        )

        # Copy weights
        attn_512.load_state_dict(attn_256.state_dict(), strict=False)

        with torch.no_grad():
            out_256, _ = attn_256(x, spectral_basis)
            out_512, _ = attn_512(x, spectral_basis)

        # Outputs should be similar (within numerical tolerance)
        assert torch.allclose(out_256, out_512, atol=1e-4, rtol=1e-4), \
            f"Outputs with different chunk sizes differ, max diff: {(out_256 - out_512).abs().max()}"
