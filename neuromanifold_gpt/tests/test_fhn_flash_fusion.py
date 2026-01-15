# neuromanifold_gpt/tests/test_fhn_flash_fusion.py
"""Tests comparing old vs new FHN modulation with Flash Attention fusion."""
import pytest
import torch
from neuromanifold_gpt.model.attention.fhn import FHNAttention


def test_output_shapes_match():
    """Old and new paths should produce matching output shapes."""
    embed_dim = 384
    n_heads = 8
    batch_size = 2
    seq_len = 20

    # Old path: manual attention with FHN weight modulation
    attn_old = FHNAttention(
        embed_dim=embed_dim,
        n_heads=n_heads,
        n_fhn_steps=2,
        use_flash_fhn_fusion=False,
    )

    # New path: Flash Attention with FHN output modulation
    attn_new = FHNAttention(
        embed_dim=embed_dim,
        n_heads=n_heads,
        n_fhn_steps=2,
        use_flash_fhn_fusion=True,
    )

    x = torch.randn(batch_size, seq_len, embed_dim)
    spectral_basis = torch.randn(batch_size, seq_len, 32)

    out_old, info_old = attn_old(x, spectral_basis)
    out_new, info_new = attn_new(x, spectral_basis)

    assert out_old.shape == out_new.shape == (batch_size, seq_len, embed_dim)
    assert 'pulse_widths' in info_old and 'pulse_widths' in info_new
    assert 'fhn_state' in info_old and 'fhn_state' in info_new


def test_gradient_flow_old_path():
    """Gradients should flow through old manual attention path."""
    attn = FHNAttention(
        embed_dim=384,
        n_heads=8,
        n_fhn_steps=2,
        use_flash_fhn_fusion=False,
    )

    x = torch.randn(2, 10, 384, requires_grad=True)
    spectral_basis = torch.randn(2, 10, 32)

    out, _ = attn(x, spectral_basis)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_gradient_flow_new_path():
    """Gradients should flow through new Flash Attention fusion path."""
    attn = FHNAttention(
        embed_dim=384,
        n_heads=8,
        n_fhn_steps=2,
        use_flash_fhn_fusion=True,
    )

    x = torch.randn(2, 10, 384, requires_grad=True)
    spectral_basis = torch.randn(2, 10, 32)

    out, _ = attn(x, spectral_basis)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_fhn_state_computed_both_paths():
    """Both paths should compute FHN state when n_fhn_steps > 0."""
    x = torch.randn(2, 10, 384)
    spectral_basis = torch.randn(2, 10, 32)

    attn_old = FHNAttention(embed_dim=384, n_heads=8, n_fhn_steps=2, use_flash_fhn_fusion=False)
    attn_new = FHNAttention(embed_dim=384, n_heads=8, n_fhn_steps=2, use_flash_fhn_fusion=True)

    _, info_old = attn_old(x, spectral_basis)
    _, info_new = attn_new(x, spectral_basis)

    # Both should have FHN state
    assert info_old['fhn_state'] is not None
    assert info_new['fhn_state'] is not None

    # FHN state should be non-zero (indicates FHN dynamics ran)
    assert info_old['fhn_state'].abs() > 0
    assert info_new['fhn_state'].abs() > 0


def test_no_fhn_steps_matches_flash_attention():
    """When n_fhn_steps=0, both should use Flash Attention path."""
    x = torch.randn(2, 10, 384)
    spectral_basis = torch.randn(2, 10, 32)

    attn_old = FHNAttention(embed_dim=384, n_heads=8, n_fhn_steps=0, use_flash_fhn_fusion=False)
    attn_new = FHNAttention(embed_dim=384, n_heads=8, n_fhn_steps=0, use_flash_fhn_fusion=True)

    _, info_old = attn_old(x, spectral_basis)
    _, info_new = attn_new(x, spectral_basis)

    # With n_fhn_steps=0, FHN state should be zero
    assert info_old['fhn_state'] == 0.0
    assert info_new['fhn_state'] == 0.0

    # Attention probs should not be computed (Flash Attention)
    assert info_old['attn_probs'] is None
    assert info_new['attn_probs'] is None


def test_output_stats_computed_both_paths():
    """Both paths should compute output statistics."""
    x = torch.randn(2, 10, 384)
    spectral_basis = torch.randn(2, 10, 32)

    attn_old = FHNAttention(embed_dim=384, n_heads=8, n_fhn_steps=2, use_flash_fhn_fusion=False)
    attn_new = FHNAttention(embed_dim=384, n_heads=8, n_fhn_steps=2, use_flash_fhn_fusion=True)

    _, info_old = attn_old(x, spectral_basis)
    _, info_new = attn_new(x, spectral_basis)

    # Both should have output_stats
    assert 'output_stats' in info_old
    assert 'output_stats' in info_new

    # Check required keys
    for key in ['variance', 'std', 'mean']:
        assert key in info_old['output_stats']
        assert key in info_new['output_stats']
        assert isinstance(info_old['output_stats'][key], float)
        assert isinstance(info_new['output_stats'][key], float)


def test_various_sequence_lengths():
    """Both paths should handle various sequence lengths."""
    embed_dim = 384
    n_heads = 8

    attn_old = FHNAttention(embed_dim=embed_dim, n_heads=n_heads, n_fhn_steps=2, use_flash_fhn_fusion=False)
    attn_new = FHNAttention(embed_dim=embed_dim, n_heads=n_heads, n_fhn_steps=2, use_flash_fhn_fusion=True)

    for seq_len in [5, 10, 20, 50]:
        x = torch.randn(2, seq_len, embed_dim)
        spectral_basis = torch.randn(2, seq_len, 32)

        out_old, _ = attn_old(x, spectral_basis)
        out_new, _ = attn_new(x, spectral_basis)

        assert out_old.shape == (2, seq_len, embed_dim)
        assert out_new.shape == (2, seq_len, embed_dim)


def test_various_batch_sizes():
    """Both paths should handle various batch sizes."""
    embed_dim = 384
    n_heads = 8
    seq_len = 10

    attn_old = FHNAttention(embed_dim=embed_dim, n_heads=n_heads, n_fhn_steps=2, use_flash_fhn_fusion=False)
    attn_new = FHNAttention(embed_dim=embed_dim, n_heads=n_heads, n_fhn_steps=2, use_flash_fhn_fusion=True)

    for batch_size in [1, 2, 4, 8]:
        x = torch.randn(batch_size, seq_len, embed_dim)
        spectral_basis = torch.randn(batch_size, seq_len, 32)

        out_old, _ = attn_old(x, spectral_basis)
        out_new, _ = attn_new(x, spectral_basis)

        assert out_old.shape == (batch_size, seq_len, embed_dim)
        assert out_new.shape == (batch_size, seq_len, embed_dim)


def test_deterministic_with_fixed_seed():
    """Both paths should be deterministic with fixed seed."""
    embed_dim = 384
    n_heads = 8

    def run_with_seed(use_fusion, seed=42):
        torch.manual_seed(seed)
        attn = FHNAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_fhn_steps=2,
            use_flash_fhn_fusion=use_fusion,
        )
        torch.manual_seed(seed + 1000)
        x = torch.randn(2, 10, embed_dim)
        spectral_basis = torch.randn(2, 10, 32)
        out, _ = attn(x, spectral_basis)
        return out

    # Test old path determinism
    out_old_1 = run_with_seed(False, seed=42)
    out_old_2 = run_with_seed(False, seed=42)
    assert torch.allclose(out_old_1, out_old_2, atol=1e-6)

    # Test new path determinism
    out_new_1 = run_with_seed(True, seed=42)
    out_new_2 = run_with_seed(True, seed=42)
    assert torch.allclose(out_new_1, out_new_2, atol=1e-6)


def test_outputs_are_different_due_to_different_proxies():
    """Old and new paths use different FHN stimulus proxies, so outputs differ."""
    torch.manual_seed(42)

    # Use same initialization for fair comparison
    attn_old = FHNAttention(embed_dim=384, n_heads=8, n_fhn_steps=2, use_flash_fhn_fusion=False)

    # Copy weights to new model
    attn_new = FHNAttention(embed_dim=384, n_heads=8, n_fhn_steps=2, use_flash_fhn_fusion=True)
    attn_new.load_state_dict(attn_old.state_dict())

    torch.manual_seed(100)
    x = torch.randn(2, 10, 384)
    spectral_basis = torch.randn(2, 10, 32)

    out_old, _ = attn_old(x, spectral_basis)
    out_new, _ = attn_new(x, spectral_basis)

    # Outputs should be different (different FHN stimulus proxies)
    # Old uses attention energy, new uses output variance
    assert not torch.allclose(out_old, out_new, atol=1e-3)

    # But they should be in similar magnitude range
    assert (out_old.abs().mean() - out_new.abs().mean()).abs() < out_old.abs().mean()


def test_pulse_widths_positive_both_paths():
    """Pulse widths should be positive in both paths."""
    x = torch.randn(2, 10, 384)
    spectral_basis = torch.randn(2, 10, 32)

    attn_old = FHNAttention(embed_dim=384, n_heads=8, n_fhn_steps=2, use_flash_fhn_fusion=False)
    attn_new = FHNAttention(embed_dim=384, n_heads=8, n_fhn_steps=2, use_flash_fhn_fusion=True)

    _, info_old = attn_old(x, spectral_basis)
    _, info_new = attn_new(x, spectral_basis)

    assert (info_old['pulse_widths'] > 0).all()
    assert (info_new['pulse_widths'] > 0).all()


def test_backward_compatibility_flag_default():
    """Default should be use_flash_fhn_fusion=True after migration to Flash fusion."""
    attn = FHNAttention(embed_dim=384, n_heads=8, n_fhn_steps=2)
    assert attn.use_flash_fhn_fusion == True


def test_new_path_attn_probs_is_none():
    """New Flash Attention path should not compute explicit attention probs."""
    x = torch.randn(2, 10, 384)
    spectral_basis = torch.randn(2, 10, 32)

    attn_new = FHNAttention(embed_dim=384, n_heads=8, n_fhn_steps=2, use_flash_fhn_fusion=True)
    _, info = attn_new(x, spectral_basis)

    # Flash Attention doesn't return attention probabilities
    assert info['attn_probs'] is None


def test_old_path_attn_probs_exists():
    """Old manual attention path should compute explicit attention probs."""
    x = torch.randn(2, 10, 384)
    spectral_basis = torch.randn(2, 10, 32)

    attn_old = FHNAttention(embed_dim=384, n_heads=8, n_fhn_steps=2, use_flash_fhn_fusion=False)
    _, info = attn_old(x, spectral_basis)

    # Manual attention computes and returns attention probabilities
    assert info['attn_probs'] is not None
    assert info['attn_probs'].shape == (2, 8, 10, 10)  # (B, H, T, T)
