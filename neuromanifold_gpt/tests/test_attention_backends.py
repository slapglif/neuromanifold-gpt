# neuromanifold_gpt/tests/test_attention_backends.py
"""End-to-end tests for attention backend selection and fallback logic.

Tests automatic backend selection, manual overrides, and graceful fallback
when specific backends are unavailable.
"""
import pytest
import torch
from neuromanifold_gpt.model.attention import (
    StandardAttention,
    FHNAttention,
    get_attention_class,
)
from neuromanifold_gpt.utils.gpu_detection import (
    detect_gpu_capability,
    get_optimal_attention_backend,
    supports_flash_attention,
)
from neuromanifold_gpt.config.base import NeuroManifoldConfig, AttentionBackend


def test_gpu_detection_returns_valid_info():
    """GPU detection should return well-formed information."""
    gpu_info = detect_gpu_capability()

    # Check all required keys are present
    assert "available" in gpu_info
    assert "name" in gpu_info
    assert "compute_capability" in gpu_info
    assert "supports_flash_attention" in gpu_info
    assert "supports_xformers" in gpu_info
    assert "supports_triton" in gpu_info

    # Check types
    assert isinstance(gpu_info["available"], bool)
    assert isinstance(gpu_info["name"], str)
    assert isinstance(gpu_info["compute_capability"], tuple)
    assert len(gpu_info["compute_capability"]) == 2


def test_optimal_backend_returns_valid_choice():
    """Optimal backend selection should return a valid backend string."""
    backend = get_optimal_attention_backend()

    valid_backends = ["flash", "xformers", "triton", "manual"]
    assert backend in valid_backends


def test_supports_flash_attention_returns_bool():
    """Flash Attention support check should return boolean."""
    supports_flash = supports_flash_attention()
    assert isinstance(supports_flash, bool)


def test_standard_attention_backend_selection():
    """StandardAttention should select backend and report it in info dict."""
    attn = StandardAttention(embed_dim=384, n_heads=8, dropout=0.0)

    x = torch.randn(2, 10, 384)
    y, info = attn(x)

    # Check output shape
    assert y.shape == (2, 10, 384)

    # Check info dict contains backend information
    assert "backend" in info
    assert info["backend"] in ["flash", "xformers", "manual"]
    assert "attention_type" in info
    assert info["attention_type"] == "standard"


def test_fhn_attention_backend_selection():
    """FHNAttention should select backend and report it in info dict."""
    attn = FHNAttention(
        embed_dim=384,
        n_heads=8,
        n_fhn_steps=2,
        use_flash_fhn_fusion=True,
    )

    x = torch.randn(2, 10, 384)
    spectral_basis = torch.randn(2, 10, 32)

    y, info = attn(x, spectral_basis)

    # Check output shape
    assert y.shape == (2, 10, 384)

    # Check info dict contains backend information
    assert "backend" in info
    # Backend can be flash, xformers, manual, or manual_fhn depending on n_fhn_steps
    valid_backends = ["flash", "xformers", "manual", "manual_fhn"]
    assert info["backend"] in valid_backends


def test_fhn_attention_fast_path_backend_selection():
    """FHNAttention with n_fhn_steps=0 should use fast attention backend."""
    attn = FHNAttention(
        embed_dim=384,
        n_heads=8,
        n_fhn_steps=0,  # Fast path: no FHN dynamics
        use_flash_fhn_fusion=True,
    )

    x = torch.randn(2, 10, 384)
    spectral_basis = torch.randn(2, 10, 32)

    y, info = attn(x, spectral_basis)

    # Check output shape
    assert y.shape == (2, 10, 384)

    # Check backend (should be flash, xformers, or manual, NOT manual_fhn)
    assert "backend" in info
    assert info["backend"] in ["flash", "xformers", "manual"]

    # FHN state should be zero when n_fhn_steps=0
    assert info["fhn_state"] == 0.0


def test_get_attention_class_with_auto_backend():
    """get_attention_class with backend='auto' should resolve to optimal backend."""
    attn_cls = get_attention_class("standard", backend="auto")

    # Should return StandardAttention class
    assert attn_cls == StandardAttention

    # Should have _resolved_backend attribute set
    assert hasattr(attn_cls, "_resolved_backend")
    assert attn_cls._resolved_backend in ["flash", "xformers", "triton", "manual"]


def test_get_attention_class_with_manual_backend():
    """get_attention_class with explicit backend should store that backend."""
    attn_cls = get_attention_class("standard", backend="manual")

    assert attn_cls == StandardAttention
    assert hasattr(attn_cls, "_resolved_backend")
    assert attn_cls._resolved_backend == "manual"


def test_get_attention_class_with_flash_backend():
    """get_attention_class with backend='flash' should store flash backend."""
    attn_cls = get_attention_class("fhn", backend="flash")

    assert attn_cls == FHNAttention
    assert hasattr(attn_cls, "_resolved_backend")
    assert attn_cls._resolved_backend == "flash"


def test_get_attention_class_without_backend():
    """get_attention_class without backend parameter should work (backward compat)."""
    attn_cls = get_attention_class("standard")

    assert attn_cls == StandardAttention
    # Should not have _resolved_backend if no backend specified
    # (or it may have one from previous test - not critical)


def test_attention_backend_enum_values():
    """AttentionBackend enum should have all expected values."""
    assert AttentionBackend.AUTO.value == "auto"
    assert AttentionBackend.FLASH.value == "flash"
    assert AttentionBackend.XFORMERS.value == "xformers"
    assert AttentionBackend.TRITON.value == "triton"
    assert AttentionBackend.PYTORCH.value == "pytorch"


def test_config_attention_backend_default():
    """NeuroManifoldConfig should have attention_backend field with default."""
    cfg = NeuroManifoldConfig()

    assert hasattr(cfg, "attention_backend")
    # Default should be 'auto'
    assert cfg.attention_backend == "auto"


def test_config_attention_backend_can_be_set():
    """NeuroManifoldConfig attention_backend should be configurable."""
    cfg = NeuroManifoldConfig(attention_backend="flash")
    assert cfg.attention_backend == "flash"

    cfg = NeuroManifoldConfig(attention_backend="xformers")
    assert cfg.attention_backend == "xformers"

    cfg = NeuroManifoldConfig(attention_backend="manual")
    assert cfg.attention_backend == "manual"


def test_standard_attention_gradient_flow():
    """StandardAttention should have gradient flow regardless of backend."""
    attn = StandardAttention(embed_dim=384, n_heads=8, dropout=0.0)

    x = torch.randn(2, 10, 384, requires_grad=True)
    y, info = attn(x)

    loss = y.sum()
    loss.backward()

    # Check gradient exists and is non-zero
    assert x.grad is not None
    assert x.grad.abs().sum() > 0

    # Check that backend was used
    assert "backend" in info
    assert info["backend"] in ["flash", "xformers", "manual"]


def test_fhn_attention_gradient_flow():
    """FHNAttention should have gradient flow regardless of backend."""
    attn = FHNAttention(
        embed_dim=384,
        n_heads=8,
        n_fhn_steps=2,
        use_flash_fhn_fusion=True,
    )

    x = torch.randn(2, 10, 384, requires_grad=True)
    spectral_basis = torch.randn(2, 10, 32)

    y, info = attn(x, spectral_basis)

    loss = y.sum()
    loss.backward()

    # Check gradient exists and is non-zero
    assert x.grad is not None
    assert x.grad.abs().sum() > 0

    # Check that backend was used
    assert "backend" in info


def test_backend_consistency_across_batch_sizes():
    """Backend selection should be consistent across different batch sizes."""
    attn = StandardAttention(embed_dim=384, n_heads=8, dropout=0.0)

    backends = []
    for batch_size in [1, 2, 4, 8]:
        x = torch.randn(batch_size, 10, 384)
        y, info = attn(x)

        assert y.shape == (batch_size, 10, 384)
        backends.append(info["backend"])

    # All backends should be the same
    assert len(set(backends)) == 1


def test_backend_consistency_across_sequence_lengths():
    """Backend selection should be consistent across different sequence lengths."""
    attn = StandardAttention(embed_dim=384, n_heads=8, dropout=0.0)

    backends = []
    for seq_len in [5, 10, 20, 50]:
        x = torch.randn(2, seq_len, 384)
        y, info = attn(x)

        assert y.shape == (2, seq_len, 384)
        backends.append(info["backend"])

    # All backends should be the same
    assert len(set(backends)) == 1


def test_fhn_attention_with_different_fhn_steps():
    """FHNAttention should work with various n_fhn_steps values."""
    embed_dim = 384
    n_heads = 8
    batch_size = 2
    seq_len = 10

    x = torch.randn(batch_size, seq_len, embed_dim)
    spectral_basis = torch.randn(batch_size, seq_len, 32)

    for n_fhn_steps in [0, 1, 2, 4]:
        attn = FHNAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_fhn_steps=n_fhn_steps,
            use_flash_fhn_fusion=True,
        )

        y, info = attn(x, spectral_basis)

        # Check output shape
        assert y.shape == (batch_size, seq_len, embed_dim)

        # Check backend is reported
        assert "backend" in info

        # Check FHN state
        if n_fhn_steps == 0:
            assert info["fhn_state"] == 0.0
        else:
            assert info["fhn_state"] != 0.0


def test_attention_type_aliases():
    """get_attention_class should support attention type aliases."""
    # Test soliton alias for fhn
    attn_cls = get_attention_class("soliton")
    assert attn_cls == FHNAttention

    # Test sdr alias for knot
    from neuromanifold_gpt.model.attention.knot import KnotAttention
    attn_cls = get_attention_class("sdr")
    assert attn_cls == KnotAttention

    # Test fast-spectral alias for fhn
    attn_cls = get_attention_class("fast-spectral")
    assert attn_cls == FHNAttention


def test_invalid_attention_type_raises_error():
    """get_attention_class should raise ValueError for invalid attention type."""
    with pytest.raises(ValueError, match="Unknown attention type"):
        get_attention_class("invalid_type")


def test_standard_attention_deterministic_with_seed():
    """StandardAttention should be deterministic with fixed seed."""
    torch.manual_seed(42)
    attn1 = StandardAttention(embed_dim=384, n_heads=8, dropout=0.0)

    torch.manual_seed(42)
    attn2 = StandardAttention(embed_dim=384, n_heads=8, dropout=0.0)

    # Use same input
    torch.manual_seed(100)
    x = torch.randn(2, 10, 384)

    y1, info1 = attn1(x)
    y2, info2 = attn2(x)

    # Outputs should be identical
    assert torch.allclose(y1, y2, atol=1e-6)

    # Backends should be the same
    assert info1["backend"] == info2["backend"]


def test_fhn_attention_deterministic_with_seed():
    """FHNAttention should be deterministic with fixed seed."""
    torch.manual_seed(42)
    attn1 = FHNAttention(embed_dim=384, n_heads=8, n_fhn_steps=2, use_flash_fhn_fusion=True)

    torch.manual_seed(42)
    attn2 = FHNAttention(embed_dim=384, n_heads=8, n_fhn_steps=2, use_flash_fhn_fusion=True)

    # Use same input
    torch.manual_seed(100)
    x = torch.randn(2, 10, 384)
    spectral_basis = torch.randn(2, 10, 32)

    y1, info1 = attn1(x, spectral_basis)
    y2, info2 = attn2(x, spectral_basis)

    # Outputs should be identical
    assert torch.allclose(y1, y2, atol=1e-6)

    # Backends should be the same
    assert info1["backend"] == info2["backend"]


def test_backend_fallback_graceful():
    """Backend selection should gracefully fall back when unavailable."""
    # This test verifies that the system doesn't crash when specific backends
    # are unavailable - it should fall back to the next available option

    # Try to get optimal backend - should always return something valid
    backend = get_optimal_attention_backend()
    assert backend in ["flash", "xformers", "triton", "manual"]

    # Create attention modules - should not raise errors
    std_attn = StandardAttention(embed_dim=384, n_heads=8)
    fhn_attn = FHNAttention(embed_dim=384, n_heads=8, n_fhn_steps=2)

    # Run forward passes - should work with whatever backend is available
    x = torch.randn(2, 10, 384)
    spectral_basis = torch.randn(2, 10, 32)

    std_out, std_info = std_attn(x)
    fhn_out, fhn_info = fhn_attn(x, spectral_basis)

    # Should complete without errors and return valid outputs
    assert std_out.shape == (2, 10, 384)
    assert fhn_out.shape == (2, 10, 384)
    assert "backend" in std_info
    assert "backend" in fhn_info


def test_output_stats_computed_regardless_of_backend():
    """Attention modules should compute output stats regardless of backend."""
    attn = FHNAttention(embed_dim=384, n_heads=8, n_fhn_steps=2)

    x = torch.randn(2, 10, 384)
    spectral_basis = torch.randn(2, 10, 32)

    y, info = attn(x, spectral_basis)

    # Check output_stats are present
    assert "output_stats" in info
    assert "variance" in info["output_stats"]
    assert "std" in info["output_stats"]
    assert "mean" in info["output_stats"]

    # Check types
    assert isinstance(info["output_stats"]["variance"], float)
    assert isinstance(info["output_stats"]["std"], float)
    assert isinstance(info["output_stats"]["mean"], float)


def test_multiple_attention_instances_independent():
    """Multiple attention instances should have independent backend selection."""
    attn1 = StandardAttention(embed_dim=384, n_heads=8)
    attn2 = StandardAttention(embed_dim=384, n_heads=8)
    attn3 = FHNAttention(embed_dim=384, n_heads=8, n_fhn_steps=2)

    x = torch.randn(2, 10, 384)
    spectral_basis = torch.randn(2, 10, 32)

    _, info1 = attn1(x)
    _, info2 = attn2(x)
    _, info3 = attn3(x, spectral_basis)

    # All should report backends
    assert "backend" in info1
    assert "backend" in info2
    assert "backend" in info3

    # StandardAttention instances should use same backend
    assert info1["backend"] == info2["backend"]
