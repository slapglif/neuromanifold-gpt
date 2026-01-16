# neuromanifold_gpt/tests/test_lazy_position_embeddings.py
"""Tests for lazy initialization of position embeddings in FHNAttention."""
import pytest
import torch
from neuromanifold_gpt.model.attention.fhn import FHNAttention


def test_rope_not_initialized_fast_path():
    """RoPE should NOT be initialized when using fast path with learned/ramanujan embeddings."""
    # Test with learned embeddings - RoPE should never be initialized
    attn = FHNAttention(
        embed_dim=256,
        n_heads=8,
        pos_emb_type='learned',
        n_fhn_steps=0,  # Fast path
        max_seq_len=128
    )

    # After __init__, rope should be None
    assert attn.rope is None, "RoPE should be None after __init__"

    # Create dummy input
    x = torch.randn(2, 32, 256)
    spectral_basis = torch.randn(2, 32, 32)

    # Run forward pass (fast path)
    out, info = attn(x, spectral_basis)

    # RoPE should still be None since pos_emb_type is 'learned'
    assert attn.rope is None, "RoPE should remain None when pos_emb_type='learned'"
    assert out.shape == (2, 32, 256)


def test_alibi_not_initialized_fast_path():
    """ALiBi should NOT be initialized when using fast path with learned/ramanujan embeddings."""
    # Test with learned embeddings - ALiBi should never be initialized
    attn = FHNAttention(
        embed_dim=256,
        n_heads=8,
        pos_emb_type='learned',
        n_fhn_steps=0,  # Fast path
        max_seq_len=128
    )

    # After __init__, alibi should be None
    assert attn.alibi is None, "ALiBi should be None after __init__"

    # Create dummy input
    x = torch.randn(2, 32, 256)
    spectral_basis = torch.randn(2, 32, 32)

    # Run forward pass (fast path)
    out, info = attn(x, spectral_basis)

    # ALiBi should still be None since pos_emb_type is 'learned'
    assert attn.alibi is None, "ALiBi should remain None when pos_emb_type='learned'"
    assert out.shape == (2, 32, 256)


def test_rope_lazy_initialization():
    """RoPE should be initialized lazily on first forward pass when needed."""
    attn = FHNAttention(
        embed_dim=256,
        n_heads=8,
        pos_emb_type='rotary',
        n_fhn_steps=0,  # Fast path (RoPE works with fast path)
        max_seq_len=128
    )

    # After __init__, rope should be None (lazy initialization)
    assert attn.rope is None, "RoPE should be None after __init__ (lazy initialization)"

    # Create dummy input
    x = torch.randn(2, 32, 256)
    spectral_basis = torch.randn(2, 32, 32)

    # Run forward pass - this should trigger lazy initialization
    out, info = attn(x, spectral_basis)

    # RoPE should now be initialized
    assert attn.rope is not None, "RoPE should be initialized after first forward pass"
    assert out.shape == (2, 32, 256)

    # Verify RoPE has expected attributes
    assert hasattr(attn.rope, 'cos_cached')
    assert hasattr(attn.rope, 'sin_cached')


def test_alibi_lazy_initialization():
    """ALiBi should be initialized lazily on first forward pass when needed."""
    attn = FHNAttention(
        embed_dim=256,
        n_heads=8,
        pos_emb_type='alibi',
        n_fhn_steps=0,  # ALiBi forces slow path
        max_seq_len=128
    )

    # After __init__, alibi should be None (lazy initialization)
    assert attn.alibi is None, "ALiBi should be None after __init__ (lazy initialization)"

    # Create dummy input
    x = torch.randn(2, 32, 256)
    spectral_basis = torch.randn(2, 32, 32)

    # Run forward pass - this should trigger lazy initialization
    out, info = attn(x, spectral_basis)

    # ALiBi should now be initialized
    assert attn.alibi is not None, "ALiBi should be initialized after first forward pass"
    assert out.shape == (2, 32, 256)

    # Verify ALiBi has expected attributes
    assert hasattr(attn.alibi, 'slopes')


def test_rope_not_initialized_with_ramanujan():
    """RoPE should NOT be initialized when using Ramanujan embeddings."""
    attn = FHNAttention(
        embed_dim=256,
        n_heads=8,
        pos_emb_type='ramanujan',
        n_fhn_steps=0,
        max_seq_len=128
    )

    # After __init__, rope should be None
    assert attn.rope is None

    # Create dummy input
    x = torch.randn(2, 32, 256)
    spectral_basis = torch.randn(2, 32, 32)

    # Run forward pass
    out, info = attn(x, spectral_basis)

    # RoPE should still be None
    assert attn.rope is None, "RoPE should remain None when pos_emb_type='ramanujan'"
    assert attn.alibi is None, "ALiBi should remain None when pos_emb_type='ramanujan'"


def test_multiple_forward_passes_single_initialization():
    """Position embeddings should only be initialized once across multiple forward passes."""
    attn = FHNAttention(
        embed_dim=256,
        n_heads=8,
        pos_emb_type='rotary',
        n_fhn_steps=0,
        max_seq_len=128
    )

    # After __init__, rope should be None
    assert attn.rope is None

    # Create dummy input
    x = torch.randn(2, 32, 256)
    spectral_basis = torch.randn(2, 32, 32)

    # First forward pass - initializes RoPE
    out1, info1 = attn(x, spectral_basis)
    rope_instance_1 = attn.rope
    assert rope_instance_1 is not None

    # Second forward pass - should reuse same RoPE instance
    out2, info2 = attn(x, spectral_basis)
    rope_instance_2 = attn.rope

    # Should be the exact same instance (not re-initialized)
    assert rope_instance_1 is rope_instance_2, "RoPE should not be re-initialized on subsequent forward passes"


def test_slow_path_with_rope():
    """RoPE should be initialized when using slow path."""
    attn = FHNAttention(
        embed_dim=256,
        n_heads=8,
        pos_emb_type='rotary',
        n_fhn_steps=2,  # Slow path
        max_seq_len=128
    )

    # After __init__, rope should be None
    assert attn.rope is None

    # Create dummy input
    x = torch.randn(2, 32, 256)
    spectral_basis = torch.randn(2, 32, 32)

    # Run forward pass with slow path
    out, info = attn(x, spectral_basis)

    # RoPE should be initialized
    assert attn.rope is not None, "RoPE should be initialized in slow path"
    assert out.shape == (2, 32, 256)


def test_slow_path_with_alibi():
    """ALiBi should be initialized when using slow path."""
    attn = FHNAttention(
        embed_dim=256,
        n_heads=8,
        pos_emb_type='alibi',
        n_fhn_steps=2,  # Slow path
        max_seq_len=128
    )

    # After __init__, alibi should be None
    assert attn.alibi is None

    # Create dummy input
    x = torch.randn(2, 32, 256)
    spectral_basis = torch.randn(2, 32, 32)

    # Run forward pass with slow path
    out, info = attn(x, spectral_basis)

    # ALiBi should be initialized
    assert attn.alibi is not None, "ALiBi should be initialized in slow path"
    assert out.shape == (2, 32, 256)


def test_no_position_embeddings_initialized_learned():
    """Neither RoPE nor ALiBi should be initialized with learned embeddings."""
    attn = FHNAttention(
        embed_dim=256,
        n_heads=8,
        pos_emb_type='learned',
        n_fhn_steps=0,
        max_seq_len=128
    )

    # After __init__, both should be None
    assert attn.rope is None
    assert attn.alibi is None

    # Create dummy input
    x = torch.randn(2, 32, 256)
    spectral_basis = torch.randn(2, 32, 32)

    # Run forward pass
    out, info = attn(x, spectral_basis)

    # Both should still be None
    assert attn.rope is None, "RoPE should never be initialized with learned embeddings"
    assert attn.alibi is None, "ALiBi should never be initialized with learned embeddings"
    assert out.shape == (2, 32, 256)


def test_rope_initialized_slow_path():
    """RoPE should be initialized lazily when needed in slow path (n_fhn_steps > 0)."""
    attn = FHNAttention(
        embed_dim=256,
        n_heads=8,
        pos_emb_type='rotary',
        n_fhn_steps=2,  # Slow path with FHN modulation
        max_seq_len=128
    )

    # After __init__, rope should be None (lazy initialization)
    assert attn.rope is None, "RoPE should be None after __init__ (lazy initialization)"

    # Create dummy input
    x = torch.randn(2, 32, 256)
    spectral_basis = torch.randn(2, 32, 32)

    # Run forward pass using slow path - should trigger lazy initialization
    out, info = attn(x, spectral_basis)

    # RoPE should now be initialized since pos_emb_type='rotary' and slow path was used
    assert attn.rope is not None, "RoPE should be initialized after first forward pass in slow path"
    assert out.shape == (2, 32, 256)

    # Verify RoPE has expected attributes
    assert hasattr(attn.rope, 'cos_cached'), "RoPE should have cos_cached attribute"
    assert hasattr(attn.rope, 'sin_cached'), "RoPE should have sin_cached attribute"


def test_alibi_initialized_slow_path():
    """ALiBi should be initialized lazily when needed in slow path."""
    attn = FHNAttention(
        embed_dim=256,
        n_heads=8,
        pos_emb_type='alibi',
        n_fhn_steps=2,  # Slow path
        max_seq_len=128
    )

    # After __init__, alibi should be None (lazy initialization)
    assert attn.alibi is None, "ALiBi should be None after __init__ (lazy initialization)"

    # Create dummy input
    x = torch.randn(2, 32, 256)
    spectral_basis = torch.randn(2, 32, 32)

    # Run forward pass using slow path - should trigger lazy initialization
    out, info = attn(x, spectral_basis)

    # ALiBi should now be initialized since pos_emb_type='alibi' and slow path was used
    assert attn.alibi is not None, "ALiBi should be initialized after first forward pass in slow path"
    assert out.shape == (2, 32, 256)

    # Verify ALiBi has expected attributes
    assert hasattr(attn.alibi, 'slopes'), "ALiBi should have slopes attribute"


def test_memory_savings_fast_path():
    """Verify memory savings by not initializing RoPE/ALiBi in fast path across multiple layers.

    In a 6-layer model with T=1024, lazy initialization saves:
    - RoPE: ~16KB per layer × 6 layers = ~96KB
    - ALiBi: ~4MB for bias matrices (when pos_emb_type='alibi')

    For fast path (n_fhn_steps=0) with learned/ramanujan embeddings,
    none of these position embedding modules should be allocated.
    """
    n_layers = 6
    embed_dim = 256
    n_heads = 8
    max_seq_len = 1024
    seq_len = 128  # Actual sequence length for test

    # Create 6 attention layers (simulating a 6-layer model)
    # Use 'learned' position embeddings (fast path, no RoPE/ALiBi needed)
    layers = [
        FHNAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            pos_emb_type='learned',
            n_fhn_steps=0,  # Fast path
            max_seq_len=max_seq_len
        )
        for _ in range(n_layers)
    ]

    # Verify no position embeddings allocated after init
    for i, layer in enumerate(layers):
        assert layer.rope is None, f"Layer {i}: RoPE should be None after __init__"
        assert layer.alibi is None, f"Layer {i}: ALiBi should be None after __init__"

    # Create dummy input
    x = torch.randn(2, seq_len, embed_dim)
    spectral_basis = torch.randn(2, seq_len, seq_len)

    # Run forward pass through all layers
    for i, layer in enumerate(layers):
        x, info = layer(x, spectral_basis)
        # Verify position embeddings still not initialized (fast path, learned embeddings)
        assert layer.rope is None, f"Layer {i}: RoPE should remain None with learned embeddings"
        assert layer.alibi is None, f"Layer {i}: ALiBi should remain None with learned embeddings"

    # Verify output shape is correct
    assert x.shape == (2, seq_len, embed_dim)

    # Test with ramanujan embeddings (also should not initialize RoPE/ALiBi)
    layers_ramanujan = [
        FHNAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            pos_emb_type='ramanujan',
            n_fhn_steps=0,  # Fast path
            max_seq_len=max_seq_len
        )
        for _ in range(n_layers)
    ]

    x = torch.randn(2, seq_len, embed_dim)
    for i, layer in enumerate(layers_ramanujan):
        x, info = layer(x, spectral_basis)
        assert layer.rope is None, f"Layer {i}: RoPE should remain None with ramanujan embeddings"
        assert layer.alibi is None, f"Layer {i}: ALiBi should remain None with ramanujan embeddings"

    # SUCCESS: No memory allocated for unused position embeddings across all layers!
    # This demonstrates the memory savings: 6 layers × (RoPE + ALiBi) = significant savings
    # when using fast path with learned/ramanujan embeddings.
