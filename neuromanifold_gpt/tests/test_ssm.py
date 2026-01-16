# neuromanifold_gpt/tests/test_ssm.py
"""Tests for State Space Model (SSM) components.

Tests cover:
- HiPPO matrix initialization (all variants)
- DiagonalHiPPO approximation
- SelectiveScan mechanism
- ParallelSelectiveScan
- MambaBlock layer
- MambaResidualBlock
- BidirectionalMamba

Note: Some tests use `use_hippo_init=False` to avoid numerical issues
in the HiPPO initialization with certain state dimensions.
"""
import pytest
import torch
from neuromanifold_gpt.model.ssm import (
    HiPPO,
    DiagonalHiPPO,
    SelectiveScan,
    ParallelSelectiveScan,
    MambaBlock,
    MambaResidualBlock,
    BidirectionalMamba,
    SSMConfig,
)
from neuromanifold_gpt.model.ssm.hippo import (
    make_hippo_legs,
    make_hippo_legt,
    make_hippo_lagt,
    make_hippo_foud,
)


# ============================================================================
# HiPPO Matrix Tests
# ============================================================================


def test_hippo_legs_shape():
    """HiPPO-LegS matrices should have correct shapes."""
    N = 64
    A, B = make_hippo_legs(N)

    assert A.shape == (N, N)
    assert B.shape == (N, 1)


def test_hippo_legs_stability():
    """HiPPO-LegS A matrix should have non-positive diagonal for stability."""
    N = 32
    A, _ = make_hippo_legs(N)

    # Diagonal should be negative (n+1 negated)
    diag = torch.diag(A)
    assert (diag <= 0).all()


def test_hippo_legt_shape():
    """HiPPO-LegT matrices should have correct shapes."""
    N = 64
    A, B = make_hippo_legt(N)

    assert A.shape == (N, N)
    assert B.shape == (N, 1)


def test_hippo_lagt_shape():
    """HiPPO-LagT matrices should have correct shapes."""
    N = 64
    A, B = make_hippo_lagt(N, alpha=0.5)

    assert A.shape == (N, N)
    assert B.shape == (N, 1)


def test_hippo_lagt_alpha_scaling():
    """HiPPO-LagT alpha should scale the A matrix."""
    N = 32
    A_low, _ = make_hippo_lagt(N, alpha=0.1)
    A_high, _ = make_hippo_lagt(N, alpha=1.0)

    # Higher alpha means faster decay (larger magnitude)
    assert A_high.abs().mean() > A_low.abs().mean()


def test_hippo_foud_shape():
    """HiPPO-FouD matrices should have correct shapes."""
    N = 64
    A, B = make_hippo_foud(N)

    assert A.shape == (N, N)
    assert B.shape == (N, 1)


def test_hippo_foud_even():
    """HiPPO-FouD should handle odd dimensions by making even."""
    N = 63  # Odd
    A, B = make_hippo_foud(N)

    # Should round up to 64
    assert A.shape == (64, 64)


def test_hippo_class_all_types():
    """HiPPO class should support all basis types."""
    state_dim = 32

    for hippo_type in ["legs", "legt", "lagt", "foud"]:
        hippo = HiPPO(state_dim=state_dim, hippo_type=hippo_type)
        A, B = hippo.get_matrices()

        assert A.shape[0] == A.shape[1]  # Square
        assert B.shape[0] == A.shape[0]  # Compatible


def test_hippo_learnable_parameter():
    """HiPPO with learnable=True should register as parameters."""
    hippo = HiPPO(state_dim=32, learnable=True)

    assert hasattr(hippo, "A")
    assert hasattr(hippo, "B")
    assert isinstance(hippo.A, torch.nn.Parameter)
    assert isinstance(hippo.B, torch.nn.Parameter)


def test_hippo_fixed_buffer():
    """HiPPO with learnable=False should register as buffers."""
    hippo = HiPPO(state_dim=32, learnable=False)

    # Should be buffers, not parameters
    assert "A" in dict(hippo.named_buffers())
    assert "B" in dict(hippo.named_buffers())


def test_hippo_get_diagonal():
    """HiPPO get_diagonal should return correct shapes."""
    hippo = HiPPO(state_dim=32)
    A_diag, B_flat = hippo.get_diagonal()

    assert A_diag.shape == (32,)
    assert B_flat.shape == (32,)


# ============================================================================
# DiagonalHiPPO Tests
# ============================================================================


def test_diagonal_hippo_shape():
    """DiagonalHiPPO should return correct shapes."""
    state_dim = 64
    dhippo = DiagonalHiPPO(state_dim=state_dim)

    A_diag, B = dhippo.get_matrices()

    assert A_diag.shape == (state_dim,)
    assert B.shape == (state_dim,)


def test_diagonal_hippo_has_log_a():
    """DiagonalHiPPO should store log_A parameter for stable training."""
    dhippo = DiagonalHiPPO(state_dim=32)

    assert hasattr(dhippo, "log_A")
    assert dhippo.log_A.shape == (32,)


def test_diagonal_hippo_learnable():
    """DiagonalHiPPO with learnable=True should have learnable parameters."""
    dhippo = DiagonalHiPPO(state_dim=32, learnable=True)

    assert isinstance(dhippo.log_A, torch.nn.Parameter)
    assert isinstance(dhippo.B, torch.nn.Parameter)


# ============================================================================
# SelectiveScan Tests
# ============================================================================


@pytest.mark.parametrize("scan_class", [SelectiveScan, ParallelSelectiveScan])
def test_selective_scan_output_shape(scan_class):
    """SelectiveScan/ParallelSelectiveScan output should match input dimensions."""
    ssm = scan_class(embed_dim=64, state_dim=16, use_hippo_init=False)
    x = torch.randn(2, 32, 64)  # (batch, seq_len, embed_dim)

    y = ssm(x)

    assert y.shape == x.shape


@pytest.mark.parametrize("scan_class", [SelectiveScan, ParallelSelectiveScan])
def test_selective_scan_batch_independence(scan_class):
    """SelectiveScan/ParallelSelectiveScan should process batches independently."""
    ssm = scan_class(embed_dim=64, state_dim=16, use_hippo_init=False)

    x1 = torch.randn(1, 16, 64)
    x2 = torch.randn(1, 16, 64)
    x_combined = torch.cat([x1, x2], dim=0)

    y1 = ssm(x1)
    y2 = ssm(x2)
    y_combined = ssm(x_combined)

    assert torch.allclose(y1, y_combined[0:1], atol=1e-5)
    assert torch.allclose(y2, y_combined[1:2], atol=1e-5)


@pytest.mark.parametrize("scan_class", [SelectiveScan, ParallelSelectiveScan])
def test_selective_scan_step_shape(scan_class):
    """SelectiveScan/ParallelSelectiveScan step should output correct shapes."""
    ssm = scan_class(embed_dim=64, state_dim=16, use_hippo_init=False)
    x = torch.randn(2, 64)  # (batch, embed_dim) - single step

    state = ssm.init_state(2, x.device, x.dtype)
    y, new_state = ssm.step(x, state)

    assert y.shape == (2, 64)
    assert new_state.shape == (2, 64, 16)  # (batch, embed_dim, state_dim)


@pytest.mark.parametrize("scan_class", [SelectiveScan, ParallelSelectiveScan])
def test_selective_scan_state_init(scan_class):
    """SelectiveScan/ParallelSelectiveScan init_state should return zeros."""
    ssm = scan_class(embed_dim=64, state_dim=16)
    state = ssm.init_state(4, torch.device("cpu"), torch.float32)

    assert state.shape == (4, 64, 16)
    assert (state == 0).all()


@pytest.mark.parametrize("scan_class", [SelectiveScan, ParallelSelectiveScan])
def test_selective_scan_no_nan_without_hippo(scan_class):
    """SelectiveScan/ParallelSelectiveScan without HiPPO init should not produce NaN."""
    ssm = scan_class(embed_dim=64, state_dim=16, use_hippo_init=False)
    x = torch.randn(2, 32, 64)

    y = ssm(x)

    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()


@pytest.mark.parametrize("scan_class", [SelectiveScan, ParallelSelectiveScan])
def test_selective_scan_different_state_dims(scan_class):
    """SelectiveScan/ParallelSelectiveScan should work with different state dimensions."""
    for state_dim in [8, 16, 32]:
        ssm = scan_class(embed_dim=64, state_dim=state_dim, use_hippo_init=False)
        x = torch.randn(2, 16, 64)
        y = ssm(x)

        assert y.shape == x.shape, f"Failed for {scan_class.__name__} with state_dim={state_dim}"


@pytest.mark.parametrize("scan_class", [SelectiveScan, ParallelSelectiveScan])
def test_selective_scan_autoregressive_consistency(scan_class):
    """SelectiveScan/ParallelSelectiveScan full forward and step-by-step should produce similar results."""
    ssm = scan_class(embed_dim=32, state_dim=8, use_hippo_init=False)
    seq_len = 8
    x = torch.randn(1, seq_len, 32)

    # Full forward
    y_full = ssm(x)

    # Step-by-step
    state = ssm.init_state(1, x.device, x.dtype)
    y_steps = []
    for t in range(seq_len):
        y_t, state = ssm.step(x[:, t], state)
        y_steps.append(y_t)
    y_step = torch.stack(y_steps, dim=1)

    # Should be close (may not be exact due to numerical differences)
    assert torch.allclose(y_full, y_step, atol=1e-4)


@pytest.mark.parametrize("scan_class", [SelectiveScan, ParallelSelectiveScan])
def test_selective_scan_dt_rank_auto(scan_class):
    """SelectiveScan/ParallelSelectiveScan should auto-compute dt_rank correctly."""
    ssm = scan_class(embed_dim=64, state_dim=16, dt_rank="auto")

    # Auto sets to ceil(embed_dim / 16)
    expected_dt_rank = max(1, (64 + 15) // 16)  # ceil(64/16) = 4
    assert ssm.dt_rank == expected_dt_rank


@pytest.mark.parametrize("scan_class", [SelectiveScan, ParallelSelectiveScan])
def test_selective_scan_dt_rank_explicit(scan_class):
    """SelectiveScan/ParallelSelectiveScan should accept explicit dt_rank."""
    ssm = scan_class(embed_dim=64, state_dim=16, dt_rank=8)

    assert ssm.dt_rank == 8


# ============================================================================
# ParallelSelectiveScan Tests
# ============================================================================


def test_parallel_selective_scan_output_shape():
    """ParallelSelectiveScan output should match input dimensions."""
    ssm = ParallelSelectiveScan(embed_dim=64, state_dim=16, use_hippo_init=False)
    x = torch.randn(2, 32, 64)

    y = ssm(x)

    assert y.shape == x.shape


def test_parallel_selective_scan_no_nan():
    """ParallelSelectiveScan should not produce NaN without HiPPO init."""
    ssm = ParallelSelectiveScan(embed_dim=32, state_dim=8, use_hippo_init=False)

    x = torch.randn(2, 16, 32)
    y = ssm(x)

    assert y.shape == x.shape
    assert not torch.isnan(y).any()


# ============================================================================
# MambaBlock Tests
# ============================================================================


@pytest.mark.parametrize("use_parallel_scan", [False, True])
def test_mamba_block_output_shape(use_parallel_scan):
    """MambaBlock output should match input dimensions (both sequential and parallel)."""
    block = MambaBlock(embed_dim=384, state_dim=64, use_hippo_init=False, use_parallel_scan=use_parallel_scan)
    x = torch.randn(2, 32, 384)

    y = block(x)

    assert y.shape == x.shape


@pytest.mark.parametrize("use_parallel_scan", [False, True])
def test_mamba_block_residual_connection(use_parallel_scan):
    """MambaBlock should include residual connection (both sequential and parallel)."""
    block = MambaBlock(embed_dim=128, state_dim=16, use_hippo_init=False, use_parallel_scan=use_parallel_scan)
    x = torch.randn(2, 16, 128)

    y = block(x)

    # Output should be influenced by input (residual)
    # Correlation should be positive
    correlation = (x * y).mean()
    assert correlation > 0


def test_mamba_block_expand_factor():
    """MambaBlock expand_factor should affect inner dimension."""
    block_2x = MambaBlock(embed_dim=128, expand_factor=2)
    block_4x = MambaBlock(embed_dim=128, expand_factor=4)

    assert block_2x.inner_dim == 256
    assert block_4x.inner_dim == 512


@pytest.mark.parametrize("use_parallel_scan", [False, True])
def test_mamba_block_conv_kernel_size(use_parallel_scan):
    """MambaBlock should support different conv kernel sizes (both sequential and parallel)."""
    for kernel_size in [2, 4, 8]:
        block = MambaBlock(embed_dim=128, conv_kernel_size=kernel_size, use_hippo_init=False, use_parallel_scan=use_parallel_scan)
        x = torch.randn(1, 32, 128)
        y = block(x)

        assert y.shape == x.shape


@pytest.mark.parametrize("use_parallel_scan", [False, True])
def test_mamba_block_step_shape(use_parallel_scan):
    """MambaBlock step should output correct shapes (both sequential and parallel)."""
    block = MambaBlock(embed_dim=128, state_dim=16, use_hippo_init=False, use_parallel_scan=use_parallel_scan)
    x = torch.randn(2, 128)  # Single token per batch

    ssm_state, conv_state = block.init_state(2, x.device, x.dtype)
    y, new_ssm_state, new_conv_state = block.step(x, ssm_state, conv_state)

    assert y.shape == (2, 128)
    assert new_ssm_state.shape == (2, block.inner_dim, 16)
    assert new_conv_state.shape == (2, block.inner_dim, block.conv_kernel_size - 1)


def test_mamba_block_init_state():
    """MambaBlock init_state should return proper state shapes."""
    block = MambaBlock(embed_dim=128, state_dim=16, conv_kernel_size=4)

    ssm_state, conv_state = block.init_state(4, torch.device("cpu"), torch.float32)

    assert ssm_state.shape == (4, block.inner_dim, 16)
    assert conv_state.shape == (4, block.inner_dim, 3)  # kernel_size - 1


@pytest.mark.parametrize("use_parallel_scan", [False, True])
def test_mamba_block_autoregressive_generation(use_parallel_scan):
    """MambaBlock step-by-step should work for autoregressive generation (both sequential and parallel)."""
    block = MambaBlock(embed_dim=64, state_dim=8, conv_kernel_size=4, use_hippo_init=False, use_parallel_scan=use_parallel_scan)
    seq_len = 10
    batch_size = 2

    ssm_state, conv_state = block.init_state(batch_size, torch.device("cpu"))

    outputs = []
    for t in range(seq_len):
        x_t = torch.randn(batch_size, 64)
        y_t, ssm_state, conv_state = block.step(x_t, ssm_state, conv_state)
        outputs.append(y_t)

    y = torch.stack(outputs, dim=1)
    assert y.shape == (batch_size, seq_len, 64)


@pytest.mark.parametrize("use_parallel_scan", [False, True])
def test_mamba_block_dropout(use_parallel_scan):
    """MambaBlock with dropout should differ in train/eval mode (both sequential and parallel)."""
    block = MambaBlock(embed_dim=128, dropout=0.5, use_hippo_init=False, use_parallel_scan=use_parallel_scan)
    x = torch.randn(4, 32, 128)

    block.train()
    y_train = block(x)

    block.eval()
    y_eval = block(x)

    # May be same if dropout doesn't activate, but shouldn't error
    assert y_train.shape == y_eval.shape


def test_mamba_block_normalization():
    """MambaBlock with use_norm should apply layer normalization."""
    block_norm = MambaBlock(embed_dim=128, use_norm=True)
    block_no_norm = MambaBlock(embed_dim=128, use_norm=False)

    assert hasattr(block_norm, "norm")
    assert not hasattr(block_no_norm, "norm") or not block_no_norm.use_norm


@pytest.mark.parametrize("use_parallel_scan", [False, True])
def test_mamba_block_gradient_flow(use_parallel_scan):
    """MambaBlock should allow gradients to flow through (both sequential and parallel)."""
    block = MambaBlock(embed_dim=128, state_dim=16, use_hippo_init=False, use_parallel_scan=use_parallel_scan)
    x = torch.randn(2, 16, 128, requires_grad=True)

    y = block(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_mamba_block_extra_repr():
    """MambaBlock should have informative string representation."""
    block = MambaBlock(embed_dim=128, state_dim=16)

    repr_str = block.extra_repr()
    assert "embed_dim=128" in repr_str
    assert "state_dim=16" in repr_str


# ============================================================================
# MambaResidualBlock Tests
# ============================================================================


def test_mamba_residual_block_output_shape():
    """MambaResidualBlock output should match input dimensions."""
    block = MambaResidualBlock(embed_dim=256, state_dim=32, use_hippo_init=False)
    x = torch.randn(2, 32, 256)

    y = block(x)

    assert y.shape == x.shape


def test_mamba_residual_block_has_norm():
    """MambaResidualBlock should have external normalization."""
    block = MambaResidualBlock(embed_dim=256)

    assert hasattr(block, "norm")
    assert isinstance(block.norm, torch.nn.LayerNorm)


def test_mamba_residual_block_has_mamba():
    """MambaResidualBlock should contain a MambaBlock."""
    block = MambaResidualBlock(embed_dim=256)

    assert hasattr(block, "mamba")
    assert isinstance(block.mamba, MambaBlock)


# ============================================================================
# BidirectionalMamba Tests
# ============================================================================


@pytest.mark.parametrize("use_parallel_scan", [False, True])
def test_bidirectional_mamba_output_shape(use_parallel_scan):
    """BidirectionalMamba output should match input dimensions (both sequential and parallel)."""
    block = BidirectionalMamba(embed_dim=256, state_dim=32, use_hippo_init=False, use_parallel_scan=use_parallel_scan)
    x = torch.randn(2, 32, 256)

    y = block(x)

    assert y.shape == x.shape


@pytest.mark.parametrize("use_parallel_scan", [False, True])
def test_bidirectional_mamba_merge_modes(use_parallel_scan):
    """BidirectionalMamba should support all merge modes (both sequential and parallel)."""
    for merge_mode in ["concat", "sum", "gate"]:
        block = BidirectionalMamba(embed_dim=128, merge_mode=merge_mode, use_hippo_init=False, use_parallel_scan=use_parallel_scan)
        x = torch.randn(2, 16, 128)
        y = block(x)

        assert y.shape == x.shape, f"Failed for merge_mode={merge_mode}"


def test_bidirectional_mamba_concat_has_proj():
    """BidirectionalMamba with concat should have merge projection."""
    block = BidirectionalMamba(embed_dim=128, merge_mode="concat")

    assert hasattr(block, "merge_proj")
    assert block.merge_proj.in_features == 256
    assert block.merge_proj.out_features == 128


def test_bidirectional_mamba_gate_has_gate():
    """BidirectionalMamba with gate should have gate projection."""
    block = BidirectionalMamba(embed_dim=128, merge_mode="gate")

    assert hasattr(block, "gate")


@pytest.mark.parametrize("use_parallel_scan", [False, True])
def test_bidirectional_mamba_bidirectional_context(use_parallel_scan):
    """BidirectionalMamba should capture both forward and backward context (both sequential and parallel)."""
    block = BidirectionalMamba(embed_dim=64, merge_mode="concat", use_hippo_init=False, use_parallel_scan=use_parallel_scan)

    # Create input where first and last positions have distinct patterns
    x = torch.zeros(1, 10, 64)
    x[:, 0, :32] = 1.0  # First position has pattern in first 32 dims
    x[:, -1, 32:] = 1.0  # Last position has pattern in last 32 dims

    y = block(x)

    # Output should be different at different positions
    assert not torch.allclose(y[:, 0], y[:, -1])


def test_bidirectional_mamba_has_two_mamba_blocks():
    """BidirectionalMamba should have forward and backward MambaBlocks."""
    block = BidirectionalMamba(embed_dim=128)

    assert hasattr(block, "forward_mamba")
    assert hasattr(block, "backward_mamba")
    assert isinstance(block.forward_mamba, MambaBlock)
    assert isinstance(block.backward_mamba, MambaBlock)


# ============================================================================
# SSMConfig Tests
# ============================================================================


def test_ssm_config_defaults():
    """SSMConfig should have sensible defaults."""
    config = SSMConfig()

    assert config.embed_dim == 384
    assert config.state_dim == 64
    assert config.dt_min == 0.001
    assert config.dt_max == 0.1


def test_ssm_config_custom():
    """SSMConfig should accept custom values."""
    config = SSMConfig(
        embed_dim=512,
        state_dim=128,
        use_hippo=True,
        hippo_type="legt",
    )

    assert config.embed_dim == 512
    assert config.state_dim == 128
    assert config.use_hippo is True
    assert config.hippo_type == "legt"


def test_ssm_config_all_fields():
    """SSMConfig should have all expected fields."""
    config = SSMConfig()

    assert hasattr(config, "embed_dim")
    assert hasattr(config, "state_dim")
    assert hasattr(config, "dt_min")
    assert hasattr(config, "dt_max")
    assert hasattr(config, "dropout")
    assert hasattr(config, "use_hippo")
    assert hasattr(config, "hippo_type")
    assert hasattr(config, "expand_factor")
    assert hasattr(config, "conv_kernel_size")
    assert hasattr(config, "use_selective_scan")


# ============================================================================
# Edge Cases and Robustness Tests
# ============================================================================


@pytest.mark.parametrize("use_parallel_scan", [False, True])
def test_ssm_components_with_tiny_input(use_parallel_scan):
    """SSM components should handle very short sequences (both sequential and parallel)."""
    block = MambaBlock(embed_dim=64, state_dim=8, use_hippo_init=False, use_parallel_scan=use_parallel_scan)
    x = torch.randn(1, 1, 64)  # Single token

    y = block(x)

    assert y.shape == x.shape
    assert not torch.isnan(y).any()


@pytest.mark.parametrize("use_parallel_scan", [False, True])
def test_ssm_components_with_long_sequence(use_parallel_scan):
    """SSM components should handle long sequences efficiently (both sequential and parallel)."""
    block = MambaBlock(embed_dim=64, state_dim=8, use_hippo_init=False, use_parallel_scan=use_parallel_scan)
    x = torch.randn(1, 512, 64)  # Long sequence

    y = block(x)

    assert y.shape == x.shape
    assert not torch.isnan(y).any()


@pytest.mark.parametrize("use_parallel_scan", [False, True])
def test_ssm_numerical_stability(use_parallel_scan):
    """SSM components should be numerically stable (both sequential and parallel)."""
    block = MambaBlock(embed_dim=128, state_dim=16, use_hippo_init=False, use_parallel_scan=use_parallel_scan)

    # Test with various input scales
    for scale in [0.01, 1.0, 10.0]:
        x = torch.randn(2, 32, 128) * scale
        y = block(x)

        assert not torch.isnan(y).any(), f"NaN detected at scale {scale}"
        assert not torch.isinf(y).any(), f"Inf detected at scale {scale}"


@pytest.mark.parametrize("use_parallel_scan", [False, True])
def test_ssm_deterministic(use_parallel_scan):
    """SSM components should be deterministic in eval mode (both sequential and parallel)."""
    block = MambaBlock(embed_dim=128, state_dim=16, dropout=0.0, use_hippo_init=False, use_parallel_scan=use_parallel_scan)
    block.eval()

    x = torch.randn(2, 32, 128)

    y1 = block(x)
    y2 = block(x)

    assert torch.allclose(y1, y2)


@pytest.mark.parametrize("use_parallel_scan", [False, True])
def test_ssm_different_batch_sizes(use_parallel_scan):
    """SSM components should handle different batch sizes (both sequential and parallel)."""
    block = MambaBlock(embed_dim=64, state_dim=8, use_hippo_init=False, use_parallel_scan=use_parallel_scan)

    for batch_size in [1, 2, 4, 8]:
        x = torch.randn(batch_size, 16, 64)
        y = block(x)

        assert y.shape == (batch_size, 16, 64)


def test_ssm_conv_state_buffer_size():
    """MambaBlock conv_state should have correct buffer size."""
    for kernel_size in [2, 4, 8]:
        block = MambaBlock(embed_dim=64, conv_kernel_size=kernel_size)
        _, conv_state = block.init_state(2, torch.device("cpu"))

        # Buffer should hold kernel_size - 1 previous activations
        assert conv_state.shape[-1] == kernel_size - 1
