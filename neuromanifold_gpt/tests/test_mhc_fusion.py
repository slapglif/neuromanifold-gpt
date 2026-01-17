# neuromanifold_gpt/tests/test_mhc_fusion.py
"""Tests comparing unfused vs fused mHC width_connection operations."""
import pytest
import torch
from neuromanifold_gpt.model.mhc import HyperConnections
from neuromanifold_gpt.model.sinkhorn import sinkhorn_log

# Try to import Triton-based fused implementation
try:
    from neuromanifold_gpt.model.mhc_fused import FusedMHCWidthConnection, fused_mhc_width_connection
    has_triton = torch.cuda.is_available()
except (ImportError, RuntimeError):
    has_triton = False

# Decorator for tests that require CUDA
requires_cuda = pytest.mark.skipif(not has_triton, reason="CUDA required for Triton kernels")


@requires_cuda
def test_output_shapes_match():
    """Unfused and fused paths should produce matching output shapes."""
    num_streams = 4
    dim = 64
    batch_size = 2
    seq_len = 10

    # Create HyperConnections for unfused path
    hc = HyperConnections(num_streams, dim=dim)

    # Create input: (B*S, T, D)
    residuals = torch.randn(batch_size * num_streams, seq_len, dim)

    # Unfused path: call width_connection
    branch_input_unfused, residuals_out_unfused, h_post_unfused = hc.width_connection(residuals)

    # Fused path: manually compute H_res and H_pre, then call fused kernel
    with torch.no_grad():
        h_res = sinkhorn_log(hc.H_res_logits, num_iters=hc.sinkhorn_iters, tau=hc.sinkhorn_tau)
        h_pre = hc.H_pre_logits.softmax(dim=-1)

    branch_input_fused, residuals_out_fused = fused_mhc_width_connection(residuals, h_res, h_pre)

    # Check shapes
    assert branch_input_unfused.shape == branch_input_fused.shape == (batch_size, seq_len, dim)
    assert residuals_out_unfused.shape == residuals_out_fused.shape == (batch_size * num_streams, seq_len, dim)
    assert h_post_unfused.shape == (1, num_streams)


@requires_cuda
def test_forward_pass_equivalence():
    """Unfused and fused paths should produce numerically equivalent outputs."""
    num_streams = 4
    dim = 64
    batch_size = 2
    seq_len = 10

    # Create HyperConnections
    hc = HyperConnections(num_streams, dim=dim)

    # Create input
    residuals = torch.randn(batch_size * num_streams, seq_len, dim)

    # Unfused path
    branch_input_unfused, residuals_out_unfused, _ = hc.width_connection(residuals)

    # Fused path
    with torch.no_grad():
        h_res = sinkhorn_log(hc.H_res_logits, num_iters=hc.sinkhorn_iters, tau=hc.sinkhorn_tau)
        h_pre = hc.H_pre_logits.softmax(dim=-1)

    branch_input_fused, residuals_out_fused = fused_mhc_width_connection(residuals, h_res, h_pre)

    # Check numerical equivalence
    assert torch.allclose(branch_input_unfused, branch_input_fused, rtol=1e-4, atol=1e-5), \
        f"branch_input mismatch: max diff = {(branch_input_unfused - branch_input_fused).abs().max()}"
    assert torch.allclose(residuals_out_unfused, residuals_out_fused, rtol=1e-4, atol=1e-5), \
        f"residuals_out mismatch: max diff = {(residuals_out_unfused - residuals_out_fused).abs().max()}"


def test_gradient_flow_unfused_path():
    """Gradients should flow through unfused width_connection path."""
    num_streams = 4
    dim = 64
    batch_size = 2
    seq_len = 10

    hc = HyperConnections(num_streams, dim=dim)
    residuals = torch.randn(batch_size * num_streams, seq_len, dim, requires_grad=True)

    branch_input, residuals_out, _ = hc.width_connection(residuals)
    loss = branch_input.sum() + residuals_out.sum()
    loss.backward()

    assert residuals.grad is not None
    assert residuals.grad.abs().sum() > 0
    assert hc.H_res_logits.grad is not None
    assert hc.H_pre_logits.grad is not None


@requires_cuda
def test_gradient_flow_fused_path():
    """Gradients should flow through fused kernel path."""
    num_streams = 4
    dim = 64
    batch_size = 2
    seq_len = 10

    residuals = torch.randn(batch_size * num_streams, seq_len, dim, requires_grad=True)
    h_res = torch.randn(num_streams, num_streams, requires_grad=True).softmax(dim=-1)
    h_pre = torch.randn(1, num_streams, requires_grad=True).softmax(dim=-1)

    branch_input, residuals_out = fused_mhc_width_connection(residuals, h_res, h_pre)
    loss = branch_input.sum() + residuals_out.sum()
    loss.backward()

    assert residuals.grad is not None
    assert residuals.grad.abs().sum() > 0
    assert h_res.grad is not None
    assert h_res.grad.abs().sum() > 0
    assert h_pre.grad is not None
    assert h_pre.grad.abs().sum() > 0


@requires_cuda
def test_backward_pass_equivalence():
    """Unfused and fused paths should produce equivalent gradients."""
    num_streams = 4
    dim = 64
    batch_size = 2
    seq_len = 10

    # Create shared inputs
    residuals_unfused = torch.randn(batch_size * num_streams, seq_len, dim, requires_grad=True)
    residuals_fused = residuals_unfused.clone().detach().requires_grad_(True)

    # Create HyperConnections
    hc = HyperConnections(num_streams, dim=dim)

    # Get H matrices
    with torch.no_grad():
        h_res = sinkhorn_log(hc.H_res_logits, num_iters=hc.sinkhorn_iters, tau=hc.sinkhorn_tau)
        h_pre = hc.H_pre_logits.softmax(dim=-1)

    # Make them require grad for comparison
    h_res_fused = h_res.clone().detach().requires_grad_(True)
    h_pre_fused = h_pre.clone().detach().requires_grad_(True)

    # Unfused path
    branch_input_unfused, residuals_out_unfused, _ = hc.width_connection(residuals_unfused)
    loss_unfused = branch_input_unfused.sum() + residuals_out_unfused.sum()
    loss_unfused.backward()

    # Fused path
    branch_input_fused, residuals_out_fused = FusedMHCWidthConnection.apply(
        residuals_fused, h_res_fused, h_pre_fused
    )
    loss_fused = branch_input_fused.sum() + residuals_out_fused.sum()
    loss_fused.backward()

    # Compare gradients
    assert torch.allclose(residuals_unfused.grad, residuals_fused.grad, rtol=1e-3, atol=1e-4), \
        f"residuals grad mismatch: max diff = {(residuals_unfused.grad - residuals_fused.grad).abs().max()}"


@requires_cuda
def test_single_stream():
    """Both paths should handle single stream (S=1) edge case."""
    num_streams = 1
    dim = 64
    batch_size = 2
    seq_len = 10

    hc = HyperConnections(num_streams, dim=dim)
    residuals = torch.randn(batch_size * num_streams, seq_len, dim)

    # Unfused path
    branch_input_unfused, residuals_out_unfused, _ = hc.width_connection(residuals)

    # Fused path
    with torch.no_grad():
        h_res = sinkhorn_log(hc.H_res_logits, num_iters=hc.sinkhorn_iters, tau=hc.sinkhorn_tau)
        h_pre = hc.H_pre_logits.softmax(dim=-1)

    branch_input_fused, residuals_out_fused = fused_mhc_width_connection(residuals, h_res, h_pre)

    assert branch_input_unfused.shape == (batch_size, seq_len, dim)
    assert residuals_out_unfused.shape == (batch_size * num_streams, seq_len, dim)
    assert torch.allclose(branch_input_unfused, branch_input_fused, rtol=1e-4, atol=1e-5)
    assert torch.allclose(residuals_out_unfused, residuals_out_fused, rtol=1e-4, atol=1e-5)


@requires_cuda
def test_large_streams():
    """Both paths should handle large number of streams (S=8)."""
    num_streams = 8
    dim = 64
    batch_size = 2
    seq_len = 10

    hc = HyperConnections(num_streams, dim=dim)
    residuals = torch.randn(batch_size * num_streams, seq_len, dim)

    # Unfused path
    branch_input_unfused, residuals_out_unfused, _ = hc.width_connection(residuals)

    # Fused path
    with torch.no_grad():
        h_res = sinkhorn_log(hc.H_res_logits, num_iters=hc.sinkhorn_iters, tau=hc.sinkhorn_tau)
        h_pre = hc.H_pre_logits.softmax(dim=-1)

    branch_input_fused, residuals_out_fused = fused_mhc_width_connection(residuals, h_res, h_pre)

    assert branch_input_unfused.shape == (batch_size, seq_len, dim)
    assert residuals_out_unfused.shape == (batch_size * num_streams, seq_len, dim)
    assert torch.allclose(branch_input_unfused, branch_input_fused, rtol=1e-4, atol=1e-5)
    assert torch.allclose(residuals_out_unfused, residuals_out_fused, rtol=1e-4, atol=1e-5)


@requires_cuda
def test_various_batch_sizes():
    """Both paths should handle various batch sizes."""
    num_streams = 4
    dim = 64
    seq_len = 10

    for batch_size in [1, 2, 4, 8]:
        hc = HyperConnections(num_streams, dim=dim)
        residuals = torch.randn(batch_size * num_streams, seq_len, dim)

        # Unfused path
        branch_input_unfused, residuals_out_unfused, _ = hc.width_connection(residuals)

        # Fused path
        with torch.no_grad():
            h_res = sinkhorn_log(hc.H_res_logits, num_iters=hc.sinkhorn_iters, tau=hc.sinkhorn_tau)
            h_pre = hc.H_pre_logits.softmax(dim=-1)

        branch_input_fused, residuals_out_fused = fused_mhc_width_connection(residuals, h_res, h_pre)

        assert branch_input_unfused.shape == (batch_size, seq_len, dim)
        assert residuals_out_unfused.shape == (batch_size * num_streams, seq_len, dim)
        assert torch.allclose(branch_input_unfused, branch_input_fused, rtol=1e-4, atol=1e-5)


@requires_cuda
def test_various_sequence_lengths():
    """Both paths should handle various sequence lengths."""
    num_streams = 4
    dim = 64
    batch_size = 2

    for seq_len in [5, 10, 20, 50]:
        hc = HyperConnections(num_streams, dim=dim)
        residuals = torch.randn(batch_size * num_streams, seq_len, dim)

        # Unfused path
        branch_input_unfused, residuals_out_unfused, _ = hc.width_connection(residuals)

        # Fused path
        with torch.no_grad():
            h_res = sinkhorn_log(hc.H_res_logits, num_iters=hc.sinkhorn_iters, tau=hc.sinkhorn_tau)
            h_pre = hc.H_pre_logits.softmax(dim=-1)

        branch_input_fused, residuals_out_fused = fused_mhc_width_connection(residuals, h_res, h_pre)

        assert branch_input_unfused.shape == (batch_size, seq_len, dim)
        assert residuals_out_unfused.shape == (batch_size * num_streams, seq_len, dim)
        assert torch.allclose(branch_input_unfused, branch_input_fused, rtol=1e-4, atol=1e-5)


@requires_cuda
def test_various_dimensions():
    """Both paths should handle various embedding dimensions."""
    num_streams = 4
    batch_size = 2
    seq_len = 10

    for dim in [32, 64, 128, 256]:
        hc = HyperConnections(num_streams, dim=dim)
        residuals = torch.randn(batch_size * num_streams, seq_len, dim)

        # Unfused path
        branch_input_unfused, residuals_out_unfused, _ = hc.width_connection(residuals)

        # Fused path
        with torch.no_grad():
            h_res = sinkhorn_log(hc.H_res_logits, num_iters=hc.sinkhorn_iters, tau=hc.sinkhorn_tau)
            h_pre = hc.H_pre_logits.softmax(dim=-1)

        branch_input_fused, residuals_out_fused = fused_mhc_width_connection(residuals, h_res, h_pre)

        assert branch_input_unfused.shape == (batch_size, seq_len, dim)
        assert residuals_out_unfused.shape == (batch_size * num_streams, seq_len, dim)
        assert torch.allclose(branch_input_unfused, branch_input_fused, rtol=1e-4, atol=1e-5)


@requires_cuda
def test_deterministic_with_fixed_seed():
    """Both paths should be deterministic with fixed seed."""
    num_streams = 4
    dim = 64
    batch_size = 2
    seq_len = 10

    def run_unfused(seed=42):
        torch.manual_seed(seed)
        hc = HyperConnections(num_streams, dim=dim)
        torch.manual_seed(seed + 1000)
        residuals = torch.randn(batch_size * num_streams, seq_len, dim)
        branch_input, residuals_out, _ = hc.width_connection(residuals)
        return branch_input, residuals_out

    def run_fused(seed=42):
        torch.manual_seed(seed)
        hc = HyperConnections(num_streams, dim=dim)
        torch.manual_seed(seed + 1000)
        residuals = torch.randn(batch_size * num_streams, seq_len, dim)

        with torch.no_grad():
            h_res = sinkhorn_log(hc.H_res_logits, num_iters=hc.sinkhorn_iters, tau=hc.sinkhorn_tau)
            h_pre = hc.H_pre_logits.softmax(dim=-1)

        branch_input, residuals_out = fused_mhc_width_connection(residuals, h_res, h_pre)
        return branch_input, residuals_out

    # Run unfused twice with same seed
    branch_input_unfused_1, residuals_out_unfused_1 = run_unfused(42)
    branch_input_unfused_2, residuals_out_unfused_2 = run_unfused(42)

    # Run fused twice with same seed
    branch_input_fused_1, residuals_out_fused_1 = run_fused(42)
    branch_input_fused_2, residuals_out_fused_2 = run_fused(42)

    # Both should be deterministic
    assert torch.allclose(branch_input_unfused_1, branch_input_unfused_2)
    assert torch.allclose(residuals_out_unfused_1, residuals_out_unfused_2)
    assert torch.allclose(branch_input_fused_1, branch_input_fused_2)
    assert torch.allclose(residuals_out_fused_1, residuals_out_fused_2)


def test_integration_with_hyperconnections():
    """Test full HyperConnections forward pass."""
    num_streams = 4
    dim = 64
    batch_size = 2
    seq_len = 10

    # Create HyperConnections with a simple identity branch
    class IdentityBranch(torch.nn.Module):
        def forward(self, x):
            return x

    hc = HyperConnections(num_streams, dim=dim, branch=IdentityBranch())

    # Create input: (B*S, T, D)
    x = torch.randn(batch_size * num_streams, seq_len, dim)

    # Forward pass through full HyperConnections
    output = hc(x)

    # Should produce output of same shape
    assert output.shape == x.shape


@requires_cuda
def test_non_contiguous_tensors():
    """Both paths should handle non-contiguous tensors."""
    num_streams = 4
    dim = 64
    batch_size = 2
    seq_len = 10

    hc = HyperConnections(num_streams, dim=dim)

    # Create non-contiguous input via transpose
    residuals_base = torch.randn(seq_len, batch_size * num_streams, dim)
    residuals = residuals_base.transpose(0, 1)  # Non-contiguous
    assert not residuals.is_contiguous()

    # Unfused path
    branch_input_unfused, residuals_out_unfused, _ = hc.width_connection(residuals)

    # Fused path (should handle non-contiguous via .contiguous() call)
    with torch.no_grad():
        h_res = sinkhorn_log(hc.H_res_logits, num_iters=hc.sinkhorn_iters, tau=hc.sinkhorn_tau)
        h_pre = hc.H_pre_logits.softmax(dim=-1)

    branch_input_fused, residuals_out_fused = fused_mhc_width_connection(residuals, h_res, h_pre)

    # Should still produce correct outputs
    assert torch.allclose(branch_input_unfused, branch_input_fused, rtol=1e-4, atol=1e-5)
    assert torch.allclose(residuals_out_unfused, residuals_out_fused, rtol=1e-4, atol=1e-5)


def test_h_res_doubly_stochastic():
    """Verify that H_res is doubly stochastic after Sinkhorn."""
    num_streams = 4
    dim = 64

    hc = HyperConnections(num_streams, dim=dim)

    # Get H_res via Sinkhorn
    with torch.no_grad():
        h_res = sinkhorn_log(hc.H_res_logits, num_iters=hc.sinkhorn_iters, tau=hc.sinkhorn_tau)

    # Check doubly stochastic properties
    # Row sums should be 1
    row_sums = h_res.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-4, atol=1e-5)

    # Column sums should be 1
    col_sums = h_res.sum(dim=0)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), rtol=1e-4, atol=1e-5)

    # All entries should be non-negative
    assert (h_res >= 0).all()


def test_h_pre_normalized():
    """Verify that H_pre is properly normalized (softmax)."""
    num_streams = 4
    dim = 64

    hc = HyperConnections(num_streams, dim=dim)

    # Get H_pre via softmax
    with torch.no_grad():
        h_pre = hc.H_pre_logits.softmax(dim=-1)

    # Should sum to 1
    assert torch.allclose(h_pre.sum(), torch.tensor(1.0), rtol=1e-5, atol=1e-6)

    # All entries should be non-negative
    assert (h_pre >= 0).all()


@requires_cuda
def test_backward_with_different_loss_functions():
    """Test backward pass with different loss functions."""
    num_streams = 4
    dim = 64
    batch_size = 2
    seq_len = 10

    residuals = torch.randn(batch_size * num_streams, seq_len, dim, requires_grad=True)
    h_res = torch.randn(num_streams, num_streams, requires_grad=True).softmax(dim=-1)
    h_pre = torch.randn(1, num_streams, requires_grad=True).softmax(dim=-1)

    # Test with different loss functions
    for loss_fn in [
        lambda x, y: x.sum() + y.sum(),
        lambda x, y: (x ** 2).mean() + (y ** 2).mean(),
        lambda x, y: x.mean() - y.mean(),
    ]:
        # Reset gradients
        if residuals.grad is not None:
            residuals.grad.zero_()
        if h_res.grad is not None:
            h_res.grad.zero_()
        if h_pre.grad is not None:
            h_pre.grad.zero_()

        branch_input, residuals_out = fused_mhc_width_connection(residuals, h_res, h_pre)
        loss = loss_fn(branch_input, residuals_out)
        loss.backward()

        # Gradients should exist and be non-zero
        assert residuals.grad is not None
        assert residuals.grad.abs().sum() > 0
        assert h_res.grad is not None
        assert h_pre.grad is not None


# CPU-only tests (these run even without CUDA)
@pytest.mark.skipif(False, reason="")  # Always run
def test_unfused_path_output_shapes():
    """Verify unfused path produces correct output shapes (CPU-safe)."""
    num_streams = 4
    dim = 64
    batch_size = 2
    seq_len = 10

    hc = HyperConnections(num_streams, dim=dim)
    residuals = torch.randn(batch_size * num_streams, seq_len, dim)

    branch_input, residuals_out, h_post = hc.width_connection(residuals)

    assert branch_input.shape == (batch_size, seq_len, dim)
    assert residuals_out.shape == (batch_size * num_streams, seq_len, dim)
    assert h_post.shape == (1, num_streams)


@pytest.mark.skipif(False, reason="")  # Always run
def test_unfused_path_gradient_flow():
    """Verify gradients flow through unfused path (CPU-safe)."""
    num_streams = 4
    dim = 64
    batch_size = 2
    seq_len = 10

    hc = HyperConnections(num_streams, dim=dim)
    residuals = torch.randn(batch_size * num_streams, seq_len, dim, requires_grad=True)

    branch_input, residuals_out, _ = hc.width_connection(residuals)
    loss = branch_input.sum() + residuals_out.sum()
    loss.backward()

    assert residuals.grad is not None
    assert residuals.grad.abs().sum() > 0
    assert hc.H_res_logits.grad is not None
    assert hc.H_pre_logits.grad is not None


@pytest.mark.skipif(False, reason="")  # Always run
def test_sinkhorn_doubly_stochastic():
    """Verify Sinkhorn produces doubly stochastic matrices (CPU-safe)."""
    num_streams = 4

    # Create random logits with fixed seed for reproducibility
    torch.manual_seed(123)
    logits = torch.randn(num_streams, num_streams)

    # Apply Sinkhorn with more iterations for convergence
    h_res = sinkhorn_log(logits, num_iters=100, tau=0.05)

    # Check doubly stochastic properties
    # Row sums should be close to 1 (iterative algorithm may not converge exactly)
    row_sums = h_res.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), rtol=2e-2, atol=2e-2)

    # Column sums should be close to 1
    col_sums = h_res.sum(dim=0)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), rtol=2e-2, atol=2e-2)

    # All entries should be non-negative
    assert (h_res >= 0).all()


@pytest.mark.skipif(False, reason="")  # Always run
def test_unfused_path_deterministic():
    """Verify unfused path is deterministic with same HyperConnections instance (CPU-safe)."""
    num_streams = 4
    dim = 64
    batch_size = 2
    seq_len = 10

    # Create a single HyperConnections instance
    torch.manual_seed(42)
    hc = HyperConnections(num_streams, dim=dim)

    # Run forward pass twice with same input
    torch.manual_seed(100)
    residuals_1 = torch.randn(batch_size * num_streams, seq_len, dim)

    torch.manual_seed(100)
    residuals_2 = torch.randn(batch_size * num_streams, seq_len, dim)

    branch_input_1, residuals_out_1, _ = hc.width_connection(residuals_1)
    branch_input_2, residuals_out_2, _ = hc.width_connection(residuals_2)

    # Should be deterministic with same model and input
    assert torch.allclose(branch_input_1, branch_input_2)
    assert torch.allclose(residuals_out_1, residuals_out_2)
