"""Tests for shared profiling utilities."""

import pytest
import torch
import torch.nn as nn


class TestCleanup:
    """Test suite for cleanup() function."""

    def test_cleanup_runs_without_error(self):
        """Test that cleanup() executes without raising exceptions."""
        from neuromanifold_gpt.utils.profiling import cleanup

        # Should not raise exception
        cleanup()

    def test_cleanup_with_cuda_available(self):
        """Test cleanup() when CUDA is available."""
        from neuromanifold_gpt.utils.profiling import cleanup

        if torch.cuda.is_available():
            # Allocate some memory
            x = torch.randn(100, 100, device="cuda")
            del x
            # Cleanup should clear cache
            cleanup()
        else:
            # Should still work without CUDA
            cleanup()

    def test_cleanup_multiple_calls(self):
        """Test that cleanup() can be called multiple times."""
        from neuromanifold_gpt.utils.profiling import cleanup

        # Multiple calls should not cause issues
        cleanup()
        cleanup()
        cleanup()


class TestProfileComponent:
    """Test suite for profile_component() function."""

    def test_profile_component_basic(self):
        """Test basic profiling of a simple component."""
        from neuromanifold_gpt.utils.profiling import profile_component

        # Create a simple module
        module = nn.Linear(10, 10)

        def make_input():
            return (torch.randn(4, 10),)

        result = profile_component(
            "TestLinear", module, make_input, n_warmup=2, n_iters=5, device="cpu"
        )

        # Check result structure
        assert "name" in result
        assert "mean_ms" in result
        assert "min_ms" in result
        assert "max_ms" in result
        assert "std_ms" in result
        assert result["name"] == "TestLinear"

        # Check timing values are reasonable
        assert result["mean_ms"] > 0
        assert result["min_ms"] > 0
        assert result["max_ms"] > 0
        assert result["std_ms"] >= 0
        assert result["min_ms"] <= result["mean_ms"] <= result["max_ms"]

    def test_profile_component_with_name(self):
        """Test that component name is preserved in results."""
        from neuromanifold_gpt.utils.profiling import profile_component

        module = nn.Linear(5, 5)

        def make_input():
            return (torch.randn(2, 5),)

        result = profile_component(
            "CustomName", module, make_input, n_warmup=1, n_iters=3, device="cpu"
        )

        assert result["name"] == "CustomName"

    def test_profile_component_auto_device(self):
        """Test that device is auto-detected when not specified."""
        from neuromanifold_gpt.utils.profiling import profile_component

        module = nn.Linear(8, 8)

        def make_input():
            device = "cuda" if torch.cuda.is_available() else "cpu"
            return (torch.randn(4, 8, device=device),)

        # Device=None should auto-detect
        result = profile_component(
            "AutoDevice", module, make_input, n_warmup=1, n_iters=2
        )

        assert "mean_ms" in result
        assert result["mean_ms"] > 0

    def test_profile_component_custom_warmup_iters(self):
        """Test custom warmup and iteration counts."""
        from neuromanifold_gpt.utils.profiling import profile_component

        module = nn.Linear(10, 10)

        def make_input():
            return (torch.randn(4, 10),)

        result = profile_component(
            "CustomIters", module, make_input, n_warmup=3, n_iters=10, device="cpu"
        )

        # Should complete successfully with custom parameters
        assert result["mean_ms"] > 0

    def test_profile_component_track_memory_cpu(self):
        """Test memory tracking on CPU (should add mem_mb with value 0)."""
        from neuromanifold_gpt.utils.profiling import profile_component

        module = nn.Linear(10, 10)

        def make_input():
            return (torch.randn(4, 10),)

        result = profile_component(
            "MemoryTest",
            module,
            make_input,
            n_warmup=1,
            n_iters=2,
            device="cpu",
            track_memory=True,
        )

        # Memory tracking adds mem_mb even on CPU, but value is 0
        assert "mem_mb" in result
        assert result["mem_mb"] == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_profile_component_cuda(self):
        """Test profiling on CUDA device."""
        from neuromanifold_gpt.utils.profiling import profile_component

        module = nn.Linear(64, 64)

        def make_input():
            return (torch.randn(8, 64, device="cuda"),)

        result = profile_component(
            "CUDATest", module, make_input, n_warmup=2, n_iters=5, device="cuda"
        )

        assert result["mean_ms"] > 0
        assert result["name"] == "CUDATest"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_profile_component_track_memory_cuda(self):
        """Test memory tracking on CUDA device."""
        from neuromanifold_gpt.utils.profiling import profile_component

        module = nn.Linear(128, 128)

        def make_input():
            return (torch.randn(16, 128, device="cuda"),)

        result = profile_component(
            "CUDAMemory",
            module,
            make_input,
            n_warmup=1,
            n_iters=2,
            device="cuda",
            track_memory=True,
        )

        # Should include memory usage on CUDA
        assert "mem_mb" in result
        assert result["mem_mb"] >= 0

    def test_profile_component_module_in_eval_mode(self):
        """Test that module is put in eval mode."""
        from neuromanifold_gpt.utils.profiling import profile_component

        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(5, 5)
                self.bn = nn.BatchNorm1d(5)

            def forward(self, x):
                return self.bn(self.linear(x))

        module = TestModule()
        module.train()  # Start in train mode

        def make_input():
            return (torch.randn(4, 5),)

        result = profile_component(
            "EvalMode", module, make_input, n_warmup=1, n_iters=2, device="cpu"
        )

        # Should complete successfully
        assert result["mean_ms"] > 0
        # Module should be in eval mode
        assert not module.training

    def test_profile_component_multiple_inputs(self):
        """Test profiling with multiple input tensors."""
        from neuromanifold_gpt.utils.profiling import profile_component

        class MultiInputModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(5, 5)
                self.linear2 = nn.Linear(5, 5)

            def forward(self, x, y):
                return self.linear1(x) + self.linear2(y)

        module = MultiInputModule()

        def make_input():
            return (torch.randn(2, 5), torch.randn(2, 5))

        result = profile_component(
            "MultiInput", module, make_input, n_warmup=1, n_iters=2, device="cpu"
        )

        assert result["mean_ms"] > 0


class TestProfileForwardBackward:
    """Test suite for profile_forward_backward() function."""

    def test_profile_forward_backward_basic(self):
        """Test basic profiling of forward and backward pass."""
        from neuromanifold_gpt.utils.profiling import profile_forward_backward

        module = nn.Linear(10, 10)

        def make_input():
            return (torch.randn(4, 10),)

        def loss_fn(output):
            return output.mean()

        result = profile_forward_backward(
            "TestLinear",
            module,
            make_input,
            loss_fn,
            n_warmup=2,
            n_iters=5,
            device="cpu",
        )

        # Check result structure
        assert "name" in result
        assert "mean_ms" in result
        assert "min_ms" in result
        assert "max_ms" in result
        assert "std_ms" in result
        assert result["name"] == "TestLinear"

        # Check timing values
        assert result["mean_ms"] > 0
        assert result["min_ms"] > 0
        assert result["max_ms"] > 0
        assert result["std_ms"] >= 0
        assert result["min_ms"] <= result["mean_ms"] <= result["max_ms"]

    def test_profile_forward_backward_with_name(self):
        """Test that component name is preserved in results."""
        from neuromanifold_gpt.utils.profiling import profile_forward_backward

        module = nn.Linear(5, 5)

        def make_input():
            return (torch.randn(2, 5),)

        def loss_fn(output):
            return output.sum()

        result = profile_forward_backward(
            "TrainingTest",
            module,
            make_input,
            loss_fn,
            n_warmup=1,
            n_iters=3,
            device="cpu",
        )

        assert result["name"] == "TrainingTest"

    def test_profile_forward_backward_auto_device(self):
        """Test that device is auto-detected when not specified."""
        from neuromanifold_gpt.utils.profiling import profile_forward_backward

        module = nn.Linear(8, 8)

        def make_input():
            device = "cuda" if torch.cuda.is_available() else "cpu"
            return (torch.randn(4, 8, device=device),)

        def loss_fn(output):
            return output.mean()

        result = profile_forward_backward(
            "AutoDevice", module, make_input, loss_fn, n_warmup=1, n_iters=2
        )

        assert "mean_ms" in result
        assert result["mean_ms"] > 0

    def test_profile_forward_backward_custom_warmup_iters(self):
        """Test custom warmup and iteration counts."""
        from neuromanifold_gpt.utils.profiling import profile_forward_backward

        module = nn.Linear(10, 10)

        def make_input():
            return (torch.randn(4, 10),)

        def loss_fn(output):
            return output.mean()

        result = profile_forward_backward(
            "CustomIters",
            module,
            make_input,
            loss_fn,
            n_warmup=3,
            n_iters=10,
            device="cpu",
        )

        assert result["mean_ms"] > 0

    def test_profile_forward_backward_track_memory_cpu(self):
        """Test memory tracking on CPU (should add mem_mb with value 0)."""
        from neuromanifold_gpt.utils.profiling import profile_forward_backward

        module = nn.Linear(10, 10)

        def make_input():
            return (torch.randn(4, 10),)

        def loss_fn(output):
            return output.mean()

        result = profile_forward_backward(
            "MemoryTest",
            module,
            make_input,
            loss_fn,
            n_warmup=1,
            n_iters=2,
            device="cpu",
            track_memory=True,
        )

        # Memory tracking adds mem_mb even on CPU, but value is 0
        assert "mem_mb" in result
        assert result["mem_mb"] == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_profile_forward_backward_cuda(self):
        """Test profiling on CUDA device."""
        from neuromanifold_gpt.utils.profiling import profile_forward_backward

        module = nn.Linear(64, 64)

        def make_input():
            return (torch.randn(8, 64, device="cuda"),)

        def loss_fn(output):
            return output.mean()

        result = profile_forward_backward(
            "CUDATest",
            module,
            make_input,
            loss_fn,
            n_warmup=2,
            n_iters=5,
            device="cuda",
        )

        assert result["mean_ms"] > 0
        assert result["name"] == "CUDATest"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_profile_forward_backward_track_memory_cuda(self):
        """Test memory tracking on CUDA device."""
        from neuromanifold_gpt.utils.profiling import profile_forward_backward

        module = nn.Linear(128, 128)

        def make_input():
            return (torch.randn(16, 128, device="cuda"),)

        def loss_fn(output):
            return output.mean()

        result = profile_forward_backward(
            "CUDAMemory",
            module,
            make_input,
            loss_fn,
            n_warmup=1,
            n_iters=2,
            device="cuda",
            track_memory=True,
        )

        # Should include memory usage on CUDA
        assert "mem_mb" in result
        assert result["mem_mb"] >= 0

    def test_profile_forward_backward_module_in_train_mode(self):
        """Test that module is put in train mode."""
        from neuromanifold_gpt.utils.profiling import profile_forward_backward

        module = nn.Linear(5, 5)
        module.eval()  # Start in eval mode

        def make_input():
            return (torch.randn(4, 5),)

        def loss_fn(output):
            return output.mean()

        result = profile_forward_backward(
            "TrainMode",
            module,
            make_input,
            loss_fn,
            n_warmup=1,
            n_iters=2,
            device="cpu",
        )

        assert result["mean_ms"] > 0
        # Module should be in train mode
        assert module.training

    def test_profile_forward_backward_gradients_computed(self):
        """Test that gradients are computed during profiling."""
        from neuromanifold_gpt.utils.profiling import profile_forward_backward

        module = nn.Linear(5, 5)

        def make_input():
            return (torch.randn(4, 5),)

        def loss_fn(output):
            return output.mean()

        # Ensure module has requires_grad
        for param in module.parameters():
            param.requires_grad = True

        result = profile_forward_backward(
            "GradientTest",
            module,
            make_input,
            loss_fn,
            n_warmup=1,
            n_iters=2,
            device="cpu",
        )

        assert result["mean_ms"] > 0
        # Gradients should be zeroed after profiling
        for param in module.parameters():
            if param.grad is not None:
                assert param.grad.abs().sum() == 0

    def test_profile_forward_backward_multiple_inputs(self):
        """Test profiling with multiple input tensors."""
        from neuromanifold_gpt.utils.profiling import profile_forward_backward

        class MultiInputModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(5, 5)
                self.linear2 = nn.Linear(5, 5)

            def forward(self, x, y):
                return self.linear1(x) + self.linear2(y)

        module = MultiInputModule()

        def make_input():
            return (torch.randn(2, 5), torch.randn(2, 5))

        def loss_fn(output):
            return output.mean()

        result = profile_forward_backward(
            "MultiInput",
            module,
            make_input,
            loss_fn,
            n_warmup=1,
            n_iters=2,
            device="cpu",
        )

        assert result["mean_ms"] > 0

    def test_profile_forward_backward_custom_loss(self):
        """Test profiling with custom loss function."""
        from neuromanifold_gpt.utils.profiling import profile_forward_backward

        module = nn.Linear(10, 5)

        def make_input():
            return (torch.randn(4, 10),)

        def custom_loss_fn(output):
            # Custom loss: sum of squares
            return (output**2).sum()

        result = profile_forward_backward(
            "CustomLoss",
            module,
            make_input,
            custom_loss_fn,
            n_warmup=1,
            n_iters=2,
            device="cpu",
        )

        assert result["mean_ms"] > 0


class TestBackwardCompatibilityAliases:
    """Test suite for backward compatibility aliases."""

    def test_profile_module_alias(self):
        """Test that profile_module is an alias for profile_component."""
        from neuromanifold_gpt.utils.profiling import profile_component, profile_module

        # Should be the same function
        assert profile_module is profile_component

    def test_profile_fwd_bwd_alias(self):
        """Test that profile_fwd_bwd is an alias for profile_forward_backward."""
        from neuromanifold_gpt.utils.profiling import (
            profile_forward_backward,
            profile_fwd_bwd,
        )

        # Should be the same function
        assert profile_fwd_bwd is profile_forward_backward

    def test_profile_module_works(self):
        """Test that profile_module alias works correctly."""
        from neuromanifold_gpt.utils.profiling import profile_module

        module = nn.Linear(8, 8)

        def make_input():
            return (torch.randn(4, 8),)

        result = profile_module(
            "AliasTest", module, make_input, n_warmup=1, n_iters=2, device="cpu"
        )

        assert result["name"] == "AliasTest"
        assert result["mean_ms"] > 0

    def test_profile_fwd_bwd_works(self):
        """Test that profile_fwd_bwd alias works correctly."""
        from neuromanifold_gpt.utils.profiling import profile_fwd_bwd

        module = nn.Linear(8, 8)

        def make_input():
            return (torch.randn(4, 8),)

        def loss_fn(output):
            return output.mean()

        result = profile_fwd_bwd(
            "AliasTest",
            module,
            make_input,
            loss_fn,
            n_warmup=1,
            n_iters=2,
            device="cpu",
        )

        assert result["name"] == "AliasTest"
        assert result["mean_ms"] > 0
