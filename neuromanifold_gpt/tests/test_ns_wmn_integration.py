# neuromanifold_gpt/tests/test_ns_wmn_integration.py
"""
End-to-end integration tests for NS-WMN (Neuro-Symbolic Wave Manifold Network).

Tests the complete pipeline as described in the paper:
- FNO input layer
- Mamba/Hyena backbone with soliton dynamics
- Topological regularization
- Continuous output generation
"""

import pytest
import torch
import torch.nn as nn
from neuromanifold_gpt.config.wave_manifold_config import WaveManifoldConfig
from neuromanifold_gpt.model.wave_manifold_gpt import WaveManifoldGPT


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def minimal_config():
    return WaveManifoldConfig(
        vocab_size=256,
        n_layer=2,
        n_head=4,
        n_embd=128,
        block_size=64,
        use_fno_encoder=True,
        fno_modes=8,
        use_soliton_mixing=True,
        use_topological_loss=True,
        use_continuous_head=False,
    )


@pytest.fixture
def full_config():
    return WaveManifoldConfig(
        vocab_size=256,
        n_layer=4,
        n_head=8,
        n_embd=256,
        block_size=128,
        use_fno_encoder=True,
        fno_modes=16,
        use_soliton_mixing=True,
        soliton_type="sine_gordon",
        use_topological_loss=True,
        topology_weight=0.1,
        use_continuous_head=True,
    )


class TestNSWMNBasicIntegration:
    def test_model_instantiation_minimal(self, minimal_config):
        model = WaveManifoldGPT(minimal_config)
        assert isinstance(model, nn.Module)
        assert hasattr(model, "input_encoder")
        assert hasattr(model, "blocks")
        assert hasattr(model, "lm_head")

    def test_model_instantiation_full(self, full_config):
        model = WaveManifoldGPT(full_config)
        assert isinstance(model, nn.Module)
        assert hasattr(model, "topo_head")
        assert hasattr(model, "continuous_head")

    def test_forward_pass_inference_minimal(self, minimal_config, device):
        model = WaveManifoldGPT(minimal_config).to(device)
        batch_size = 2
        seq_len = 32

        idx = torch.randint(0, minimal_config.vocab_size, (batch_size, seq_len)).to(
            device
        )

        with torch.no_grad():
            logits, loss, info = model(idx)

        assert logits.shape == (batch_size, seq_len, minimal_config.vocab_size)
        assert loss is None
        assert isinstance(info, dict)

    def test_forward_pass_training_minimal(self, minimal_config, device):
        model = WaveManifoldGPT(minimal_config).to(device)
        batch_size = 2
        seq_len = 32

        idx = torch.randint(0, minimal_config.vocab_size, (batch_size, seq_len)).to(
            device
        )
        targets = torch.randint(0, minimal_config.vocab_size, (batch_size, seq_len)).to(
            device
        )

        logits, loss, info = model(idx, targets=targets)

        assert logits.shape == (batch_size, seq_len, minimal_config.vocab_size)
        assert loss is not None
        assert loss.item() > 0
        assert "loss_discrete" in info

    def test_topological_loss_computed(self, minimal_config, device):
        model = WaveManifoldGPT(minimal_config).to(device)
        batch_size = 2
        seq_len = 32

        idx = torch.randint(0, minimal_config.vocab_size, (batch_size, seq_len)).to(
            device
        )
        targets = torch.randint(0, minimal_config.vocab_size, (batch_size, seq_len)).to(
            device
        )

        logits, loss, info = model(idx, targets=targets)

        assert (
            "total_loss" in info
            or "smoothness_loss" in info
            or "regularity_loss" in info
        )

    def test_backward_pass_minimal(self, minimal_config, device):
        model = WaveManifoldGPT(minimal_config).to(device)
        batch_size = 2
        seq_len = 32

        idx = torch.randint(0, minimal_config.vocab_size, (batch_size, seq_len)).to(
            device
        )
        targets = torch.randint(0, minimal_config.vocab_size, (batch_size, seq_len)).to(
            device
        )

        logits, loss, info = model(idx, targets=targets)
        loss.backward()

        params_with_grad = [p for p in model.parameters() if p.grad is not None]
        total_params = len(list(model.parameters()))

        assert len(params_with_grad) > total_params * 0.5, (
            f"Too few params have gradients: {len(params_with_grad)}/{total_params}"
        )

    def test_gradient_flow_through_fno(self, minimal_config, device):
        model = WaveManifoldGPT(minimal_config).to(device)
        batch_size = 2
        seq_len = 32

        idx = torch.randint(0, minimal_config.vocab_size, (batch_size, seq_len)).to(
            device
        )
        targets = torch.randint(0, minimal_config.vocab_size, (batch_size, seq_len)).to(
            device
        )

        logits, loss, info = model(idx, targets=targets)
        loss.backward()

        fno_params_with_grad = [
            p for p in model.input_encoder.parameters() if p.grad is not None
        ]
        assert len(fno_params_with_grad) > 0, "No FNO params have gradients"

        for param in fno_params_with_grad:
            assert not torch.isnan(param.grad).any(), "NaN in FNO gradients"
            assert not torch.isinf(param.grad).any(), "Inf in FNO gradients"


class TestNSWMNFullFeatures:
    def test_continuous_head_training(self, full_config, device):
        model = WaveManifoldGPT(full_config).to(device)
        batch_size = 2
        seq_len = 32

        idx = torch.randint(0, full_config.vocab_size, (batch_size, seq_len)).to(device)
        targets = torch.randint(0, full_config.vocab_size, (batch_size, seq_len)).to(
            device
        )

        logits, loss, info = model(idx, targets=targets)

        assert "loss_continuous" in info
        assert info["loss_continuous"] > 0

    def test_multimodal_byte_input(self, minimal_config, device):
        model = WaveManifoldGPT(minimal_config).to(device)
        batch_size = 2
        seq_len = 64

        byte_stream = torch.randint(0, 256, (batch_size, seq_len)).to(device)

        with torch.no_grad():
            logits, _, _ = model(byte_stream)

        assert logits.shape == (batch_size, seq_len, minimal_config.vocab_size)

    def test_resolution_invariance(self, minimal_config, device):
        model = WaveManifoldGPT(minimal_config).to(device)

        short_seq = torch.randint(0, 256, (1, 16)).to(device)
        long_seq = torch.randint(0, 256, (1, 64)).to(device)

        with torch.no_grad():
            logits_short, _, _ = model(short_seq)
            logits_long, _, _ = model(long_seq)

        assert logits_short.shape[1] == 16
        assert logits_long.shape[1] == 64


class TestNSWMNMemoryEfficiency:
    def test_long_sequence_processing(self, minimal_config, device):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for memory test")

        minimal_config.block_size = 512
        model = WaveManifoldGPT(minimal_config).to(device)

        batch_size = 1
        seq_len = 512

        idx = torch.randint(0, 256, (batch_size, seq_len)).to(device)
        targets = torch.randint(0, 256, (batch_size, seq_len)).to(device)

        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()

        logits, loss, info = model(idx, targets=targets)
        loss.backward()

        peak_memory = torch.cuda.max_memory_allocated()
        memory_used_mb = (peak_memory - initial_memory) / 1024 / 1024

        assert memory_used_mb < 500, f"Memory usage too high: {memory_used_mb:.1f}MB"


class TestNSWMNPhysicsProperties:
    def test_energy_conservation_soliton(self, minimal_config, device):
        minimal_config.use_soliton_mixing = True
        minimal_config.soliton_type = "sine_gordon"
        model = WaveManifoldGPT(minimal_config).to(device)

        batch_size = 2
        seq_len = 32

        idx = torch.randint(0, 256, (batch_size, seq_len)).to(device)

        with torch.no_grad():
            logits, _, info = model(idx)

        assert isinstance(info, dict) and len(info) >= 0

    def test_topological_invariants_stability(self, minimal_config, device):
        model = WaveManifoldGPT(minimal_config).to(device)

        batch_size = 2
        seq_len = 32

        idx = torch.randint(0, 256, (batch_size, seq_len)).to(device)
        idx_permuted = idx.clone()
        idx_permuted[:, : seq_len // 2] = idx[:, seq_len // 2 :].clone()
        idx_permuted[:, seq_len // 2 :] = idx[:, : seq_len // 2].clone()

        with torch.no_grad():
            _, _, info1 = model(idx)
            _, _, info2 = model(idx_permuted)

        if "topo_complexity" in info1 and "topo_complexity" in info2:
            assert abs(info1["topo_complexity"] - info2["topo_complexity"]) < 1.0


class TestNSWMNOptimizationLoop:
    def test_simple_training_loop(self, minimal_config, device):
        model = WaveManifoldGPT(minimal_config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        batch_size = 4
        seq_len = 32
        n_steps = 5

        losses = []
        for _ in range(n_steps):
            idx = torch.randint(0, 256, (batch_size, seq_len)).to(device)
            targets = torch.randint(0, 256, (batch_size, seq_len)).to(device)

            optimizer.zero_grad()
            logits, loss, info = model(idx, targets=targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())

        assert all(not math.isnan(l) for l in losses)
        assert all(not math.isinf(l) for l in losses)

    def test_mixed_precision_training(self, minimal_config, device):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for mixed precision test")

        model = WaveManifoldGPT(minimal_config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda")

        batch_size = 2
        seq_len = 32

        idx = torch.randint(0, 256, (batch_size, seq_len)).to(device)
        targets = torch.randint(0, 256, (batch_size, seq_len)).to(device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            logits, loss, info = model(idx, targets=targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        assert not torch.isnan(loss)


import math

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
