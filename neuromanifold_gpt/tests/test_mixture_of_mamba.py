import pytest
import torch
from neuromanifold_gpt.model.ssm.mixture_of_mamba import (
    ModalityRouter,
    MixtureOfMambaBlock,
    AdaptiveMixtureOfMamba,
)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestModalityRouter:
    def test_routing_shape(self, device):
        router = ModalityRouter(embed_dim=128, num_experts=8, top_k=2).to(device)
        x = torch.randn(2, 32, 128).to(device)

        gates, indices, load = router(x)

        assert gates.shape == (2, 32, 2)
        assert indices.shape == (2, 32, 2)
        assert load.shape == (8,)

    def test_gating_properties(self, device):
        router = ModalityRouter(embed_dim=128, num_experts=8, top_k=2).to(device)
        x = torch.randn(2, 32, 128).to(device)

        gates, indices, load = router(x)

        assert (gates >= 0).all()
        assert (gates <= 1).all()
        assert torch.allclose(
            gates.sum(dim=-1), torch.ones(2, 32).to(device), atol=1e-5
        )

    def test_load_distribution(self, device):
        router = ModalityRouter(embed_dim=128, num_experts=8, top_k=2).to(device)
        x = torch.randn(2, 32, 128).to(device)

        gates, indices, load = router(x)

        assert load.sum() > 0
        assert (load >= 0).all()


class TestMixtureOfMambaBlock:
    def test_forward_shape(self, device):
        mom = MixtureOfMambaBlock(
            embed_dim=128, num_experts=4, top_k=2, state_dim=16
        ).to(device)

        x = torch.randn(2, 32, 128).to(device)
        output, info = mom(x)

        assert output.shape == x.shape
        assert "load_balance_loss" in info
        assert "expert_load" in info

    def test_expert_specialization(self, device):
        mom = MixtureOfMambaBlock(
            embed_dim=128, num_experts=8, top_k=2, state_dim=16
        ).to(device)

        x = torch.randn(4, 64, 128).to(device)
        output, info = mom(x)

        assert info["num_experts_used"] > 0
        assert info["num_experts_used"] <= 8

    def test_backward_pass(self, device):
        mom = MixtureOfMambaBlock(
            embed_dim=128, num_experts=4, top_k=2, state_dim=16
        ).to(device)

        x = torch.randn(2, 32, 128, requires_grad=True).to(device)
        output, info = mom(x)

        loss = output.mean()
        loss.backward()

        params_with_grad = [p for p in mom.parameters() if p.grad is not None]
        assert len(params_with_grad) > 0

        for p in params_with_grad:
            assert not torch.isnan(p.grad).any()


class TestAdaptiveMixtureOfMamba:
    def test_forward_with_spectral_routing(self, device):
        amom = AdaptiveMixtureOfMamba(
            embed_dim=128, num_experts=4, top_k=2, use_spectral_routing=True
        ).to(device)

        x = torch.randn(2, 32, 128).to(device)
        output, info = amom(x)

        assert output.shape == x.shape
        assert "modality_distribution" in info
        assert info["modality_distribution"].shape == (3,)

    def test_modality_detection(self, device):
        amom = AdaptiveMixtureOfMamba(
            embed_dim=256, num_experts=8, top_k=2, use_spectral_routing=True
        ).to(device)

        text_like = torch.randn(1, 64, 256).to(device) * 0.1
        audio_like = torch.randn(1, 64, 256).to(device) * 2.0

        _, text_info = amom(text_like)
        _, audio_info = amom(audio_like)

        assert "modality_distribution" in text_info
        assert "modality_distribution" in audio_info

    def test_without_spectral_routing(self, device):
        amom = AdaptiveMixtureOfMamba(
            embed_dim=128, num_experts=4, top_k=2, use_spectral_routing=False
        ).to(device)

        x = torch.randn(2, 32, 128).to(device)
        output, info = amom(x)

        assert output.shape == x.shape
        assert "modality_distribution" not in info


class TestMoMIntegration:
    def test_multimodal_batch_processing(self, device):
        mom = AdaptiveMixtureOfMamba(
            embed_dim=256, num_experts=8, top_k=2, use_spectral_routing=True
        ).to(device)

        batch_size = 4
        seq_len = 128
        x = torch.randn(batch_size, seq_len, 256).to(device)

        output, info = mom(x)

        assert output.shape == (batch_size, seq_len, 256)
        assert info["expert_load"].shape == (8,)
        assert (info["expert_load"] > 0).sum() >= 2

    def test_gradient_flow_through_experts(self, device):
        mom = MixtureOfMambaBlock(
            embed_dim=128, num_experts=4, top_k=2, state_dim=16
        ).to(device)

        x = torch.randn(2, 32, 128, requires_grad=True).to(device)
        output, info = mom(x)

        loss = output.mean() + info["load_balance_loss"]
        loss.backward()

        expert_grads = [
            p.grad
            for expert in mom.experts
            for p in expert.parameters()
            if p.grad is not None
        ]

        assert len(expert_grads) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
