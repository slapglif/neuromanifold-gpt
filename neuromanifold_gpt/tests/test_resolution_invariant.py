import pytest
import torch

from neuromanifold_gpt.model.sampling.resolution_invariant import (
    AdaptiveRateEncoder,
    ResolutionInvariantSampler,
)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestResolutionInvariantSampler:
    def test_upsample(self, device):
        sampler = ResolutionInvariantSampler(embed_dim=64, interpolation="linear").to(
            device
        )

        x = torch.randn(2, 100, 64, device=device)
        upsampled = sampler.upsample(x, target_length=200)

        assert upsampled.shape == (2, 200, 64)

    def test_downsample(self, device):
        sampler = ResolutionInvariantSampler(embed_dim=64, interpolation="linear").to(
            device
        )

        x = torch.randn(2, 200, 64, device=device)
        downsampled = sampler.downsample(x, target_length=100)

        assert downsampled.shape == (2, 100, 64)

    def test_resample_upsample(self, device):
        sampler = ResolutionInvariantSampler(embed_dim=64, interpolation="linear").to(
            device
        )

        x = torch.randn(2, 160, 64, device=device)
        resampled = sampler.resample(x, source_rate=16000, target_rate=44100)

        expected_length = int(160 * 44100 / 16000)
        assert resampled.shape == (2, expected_length, 64)

    def test_resample_downsample(self, device):
        sampler = ResolutionInvariantSampler(embed_dim=64, interpolation="linear").to(
            device
        )

        x = torch.randn(2, 441, 64, device=device)
        resampled = sampler.resample(x, source_rate=44100, target_rate=16000)

        expected_length = int(441 * 16000 / 44100)
        assert resampled.shape == (2, expected_length, 64)

    def test_learned_interpolation(self, device):
        sampler = ResolutionInvariantSampler(embed_dim=64, interpolation="learned").to(
            device
        )

        x = torch.randn(2, 100, 64, device=device)
        upsampled = sampler.upsample(x, target_length=200)

        assert upsampled.shape == (2, 200, 64)

    def test_forward_with_rates(self, device):
        sampler = ResolutionInvariantSampler(embed_dim=64, interpolation="linear").to(
            device
        )

        x = torch.randn(2, 100, 64, device=device)
        output = sampler(x, source_rate=16000, target_rate=48000)

        expected_length = int(100 * 48000 / 16000)
        assert output.shape == (2, expected_length, 64)

    def test_forward_with_target_length(self, device):
        sampler = ResolutionInvariantSampler(embed_dim=64, interpolation="linear").to(
            device
        )

        x = torch.randn(2, 100, 64, device=device)
        output = sampler(x, target_length=150)

        assert output.shape == (2, 150, 64)

    def test_identity_sampling(self, device):
        sampler = ResolutionInvariantSampler(embed_dim=64, interpolation="linear").to(
            device
        )

        x = torch.randn(2, 100, 64, device=device)
        output = sampler(x, source_rate=16000, target_rate=16000)

        assert torch.allclose(output, x)

    def test_differentiability(self, device):
        sampler = ResolutionInvariantSampler(embed_dim=64, interpolation="learned").to(
            device
        )

        x = torch.randn(2, 100, 64, device=device, requires_grad=True)
        output = sampler.upsample(x, target_length=200)

        loss = output.sum()
        loss.backward()

        assert x.grad is not None


class TestAdaptiveRateEncoder:
    def test_forward(self, device):
        encoder = AdaptiveRateEncoder(embed_dim=64, num_rates=4).to(device)

        x = torch.randn(2, 100, 64, device=device)
        encoded = encoder(x, rate_id=1)

        assert encoded.shape == x.shape

    def test_different_rates(self, device):
        encoder = AdaptiveRateEncoder(embed_dim=64, num_rates=4).to(device)

        x = torch.randn(2, 100, 64, device=device)
        encoded_0 = encoder(x, rate_id=0)
        encoded_1 = encoder(x, rate_id=1)

        assert not torch.allclose(encoded_0, encoded_1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
