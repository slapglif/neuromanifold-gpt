import pytest
import torch

from neuromanifold_gpt.data.synthetic_solitons import SolitonDataset
from neuromanifold_gpt.experiments.ablation import AblationConfig, AblationStudy
from neuromanifold_gpt.inference.optimization import (
    ContinuousStateCache,
    StreamingProcessor,
)


def test_ablation_study():
    study = AblationStudy({"lr": 0.001})
    study.generate_standard_ablations()

    assert len(study.ablation_configs) == 7
    assert study.ablation_configs[0].name == "baseline"
    assert study.ablation_configs[-1].name == "minimal"


def test_soliton_dataset_sine_gordon():
    dataset = SolitonDataset(num_samples=10, seq_length=64, spatial_dim=32)
    soliton = dataset.generate_sine_gordon_soliton()

    assert soliton.shape == (64, 32)


def test_soliton_dataset_kdv():
    dataset = SolitonDataset(num_samples=10, seq_length=64, spatial_dim=32)
    soliton = dataset.generate_kdv_soliton()

    assert soliton.shape == (64, 32)


def test_soliton_collision():
    dataset = SolitonDataset(num_samples=10)
    collision, metadata = dataset.generate_collision()

    assert collision.shape == (256, 64)
    assert metadata["elastic"]
    assert metadata["num_solitons"] == 2


def test_soliton_dataset_getitem():
    dataset = SolitonDataset(num_samples=10)
    collision, metadata = dataset[0]

    assert collision.shape == (1, 256, 64)


def test_continuous_state_cache():
    cache = ContinuousStateCache(max_length=100, embed_dim=64)

    state1 = torch.randn(2, 30, 64)
    result1 = cache.update(state1)
    assert result1.shape == (2, 30, 64)

    state2 = torch.randn(2, 30, 64)
    result2 = cache.update(state2)
    assert result2.shape == (2, 60, 64)


def test_cache_overflow():
    cache = ContinuousStateCache(max_length=100, embed_dim=64)

    for _ in range(5):
        state = torch.randn(2, 30, 64)
        cache.update(state)

    cached = cache.get()
    assert cached.shape[1] <= 100


def test_streaming_processor():
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return x

    model = DummyModel()
    processor = StreamingProcessor(model, chunk_size=32)

    chunk = torch.randn(2, 32, 64)
    output = processor.process_chunk(chunk)

    assert output.shape == chunk.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
