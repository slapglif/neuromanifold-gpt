import pytest
import torch
import torch.nn as nn

from neuromanifold_gpt.profiling.complexity import ComplexityProfiler, analyze_scaling
from neuromanifold_gpt.training.checkpointing import (
    GradientAccumulator,
    GradientCheckpointing,
    SequenceChunker,
)


class SimpleModel(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_complexity_profiler_memory(device):
    profiler = ComplexityProfiler(device=device)
    model = SimpleModel(64).to(device)
    x = torch.randn(2, 100, 64).to(device)

    memory = profiler.profile_memory(model, x)
    assert memory >= 0


def test_complexity_profiler_time(device):
    profiler = ComplexityProfiler(device=device)
    model = SimpleModel(64).to(device)
    x = torch.randn(2, 100, 64).to(device)

    time_ms = profiler.profile_time(model, x, num_iterations=10)
    assert time_ms > 0


def test_complexity_profiler_params():
    model = SimpleModel(64)
    profiler = ComplexityProfiler()

    params = profiler.count_params(model)
    assert params == 64 * 64 + 64


def test_gradient_accumulator():
    model = SimpleModel(64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    accumulator = GradientAccumulator(model, optimizer, accumulation_steps=4)

    for i in range(4):
        x = torch.randn(2, 10, 64)
        y = model(x)
        loss = y.sum()
        accumulator.step(loss)

    assert accumulator.current_step == 4


def test_sequence_chunker():
    chunker = SequenceChunker(chunk_size=100, overlap=10)
    x = torch.randn(2, 250, 64)

    chunks = chunker.chunk(x)
    assert len(chunks) == 3
    assert chunks[0].shape[1] == 100


def test_sequence_chunker_merge():
    chunker = SequenceChunker(chunk_size=100, overlap=10)
    x = torch.randn(2, 250, 64)

    chunks = chunker.chunk(x)
    merged = chunker.merge(chunks, original_length=250)

    assert merged.shape == x.shape


def test_gradient_checkpointing():
    module = SimpleModel(64)
    checkpointed = GradientCheckpointing(module)

    checkpointed.train()
    x = torch.randn(2, 100, 64, requires_grad=True)

    y = checkpointed(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
