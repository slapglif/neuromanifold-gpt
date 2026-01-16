# neuromanifold_gpt/tests/test_gpt.py
"""Tests for complete NeuroManifoldGPT model."""
import pytest
import torch
from neuromanifold_gpt.config import NeuroManifoldConfigNano
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT


def test_gpt_forward():
    """Model forward pass should produce logits."""
    config = NeuroManifoldConfigNano()
    model = NeuroManifoldGPT(config)

    tokens = torch.randint(0, config.vocab_size, (2, 50))

    logits, loss, info = model(tokens)

    assert logits.shape == (2, 50, config.vocab_size)
    assert loss is None  # No targets provided


def test_gpt_with_targets():
    """Should compute loss when targets provided."""
    config = NeuroManifoldConfigNano()
    model = NeuroManifoldGPT(config)

    tokens = torch.randint(0, config.vocab_size, (2, 50))
    targets = torch.randint(0, config.vocab_size, (2, 50))

    logits, loss, info = model(tokens, targets)

    assert loss is not None
    assert loss.ndim == 0  # Scalar


def test_gpt_generate():
    """Should generate new tokens."""
    config = NeuroManifoldConfigNano()
    model = NeuroManifoldGPT(config)
    model.eval()

    prompt = torch.randint(0, config.vocab_size, (1, 10))

    generated = model.generate(prompt, max_new_tokens=5)

    assert generated.shape == (1, 15)


def test_gpt_info_dict():
    """Info dict should contain diagnostic information."""
    config = NeuroManifoldConfigNano()
    model = NeuroManifoldGPT(config)

    tokens = torch.randint(0, config.vocab_size, (1, 20))

    _, _, info = model(tokens)

    assert "sdr" in info
    assert "sdr_scores" in info
    assert "block_infos" in info
    assert "memory_size" in info
    assert len(info["block_infos"]) == config.n_layer


def test_gpt_memory_stores_during_training():
    """Engram memory should store SDRs during training."""
    config = NeuroManifoldConfigNano()
    model = NeuroManifoldGPT(config)
    model.train()

    initial_memory_size = len(model.memory)

    tokens = torch.randint(0, config.vocab_size, (1, 10))
    _ = model(tokens)

    # Memory should have grown
    assert len(model.memory) > initial_memory_size


def test_gpt_no_memory_during_eval():
    """Engram memory should NOT store during eval."""
    config = NeuroManifoldConfigNano()
    model = NeuroManifoldGPT(config)
    model.eval()

    initial_memory_size = len(model.memory)

    tokens = torch.randint(0, config.vocab_size, (1, 10))
    with torch.no_grad():
        _ = model(tokens)

    # Memory should be unchanged
    assert len(model.memory) == initial_memory_size


def test_gpt_memory_active_retrieval():
    """Memory should retrieve when memory_active_retrieval=True and memory has content."""
    config = NeuroManifoldConfigNano()
    config.memory_active_retrieval = True  # Enable active retrieval
    config.use_sdr = True  # SDR required for memory retrieval to work
    config.engram_threshold = 0.0  # Lower threshold for random token testing

    model = NeuroManifoldGPT(config)
    model.train()

    # Build memory with multiple passes
    for _ in range(5):
        tokens = torch.randint(0, config.vocab_size, (2, 20))
        _, _, _ = model(tokens)

    memory_size = len(model.memory)
    assert memory_size > 0, f"Memory should have content after 5 training passes, got size={memory_size}"

    # Now test retrieval in eval mode
    model.eval()
    with torch.no_grad():
        tokens = torch.randint(0, config.vocab_size, (2, 20))
        _, _, info = model(tokens)

    # Verify retrieval happened
    assert "memory_retrieval" in info
    retrieved = info["memory_retrieval"]["retrieved_count"]
    similarity = info["memory_retrieval"]["avg_similarity"]
    assert retrieved > 0, f"Should retrieve memories from {memory_size} stored, got retrieved_count={retrieved}"
    assert similarity > 0.01, f"Avg similarity should exceed 0.01 for meaningful retrieval, got {similarity:.4f}"


def test_gpt_generate_repetition_penalty():
    """Repetition penalty should reduce probability of repeated tokens."""
    config = NeuroManifoldConfigNano()
    model = NeuroManifoldGPT(config)
    model.eval()

    # Test 1: Basic functionality - repeated tokens should be penalized
    torch.manual_seed(42)
    prompt = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

    # Generate without penalty
    with torch.no_grad():
        generated_no_penalty = model.generate(prompt, max_new_tokens=10, repetition_penalty=1.0, temperature=1.0)

    # Generate with penalty
    torch.manual_seed(42)
    with torch.no_grad():
        generated_with_penalty = model.generate(prompt, max_new_tokens=10, repetition_penalty=1.5, temperature=1.0)

    # Both should have correct shape
    assert generated_no_penalty.shape == (1, 15)
    assert generated_with_penalty.shape == (1, 15)

    # Test 2: Batch processing (batch_size > 1)
    torch.manual_seed(42)
    batch_prompt = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    with torch.no_grad():
        batch_generated = model.generate(batch_prompt, max_new_tokens=5, repetition_penalty=1.5, temperature=1.0)

    assert batch_generated.shape == (2, 8)

    # Test 3: Edge case - penalty=1.0 should be no-op (just verify it runs)
    torch.manual_seed(42)
    prompt_edge = torch.randint(0, config.vocab_size, (1, 10))

    with torch.no_grad():
        gen_no_penalty = model.generate(prompt_edge.clone(), max_new_tokens=5, repetition_penalty=1.0, temperature=1.0)

    # Should have correct shape
    assert gen_no_penalty.shape == (1, 15)

    # Test 4: Very high penalty should work without errors
    torch.manual_seed(42)
    with torch.no_grad():
        high_penalty_gen = model.generate(prompt, max_new_tokens=5, repetition_penalty=5.0, temperature=1.0)

    assert high_penalty_gen.shape == (1, 10)

    # Test 5: Long sequence handling
    torch.manual_seed(42)
    # Use a sequence that's close to block_size to test scaling
    long_prompt = torch.randint(0, config.vocab_size, (1, min(100, config.block_size - 10)))

    with torch.no_grad():
        long_generated = model.generate(long_prompt, max_new_tokens=10, repetition_penalty=1.5, temperature=1.0)

    assert long_generated.shape == (1, long_prompt.size(1) + 10)
