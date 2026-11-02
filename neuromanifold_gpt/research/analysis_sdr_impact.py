import torch

from neuromanifold_gpt.config.base import NeuroManifoldConfig
from neuromanifold_gpt.model.semantic_folding import SemanticFoldingEncoder


def analyze_parameters():
    print("--- SDR Parameter Analysis ---")

    # 1. Base Config
    cfg_base = NeuroManifoldConfig()
    print(
        f"Base: sdr_size={cfg_base.sdr_size}, sdr_sparsity={cfg_base.sdr_sparsity}, sdr_n_active={cfg_base.sdr_n_active}, context_size={cfg_base.sdr_context_size}"
    )

    # 2. Impact of Increasing Sparsity (Decreasing sdr_sparsity ratio)
    # If we want MORE sparsity, sdr_sparsity goes DOWN (e.g., 0.02 -> 0.01)
    # This means FEWER active bits (sdr_n_active goes down)
    cfg_sparse = NeuroManifoldConfig(sdr_sparsity=0.01)
    print(
        f"High Sparsity: sdr_sparsity={cfg_sparse.sdr_sparsity}, sdr_n_active={cfg_sparse.sdr_n_active}"
    )

    # 3. Context Size Impact
    # context_size affects the local attention mask in ContextEncoder
    # Window width = 2 * context_size + 1
    # For context_size=5, window=11. For context_size=1, window=3.
    print(f"Context Window (size=5): {2*cfg_base.sdr_context_size + 1}")

    # 4. Encoder Behavior
    encoder = SemanticFoldingEncoder(
        vocab_size=100,
        sdr_size=cfg_base.sdr_size,
        n_active=cfg_base.sdr_n_active,
        context_size=cfg_base.sdr_context_size,
    )

    # Simulate forward pass
    tokens = torch.randint(0, 100, (1, 10))
    sdr, scores = encoder(tokens)

    print(f"SDR output shape: {sdr.shape}")
    actual_active = sdr.sum(dim=-1).unique()
    print(f"Actual active bits per token: {actual_active.tolist()}")

    # Similarity test
    sim = encoder.semantic_similarity(sdr[:, 0:1], sdr[:, 1:2])
    print(f"Sample similarity between token 0 and 1: {sim.item():.4f}")


if __name__ == "__main__":
    analyze_parameters()
