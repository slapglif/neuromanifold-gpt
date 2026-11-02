from neuromanifold_gpt.config.base import NeuroManifoldConfig


def test_char_sparsity():
    # Character level often needs higher resolution (more bits) or specific context size
    # to differentiate small sets of patterns.

    # 1. Standard "Biological" Sparsity (2%)
    cfg_standard = NeuroManifoldConfig(
        sdr_size=2048, sdr_sparsity=0.02, sdr_context_size=5
    )

    # 2. Increased Sparsity (1%) for Char-level
    # We want to increase RESOLUTION by making each bit represent a more specific pattern.
    # Often for char-level, we increase sdr_size OR decrease sparsity.
    cfg_char = NeuroManifoldConfig(
        sdr_size=4096, sdr_sparsity=0.01, sdr_context_size=10
    )

    print(
        f"Standard: bits={cfg_standard.sdr_size}, active={cfg_standard.sdr_n_active}, sparsity={cfg_standard.sdr_sparsity}"
    )
    print(
        f"Char-Optim: bits={cfg_char.sdr_size}, active={cfg_char.sdr_n_active}, sparsity={cfg_char.sdr_sparsity}"
    )

    # Verify sdr_n_active computation
    assert cfg_char.sdr_n_active == 40  # 4096 * 0.01 = 40.96 -> 40

    # Check Context Window impact on characters
    # chars need larger window to see full words? Or smaller to see local n-grams?
    # Usually sdr_context_size=10 sees 21 characters (~3-4 words).
    print(f"Char Context Window: {2*cfg_char.sdr_context_size + 1} chars")


if __name__ == "__main__":
    test_char_sparsity()
