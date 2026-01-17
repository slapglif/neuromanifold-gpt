import torch
from neuromanifold_gpt.model.neuro_symbolic_wave import NeuroSymbolicWaveNet

def test_neuro_symbolic_wave_init_and_forward():
    print("Initializing NeuroSymbolicWaveNet...")
    vocab_size = 1000
    matrix_dim = 16
    depth = 2
    num_heads = 4
    seq_len = 16
    
    model = NeuroSymbolicWaveNet(
        vocab_size=vocab_size,
        matrix_dim=matrix_dim,
        depth=depth,
        num_heads=num_heads,
        max_seq_len=seq_len
    )
    print("Model initialized.")
    
    input_ids = torch.randint(0, vocab_size, (2, seq_len))
    print(f"Input shape: {input_ids.shape}")
    
    output = model(input_ids)
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (2, seq_len, vocab_size)
    assert not torch.isnan(output).any(), "NaNs in output"
    print("Verification successful: Forward pass complete without errors.")

if __name__ == "__main__":
    test_neuro_symbolic_wave_init_and_forward()
