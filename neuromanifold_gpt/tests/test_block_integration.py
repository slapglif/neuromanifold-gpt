
import torch
from neuromanifold_gpt.model.block import NeuroManifoldBlock
from neuromanifold_gpt.config.base import NeuroManifoldConfigNano

def test_full_block_integration():
    """Smoke test for NeuroManifoldBlock with WaveKAN."""
    print("Initializing NeuroManifoldBlock with WaveKAN...")
    
    # Configure for WaveKAN
    config = NeuroManifoldConfigNano()
    config.use_kan = True
    config.kan_type = "wave"
    config.kan_wavelet = "dog"
    config.use_fast_wavekan = True
    
    # Initialize block
    block = NeuroManifoldBlock(
        sdr_size=config.sdr_size,
        embed_dim=config.n_embd,
        manifold_dim=config.manifold_dim,
        n_eigenvectors=config.n_eigenvectors,
        n_heads=config.n_heads,
        dropout=config.dropout,
        fhn_threshold=config.fhn_threshold,
        fhn_tau=config.fhn_tau,
        pulse_width_base=config.pulse_width_base,
        n_fhn_steps=config.n_fhn_steps,
        use_fhn_imex=config.use_fhn_imex,
        use_kan=config.use_kan,
        kan_type=config.kan_type,
        kan_degree=config.kan_degree,
        kan_wavelet=config.kan_wavelet,
        use_fast_wavekan=config.use_fast_wavekan
    )
    
    # Create input SDR (Batch, Time, SDR_Size)
    B, T, S = 2, 64, config.sdr_size
    sdr = torch.randn(B, T, S)
    
    print(f"Forward pass with input shape {sdr.shape}...")
    output, info = block(sdr)
    
    print(f"Output shape: {output.shape}")
    print(f"Info keys: {info.keys()}")
    
    # Check for NaNs
    assert not torch.isnan(output).any(), "NaNs in output"
    print("Forward pass successful (no NaNs)")
    
    # Backward pass check
    print("Testing backward pass...")
    loss = output.sum()
    loss.backward()
    
    # Check gradients
    has_grads = True
    for name, param in block.named_parameters():
        if param.requires_grad and param.grad is None:
            print(f"WARNING: No gradient for {name}")
            has_grads = False
            
    if has_grads:
        print("Backward pass successful (gradients flow)")
    
    print("\nIntegration test PASSED!")

if __name__ == "__main__":
    test_full_block_integration()
