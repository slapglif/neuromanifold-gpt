
import torch
from neuromanifold_gpt.model.block import NeuroManifoldBlock
from neuromanifold_gpt.config.base import NeuroManifoldConfigNano
from neuromanifold_gpt.config.block_config import NeuroManifoldBlockConfig

def test_full_block_integration():
    """Smoke test for NeuroManifoldBlock with WaveKAN."""
    print("Initializing NeuroManifoldBlock with WaveKAN...")

    # Create model config
    model_config = NeuroManifoldConfigNano()
    model_config.use_kan = True
    model_config.kan_type = "wave"
    model_config.kan_wavelet = "dog"
    model_config.use_fast_wavekan = True

    # Create block config from model config
    block_config = NeuroManifoldBlockConfig.from_model_config(model_config, layer_idx=0)

    # Initialize block with config object
    block = NeuroManifoldBlock(config=block_config)
    
    # Create input SDR (Batch, Time, SDR_Size)
    B, T, S = 2, 64, block_config.sdr_size
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
