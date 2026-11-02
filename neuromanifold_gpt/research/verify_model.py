# verify_model.py
import torch

from neuromanifold_gpt.config import NeuroManifoldConfigNano
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT


def test_model_forward():
    print("Initializing Nano config...")
    config = NeuroManifoldConfigNano()

    print("Initializing NeuroManifoldGPT model...")
    print(
        f"Config: FHN={config.fhn_tau}, KAN={config.use_kan}, Degree={config.kan_degree}"
    )
    model = NeuroManifoldGPT(config)
    print(f"Model parameters: {model.num_parameters()/1e6:.2f}M")

    # Create dummy input
    B, T = 2, 64
    x = torch.randint(0, config.vocab_size, (B, T))

    print("Running forward pass...")
    logits, loss, info = model(x)

    print(f"Logits shape: {logits.shape}")
    print("Forward pass successful!")

    # Check if KAN was used
    has_kan = False
    for name, mod in model.named_modules():
        if "ChebyKAN" in str(type(mod)):
            has_kan = True
            break

    if has_kan:
        print("✅ ChebyKAN detected in model hierarchy")
    else:
        print("❌ ChebyKAN NOT detected (SwiGLU used?)")

    # Check causality in spectral attention
    # We can't easily check gradient flow here without backward,
    # but we can check if it runs without error.


if __name__ == "__main__":
    test_model_forward()
