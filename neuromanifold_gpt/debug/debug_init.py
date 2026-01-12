import time
import torch
from neuromanifold_gpt.config.base import NeuroManifoldConfig
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT

print("Setting up config...")
config = NeuroManifoldConfig(
    vocab_size=65,
    block_size=256,
    n_layer=4,
    n_heads=4,
    n_embd=128,
    manifold_dim=32,
    n_eigenvectors=16,
    use_sdr=True,
    sdr_size=1024,
    use_kan=True,
    kan_type="faster",
    kan_num_centers=3,
    use_mhc=True,
    use_full_mhc=True,
    mhc_n_streams=2,
)

print("Initializing model...")
start = time.time()
model = NeuroManifoldGPT(config)
print(f"Model init took {time.time() - start:.4f}s")

print("Moving to CUDA...")
start = time.time()
model.to("cuda")
print(f"Move to CUDA took {time.time() - start:.4f}s")

print("Checking parameter count...")
print(f"Params: {sum(p.numel() for p in model.parameters())}")
