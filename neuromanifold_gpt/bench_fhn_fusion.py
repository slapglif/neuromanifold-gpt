
import torch
import time
from neuromanifold_gpt.model.attention.fhn import FHNDynamics

def benchmark_fhn_fusion():
    """Benchmark Fused Triton vs PyTorch FHN Solver."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("Skipping fusion benchmark (CUDA not available)")
        return

    B, H, T, D = 64, 8, 256, 32
    inputs = torch.randn(B, H, T, D, device=device)
    
    # PyTorch Solver
    fhn_torch = FHNDynamics(D, use_fused=False).to(device)
    
    # Triton Solver
    fhn_fused = FHNDynamics(D, use_fused=True).to(device)
    
    # Warmup
    for _ in range(10):
        fhn_torch(inputs)
        fhn_fused(inputs)
        
    torch.cuda.synchronize()
    
    # Benchmark PyTorch
    start = time.time()
    for _ in range(100):
        fhn_torch(inputs)
    torch.cuda.synchronize()
    time_torch = (time.time() - start) * 1000 / 100
    
    # Benchmark Triton
    start = time.time()
    for _ in range(100):
        fhn_fused(inputs)
    torch.cuda.synchronize()
    time_fused = (time.time() - start) * 1000 / 100
    
    print(f"FHN Solver Benchmark (B={B}, T={T}, D={D})")
    print(f"PyTorch: {time_torch:.3f} ms")
    print(f"Triton:  {time_fused:.3f} ms")
    print(f"Speedup: {time_torch / time_fused:.2f}x")

if __name__ == "__main__":
    benchmark_fhn_fusion()
