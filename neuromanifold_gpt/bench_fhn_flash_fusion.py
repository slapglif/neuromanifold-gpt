import torch
import time
from neuromanifold_gpt.model.attention.fhn import FHNAttention

def benchmark_fhn_flash_fusion():
    """Benchmark Flash Attention + FHN Fusion vs Manual Attention + FHN."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("Skipping Flash fusion benchmark (CUDA not available)")
        return

    B, T, D = 32, 512, 384
    H = 8

    # Input tensors
    x = torch.randn(B, T, D, device=device, requires_grad=True)
    spec = torch.randn(B, T, 32, device=device)

    # Baseline: Flash Attention only (n_fhn_steps=0)
    attn_baseline = FHNAttention(D, H, n_fhn_steps=0).to(device)

    # Old path: Manual Attention + FHN (use_flash_fhn_fusion=False)
    attn_manual = FHNAttention(D, H, n_fhn_steps=2, use_flash_fhn_fusion=False).to(device)

    # New path: Flash Attention + Output Modulation (use_flash_fhn_fusion=True)
    attn_flash = FHNAttention(D, H, n_fhn_steps=2, use_flash_fhn_fusion=True).to(device)

    # Warmup
    for _ in range(10):
        out_baseline, _ = attn_baseline(x.clone(), spec)
        out_manual, _ = attn_manual(x.clone(), spec)
        out_flash, _ = attn_flash(x.clone(), spec)

    torch.cuda.synchronize()

    # Benchmark Baseline (Flash only, no FHN)
    start = time.time()
    for _ in range(100):
        x_baseline = x.clone()
        x_baseline.grad = None
        out_baseline, _ = attn_baseline(x_baseline, spec)
        loss = out_baseline.sum()
        loss.backward()
    torch.cuda.synchronize()
    time_baseline = (time.time() - start) * 1000 / 100

    # Benchmark Manual Attention + FHN (old path)
    start = time.time()
    for _ in range(100):
        x_manual = x.clone()
        x_manual.grad = None
        out_manual, _ = attn_manual(x_manual, spec)
        loss = out_manual.sum()
        loss.backward()
    torch.cuda.synchronize()
    time_manual = (time.time() - start) * 1000 / 100

    # Benchmark Flash Attention + Output Modulation (new path)
    start = time.time()
    for _ in range(100):
        x_flash = x.clone()
        x_flash.grad = None
        out_flash, _ = attn_flash(x_flash, spec)
        loss = out_flash.sum()
        loss.backward()
    torch.cuda.synchronize()
    time_flash = (time.time() - start) * 1000 / 100

    print(f"\nFHN Flash Fusion Benchmark (B={B}, T={T}, D={D}, H={H})")
    print(f"{'='*60}")
    print(f"Baseline (Flash only, no FHN):     {time_baseline:.3f} ms")
    print(f"Manual Attention + FHN (old):      {time_manual:.3f} ms")
    print(f"Flash Attention + FHN Fusion (new): {time_flash:.3f} ms")
    print(f"{'='*60}")
    print(f"Speedup (new vs old):              {time_manual / time_flash:.2f}x")
    print(f"Speedup (new vs baseline):         {time_baseline / time_flash:.2f}x")
    print(f"Overhead of old FHN:               {time_manual / time_baseline:.2f}x")
    print(f"Overhead of new FHN:               {time_flash / time_baseline:.2f}x")

if __name__ == "__main__":
    benchmark_fhn_flash_fusion()
