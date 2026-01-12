#!/usr/bin/env python3
"""
Simple FHN test without JIT to compare.
"""

import time
import torch
from rich.console import Console

console = Console()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def fhn_update_no_jit(v, w, I, a, b, tau, dt, n_steps):
    """Plain Python FHN update - no JIT."""
    dt_val = dt.item()
    a_val = a.item()
    b_val = b.item()

    for _ in range(n_steps):
        # dv = v - v^3/3 - w + I
        v3 = v * v * v
        dv = v - (v3 / 3.0) - w + I

        # Update v
        v_next = v + dt_val * dv

        # Update w (implicit)
        alpha = dt_val / tau
        denom = 1.0 + alpha * b_val
        w_next = (w + alpha * (v_next + a_val)) / denom

        # Clamp
        v = torch.clamp(v_next, -3.0, 3.0)
        w = torch.clamp(w_next, -3.0, 3.0)

    return v, w


def main():
    console.print(f"[bold]Simple FHN Comparison[/bold]")
    console.print(f"Device: {DEVICE}")

    # Small test
    B, H, K, D = 4, 8, 32, 48  # Reasonable size
    console.print(f"Shape: ({B}, {H}, {K}, {D}) = {B*H*K*D} elements\n")

    v = torch.zeros(B, H, K, D, device=DEVICE)
    w = torch.zeros(B, H, K, D, device=DEVICE)
    I = torch.randn(B, H, K, D, device=DEVICE)
    a = torch.tensor(0.7, device=DEVICE)
    b = torch.tensor(0.8, device=DEVICE)
    tau = 12.5
    dt = torch.tensor(1.0, device=DEVICE)
    n_steps = 2

    # Test plain Python version
    console.print("1. Plain Python FHN (no JIT)...")
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        v_out, w_out = fhn_update_no_jit(v.clone(), w.clone(), I, a, b, tau, dt, n_steps)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t_python = (time.perf_counter() - start) * 1000
    console.print(f"   Time: {t_python:.3f} ms")

    # Second call (warm)
    console.print("2. Plain Python (warm)...")
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        v_out2, w_out2 = fhn_update_no_jit(v.clone(), w.clone(), I, a, b, tau, dt, n_steps)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t_python_warm = (time.perf_counter() - start) * 1000
    console.print(f"   Time: {t_python_warm:.3f} ms")

    # Test JIT version
    console.print("\n3. JIT FHN (first call - compile)...")
    from neuromanifold_gpt.model.attention.fhn import fhn_update_step

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        v_jit, w_jit = fhn_update_step(v.clone(), w.clone(), I, a, b, tau, dt, n_steps)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t_jit_cold = (time.perf_counter() - start) * 1000
    console.print(f"   Time: {t_jit_cold:.3f} ms")

    console.print("4. JIT FHN (warm call)...")
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        v_jit2, w_jit2 = fhn_update_step(v.clone(), w.clone(), I, a, b, tau, dt, n_steps)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t_jit_warm = (time.perf_counter() - start) * 1000
    console.print(f"   Time: {t_jit_warm:.3f} ms")

    # Analysis
    console.print("\n[bold yellow]Analysis:[/bold yellow]")
    console.print(f"   Python warm:  {t_python_warm:.3f} ms")
    console.print(f"   JIT cold:     {t_jit_cold:.3f} ms (compile overhead)")
    console.print(f"   JIT warm:     {t_jit_warm:.3f} ms")

    if t_jit_cold > 100:
        console.print(f"\n[bold red]JIT compile is {t_jit_cold/t_python_warm:.1f}x slower than runtime![/bold red]")
        console.print("Consider: compile JIT script once at module load, not per-call")

    # Scale estimates
    console.print("\n[bold]Estimated @ B=64, T=256:[/bold]")
    scale = (64 / B) * (256 / K)
    console.print(f"   Python:  {t_python_warm * scale:.1f} ms")
    console.print(f"   JIT:     {t_jit_warm * scale:.1f} ms")


if __name__ == "__main__":
    main()
