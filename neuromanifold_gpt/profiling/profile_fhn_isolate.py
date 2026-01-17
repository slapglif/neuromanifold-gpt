#!/usr/bin/env python3
"""
Isolate the JIT bug - step by step.
"""

import time

import torch
from rich.console import Console

console = Console()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Fixed version - use scalar multiplication correctly
@torch.jit.script
def fhn_update_step_fixed(
    v: torch.Tensor,
    w: torch.Tensor,
    I: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    tau: float,
    dt: torch.Tensor,
    n_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed JIT script - use scalar values properly."""
    dt_val = dt.item()
    a_val = a.item()
    b_val = b.item()

    alpha = dt_val / tau
    denom = 1.0 + alpha * b_val

    for _ in range(n_steps):
        v3 = v * v * v
        dv_explicit = v - (v3 / 3.0) - w + I

        # Update v: scalar * tensor (proper order)
        v_next = v + dv_explicit * dt_val

        # Update w with scalars
        w_next = (w + (v_next + a_val) * alpha) / denom

        # Clamp
        v = torch.clamp(v_next, -3.0, 3.0)
        w = torch.clamp(w_next, -3.0, 3.0)

    return v, w


def main():
    console.print("[bold]JIT Bug Isolation[/bold]")
    console.print(f"Device: {DEVICE}\n")

    # Create inputs
    B, H, K, D = 4, 8, 32, 48
    v = torch.zeros(B, H, K, D, device=DEVICE)
    w = torch.zeros(B, H, K, D, device=DEVICE)
    I = torch.randn(B, H, K, D, device=DEVICE)
    a = torch.tensor(0.7, device=DEVICE)
    b = torch.tensor(0.8, device=DEVICE)
    tau = 12.5
    dt = torch.tensor(1.0, device=DEVICE)
    n_steps = 2

    # Test fixed version
    console.print("1. Fixed JIT (first call - compile)...")
    start = time.perf_counter()
    with torch.no_grad():
        v1, w1 = fhn_update_step_fixed(v.clone(), w.clone(), I, a, b, tau, dt, n_steps)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t1 = (time.perf_counter() - start) * 1000
    console.print(f"   Time: {t1:.3f} ms")

    console.print("2. Fixed JIT (warm call)...")
    start = time.perf_counter()
    with torch.no_grad():
        v2, w2 = fhn_update_step_fixed(v.clone(), w.clone(), I, a, b, tau, dt, n_steps)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t2 = (time.perf_counter() - start) * 1000
    console.print(f"   Time: {t2:.3f} ms")

    console.print("3. Fixed JIT (10 iterations)...")
    times = []
    for _ in range(10):
        start = time.perf_counter()
        with torch.no_grad():
            v3, w3 = fhn_update_step_fixed(
                v.clone(), w.clone(), I, a, b, tau, dt, n_steps
            )
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    console.print(f"   Mean: {sum(times)/len(times):.3f} ms, Min: {min(times):.3f} ms")

    console.print("\n[bold green]Fixed JIT works![/bold green]")
    console.print("Issue: The original used tensor operations (dt * x) in loop")
    console.print("Fix: Extract .item() scalars before the loop, use proper order")

    # Now profile the rest of the components
    console.print("\n[bold]=== Full Component Profiling ===[/bold]")

    # Scale estimates
    mean_time = sum(times) / len(times)
    console.print("\n[bold]FHN Dynamics @ B=4, H=8, K=32, D=48:[/bold]")
    console.print(f"   Time: {mean_time:.2f} ms")

    # Now let's profile the full attention pipeline
    console.print("\n[bold]Profile Full FHN Attention with Fix...[/bold]")

    # We need to test with a proper FHNAttention setup
    # For now, estimate the scaling
    B_full, T_full = 64, 256
    scale = (B_full / B) * (T_full / K)  # K is like spectral dim
    console.print(f"   Estimated @ B={B_full}, T={T_full}: {mean_time * scale:.1f} ms")


if __name__ == "__main__":
    main()
