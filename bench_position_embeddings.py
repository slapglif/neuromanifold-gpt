"""
Benchmark script for position embedding performance comparison.
Compares learned, RoPE, and ALiBi position embeddings to measure overhead.
"""
import os
from contextlib import nullcontext
import numpy as np
import time
import torch
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
batch_size = 12
block_size = 1024
bias = False
real_data = False  # Use synthetic data for consistent benchmarking
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True  # use PyTorch 2.0 to compile the model to be faster
quick = False  # quick mode: fewer iterations
exec(open('configurator.py').read())  # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# data loading init - use synthetic data for consistent benchmarking
x = torch.randint(50304, (batch_size, block_size), device=device)
y = torch.randint(50304, (batch_size, block_size), device=device)
get_batch = lambda split: (x, y)

# Determine iteration counts based on quick mode
if quick:
    burnin_steps = 5
    bench_steps = 10
    print("Quick mode: using fewer iterations")
else:
    burnin_steps = 10
    bench_steps = 20

# Position embedding types to benchmark
pos_emb_types = ['learned', 'rotary', 'alibi']
results = {}

print(f"Benchmarking position embeddings on {device}")
print(f"Batch size: {batch_size}, Block size: {block_size}")
print(f"Model: 12 layers, 12 heads, 768 embed dim")
print("-" * 70)

for pos_emb_type in pos_emb_types:
    print(f"\nBenchmarking {pos_emb_type} position embeddings...")

    # model init
    gptconf = GPTConfig(
        block_size=block_size,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0,
        bias=bias,
        pos_emb_type=pos_emb_type,
    )
    model = GPT(gptconf)
    model.to(device)

    optimizer = model.configure_optimizers(
        weight_decay=1e-2,
        learning_rate=1e-4,
        betas=(0.9, 0.95),
        device_type=device_type
    )

    if compile:
        print(f"  Compiling model...")
        model = torch.compile(model)

    # Burnin phase
    if device_type == 'cuda':
        torch.cuda.synchronize()

    X, Y = get_batch('train')
    for k in range(burnin_steps):
        with ctx:
            logits, loss = model(X, Y)
        X, Y = get_batch('train')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    if device_type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark phase
    t0 = time.time()
    X, Y = get_batch('train')
    for k in range(bench_steps):
        with ctx:
            logits, loss = model(X, Y)
        X, Y = get_batch('train')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    if device_type == 'cuda':
        torch.cuda.synchronize()

    t1 = time.time()
    dt = t1 - t0
    time_per_iter = dt / bench_steps * 1000  # in ms

    results[pos_emb_type] = {
        'total_time': dt,
        'time_per_iter': time_per_iter,
    }

    print(f"  Time per iteration: {time_per_iter:.4f}ms")

    # Clean up
    del model
    del optimizer
    if device_type == 'cuda':
        torch.cuda.empty_cache()

# Calculate overhead
print("\n" + "=" * 70)
print("BENCHMARK RESULTS")
print("=" * 70)

baseline_time = results['learned']['time_per_iter']
print(f"\nBaseline (learned embeddings): {baseline_time:.4f}ms per iteration")

for pos_emb_type in ['rotary', 'alibi']:
    time_per_iter = results[pos_emb_type]['time_per_iter']
    overhead = ((time_per_iter - baseline_time) / baseline_time) * 100
    overhead_sign = '+' if overhead >= 0 else ''

    print(f"{pos_emb_type.capitalize():8s} embeddings: {time_per_iter:.4f}ms "
          f"({overhead_sign}{overhead:.2f}% overhead)")

    # Check if overhead meets acceptance criteria
    if abs(overhead) < 5.0:
        status = "✓ PASS"
    else:
        status = "✗ FAIL"

    print(f"  Performance overhead < 5%: {status}")

print("\n" + "=" * 70)
print("Summary:")
max_overhead = max(
    abs((results['rotary']['time_per_iter'] - baseline_time) / baseline_time * 100),
    abs((results['alibi']['time_per_iter'] - baseline_time) / baseline_time * 100)
)
if max_overhead < 5.0:
    print(f"✓ All position embeddings meet < 5% overhead requirement")
    print(f"  Maximum overhead: {max_overhead:.2f}%")
else:
    print(f"✗ Some position embeddings exceed 5% overhead")
    print(f"  Maximum overhead: {max_overhead:.2f}%")
print("=" * 70)
