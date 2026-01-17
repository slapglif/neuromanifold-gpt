import os
import time
import math
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from neuromanifold_gpt.model.neuro_symbolic_wave import NeuroSymbolicWaveNet

# -----------------------------------------------------------------------------
# Configuration & Matmul Precision
# -----------------------------------------------------------------------------
torch.set_float32_matmul_precision("medium")
DATA_DIR = "data/shakespeare_char"
OUT_DIR = "out_bench"
os.makedirs(OUT_DIR, exist_ok=True)

# Training Hyperparameters (Optimized from train.py)
batch_size = 16
accumulation_steps = 4  # Effective batch size = 64
block_size = 256
max_iters = 5000
learning_rate = 1e-3
min_lr = 1e-4
warmup_iters = 100
lr_decay_iters = 5000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0
device = "cuda" if torch.cuda.is_available() else "cpu"

# Intervals
eval_interval = 250
eval_iters = 50
sample_interval = 250
save_interval = 500

# Model Hyperparameters
matrix_dim = 16  # model_dim = 16*16*2 = 512
depth = 4
num_heads = 8
fno_modes = 16

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
train_data = np.memmap(os.path.join(DATA_DIR, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(DATA_DIR, "val.bin"), dtype=np.uint16, mode="r")


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )
    if device == "cuda":
        x, y = (
            x.pin_memory().to(device, non_blocking=True),
            y.pin_memory().to(device, non_blocking=True),
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits = model(X)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Load Vocab
with open(os.path.join(DATA_DIR, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
vocab_size = meta["vocab_size"]
itos = meta["itos"]
stoi = meta["stoi"]
decode = lambda l: "".join([itos[i] for i in l])

# -----------------------------------------------------------------------------
# Model Initialization
# -----------------------------------------------------------------------------
print(f"Initializing NS-WMN with model_dim=512, vocab_size={vocab_size}")
if device == "cuda":
    torch.cuda.empty_cache()

model = NeuroSymbolicWaveNet(
    vocab_size=vocab_size,
    matrix_dim=matrix_dim,
    depth=depth,
    num_heads=num_heads,
    max_seq_len=block_size,
    fno_modes=fno_modes,
    use_checkpointing=True,  # Memory optimization
)
model.to(device)

# -----------------------------------------------------------------------------
# Optimizer (Optimized Grouping from train.py)
# -----------------------------------------------------------------------------
decay_params = []
nodecay_params = []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if param.dim() < 2 or "ln" in name or "norm" in name:
        nodecay_params.append(param)
    else:
        decay_params.append(param)

optimizer = torch.optim.AdamW(
    [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ],
    lr=learning_rate,
    betas=(beta1, beta2),
)


# -----------------------------------------------------------------------------
# LR Scheduler (Cosine with Warmup from train.py)
# -----------------------------------------------------------------------------
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
print("Starting Optimized Training Run (Target: 10 mins deep convergence)")
start_time = time.time()
best_val_loss = float("inf")

for iter_num in range(max_iters):
    # Set current LR
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    t0 = time.time()

    # Evaluation loop
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        elapsed = time.time() - start_time
        print(
            f"Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.2e}, elapsed {elapsed:.2f}s"
        )
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "model_best.pt"))
            print(f"Saving best model (val_loss: {best_val_loss:.4f})")

    # Sampling loop
    if iter_num > 0 and iter_num % sample_interval == 0:
        model.eval()
        with torch.no_grad():
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            # Use newline as start token if available
            start_idx = stoi.get("\n", 0)
            context[0, 0] = start_idx
            generated = []
            for _ in range(200):
                logits = model(context)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                context = torch.cat((context, idx_next), dim=1)
                if context.size(1) > block_size:
                    context = context[:, 1:]
                generated.append(itos[idx_next.item()])
            print(
                f"\n--- Sample (Step {iter_num}) ---\n{''.join(generated)}\n--- End Sample ---\n"
            )
        model.train()

    # Gradient accumulation loop
    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0
    for _ in range(accumulation_steps):
        X, Y = get_batch("train")
        logits = model(X)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
        loss = loss / accumulation_steps
        accum_loss += loss.item()
        loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    t1 = time.time()
    dt = t1 - t0
    elapsed = t1 - start_time

    if iter_num % 50 == 0:
        print(
            f"Iter {iter_num:4d} | Loss: {accum_loss:.4f} | Time: {dt * 1000:.2f}ms | Elapsed: {elapsed:.2f}s"
        )

    # Final Exit condition (10 mins)
    if elapsed > 600:
        print(f"10-Minute Time Limit Reached ({elapsed:.2f}s). Finalizing...")
        break

print(f"\nTraining Complete. Total Time: {time.time() - start_time:.2f}s")
