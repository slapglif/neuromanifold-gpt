"""
Sample from a trained NeuroManifoldGPT model.

Follows Karpathy's exact nanoGPT sampling methodology.

Usage:
    python neuromanifold_gpt/sample_nanogpt.py --out_dir=out-neuromanifold
    python neuromanifold_gpt/sample_nanogpt.py --prompt="Hello, world!"
    python neuromanifold_gpt/sample_nanogpt.py --num_samples=5 --max_new_tokens=500
"""
import os
import pickle
from contextlib import nullcontext

import tiktoken
import torch

from neuromanifold_gpt.config import NeuroManifoldConfig
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT

# -----------------------------------------------------------------------------
# Default sampling parameters
out_dir = "out-neuromanifold"
prompt = "\n"  # Start token or custom prompt
num_samples = 1
max_new_tokens = 500
temperature = 0.8
top_k = 200
seed = 1337
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
compile_model = False

# -----------------------------------------------------------------------------
# Parse command line
exec(open(os.path.join(os.path.dirname(__file__), "configurator.py")).read())

# -----------------------------------------------------------------------------
# Setup
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# Load model
ckpt_path = os.path.join(out_dir, "ckpt.pt")
checkpoint = torch.load(ckpt_path, map_location=device)

# Recreate config
checkpoint_config = checkpoint["model_config"]
model_config = NeuroManifoldConfig(**checkpoint_config)

# Create model
model = NeuroManifoldGPT(model_config)

# Load weights
state_dict = checkpoint["model"]
unwanted_prefix = "_orig_mod."
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)

if compile_model:
    model = torch.compile(model)

print(f"Loaded model from {ckpt_path}")
print(f"Model has {model.num_parameters() / 1e6:.2f}M parameters")
print(f"Best val loss: {checkpoint['best_val_loss']:.4f}")
print(f"Trained for {checkpoint['iter_num']} iterations")

# -----------------------------------------------------------------------------
# Encoder/decoder setup
# Try to load dataset-specific encoder, fall back to tiktoken
data_dir = os.path.join("data", checkpoint.get("config", {}).get("dataset", "openwebtext"))
meta_path = os.path.join(data_dir, "meta.pkl")
if os.path.exists(meta_path):
    print(f"Loading tokenizer from {meta_path}...")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])
else:
    print("Using tiktoken gpt2 encoding...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# -----------------------------------------------------------------------------
# Encode prompt
start_ids = encode(prompt)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

# -----------------------------------------------------------------------------
# Generate
print(f"\nGenerating {num_samples} sample(s) with {max_new_tokens} tokens each...")
print(f"Temperature: {temperature}, Top-k: {top_k}")
print("-" * 50)

with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(f"\n--- Sample {k + 1} ---")
            print(decode(y[0].tolist()))
            print("-" * 50)
