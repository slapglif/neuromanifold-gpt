"""
Sample from a trained NeuroManifoldGPT model
"""
import os
import sys
from neuromanifold_gpt.cli.help_formatter import (
    create_parser_from_defaults,
    parse_args_with_config_override,
)

# -----------------------------------------------------------------------------
# Default Configuration
# -----------------------------------------------------------------------------
defaults = {
    # Model
    'out_dir': 'out-neuromanifold',

    # Sampling
    'prompt': "\n",  # Start token or custom prompt
    'num_samples': 1,  # number of samples to draw
    'max_new_tokens': 500,  # number of tokens generated in each sample
    'temperature': 0.8,  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random
    'top_k': 200,  # retain only the top_k most likely tokens, clamp others to have 0 probability
    'seed': 1337,

    # Hardware
    'device': 'cuda',  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    'dtype': 'bfloat16',  # 'float32' or 'bfloat16' or 'float16'
    'compile': False,  # use PyTorch 2.0 to compile the model to be faster
}

# Create argument parser with rich formatting
parser = create_parser_from_defaults(
    defaults=defaults,
    description="Sample from a trained NeuroManifoldGPT model",
    groups={
        'Model': ['out_dir'],
        'Sampling': ['prompt', 'num_samples', 'max_new_tokens', 'temperature', 'top_k', 'seed'],
        'Hardware': ['device', 'dtype', 'compile'],
    },
    examples=[
        "python neuromanifold_gpt/sample_nanogpt.py",
        "python neuromanifold_gpt/sample_nanogpt.py --num_samples=5 --temperature=1.0",
        "python neuromanifold_gpt/sample_nanogpt.py --out_dir=out-neuromanifold",
        "python neuromanifold_gpt/sample_nanogpt.py --prompt='Hello, world!'",
    ],
)

# Parse arguments with config file override support
args = parse_args_with_config_override(parser)

# Extract configuration values
out_dir = args.out_dir
prompt = args.prompt
num_samples = args.num_samples
max_new_tokens = args.max_new_tokens
temperature = args.temperature
top_k = args.top_k
seed = args.seed
device = args.device
dtype = args.dtype
compile_model = args.compile

# -----------------------------------------------------------------------------
# Import heavy dependencies after argparse (so --help works without them)
# -----------------------------------------------------------------------------
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from neuromanifold_gpt.config import NeuroManifoldConfig
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT
# -----------------------------------------------------------------------------

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
