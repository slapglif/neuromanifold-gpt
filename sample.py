"""
Sample from a trained model
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
    'init_from': 'resume',  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
    'out_dir': 'out',  # ignored if init_from is not 'resume'

    # Sampling
    'start': "\n",  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
    'num_samples': 10,  # number of samples to draw
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
    description="Sample from a trained NeuroManifoldGPT or GPT model",
    groups={
        'Model': ['init_from', 'out_dir'],
        'Sampling': ['start', 'num_samples', 'max_new_tokens', 'temperature', 'top_k', 'seed'],
        'Hardware': ['device', 'dtype', 'compile'],
    },
    examples=[
        "python sample.py",
        "python sample.py --num_samples=5 --temperature=1.0",
        "python sample.py config/sample_config.py --max_new_tokens=1000",
        "python sample.py --start='Once upon a time'",
    ],
)

# Parse arguments with config file override support
args = parse_args_with_config_override(parser)

# Extract configuration values
init_from = args.init_from
out_dir = args.out_dir
start = args.start
num_samples = args.num_samples
max_new_tokens = args.max_new_tokens
temperature = args.temperature
top_k = args.top_k
seed = args.seed
device = args.device
dtype = args.dtype
compile = args.compile

# -----------------------------------------------------------------------------
# Import heavy dependencies after argparse (so --help works without them)
# -----------------------------------------------------------------------------
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT
from neuromanifold_gpt.config.base import NeuroManifoldConfig
from model import GPTConfig, GPT
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    # Weights only load issue in PyTorch 2.6+ with custom configs (trust local source)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Check if it's a NeuroManifold checkpoint (has 'config' object) or legacy nanoGPT
    if 'config' in checkpoint and isinstance(checkpoint['config'], (NeuroManifoldConfig, type(None))):
        print("Loading NeuroManifoldGPT model...")
        nm_config = checkpoint['config']
        model = NeuroManifoldGPT(nm_config)
    else:
        # Legacy/Standard GPT
        print("Loading standard GPT model...")
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
# Handle NeuroManifold config (object) vs Dict config
dataset_name = None
if 'config' in checkpoint:
    if hasattr(checkpoint['config'], 'dataset'):
        dataset_name = checkpoint['config'].dataset
    elif isinstance(checkpoint['config'], dict) and 'dataset' in checkpoint['config']:
        dataset_name = checkpoint['config']['dataset']
    else:
        # Fallback to shakespeare_char if likely
        dataset_name = 'shakespeare_char'

if dataset_name:
    meta_path = os.path.join('data', dataset_name, 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
