"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT
from neuromanifold_gpt.config.base import NeuroManifoldConfig
from neuromanifold_gpt.config.training import SamplingConfig
from neuromanifold_gpt.config.loader import load_config
from neuromanifold_gpt.utils.checkpoints import select_checkpoint
from neuromanifold_gpt.utils.progress import checkpoint_progress
from neuromanifold_gpt.utils.logging import get_logger
from model import GPT, GPTConfig

logger = get_logger(__name__)

# -----------------------------------------------------------------------------
# Load configuration with type-safe CLI overrides
config = load_config(SamplingConfig)

# Extract config values for local use
init_from = config.init_from
out_dir = config.out_dir
start = config.start
num_samples = config.num_samples
max_new_tokens = config.max_new_tokens
temperature = config.temperature
top_k = config.top_k
seed = config.seed
device = config.device
dtype = config.dtype
compile = config.compile
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
    ckpt_path = select_checkpoint(out_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoints found in {out_dir}")
    # Weights only load issue in PyTorch 2.6+ with custom configs (trust local source)
    with checkpoint_progress("Loading checkpoint from disk"):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Check if it's a NeuroManifold checkpoint (has 'config' object) or legacy nanoGPT
    if 'config' in checkpoint and isinstance(checkpoint['config'], (NeuroManifoldConfig, type(None))):
        logger.info("Loading NeuroManifoldGPT model...")
        nm_config = checkpoint['config']
        model = NeuroManifoldGPT(nm_config)
    else:
        # Legacy/Standard GPT
        logger.info("Loading standard GPT model...")
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    with checkpoint_progress("Loading model weights"):
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
    logger.info(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    logger.info("No meta.pkl found, assuming GPT-2 encodings...")
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
            logger.section(f"Sample {k+1}/{num_samples}")
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            logger.info(decode(y[0].tolist()))
