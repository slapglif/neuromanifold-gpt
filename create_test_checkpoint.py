"""
Create a test checkpoint with random weights for pipeline verification.
"""
import os
import torch
from model import GPT, GPTConfig

# Create output directory
os.makedirs('out', exist_ok=True)

# Create a small GPT-2 model configuration
config = GPTConfig(
    block_size=1024,
    vocab_size=50257,  # GPT-2 vocab size
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.0,
    bias=False,
)

# Initialize model with random weights
model = GPT(config)
model.eval()

# Create checkpoint dictionary
checkpoint = {
    'model': model.state_dict(),
    'model_args': {
        'block_size': config.block_size,
        'vocab_size': config.vocab_size,
        'n_layer': config.n_layer,
        'n_head': config.n_head,
        'n_embd': config.n_embd,
        'dropout': config.dropout,
        'bias': config.bias,
    },
    'iter_num': 0,
    'best_val_loss': float('inf'),
    'config': {
        'dataset': 'test',
    }
}

# Save checkpoint
checkpoint_path = 'out/ckpt.pt'
print(f"Creating test checkpoint at {checkpoint_path}")
torch.save(checkpoint, checkpoint_path)
print(f"Checkpoint created successfully!")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
