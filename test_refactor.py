#!/usr/bin/env python3
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT
import torch

config = type('C', (), {
    'use_sdr': False,
    'vocab_size': 100,
    'n_embd': 64,
    'block_size': 32,
    'n_layer': 2,
    'n_heads': 4,
    'manifold_dim': 32,
    'n_eigenvectors': 16,
    'dropout': 0.0,
    'memory_active_retrieval': False,
    'use_memory': False
})()

model = NeuroManifoldGPT(config)
tokens = torch.randint(0, 100, (2, 10))
logits, loss, info = model(tokens)
print('OK')
