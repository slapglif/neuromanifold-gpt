from neuromanifold_gpt.model.gpt import NeuroManifoldGPT
from neuromanifold_gpt.config import NeuroManifoldConfigNano
import torch

cfg = NeuroManifoldConfigNano()
m = NeuroManifoldGPT(cfg)
m.eval()
x = torch.randint(0, cfg.vocab_size, (2, 10))
y = m.generate(x, max_new_tokens=5, repetition_penalty=1.5)
print('OK' if y.shape == (2, 15) else 'FAIL')
