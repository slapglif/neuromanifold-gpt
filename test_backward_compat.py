"""Test backward compatibility of boolean flags."""
from neuromanifold_gpt.config.base import NeuroManifoldConfig
from neuromanifold_gpt.model.gpt import NeuroManifoldGPT

# Test use_kaufmann_attention flag
c1 = NeuroManifoldConfig(use_kaufmann_attention=True, n_layer=1, use_sdr=False)
m1 = NeuroManifoldGPT(c1)

# Test use_knot_attention flag
c2 = NeuroManifoldConfig(use_knot_attention=True, n_layer=1, use_sdr=False)
m2 = NeuroManifoldGPT(c2)

print('OK')
