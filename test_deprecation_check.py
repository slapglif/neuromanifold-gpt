import warnings
warnings.simplefilter('always')
from neuromanifold_gpt.config.base import NeuroManifoldConfig

print("Testing use_kaufmann_attention deprecation...")
c1 = NeuroManifoldConfig(use_kaufmann_attention=True, n_layer=1, use_sdr=False)
print("OK")

print("Testing use_knot_attention deprecation...")
c2 = NeuroManifoldConfig(use_knot_attention=True, n_layer=1, use_sdr=False)
print("OK")
