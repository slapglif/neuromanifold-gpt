from neuromanifold_gpt.model.gpt import NeuroManifoldGPT
from neuromanifold_gpt.model.system_two_mixin import SystemTwoReasoningMixin

assert issubclass(NeuroManifoldGPT, SystemTwoReasoningMixin)
print('OK')
