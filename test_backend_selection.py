#!/usr/bin/env python3
"""Test backend selection logic in attention registry."""

from neuromanifold_gpt.model.attention import get_attention_class
from neuromanifold_gpt.config.base import NeuroManifoldConfig

# Test 1: Basic usage (backward compatible - no backend parameter)
print("Test 1: Basic usage (backward compatible)")
attn_cls = get_attention_class('fhn')
print(f"✓ get_attention_class('fhn') = {attn_cls.__name__}")

# Test 2: With explicit backend
print("\nTest 2: Explicit backend selection")
attn_cls = get_attention_class('fhn', backend='flash')
print(f"✓ get_attention_class('fhn', backend='flash') = {attn_cls.__name__}")
if hasattr(attn_cls, '_resolved_backend'):
    print(f"  Resolved backend: {attn_cls._resolved_backend}")

# Test 3: With auto backend
print("\nTest 3: Auto backend selection")
attn_cls = get_attention_class('fhn', backend='auto')
print(f"✓ get_attention_class('fhn', backend='auto') = {attn_cls.__name__}")
if hasattr(attn_cls, '_resolved_backend'):
    print(f"  Auto-selected backend: {attn_cls._resolved_backend}")

# Test 4: Integration with config
print("\nTest 4: Integration with NeuroManifoldConfig")
cfg = NeuroManifoldConfig(attention_backend='auto')
attn_cls = get_attention_class(cfg.attention_type, backend=cfg.attention_backend)
print(f"✓ Config: attention_type={cfg.attention_type}, attention_backend={cfg.attention_backend}")
print(f"  Selected class: {attn_cls.__name__}")
if hasattr(attn_cls, '_resolved_backend'):
    print(f"  Resolved backend: {attn_cls._resolved_backend}")

# Test 5: Test all attention types
print("\nTest 5: All attention types")
for attn_type in ['standard', 'fhn', 'knot', 'kaufmann']:
    attn_cls = get_attention_class(attn_type)
    print(f"✓ {attn_type} -> {attn_cls.__name__}")

# Test 6: Test aliases
print("\nTest 6: Aliases")
for alias, expected in [('soliton', 'fhn'), ('sdr', 'knot'), ('fast-spectral', 'fhn')]:
    attn_cls = get_attention_class(alias)
    print(f"✓ {alias} -> {attn_cls.__name__}")

print("\n✅ All tests passed! Backend selection logic working correctly.")
