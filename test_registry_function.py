from neuromanifold_gpt.model.attention import get_attention_class, FHNAttention, KnotAttention, KaufmannAttention, StandardAttention

print("Testing get_attention_class registry...")

# Test direct names
assert get_attention_class('fhn') == FHNAttention
print("✓ fhn -> FHNAttention")

assert get_attention_class('knot') == KnotAttention
print("✓ knot -> KnotAttention")

assert get_attention_class('kaufmann') == KaufmannAttention
print("✓ kaufmann -> KaufmannAttention")

assert get_attention_class('standard') == StandardAttention
print("✓ standard -> StandardAttention")

# Test aliases
assert get_attention_class('soliton') == FHNAttention
print("✓ soliton -> FHNAttention (alias)")

assert get_attention_class('sdr') == KnotAttention
print("✓ sdr -> KnotAttention (alias)")

assert get_attention_class('fast-spectral') == FHNAttention
print("✓ fast-spectral -> FHNAttention (alias)")

# Test unknown type
try:
    get_attention_class('unknown')
    print("✗ FAILED: Should raise ValueError for unknown type")
except ValueError as e:
    print(f"✓ unknown -> ValueError: {e}")

print("\nAll registry tests passed!")
