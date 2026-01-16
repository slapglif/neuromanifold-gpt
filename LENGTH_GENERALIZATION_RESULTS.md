# Length Generalization Test Results

## Overview

- **Training sequence length**: 64
- **Test sequence length**: 128 (2x training)
- **Dataset**: Shakespeare (character-level)
- **Training iterations**: 300
- **Model size**: 4 layers, 4 heads, 256 embedding dim
- **Device**: cuda

## Results Summary

| Position Embedding | Train Length PPL | Test Length PPL (2x) | Degradation | Status |
|-------------------|------------------|---------------------|-------------|--------|
| Learned           |            15.69 |               18.36 |      +17.0% | ⚠️ Expected degradation |
| Rotary            |            15.60 |               16.24 |       +4.1% | ✅ Good extrapolation |
| Alibi             |            15.38 |               16.24 |       +5.6% | ✅ Good extrapolation |

## Detailed Results

### Learned

**At training length (64):**
- Loss: 2.7531
- Perplexity: 15.6919

**At test length (128):**
- Loss: 2.9101
- Perplexity: 18.3586

**Degradation**: +16.99%

### Rotary

**At training length (64):**
- Loss: 2.7474
- Perplexity: 15.6017

**At test length (128):**
- Loss: 2.7878
- Perplexity: 16.2450

**Degradation**: +4.12%

### Alibi

**At training length (64):**
- Loss: 2.7331
- Perplexity: 15.3810

**At test length (128):**
- Loss: 2.7876
- Perplexity: 16.2426

**Degradation**: +5.60%

## Analysis

### Expected Behavior

1. **RoPE (Rotary Position Embeddings)**:
   - Should extrapolate well due to relative position encoding
   - Rotation-based mechanism works at any sequence length
   - Expected: Low degradation (< 50%)

2. **ALiBi (Attention with Linear Biases)**:
   - Should extrapolate well due to linear bias design
   - Bias slopes are position-agnostic
   - Expected: Low degradation (< 50%)

3. **Learned Position Embeddings**:
   - Limited to training sequence length
   - No learned representation for positions > 64
   - Expected: Higher degradation (baseline for comparison)

### Comparative Analysis

- **RoPE vs Learned**: +12.9 percentage points better extrapolation
- **ALiBi vs Learned**: +11.4 percentage points better extrapolation

### Conclusion

✅ **SUCCESS**: Both RoPE and ALiBi demonstrate better length generalization than learned position embeddings, confirming their extrapolation capabilities.
