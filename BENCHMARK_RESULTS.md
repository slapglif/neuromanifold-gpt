# Zero-Shot Benchmark Evaluation Results

## Executive Summary

This document provides comprehensive documentation for evaluating NeuroManifoldGPT and standard GPT models on established NLP benchmarks using zero-shot evaluation. The evaluation suite includes four standard benchmarks that enable reproducible comparisons with published results.

**Benchmarks Evaluated:**
- **LAMBADA**: Language modeling with final word prediction (perplexity metric)
- **HellaSwag**: Commonsense reasoning (accuracy metric)
- **PIQA**: Physical commonsense reasoning (accuracy metric)
- **WinoGrande**: Winograd schema challenge (accuracy metric)

**Purpose**: Validate that trained models achieve quality comparable to published baselines, ensuring training succeeds and produces research-grade models.

---

## Evaluation Methodology

### Zero-Shot Evaluation Approach

All benchmarks use **zero-shot evaluation**, meaning the model is tested directly without any task-specific training or fine-tuning. This measures the model's general language understanding capabilities.

#### LAMBADA Perplexity

**Task**: Predict the final word of a passage given full context.

**Metric**: Perplexity on final token prediction
- Lower perplexity = better performance
- Measures how well the model predicts the last word

**Evaluation Process**:
1. Load each LAMBADA example (5,153 examples total)
2. Tokenize the full context
3. Compute log-likelihood of the final token
4. Calculate perplexity: exp(mean(-log_likelihood))

**Implementation**: `neuromanifold_gpt/benchmarks/zero_shot.py::evaluate_lambada()`

#### Multiple-Choice Accuracy (HellaSwag, PIQA, WinoGrande)

**Task**: Select the most plausible completion from multiple choices.

**Metric**: Accuracy (fraction of correct predictions)
- Higher accuracy = better performance
- Random baseline: 25% (4 choices) or 50% (2 choices)

**Evaluation Process**:
1. For each example, concatenate context with each choice
2. Compute log-likelihood for each choice sequence
3. Select choice with highest likelihood
4. Calculate accuracy: correct / total

**Implementation**: `neuromanifold_gpt/benchmarks/zero_shot.py::evaluate_multiple_choice()`

### Technical Details

**Tokenization**: GPT-2 BPE tokenizer (tiktoken) for consistency with published baselines

**Precision**: Mixed precision (bfloat16/float16) with autocast for efficient evaluation

**Device**: CUDA-enabled GPU (CPU fallback available)

**Batch Processing**: Evaluates examples individually for accurate per-example metrics

---

## Running Evaluations

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure you have a trained checkpoint
ls out/ckpt.pt  # or out/ckpt_final.pt
```

### Basic Evaluation Commands

```bash
# Evaluate single benchmark (LAMBADA)
python eval.py --out_dir=out --benchmark=lambada

# Evaluate all benchmarks
python eval.py --out_dir=out --benchmark=all

# Quick test with limited examples
python eval.py --out_dir=out --benchmark=all --eval_iters=100

# With WandB logging
python eval.py --out_dir=out --benchmark=all --wandb_log=True --wandb_project=my-project

# CPU evaluation (if no GPU)
python eval.py --out_dir=out --benchmark=all --device=cpu

# Using config file
python eval.py config/eval_lambada.py
```

### Configuration Parameters

```bash
--out_dir=<path>           # Checkpoint directory (default: 'out')
--benchmark=<name>         # lambada|hellaswag|piqa|winogrande|all (default: 'lambada')
--eval_iters=<int>         # Max examples to evaluate, None=all (default: None)
--device=<str>             # 'cpu', 'cuda', 'cuda:0', etc. (default: 'cuda')
--dtype=<str>              # 'float32', 'bfloat16', 'float16' (default: auto)
--wandb_log=<bool>         # Log results to wandb (default: False)
--compile=<bool>           # Use PyTorch 2.0 compilation (default: False)
```

### Example Output

```
========================================
Loading model from out
========================================
Loading checkpoint from disk... ✓
Loading standard GPT model...
Loading model weights... ✓
Model loaded successfully!

No meta.pkl found, using GPT-2 tokenizer...

========================================
Evaluating LAMBADA
========================================

Evaluating LAMBADA (5153 examples)...
  Progress: 1000/5153 - PPL: 21.34, Acc: 0.4523
  Progress: 2000/5153 - PPL: 20.98, Acc: 0.4578
  Progress: 3000/5153 - PPL: 20.76, Acc: 0.4601
  Progress: 4000/5153 - PPL: 20.62, Acc: 0.4618
  Progress: 5000/5153 - PPL: 20.51, Acc: 0.4629

LAMBADA Results:
  Perplexity: 20.48
  Loss: 3.0196
  Accuracy: 0.4634
  Examples: 5153

========================================
Evaluating HELLASWAG
========================================

Evaluating HellaSwag (10042 examples)...
  Progress: 2000/10042 - Acc: 0.3089
  Progress: 4000/10042 - Acc: 0.3102
  Progress: 6000/10042 - Acc: 0.3095
  Progress: 8000/10042 - Acc: 0.3108
  Progress: 10000/10042 - Acc: 0.3112

HellaSwag Results:
  Accuracy: 0.3115
  Examples: 10042

========================================
Evaluating PIQA
========================================

Evaluating PIQA (1838 examples)...
  Progress: 500/1838 - Acc: 0.6380
  Progress: 1000/1838 - Acc: 0.6410
  Progress: 1500/1838 - Acc: 0.6387

PIQA Results:
  Accuracy: 0.6398
  Examples: 1838

========================================
Evaluating WINOGRANDE
========================================

Evaluating WinoGrande (1267 examples)...
  Progress: 500/1267 - Acc: 0.5180
  Progress: 1000/1267 - Acc: 0.5210

WinoGrande Results:
  Accuracy: 0.5225
  Examples: 1267

========================================
Evaluation Summary
========================================

┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Benchmark     ┃ Metric                  ┃         Value ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ LAMBADA       │ perplexity              │       20.4800 │
│               │ loss                    │        3.0196 │
│               │ accuracy                │        0.4634 │
│               │ num_examples            │         5153  │
│               │                         │               │
│ HELLASWAG     │ accuracy                │        0.3115 │
│               │ num_examples            │        10042  │
│               │                         │               │
│ PIQA          │ accuracy                │        0.6398 │
│               │ num_examples            │         1838  │
│               │                         │               │
│ WINOGRANDE    │ accuracy                │        0.5225 │
│               │ num_examples            │         1267  │
└───────────────┴─────────────────────────┴───────────────┘

Evaluation complete!
```

---

## Published Baselines

### GPT-2 124M (Trained on OpenWebText)

These are the expected results for a properly trained GPT-2 124M model, as reported in published literature and OpenAI's original work:

| Benchmark | Metric | Published Value | Acceptable Range (±10%) |
|-----------|--------|-----------------|-------------------------|
| **LAMBADA** | Perplexity | 20.5 | 18.5 - 22.6 |
| **HellaSwag** | Accuracy | 0.31 (31%) | 0.28 - 0.34 |
| **PIQA** | Accuracy | 0.64 (64%) | 0.58 - 0.70 |
| **WinoGrande** | Accuracy | 0.52 (52%) | 0.47 - 0.57 |

**Sources**:
- LAMBADA: Radford et al. (2019) "Language Models are Unsupervised Multitask Learners"
- HellaSwag: Zellers et al. (2019) "HellaSwag: Can a Machine Really Finish Your Sentence?"
- PIQA: Bisk et al. (2020) "PIQA: Reasoning about Physical Commonsense in Natural Language"
- WinoGrande: Sakaguchi et al. (2020) "WinoGrande: An Adversarial Winograd Schema Challenge at Scale"

### Baseline Interpretation

**Within 10% of baseline**: ✅ Training successful, model quality validated
- This indicates the model has learned proper language modeling capabilities
- Results are reproducible and comparable to published work

**Outside 10% of baseline**: ⚠️ Investigate potential issues
- Check training convergence (loss curves, perplexity over time)
- Verify dataset preprocessing and tokenization
- Review hyperparameters (learning rate, batch size, etc.)
- Ensure sufficient training steps (GPT-2 124M typically needs ~100K steps)

### Expected Results by Model Type

**Random/Untrained Model**:
- LAMBADA perplexity: 100-300+ (very high)
- Multiple-choice accuracy: 25-50% (near random chance)

**Partially Trained Model** (early checkpoint):
- LAMBADA perplexity: 40-80 (improving)
- Multiple-choice accuracy: 25-40% (above random)

**Well-Trained Model** (converged):
- LAMBADA perplexity: 18-23 (low)
- Multiple-choice accuracy: 30-65% (task-dependent)

**Pretrained GPT-2 124M** (from OpenAI):
- Should match published baselines closely

---

## Benchmark Dataset Details

### LAMBADA

**Source**: OpenAI public blob storage
**Examples**: 5,153 test examples
**Format**: Text passages with final word to predict
**Task**: Language modeling with long-range dependencies

**Example**:
```
Context: "Yes, I thought I was going to lose the baby." "I was scared too," he stated, sincerity flooding his eyes. "You were ?" "Yes, of course. Why do you even ask?" "This baby wasn't exactly planned for..."
Target: either
```

**Characteristics**:
- Tests long-range dependency understanding
- Requires maintaining context over 4-5 sentences
- Final word often requires global passage understanding

### HellaSwag

**Source**: GitHub (rowanz/hellaswag)
**Examples**: 10,042 validation examples
**Format**: Context + 4 completion choices
**Task**: Commonsense reasoning about event sequences

**Example**:
```
Context: "A woman is outside with a bucket and a dog. The dog is running around trying to avoid a bath. She..."
Choices:
  A) rinses the bucket off with soap and blow dry the dog's head.
  B) uses a hose to keep it from getting soapy.
  C) gets the dog wet, then it runs away again.  [CORRECT]
  D) gets into the bath tub with the dog.
```

**Characteristics**:
- Tests physical and social commonsense
- Adversarial mining: hard negatives from language models
- Random baseline: 25% (4 choices)

### PIQA

**Source**: yonatanbisk.com
**Examples**: 1,838 validation examples
**Format**: Goal + 2 solution choices
**Task**: Physical commonsense reasoning

**Example**:
```
Goal: "To separate egg whites from the yolk using a water bottle, you can"
Choices:
  A) Squeeze the water bottle and press it against the yolk. Release, which creates suction and lifts the yolk.  [CORRECT]
  B) Place the water bottle and press it against the yolk. Keep pressing, which creates suction and lifts the yolk.
```

**Characteristics**:
- Tests understanding of physical interactions
- Requires knowledge of cause-and-effect
- Random baseline: 50% (2 choices)

### PIQA

**Source**: Google Cloud Storage (AI2)
**Examples**: 1,267 validation examples
**Format**: Sentence + 2 pronoun resolution choices
**Task**: Commonsense reasoning with ambiguous pronouns

**Example**:
```
Sentence: "The trophy doesn't fit into the brown suitcase because it is too large."
Question: What is too large?
Choices:
  A) the trophy  [CORRECT]
  B) the suitcase
```

**Characteristics**:
- Winograd schema challenge (pronoun disambiguation)
- Requires semantic understanding and reasoning
- Random baseline: 50% (2 choices)

---

## Results Analysis Guide

### Interpreting Your Results

After running evaluations, compare your results to the published baselines above. Here's how to interpret different outcomes:

#### ✅ Success Criteria

**All benchmarks within ±10% of baselines**:
- Training is successful and reproducible
- Model quality is research-grade
- Ready for publication or production use

**Example passing results for GPT-2 124M**:
- LAMBADA perplexity: 19.2 (within 18.5-22.6 range) ✅
- HellaSwag accuracy: 0.32 (within 0.28-0.34 range) ✅
- PIQA accuracy: 0.66 (within 0.58-0.70 range) ✅
- WinoGrande accuracy: 0.51 (within 0.47-0.57 range) ✅

#### ⚠️ Potential Issues

**LAMBADA perplexity >> 22.6**:
- Training may not have converged
- Check loss curves: should plateau at ~3.0 for GPT-2 124M
- May need more training steps or better hyperparameters

**All accuracies near random chance**:
- Model may have random weights (not trained)
- Verify checkpoint loading is working correctly
- Check that model is in eval mode (no dropout)

**One benchmark significantly worse**:
- Tokenization issues may affect specific datasets
- Verify dataset downloaded correctly
- Check evaluation implementation for that benchmark

#### 🔬 Research Notes

**NeuroManifold models vs Standard GPT**:
- Novel architectures may show different performance profiles
- SDR encoding may benefit long-range tasks (LAMBADA)
- Manifold projections may help with structured reasoning (WinoGrande)
- Compare results directly using same evaluation code

**Model scaling**:
- Larger models (355M, 774M) should show significant improvements
- Expected improvements: ~30-50% better across all benchmarks
- If scaling doesn't help, investigate architecture or training issues

---

## WandB Integration

### Logging to Weights & Biases

Enable WandB logging to track evaluation results across experiments:

```bash
# Set up WandB (first time only)
wandb login

# Run evaluation with logging
python eval.py --out_dir=out --benchmark=all \
  --wandb_log=True \
  --wandb_project=neuromanifold-evals \
  --wandb_run_name=gpt2-124m-baseline
```

### WandB Metrics Logged

**Per-benchmark metrics** (logged during evaluation):
- `lambada/perplexity`
- `lambada/loss`
- `lambada/accuracy`
- `hellaswag/accuracy`
- `piqa/accuracy`
- `winogrande/accuracy`

**Summary metrics** (logged at end):
- `summary/lambada_perplexity`
- `summary/lambada_accuracy`
- `summary/hellaswag_accuracy`
- `summary/piqa_accuracy`
- `summary/winogrande_accuracy`

**Run configuration**:
- `out_dir`, `benchmark`, `eval_iters`
- `device`, `dtype`, `seed`
- `checkpoint` (filename)

### Comparing Results

In WandB dashboard:
1. Navigate to your project
2. Create a table with runs to compare
3. Add columns for each benchmark metric
4. Filter/sort by perplexity or accuracy
5. Create charts showing results over training time

---

## Troubleshooting

### Common Issues

#### No checkpoints found
```
FileNotFoundError: No checkpoints found in out
```
**Solution**: Train a model first or specify correct checkpoint directory
```bash
# Train a model
python train.py config/train_gpt2.py

# Or use different directory
python eval.py --out_dir=path/to/checkpoints --benchmark=lambada
```

#### CUDA out of memory
```
RuntimeError: CUDA out of memory
```
**Solution**: Use CPU evaluation or reduce batch size (evaluation is done per-example by default, so this is rare)
```bash
# Use CPU
python eval.py --out_dir=out --benchmark=all --device=cpu

# Or use float16 instead of bfloat16
python eval.py --out_dir=out --benchmark=all --dtype=float16
```

#### Dataset download fails
```
requests.exceptions.ConnectionError: Failed to download dataset
```
**Solution**: Check internet connection and retry. Datasets are cached after first download in `~/.cache/neuromanifold_benchmarks/`

Manual download locations:
- LAMBADA: https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl
- HellaSwag: https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl
- PIQA: https://yonatanbisk.com/piqa/data/valid.jsonl + valid-labels.lst
- WinoGrande: https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1.zip

#### Perplexity is NaN or Inf
```
LAMBADA Results: Perplexity: nan
```
**Solution**: Model may have numerical instability
- Check model weights are loaded correctly
- Verify model is in eval mode (dropout disabled)
- Try float32 instead of float16: `--dtype=float32`
- Check for NaN in checkpoint: corrupted training run

#### Very high perplexity (100+)
```
LAMBADA Results: Perplexity: 142.35
```
**Explanation**: Model is not well-trained
- Random weights: 100-300+ perplexity
- Early checkpoint: 40-80 perplexity
- Well-trained: 18-23 perplexity

**Solution**: Continue training or use a later checkpoint

---

## Implementation Details

### Code Structure

```
neuromanifold_gpt/benchmarks/
├── __init__.py                 # Benchmark module exports
├── datasets.py                 # Dataset downloaders and loaders
├── zero_shot.py               # Evaluation implementations
└── test_benchmarks.py         # Unit tests

eval.py                         # CLI evaluation script
config/eval_lambada.py         # Example evaluation config
```

### Key Functions

**`evaluate_lambada(model, tokenizer, device, dtype, max_examples, verbose)`**
- Evaluates LAMBADA perplexity
- Returns: `{perplexity, loss, accuracy, num_examples}`

**`evaluate_multiple_choice(model, tokenizer, benchmark, device, dtype, max_examples, verbose)`**
- Evaluates HellaSwag, PIQA, or WinoGrande
- Returns: `{accuracy, num_examples}`

**`download_lambada()` / `download_hellaswag()` / `download_piqa()` / `download_winogrande()`**
- Downloads and caches datasets
- Returns: path to cached dataset file

### Model Compatibility

The evaluation suite supports:
- ✅ **NeuroManifoldGPT**: Models with NeuroManifoldConfig
- ✅ **Standard GPT**: nanoGPT-style models with GPTConfig
- ✅ **Mixed precision**: bfloat16, float16, float32
- ✅ **Device flexibility**: CUDA, CPU, multi-GPU
- ✅ **PyTorch 2.0**: Optional torch.compile support

---

## Validation Checklist

Before claiming your model is properly trained, verify:

- [ ] All benchmarks complete without errors
- [ ] No NaN or Inf values in results
- [ ] LAMBADA perplexity within 10% of baseline (18.5-22.6 for GPT-2 124M)
- [ ] HellaSwag accuracy within 10% of baseline (0.28-0.34 for GPT-2 124M)
- [ ] PIQA accuracy within 10% of baseline (0.58-0.70 for GPT-2 124M)
- [ ] WinoGrande accuracy within 10% of baseline (0.47-0.57 for GPT-2 124M)
- [ ] Results are reproducible (same seed = same results)
- [ ] Rich table displays correctly with all metrics
- [ ] WandB logging works (if enabled)
- [ ] Training loss converged to ~3.0 for GPT-2 124M

---

## Future Work

### Potential Extensions

1. **Additional Benchmarks**
   - ARC (AI2 Reasoning Challenge)
   - OpenBookQA
   - BoolQ, SQuAD, RACE
   - Math reasoning benchmarks (GSM8K, MATH)

2. **Few-Shot Evaluation**
   - Support for 1-shot, 5-shot, 10-shot evaluation
   - In-context learning performance tracking

3. **lm-evaluation-harness Integration**
   - Adapter for EleutherAI's lm-evaluation-harness
   - Access to 50+ benchmarks
   - Standardized evaluation protocol

4. **Performance Optimizations**
   - Batched evaluation for faster processing
   - Multi-GPU support for large models
   - Flash attention for long context

5. **Analysis Tools**
   - Per-example error analysis
   - Confidence calibration metrics
   - Benchmark correlation studies

---

## References

### Papers

1. **Language Models are Unsupervised Multitask Learners** (GPT-2)
   Radford et al., OpenAI, 2019
   https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

2. **HellaSwag: Can a Machine Really Finish Your Sentence?**
   Zellers et al., ACL 2019
   https://arxiv.org/abs/1905.07830

3. **PIQA: Reasoning about Physical Commonsense in Natural Language**
   Bisk et al., AAAI 2020
   https://arxiv.org/abs/1911.11641

4. **WinoGrande: An Adversarial Winograd Schema Challenge at Scale**
   Sakaguchi et al., AAAI 2020
   https://arxiv.org/abs/1907.10641

5. **The LAMBADA Dataset: Word Prediction Requiring Broad Context**
   Paperno et al., ACL 2016
   https://arxiv.org/abs/1606.06031

### Resources

- **OpenAI GPT-2 Repository**: https://github.com/openai/gpt-2
- **lm-evaluation-harness**: https://github.com/EleutherAI/lm-evaluation-harness
- **NeuroManifoldGPT Documentation**: See `README.md` and `docs/`

---

## Conclusion

This zero-shot benchmark evaluation suite provides a standardized, reproducible framework for validating language model quality. By comparing results against published baselines, researchers can confidently assess whether their training runs have succeeded and their models are ready for research or production use.

**Key Takeaways**:
- ✅ Four standard benchmarks (LAMBADA, HellaSwag, PIQA, WinoGrande)
- ✅ Zero-shot evaluation methodology matching published protocols
- ✅ Clear success criteria (within 10% of baselines)
- ✅ WandB integration for experiment tracking
- ✅ Support for both NeuroManifoldGPT and standard GPT models
- ✅ Comprehensive documentation and troubleshooting guide

For questions, issues, or contributions, please see the main repository README or open an issue on GitHub.

---

**Document Version**: 1.0
**Last Updated**: 2026-01-16
**Evaluation Code Version**: See `eval.py` and `neuromanifold_gpt/benchmarks/`
