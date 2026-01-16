# Finetuning Configuration Recipes

This directory contains tested hyperparameter configurations for finetuning pretrained GPT-2 models on custom datasets. These recipes provide good defaults to reduce the barrier to entry for finetuning and help you achieve good results without extensive hyperparameter search.

## Available Recipes

Each recipe is optimized for a specific GPT-2 model size:

- **gpt2_small.py** - GPT-2 Small (124M parameters)
- **gpt2_medium.py** - GPT-2 Medium (350M parameters)
- **gpt2_large.py** - GPT-2 Large (774M parameters)
- **gpt2_xl.py** - GPT-2 XL (1.5B parameters)

## Usage

To finetune using one of these recipes:

```sh
python train.py config/finetune/gpt2_small.py
```

## What's Included

Each recipe includes:

- **Optimized hyperparameters** - Learning rate, batch size, gradient accumulation steps tuned for finetuning (not training from scratch)
- **Memory efficiency** - Batch sizes configured for common GPU memory constraints
- **Training guidance** - Expected training time and final loss ranges documented in comments
- **Dataset size recommendations** - Suggested adjustments based on your dataset size

## Choosing a Recipe

**Model size selection:**
- **Small (124M)** - Fast iteration, good for experimentation and small datasets
- **Medium (350M)** - Better quality, reasonable training time
- **Large (774M)** - High quality, requires more GPU memory
- **XL (1.5B)** - Best quality, significant memory requirements

**Dataset size considerations:**
- **Small datasets (<1M tokens)** - Use smaller models (Small/Medium) with more training iterations
- **Medium datasets (1-10M tokens)** - Medium/Large models work well
- **Large datasets (>10M tokens)** - Can leverage larger models effectively

## Key Differences from Training from Scratch

These finetuning recipes differ from training configs in several ways:

1. **Lower learning rates** - Typically 10-100x lower than training from scratch to avoid catastrophic forgetting
2. **Shorter training** - Fewer iterations since we start from a pretrained checkpoint
3. **No learning rate decay** - Often use constant LR for finetuning
4. **Checkpoint initialization** - Start from pretrained weights via `init_from` parameter

## Example Workflow

1. Prepare your dataset (see `data/shakespeare/prepare.py` for an example)
2. Choose a recipe based on model size and dataset
3. Optionally adjust hyperparameters for your specific use case
4. Run training: `python train.py config/finetune/gpt2_small.py`
5. Sample from your finetuned model: `python sample.py --out_dir=out-finetune`

For a complete guide, see `docs/finetuning-guide.md`.

## Common Finetuning Scenarios

- **Domain adaptation** - Adapt a general-purpose model to a specific domain (legal, medical, code, etc.)
- **Instruction tuning** - Teach the model to follow instructions or perform specific tasks
- **Style transfer** - Finetune on text with a particular writing style (Shakespeare, technical writing, etc.)

## Performance Expectations

Expected results vary based on dataset and model size. As a baseline reference using the Shakespeare dataset (~300K tokens):

- Training typically completes in minutes to hours (depending on model size and hardware)
- Validation loss should decrease steadily during training
- Final loss typically in the 0.8-1.5 range for Shakespeare (lower is better)
- Generated samples should show clear characteristics of the training data

See individual recipe files for specific benchmarks and timing estimates.
