# Suggested Commands for NeuroManifoldGPT

## Setup and Environment
```sh
# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

## Data Preparation
```sh
# Prepare Shakespeare character-level dataset
python data/shakespeare_char/prepare.py

# Prepare OpenWebText (for larger models)
python data/openwebtext/prepare.py
```

## Training
```sh
# Basic training on Shakespeare
python train.py --config config/train_neuromanifold_shakespeare.py

# Training with overrides
python train.py --config config/train_neuromanifold_shakespeare.py --batch_size 32 --max_iters 2000

# Training with baseline GPT (non-NeuroManifold)
python train.py --config config/train_neuromanifold_shakespeare.py --model_type gpt
```

## Inference / Sampling
```sh
# Sample from a trained model
python sample.py --out_dir=out-shakespeare-char

# Sample from a specific prompt
python sample.py --out_dir=out-shakespeare-char --start="ROMEO:"

# Sample from a pretrained GPT-2 model
python sample.py --init_from=gpt2-xl
```

## Benchmarking and Profiling
```sh
# Run basic benchmark
python bench.py

# Benchmarking specific attention variants
python neuromanifold_gpt/bench_attention_variants.py
```

## Quality Control
```sh
# Run all tests
pytest

# Run tests with coverage
pytest --cov=neuromanifold_gpt

# Linting
ruff check .

# Formatting
black .
```
