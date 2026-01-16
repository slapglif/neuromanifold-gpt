# NeuroManifoldGPT Project Overview

NeuroManifoldGPT is a research-oriented large language model project that extends Karpathy's `nanoGPT`. It incorporates advanced architectural features such as:
- **Soliton-Spectral Attention**: Utilizing physical wave dynamics (KdV, Sine-Gordon) for attention mechanisms.
- **SDR (Sparse Distributed Representation) Memory**: A biologically inspired memory system.
- **KAN (Kolmogorov-Arnold Networks)**: An alternative to standard MLPs.
- **MHC (Multi-Head Composition)**: A novel way of combining attention heads.
- **Kaufmann Attention**: A specific attention variant.

The project is designed to be simple, fast, and hackable, while achieving high performance (val_loss < 1.5 on Shakespeare in under 100s).

## Tech Stack
- **Language**: Python 3.10+
- **Deep Learning**: PyTorch 2.2+, PyTorch Lightning 2.2+
- **Utilities**: NumPy, SciPy, Einops, Tqdm
- **Logging/CLI**: Loguru, Rich, WandB
- **Tokenization**: Tiktoken, Transformers (for BPE/GPT-2)
- **Tooling**: Black (formatting), Ruff (linting), Pytest (testing)

## Codebase Structure
- `neuromanifold_gpt/`: Core package.
  - `model/`: Implementation of various attention types (FHN, Kaufmann, Soliton), KAN layers, memory systems, and the GPT architecture.
  - `training/`: Training utilities, Lightning modules, and data modules.
  - `config/`: Configuration system with a refactored Ralph Loop iteration framework.
  - `utils/`: Logging, checkpointing, and performance monitoring utilities.
  - `tests/`: Comprehensive test suite.
- `train.py`: Main entry point for training using PyTorch Lightning.
- `sample.py`: Inference script for generating text from trained models.
- `data/`: Data preparation scripts for Shakespeare, OpenWebText, etc.
- `bench.py`: Benchmarking and profiling script.
- `config/`: Original configuration files (partially refactored into `neuromanifold_gpt/config/`).
- `docs/`: Extensive documentation on finetuning, config system, etc.
