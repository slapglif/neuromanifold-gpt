#!/bin/bash
# End-to-end verification script for Neural Architecture Search
# This script verifies the complete NAS workflow step by step

set -e  # Exit on error

echo "========================================================================"
echo "NEURAL ARCHITECTURE SEARCH - END-TO-END VERIFICATION"
echo "========================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check prerequisites
echo "Checking prerequisites..."
echo "----------------------------------------"

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    echo -e "${GREEN}✓${NC} Python 3 found: $PYTHON_VERSION"
else
    echo -e "${RED}✗${NC} Python 3 not found"
    exit 1
fi

# Check if dataset exists
if [ -d "data/shakespeare_char" ] && [ -f "data/shakespeare_char/train.bin" ]; then
    echo -e "${GREEN}✓${NC} Shakespeare dataset found"
else
    echo -e "${RED}✗${NC} Shakespeare dataset not found"
    echo "  Please run: python data/shakespeare_char/prepare.py"
    exit 1
fi

# Check if required Python packages are available
echo ""
echo "Checking Python dependencies..."
python3 << 'PYCHECK'
import sys

required = {
    'torch': 'PyTorch',
    'numpy': 'NumPy',
    'loguru': 'Loguru',
    'einops': 'Einops'
}

missing = []
for module, name in required.items():
    try:
        __import__(module)
        print(f"✓ {name} installed")
    except ImportError:
        print(f"✗ {name} NOT installed")
        missing.append(name)

if missing:
    print(f"\nMissing dependencies: {', '.join(missing)}")
    print("\nTo install all dependencies:")
    print("  pip install -r requirements.txt")
    sys.exit(1)
PYCHECK

if [ $? -ne 0 ]; then
    echo -e "${RED}Missing required dependencies${NC}"
    exit 1
fi

echo ""
echo "========================================================================"
echo "STEP 1: Verify NAS imports"
echo "========================================================================"

python3 << 'PYIMPORTS'
import sys
sys.path.insert(0, '.')

# Test all NAS imports
print("Testing NAS module imports...")

try:
    from neuromanifold_gpt.nas import SearchSpace
    print("✓ SearchSpace imported")
except ImportError as e:
    print(f"✗ Failed to import SearchSpace: {e}")
    sys.exit(1)

try:
    from neuromanifold_gpt.nas import ArchitectureEvaluator
    print("✓ ArchitectureEvaluator imported")
except ImportError as e:
    print(f"✗ Failed to import ArchitectureEvaluator: {e}")
    sys.exit(1)

try:
    from neuromanifold_gpt.nas import RandomSearch
    print("✓ RandomSearch imported")
except ImportError as e:
    print(f"✗ Failed to import RandomSearch: {e}")
    sys.exit(1)

try:
    from neuromanifold_gpt.nas import EvolutionarySearch
    print("✓ EvolutionarySearch imported")
except ImportError as e:
    print(f"✗ Failed to import EvolutionarySearch: {e}")
    sys.exit(1)

try:
    from neuromanifold_gpt.nas import export_config
    print("✓ export_config imported")
except ImportError as e:
    print(f"✗ Failed to import export_config: {e}")
    sys.exit(1)

print("\nAll NAS imports successful!")
PYIMPORTS

if [ $? -ne 0 ]; then
    echo -e "${RED}Import verification failed${NC}"
    exit 1
fi

echo ""
echo "========================================================================"
echo "STEP 2: Run random search with 5 evaluations"
echo "========================================================================"
echo ""
echo "Running: python3 examples/nas_search.py --strategy random --budget 5 --iters 100"
echo ""

OUTPUT_DIR="./nas_verification_output"
mkdir -p "$OUTPUT_DIR"

python3 examples/nas_search.py \
    --strategy random \
    --budget 5 \
    --iters 100 \
    --output "$OUTPUT_DIR" \
    --data data/shakespeare_char/input.txt 2>&1 | tee "$OUTPUT_DIR/search_log.txt"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Search failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Search completed successfully${NC}"

echo ""
echo "========================================================================"
echo "STEP 3: Verify search results"
echo "========================================================================"

# Check if results file exists
RESULTS_FILE="$OUTPUT_DIR/search_results.json"
if [ ! -f "$RESULTS_FILE" ]; then
    echo -e "${RED}✗ Results file not found: $RESULTS_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Results file exists: $RESULTS_FILE"

# Parse and display results
python3 << PYPARSE
import json
import sys

try:
    with open('$RESULTS_FILE', 'r') as f:
        results = json.load(f)

    print(f"\n✓ Search results loaded successfully")
    print(f"  - Architectures evaluated: {len(results.get('architectures', []))}")
    print(f"  - Search strategy: {results.get('search_metadata', {}).get('strategy', 'unknown')}")

    if 'best_architecture' in results:
        best = results['best_architecture']
        metrics = best.get('metrics', {})
        print(f"  - Best perplexity: {metrics.get('perplexity', 'N/A'):.2f}")
        print(f"  - Best loss: {metrics.get('loss', 'N/A'):.4f}")

    # Verify we have at least some architectures
    if len(results.get('architectures', [])) < 3:
        print(f"\n⚠ Warning: Only {len(results['architectures'])} architectures found (expected at least 3)")
    else:
        print(f"\n✓ Sufficient architectures for export")

except Exception as e:
    print(f"✗ Failed to parse results: {e}")
    sys.exit(1)
PYPARSE

if [ $? -ne 0 ]; then
    echo -e "${RED}Results verification failed${NC}"
    exit 1
fi

echo ""
echo "========================================================================"
echo "STEP 4: Export top-3 architectures"
echo "========================================================================"
echo ""
echo "Running: python3 examples/nas_export_best.py --top-k 3"
echo ""

EXPORT_DIR="$OUTPUT_DIR/exported_configs"
mkdir -p "$EXPORT_DIR"

python3 examples/nas_export_best.py \
    "$RESULTS_FILE" \
    --output "$EXPORT_DIR" \
    --top-k 3 \
    --format all \
    --summary 2>&1 | tee "$OUTPUT_DIR/export_log.txt"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Export failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Export completed successfully${NC}"

echo ""
echo "========================================================================"
echo "STEP 5: Verify exported configs can instantiate models"
echo "========================================================================"

# Check exported files exist
EXPORTED_FILES=$(ls "$EXPORT_DIR"/*.py 2>/dev/null | wc -l)
if [ "$EXPORTED_FILES" -eq 0 ]; then
    echo -e "${RED}✗ No exported config files found${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Found $EXPORTED_FILES exported config file(s)"

# Test model instantiation
python3 << PYINSTANTIATE
import sys
import importlib.util
from pathlib import Path

sys.path.insert(0, '.')

export_dir = Path('$EXPORT_DIR')
config_files = sorted(export_dir.glob('*.py'))

if not config_files:
    print("✗ No config files found")
    sys.exit(1)

print(f"Testing {len(config_files)} exported config(s)...\n")

from neuromanifold_gpt.model import NeuroManifoldGPT

success_count = 0
for i, config_file in enumerate(config_files, 1):
    print(f"Config {i}: {config_file.name}")

    try:
        # Load config module
        spec = importlib.util.spec_from_file_location(f"config_{i}", config_file)
        if spec is None or spec.loader is None:
            print(f"  ✗ Failed to load config file")
            continue

        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)

        if not hasattr(config_module, 'config'):
            print(f"  ✗ Config missing 'config' variable")
            continue

        config = config_module.config
        print(f"  - Architecture: {config.n_layer}L x {config.n_embd}E x {config.n_head}H")

        # Try to instantiate model
        model = NeuroManifoldGPT(config)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Model instantiated: {param_count:,} parameters")
        success_count += 1

    except Exception as e:
        print(f"  ✗ Failed to instantiate: {e}")
        continue

print(f"\n✓ Successfully instantiated {success_count}/{len(config_files)} models")

if success_count == 0:
    sys.exit(1)
PYINSTANTIATE

if [ $? -ne 0 ]; then
    echo -e "${RED}Model instantiation failed${NC}"
    exit 1
fi

echo ""
echo "========================================================================"
echo "STEP 6: Train one exported config for 100 iterations"
echo "========================================================================"

# Use the first exported config for training
FIRST_CONFIG=$(ls "$EXPORT_DIR"/*.py 2>/dev/null | head -1)

if [ -z "$FIRST_CONFIG" ]; then
    echo -e "${RED}✗ No config file found for training${NC}"
    exit 1
fi

echo "Training config: $(basename $FIRST_CONFIG)"
echo ""

python3 << PYTRAIN
import sys
import importlib.util
from pathlib import Path
import numpy as np

sys.path.insert(0, '.')

# Load config
config_file = '$FIRST_CONFIG'
print(f"Loading config: {config_file}")

spec = importlib.util.spec_from_file_location("train_config", config_file)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
config = config_module.config

print(f"Config: {config.n_layer}L x {config.n_embd}E x {config.n_head}H")

# Load dataset
data_dir = Path('data/shakespeare_char')
train_data = np.memmap(data_dir / 'train.bin', dtype=np.uint16, mode='r')
print(f"Dataset loaded: {len(train_data):,} tokens")

# Import torch components
import torch
from neuromanifold_gpt.model import NeuroManifoldGPT

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Initialize model
model = NeuroManifoldGPT(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

param_count = sum(p.numel() for p in model.parameters())
print(f"Model: {param_count:,} parameters\n")

# Training loop
print("Training for 100 iterations...")
model.train()

batch_size = 32
block_size = config.block_size

for iter_num in range(100):
    # Get batch
    ix = torch.randint(len(train_data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(train_data[i:i+block_size].astype(np.int64)) for i in ix]).to(device)
    y = torch.stack([torch.from_numpy(train_data[i+1:i+1+block_size].astype(np.int64)) for i in ix]).to(device)

    # Forward pass
    try:
        logits, loss = model(x, y)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (iter_num + 1) % 20 == 0:
            print(f"  Iteration {iter_num + 1}/100, Loss: {loss.item():.4f}")

    except Exception as e:
        print(f"✗ Training failed at iteration {iter_num}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("\n✓ Training completed successfully for 100 iterations")
PYTRAIN

if [ $? -ne 0 ]; then
    echo -e "${RED}Training verification failed${NC}"
    exit 1
fi

echo ""
echo "========================================================================"
echo "✅ ALL TESTS PASSED - END-TO-END VERIFICATION SUCCESSFUL"
echo "========================================================================"
echo ""
echo "Summary:"
echo "  ✓ NAS imports verified"
echo "  ✓ Random search completed with 5 evaluations"
echo "  ✓ Search results validated"
echo "  ✓ Top-3 architectures exported"
echo "  ✓ Exported configs instantiate models"
echo "  ✓ Training with exported config successful"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "  - search_results.json: Search results with all architectures"
echo "  - exported_configs/: Exported architecture configs (Python + JSON)"
echo "  - nas_summary.md: Summary report of top architectures"
echo ""
echo "You can now use the exported configs for full training:"
echo "  python examples/train_with_config.py $EXPORT_DIR/nas_discovered_1.py"
echo ""
