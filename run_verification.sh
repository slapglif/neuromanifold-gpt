#!/bin/bash
# Wait for pip to finish and run verification

set -e

echo "Waiting for pip install to complete..."
sleep 90

source .venv/bin/activate

echo "Checking if torch is installed..."
python3 -c "import torch; print('✓ torch version:', torch.__version__)"

echo "Running integration test..."
python3 test_integration.py

echo "Done!"
