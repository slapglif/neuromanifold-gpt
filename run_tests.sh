#!/bin/bash
# Test runner for neuromanifold-gpt
# Ensures tests run with pinned dependencies in virtual environment
# without external PYTHONPATH interference

set -e

# Unset PYTHONPATH to avoid external package conflicts
unset PYTHONPATH

# Activate virtual environment and run tests
if [ -f .venv/bin/python3 ]; then
    echo "Running tests with pinned dependencies..."
    ./.venv/bin/python3 -m pytest neuromanifold_gpt/tests/ "$@"
else
    echo "Error: Virtual environment not found at .venv/"
    echo "Please run: python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt"
    exit 1
fi
