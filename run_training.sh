#!/bin/bash
# Training wrapper script with proper environment setup

# Activate virtual environment
source .venv/bin/activate

# Set PYTHONPATH to prioritize venv packages but include system packages as fallback
VENV_SITE_PACKAGES="$(pwd)/.venv/lib/python3.12/site-packages"
SYSTEM_PACKAGES="/home/mikeb/Applications/Auto-Claude/resources/python-site-packages"
export PYTHONPATH="$VENV_SITE_PACKAGES:$SYSTEM_PACKAGES"

# Run training
python3 train.py "$@"
