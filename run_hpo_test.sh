#!/bin/bash
# Wrapper script to run HPO test with isolated Python environment

# Run with clean environment and -s flag to avoid user site packages
env -i PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" HOME="$HOME" \
    .venv/bin/python -s run_hpo.py --config hpo_test.yaml --verbose
