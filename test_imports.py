#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

try:
    from neuromanifold_gpt.config.ralph_base import RalphBaseConfig
    from neuromanifold_gpt.config.ralph_builder import RalphConfigBuilder
    print("Direct config imports: OK")
except Exception as e:
    print(f"Direct imports failed: {e}")
    sys.exit(1)

try:
    from neuromanifold_gpt.config.ralph_configs import get_ralph_config
    print("Registry import: OK")
except Exception as e:
    print(f"Registry import failed: {e}")
    sys.exit(1)

print("All imports successful!")
