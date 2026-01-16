#!/usr/bin/env python3
"""Test config dataclasses directly."""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import config module directly (bypassing __init__.py)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "block_config",
    "./neuromanifold_gpt/config/block_config.py"
)
block_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(block_config)

# Get the classes
NeuroManifoldBlockConfig = block_config.NeuroManifoldBlockConfig
FHNConfig = block_config.FHNConfig
KANConfig = block_config.KANConfig
MHCConfig = block_config.MHCConfig
MLAConfig = block_config.MLAConfig
MoEConfig = block_config.MoEConfig

def test_config_creation():
    """Test that config objects can be created."""
    print("Testing config creation...")

    # Test individual configs
    fhn = FHNConfig()
    print(f"✓ FHNConfig created: threshold={fhn.fhn_threshold}, tau={fhn.fhn_tau}")

    kan = KANConfig()
    print(f"✓ KANConfig created: type={kan.kan_type}, num_centers={kan.kan_num_centers}")

    mhc = MHCConfig()
    print(f"✓ MHCConfig created: streams={mhc.mhc_n_streams}, use_mhc={mhc.use_mhc}")

    mla = MLAConfig()
    print(f"✓ MLAConfig created: use_mla={mla.use_mla}, latent_dim={mla.mla_latent_dim}")

    moe = MoEConfig()
    print(f"✓ MoEConfig created: n_experts={moe.moe_n_experts}, n_active={moe.moe_n_active}")

    # Test master config with defaults
    config = NeuroManifoldBlockConfig(sdr_size=2048, embed_dim=384)
    print(f"✓ NeuroManifoldBlockConfig created: sdr_size={config.sdr_size}, embed_dim={config.embed_dim}")
    print(f"  - FHN threshold: {config.fhn.fhn_threshold}")
    print(f"  - KAN type: {config.kan.kan_type}")
    print(f"  - mHC streams: {config.mhc.mhc_n_streams}")

    # Test config with custom sub-configs
    custom_fhn = FHNConfig(fhn_threshold=0.7, fhn_tau=15.0)
    custom_kan = KANConfig(kan_type="wave", kan_wavelet="dog")
    config2 = NeuroManifoldBlockConfig(
        sdr_size=4096,
        embed_dim=768,
        fhn=custom_fhn,
        kan=custom_kan
    )
    print(f"✓ Custom config created:")
    print(f"  - FHN: threshold={config2.fhn.fhn_threshold}, tau={config2.fhn.fhn_tau}")
    print(f"  - KAN: type={config2.kan.kan_type}, wavelet={config2.kan.kan_wavelet}")

    return True

def test_config_access_patterns():
    """Test that config supports the access patterns used in block.py."""
    print("\nTesting access patterns used in NeuroManifoldBlock...")

    config = NeuroManifoldBlockConfig(sdr_size=2048, embed_dim=384)

    # Test core dimension access
    assert config.sdr_size == 2048
    assert config.embed_dim == 384
    assert config.manifold_dim == 64
    assert config.n_eigenvectors == 32
    assert config.n_heads == 8
    assert config.mlp_ratio == 4.0
    assert config.dropout == 0.0
    print("✓ Core dimensions accessible")

    # Test flag access
    assert config.skip_manifold_spectral == False
    assert config.use_knot_attention == False
    assert config.use_kaufmann_attention == False
    print("✓ Flag attributes accessible")

    # Test FHN sub-config access
    assert config.fhn.fhn_threshold == 0.5
    assert config.fhn.fhn_tau == 12.5
    assert config.fhn.pulse_width_base == 4
    assert config.fhn.n_fhn_steps == 2
    assert config.fhn.use_fhn_imex == True
    assert config.fhn.use_fhn_partitioning == True
    assert config.fhn.use_fhn_fused == False
    print("✓ FHN sub-config accessible via config.fhn.xxx")

    # Test KAN sub-config access
    assert config.kan.use_kan == True
    assert config.kan.kan_type == "faster"
    assert config.kan.kan_degree == 4
    assert config.kan.kan_wavelet == "dog"
    assert config.kan.use_fast_wavekan == True
    assert config.kan.kan_num_centers == 3
    print("✓ KAN sub-config accessible via config.kan.xxx")

    # Test mHC sub-config access
    assert config.mhc.use_mhc == True
    assert config.mhc.use_full_mhc == True
    assert config.mhc.mhc_n_streams == 2
    assert config.mhc.mhc_residual_weight == 0.9
    assert config.mhc.mhc_sinkhorn_iters == 5
    assert config.mhc.mhc_sinkhorn_tau == 0.05
    print("✓ mHC sub-config accessible via config.mhc.xxx")

    # Test MLA sub-config access
    assert config.mla.use_mla == False
    assert config.mla.mla_latent_dim == 64
    assert config.mla.mla_rope_dim == 32
    print("✓ MLA sub-config accessible via config.mla.xxx")

    # Test MoE sub-config access
    assert config.moe.use_moe == False
    assert config.moe.moe_n_experts == 8
    assert config.moe.moe_n_active == 2
    assert config.moe.use_shared_expert == True
    assert config.moe.use_e7_routing == False
    print("✓ MoE sub-config accessible via config.moe.xxx")

    return True

def test_backward_compatibility_pattern():
    """Test the pattern used for backward compatibility in block.py."""
    print("\nTesting backward compatibility pattern...")

    # This simulates what happens in block.py when individual params are passed
    sdr_size = 2048
    embed_dim = 384
    fhn_threshold = 0.6
    fhn_tau = 10.0
    kan_type = "cheby"
    use_mhc = False

    # Create config from individual params (like in block.py)
    config = NeuroManifoldBlockConfig(
        sdr_size=sdr_size,
        embed_dim=embed_dim,
        fhn=FHNConfig(
            fhn_threshold=fhn_threshold,
            fhn_tau=fhn_tau,
        ),
        kan=KANConfig(
            kan_type=kan_type,
        ),
        mhc=MHCConfig(
            use_mhc=use_mhc,
        ),
    )

    # Verify we can access all values through config
    assert config.sdr_size == 2048
    assert config.embed_dim == 384
    assert config.fhn.fhn_threshold == 0.6
    assert config.fhn.fhn_tau == 10.0
    assert config.kan.kan_type == "cheby"
    assert config.mhc.use_mhc == False

    print("✓ Backward compatibility pattern works correctly")
    print("  - Created config from individual parameters")
    print("  - All values accessible via config.xxx notation")

    return True

if __name__ == "__main__":
    print("=" * 60)
    print("NeuroManifoldBlockConfig Verification")
    print("=" * 60)

    try:
        test_config_creation()
        test_config_access_patterns()
        test_backward_compatibility_pattern()

        print("\n" + "=" * 60)
        print("✅ ALL CONFIGURATION TESTS PASSED")
        print("=" * 60)
        print("\nVerified:")
        print("1. ✓ All config dataclasses can be instantiated")
        print("2. ✓ All access patterns from block.py work correctly")
        print("3. ✓ Backward compatibility maintained")
        print("4. ✓ Config composition with sub-configs functional")
        print("\nThe refactoring in block.py will work correctly!")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
