# Compatibility Matrix

This document outlines tested configurations, known compatibility issues, and recommended versions for NeuroManifoldGPT.

## Tested Configurations

NeuroManifoldGPT is continuously tested against multiple Python and PyTorch versions to ensure broad compatibility. Our CI pipeline runs all tests against the following configuration matrix:

### Version Matrix

| Python Version | PyTorch 2.0.1 | PyTorch 2.1.2 | PyTorch 2.2.2 | PyTorch 2.3.1 |
|----------------|---------------|---------------|---------------|---------------|
| 3.10           | ✅ Tested     | ✅ Tested     | ✅ Tested     | ✅ Tested     |
| 3.11           | ✅ Tested     | ✅ Tested     | ✅ Tested     | ✅ Tested     |

All combinations above are automatically tested on every commit and pull request.

### Core Dependencies

The following dependencies are pinned to specific tested versions:

| Package    | Version | Purpose                          |
|------------|---------|----------------------------------|
| torch      | 2.2.2   | Core deep learning framework     |
| einops     | 0.7.0   | Tensor operations                |
| lightning  | 2.2.0   | Training framework               |
| scipy      | 1.11.4  | Scientific computing             |
| loguru     | 0.7.2   | Logging utilities                |
| rich       | 13.7.0  | Terminal formatting              |

### Development Dependencies

| Package      | Version | Purpose                    |
|--------------|---------|----------------------------|
| pytest       | 7.4.4   | Testing framework          |
| pytest-cov   | 4.1.0   | Coverage reporting         |
| ruff         | 0.1.14  | Fast Python linter         |
| black        | 23.12.1 | Code formatter             |

## Recommended Configuration

For the best stability and performance, we recommend:

- **Python:** 3.11
- **PyTorch:** 2.2.2
- **CUDA:** 11.8 or 12.1 (if using GPU)
- **Install method:** `pip install -r requirements.txt`

This configuration receives the most thorough testing, including coverage analysis.

### Quick Start

```sh
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install pinned dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

## Known Issues

### PyTorch Version Compatibility

- **PyTorch 2.0.x:** Fully supported. Note that some newer PyTorch 2.3 features are not available.
- **PyTorch 2.1.x:** Fully supported. Recommended for stability.
- **PyTorch 2.2.x:** **Recommended version.** Best tested configuration.
- **PyTorch 2.3.x:** Fully supported. Latest features available.

### Python Version Compatibility

- **Python 3.9:** Not tested. May work but compatibility is not guaranteed.
- **Python 3.10:** Fully supported and tested.
- **Python 3.11:** **Recommended.** Fully supported with best performance.
- **Python 3.12:** Not yet tested. Compatibility unknown.

### Platform-Specific Notes

#### macOS Apple Silicon

When running on Apple Silicon (M1/M2/M3), you may want to use PyTorch with Metal Performance Shaders (MPS) support:

```sh
# Install PyTorch with MPS support
pip install torch==2.2.2  # MPS is included by default in 2.0+
```

Note: MPS support is experimental and some operations may fall back to CPU.

#### Windows

- CUDA support requires CUDA 11.8 or 12.1 toolkit installed
- Some file path operations may require adjustments for Windows path separators
- WSL2 is recommended for best compatibility

#### Linux

- Recommended platform for production use
- CUDA support requires matching CUDA toolkit (11.8 or 12.1)
- All features fully supported

### Dependency Conflicts

If you encounter dependency conflicts:

1. **Use a clean virtual environment:** Don't mix with other projects
2. **Install in order:** Install `requirements.txt` first, then `requirements-dev.txt`
3. **Check pip version:** Ensure pip is up to date: `pip install --upgrade pip`
4. **Verify pinned versions:** Run `pip freeze` and compare with `requirements.txt`

### Known Incompatibilities

- **PyTorch 1.x:** Not supported. Minimum version is PyTorch 2.0.1.
- **Lightning 1.x:** Not supported. Minimum version is Lightning 2.2.0.
- **Python 3.8 and earlier:** Not supported. Minimum version is Python 3.10.

## Testing Your Configuration

To verify your environment is correctly configured:

```sh
# Run the test suite
pytest neuromanifold_gpt/tests/ -v

# Check installed versions
pip freeze | grep -E "(torch|einops|lightning|scipy|loguru|rich)"

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

All tests should pass. If you encounter failures, please check:

1. You're using a tested configuration from the matrix above
2. All dependencies are at the pinned versions
3. Your Python version matches a tested version
4. CUDA toolkit (if using GPU) matches PyTorch CUDA version

## Upgrading Dependencies

When upgrading to newer versions of dependencies, please see [UPGRADE.md](UPGRADE.md) for detailed instructions on:

- Testing new versions locally
- Updating pinned versions in requirements files
- Running the full CI test matrix
- Rollback procedures if issues arise

## Reporting Compatibility Issues

If you discover a compatibility issue:

1. Check that you're using a tested configuration from this document
2. Verify all dependencies match the pinned versions
3. Search existing GitHub issues for similar reports
4. Open a new issue with:
   - Your Python version (`python --version`)
   - Your PyTorch version (`python -c "import torch; print(torch.__version__)"`)
   - Output of `pip freeze`
   - Full error message and stack trace
   - Steps to reproduce

## Version History

| Date       | Python Versions | PyTorch Versions      | Notes                           |
|------------|-----------------|----------------------|----------------------------------|
| 2026-01    | 3.10, 3.11      | 2.0.1, 2.1.2, 2.2.2, 2.3.1 | Initial compatibility matrix |

## References

- [PyTorch Compatibility Matrix](https://pytorch.org/get-started/previous-versions/)
- [Python Version Schedule](https://devguide.python.org/versions/)
- [requirements.txt](requirements.txt) - Pinned runtime dependencies
- [requirements-dev.txt](requirements-dev.txt) - Pinned development dependencies
- [.github/workflows/ci.yml](.github/workflows/ci.yml) - CI test configuration
