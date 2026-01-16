# Dependency Upgrade Guide

This guide explains how to safely upgrade dependencies in NeuroManifoldGPT, test changes thoroughly, and rollback if needed.

## Table of Contents

- [Overview](#overview)
- [Before You Start](#before-you-start)
- [Upgrade Process](#upgrade-process)
- [Testing Procedure](#testing-procedure)
- [Updating Requirements Files](#updating-requirements-files)
- [CI Verification](#ci-verification)
- [Rollback Instructions](#rollback-instructions)
- [Common Scenarios](#common-scenarios)

## Overview

NeuroManifoldGPT uses pinned dependencies to ensure stability and reproducibility. When upgrading dependencies:

- **Always test locally first** before updating pinned versions
- **Run the full test suite** against multiple Python and PyTorch versions
- **Update documentation** to reflect any compatibility changes
- **Have a rollback plan** ready in case issues arise

## Before You Start

### Prerequisites

1. **Clean environment:** Use a fresh virtual environment for testing
2. **Backup current state:** Commit any uncommitted changes
3. **Review changelogs:** Check release notes for breaking changes
4. **Check CI status:** Ensure current main branch is passing

### Create a Test Branch

```sh
# Create and switch to upgrade branch
git checkout -b upgrade/dependencies-YYYY-MM-DD

# Verify you're on the new branch
git branch --show-current
```

## Upgrade Process

### Step 1: Identify Candidate Versions

Check for available updates:

```sh
# Activate your virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Check for outdated packages
pip list --outdated

# Or check specific packages
pip index versions torch
pip index versions lightning
pip index versions einops
```

Review the changelogs for each package you plan to upgrade:

- **PyTorch:** https://github.com/pytorch/pytorch/releases
- **Lightning:** https://github.com/Lightning-AI/pytorch-lightning/releases
- **Einops:** https://github.com/arogozhnikov/einops/releases
- **Scipy:** https://github.com/scipy/scipy/releases

### Step 2: Create Test Environment

```sh
# Create a fresh test environment
python3.11 -m venv test-venv
source test-venv/bin/activate  # On Windows: test-venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install current pinned dependencies first (baseline)
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Step 3: Test New Versions

Test one dependency at a time to isolate issues:

```sh
# Example: Testing PyTorch 2.3.1 (newer than pinned 2.2.2)
pip install torch==2.3.1

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# Run quick sanity check
python -c "import torch; x = torch.randn(2, 3); print('Torch OK')"
```

## Testing Procedure

### Local Testing

Run comprehensive tests to verify compatibility:

```sh
# 1. Run unit tests
pytest neuromanifold_gpt/tests/ -v

# 2. Run with coverage
pytest neuromanifold_gpt/tests/ --cov=neuromanifold_gpt --cov-report=term-missing

# 3. Check code quality (should still pass with new deps)
ruff check .
black --check .

# 4. Test training script (quick iteration)
python train.py config/train_shakespeare_char.py \
    --max_iters=100 \
    --eval_interval=50 \
    --eval_iters=10

# 5. Test model loading
python -c "from model import GPT; model = GPT.from_pretrained('gpt2'); print('Model OK')"
```

### Matrix Testing

Test against multiple Python and PyTorch versions locally:

```sh
# Test with different Python versions using pyenv or conda
# Python 3.10
python3.10 -m venv test-py310
source test-py310/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pytest neuromanifold_gpt/tests/ -v
deactivate

# Python 3.11
python3.11 -m venv test-py311
source test-py311/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pytest neuromanifold_gpt/tests/ -v
deactivate
```

### Document Test Results

Create a test log in your upgrade branch:

```sh
# Create test results file
cat > upgrade-test-results.md << 'EOF'
# Upgrade Test Results - YYYY-MM-DD

## Tested Configurations

### Python 3.10
- PyTorch 2.3.1: ✅ All tests pass
- Notes: [any issues or warnings]

### Python 3.11
- PyTorch 2.3.1: ✅ All tests pass
- Notes: [any issues or warnings]

## Dependency Changes
- torch: 2.2.2 → 2.3.1
- [other changes]

## Breaking Changes
- [list any breaking changes or required code updates]

## Performance Impact
- [note any performance changes observed]
EOF
```

## Updating Requirements Files

Once testing passes, update the pinned versions:

### Update requirements.txt

```sh
# Edit requirements.txt
# Change:
#   torch==2.2.2
# To:
#   torch==2.3.1

# Verify syntax
pip install -r requirements.txt --dry-run
```

### Update requirements-dev.txt

```sh
# Edit requirements-dev.txt if dev dependencies changed
# e.g., pytest==7.4.4 → pytest==8.0.0

# Verify syntax
pip install -r requirements-dev.txt --dry-run
```

### Update COMPATIBILITY.md

Update the compatibility matrix and version history:

```markdown
## Version History

| Date       | Python Versions | PyTorch Versions      | Notes                           |
|------------|-----------------|----------------------|----------------------------------|
| YYYY-MM    | 3.10, 3.11      | 2.0.1, 2.1.2, 2.2.2, 2.3.1 | Upgraded to PyTorch 2.3.1 |
| 2026-01    | 3.10, 3.11      | 2.0.1, 2.1.2, 2.2.2, 2.3.1 | Initial compatibility matrix |
```

## CI Verification

### Push and Monitor CI

```sh
# Commit your changes
git add requirements.txt requirements-dev.txt COMPATIBILITY.md
git commit -m "chore: upgrade dependencies to tested versions

- torch: 2.2.2 → 2.3.1
- [other changes]

Tested on Python 3.10, 3.11 with full test suite passing.
See upgrade-test-results.md for details."

# Push to GitHub
git push origin upgrade/dependencies-YYYY-MM-DD
```

### Monitor CI Pipeline

The CI pipeline (`.github/workflows/ci.yml`) will automatically test:

- Python 3.10 with PyTorch 2.0.1, 2.1.2, 2.2.2, 2.3.1
- Python 3.11 with PyTorch 2.0.1, 2.1.2, 2.2.2, 2.3.1
- All combinations (8 total test jobs)

**Wait for all CI jobs to pass** before merging. If any fail:

1. Review the CI logs to identify the failure
2. Reproduce locally with the same Python/PyTorch versions
3. Fix the issue or adjust the pinned versions
4. Push the fix and wait for CI to pass

### Create Pull Request

```sh
# Using GitHub CLI
gh pr create \
    --title "chore: upgrade dependencies to tested versions" \
    --body "$(cat upgrade-test-results.md)"

# Or manually on GitHub
# Include upgrade-test-results.md in the PR description
```

## Rollback Instructions

If you encounter issues after upgrading, here's how to rollback:

### Option 1: Rollback in Development

If you haven't committed yet:

```sh
# Discard changes to requirements files
git checkout requirements.txt requirements-dev.txt

# Reinstall pinned versions
pip install -r requirements.txt -r requirements-dev.txt --force-reinstall
```

### Option 2: Rollback Committed Changes

If you've committed but not pushed:

```sh
# Undo the last commit but keep changes
git reset HEAD~1

# Or discard the commit and changes
git reset --hard HEAD~1

# Reinstall pinned versions
pip install -r requirements.txt -r requirements-dev.txt --force-reinstall
```

### Option 3: Rollback Merged PR

If the upgrade was merged and caused issues in production:

```sh
# Find the commit hash before the upgrade
git log --oneline

# Create a revert commit
git revert <upgrade-commit-hash>

# Or cherry-pick the old requirements files
git checkout <commit-before-upgrade> -- requirements.txt requirements-dev.txt

# Commit the rollback
git add requirements.txt requirements-dev.txt
git commit -m "chore: rollback dependency upgrade due to [issue]

Reverts to previous tested versions:
- torch: 2.3.1 → 2.2.2
- [other changes]

Issue: [describe the problem encountered]"

# Push and create PR
git push origin rollback/dependencies-YYYY-MM-DD
gh pr create --title "Rollback dependency upgrade" --body "[explain issue]"
```

### Emergency Rollback

If you need to immediately fix a broken environment:

```sh
# Clone a fresh copy from main
git clone <repo-url> emergency-fix
cd emergency-fix

# Install known-good dependencies
pip install -r requirements.txt -r requirements-dev.txt

# Or use git to get previous requirements
git show main~5:requirements.txt > requirements-old.txt
pip install -r requirements-old.txt
```

### Verify Rollback

After rollback, verify everything works:

```sh
# Run tests
pytest neuromanifold_gpt/tests/ -v

# Check installed versions match requirements
pip freeze | grep -E "(torch|einops|lightning|scipy|loguru|rich)"
diff <(pip freeze | sort) <(cat requirements.txt | sort)

# Quick functionality check
python -c "import torch; from model import GPT; print('OK')"
```

## Common Scenarios

### Upgrading PyTorch

PyTorch upgrades are the most impactful. Special considerations:

```sh
# Check CUDA compatibility
python -c "import torch; print(torch.version.cuda)"

# Test CUDA still works after upgrade
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Benchmark performance (should be similar or better)
python bench.py  # if you have benchmarking scripts
```

### Upgrading Minor Versions

For minor version bumps (e.g., 2.2.2 → 2.2.3):

```sh
# Usually safe, but still test
pip install torch==2.2.3
pytest neuromanifold_gpt/tests/ -v

# If tests pass, update requirements.txt
```

### Upgrading Multiple Dependencies

Upgrade one at a time to isolate issues:

```sh
# Bad: upgrading everything at once
pip install --upgrade torch lightning einops scipy

# Good: upgrade incrementally
pip install torch==2.3.1
pytest -v  # verify
pip install lightning==2.3.0
pytest -v  # verify
pip install einops==0.8.0
pytest -v  # verify
```

### Security Updates

For security vulnerabilities, prioritize speed but maintain safety:

```sh
# Check for security issues
pip-audit  # requires: pip install pip-audit

# Apply security updates
pip install <package>==<patched-version>

# Run quick smoke tests
pytest neuromanifold_gpt/tests/test_core.py -v

# If passing, update requirements and merge quickly
git add requirements.txt
git commit -m "security: upgrade <package> to fix CVE-XXXX-XXXXX"
git push
gh pr create --title "Security: Fix CVE-XXXX-XXXXX" --body "Urgent security update"
```

## Best Practices

1. **Test locally first:** Never update requirements.txt without local testing
2. **One dependency at a time:** Easier to identify breaking changes
3. **Document everything:** Keep upgrade-test-results.md up to date
4. **Monitor CI:** Don't merge until all matrix tests pass
5. **Communicate:** Notify team members of planned upgrades
6. **Schedule appropriately:** Don't upgrade right before a release
7. **Keep backups:** Tag or branch before major upgrades

## Getting Help

If you encounter issues during an upgrade:

1. **Check existing issues:** Search GitHub issues for similar problems
2. **Review package changelogs:** Look for breaking changes
3. **Check compatibility matrix:** Verify your config is tested in COMPATIBILITY.md
4. **Ask the community:** Open a GitHub discussion
5. **Report bugs:** If you find a genuine issue, report it upstream

## References

- [COMPATIBILITY.md](COMPATIBILITY.md) - Tested configurations and known issues
- [requirements.txt](requirements.txt) - Pinned runtime dependencies
- [requirements-dev.txt](requirements-dev.txt) - Pinned development dependencies
- [.github/workflows/ci.yml](.github/workflows/ci.yml) - CI test matrix configuration
- [PyTorch Version Compatibility](https://pytorch.org/get-started/previous-versions/)
- [pip-audit Documentation](https://pypi.org/project/pip-audit/)
