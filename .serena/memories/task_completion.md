# What to do when a task is completed

When finishing a development task in NeuroManifoldGPT, follow these steps to ensure quality and consistency:

1. **Format Code**: Run `black` on all modified files.
   ```sh
   black path/to/file.py
   ```

2. **Lint Code**: Run `ruff` to catch any stylistic or logical issues.
   ```sh
   ruff check path/to/file.py
   ```

3. **Run Tests**:
   - If you modified core model logic, run the relevant tests in `neuromanifold_gpt/tests/`.
   - Run the full test suite to ensure no regressions.
   ```sh
   pytest neuromanifold_gpt/tests/test_modified_component.py
   pytest
   ```

4. **Verify Benchmarks (if applicable)**: If the change affects performance, run `bench.py` or the relevant script in `neuromanifold_gpt/benchmarks/` to verify there is no regression in speed or MFU (Model FLOPs Utilization).

5. **Update Documentation**: If you added new features or changed configurations, update the relevant files in `docs/` and the `README.md` if necessary.

6. **Checkpoint Cleanup**: Ensure that any temporary training outputs or logs generated during testing are cleaned up or moved to the appropriate `out-*` directory.
