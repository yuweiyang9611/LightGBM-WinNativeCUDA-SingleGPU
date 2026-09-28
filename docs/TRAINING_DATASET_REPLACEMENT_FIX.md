# Training Dataset replacement fixes — 2026-09-29

## Reproduced failures

- Quantized training retained buffers and sampling ranges sized for the original Dataset. Growing the training data reproduced CPU heap corruption and CUDA training failures.
- Python replaced its reference to the old Dataset before the native reset completed. Repeated replacements could free the old native Dataset while its bin mappers were still being checked, producing access violations or spurious alignment failures.
- Native reset replaced the objective before rejecting incompatible bin mappers. The rejection could leave the Booster unusable and crash a later operation.

## Repairs

- Reinitialize row-dependent quantization storage and Hessian constancy when replacing the training Dataset, on both CPU and CUDA. Preserve the random generator and iteration progress while updating the sampling range.
- Keep the previous Python Dataset alive through construction and native reset, and replace the Python reference only after success.
- Check bin-mapper compatibility before replacing native training data or the objective.

## Validation

- Twelve focused cases pass: ordinary/quantized CPU/CUDA training across row counts `256 → 4096 → 128 → 8192`, with weight changes; identical-data replacement preserving the random sequence; and incompatible Dataset rejection followed by successful continued training.
- Three general CPU regressions also cover repeated replacement and error recovery.
- **143 Windows CUDA/source/version/build tests and all 38 C++ tests passed.**
- **87 regression bodies passed Compute Sanitizer with 0 errors**, using direct-interpreter application-only tracking.
- Applicable pre-commit checks passed.

## Local artifacts

- DLL: `.builds/cuda-dataset-replacement-20260929/artifacts/lib_lightgbm.dll`
- SHA256: `c5eda54aa5430d955ddd4f44f925c1ff9a506fe8f04b6e5c988ec62f042cddb8`
- Full suite: `.builds/dataset-replacement-runtime.log`
- CPU regression: `.builds/dataset-replacement-engine.log`
- C++ suite: `.builds/dataset-replacement-cpp.log`
- Memory check: `.builds/dataset-replacement-memcheck.log`
- Local memory-check runner: `.builds/cuda-dataset-replacement-20260929/memcheck_regressions.py`

Earlier binaries and installed packages were preserved. This DLL contains the preceding fixes.
