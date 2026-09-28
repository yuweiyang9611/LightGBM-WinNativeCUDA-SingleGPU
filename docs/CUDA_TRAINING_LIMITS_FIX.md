# CUDA training and version synchronization fixes — 2026-09-29

This change fixes the three issues reproduced after merging upstream through `f4777388`. The CUDA defects also reproduced with the pre-merge DLL.

## Changes

- Quantized gradients now use an `int16_t` device buffer, matching the two 16-bit values written per row. The inherited byte-pointer interface is preserved. Previously, the buffer held only half the required bytes.
- CUDA keeps host leaf depths current for numerical and categorical splits. Split search excludes leaves at `max_depth` and invalidates their cached candidates. Nonpositive limits remain unlimited; updated configuration applies to subsequent trees.
- The version synchronization script uses portable basic-regex repetition, so Python metadata updates alongside R and AppVeyor metadata.
- Memory validation exposed another overflow in the same training path: a child partition can require more blocks than its root after block-size rounding. Partition offset buffers now reserve space for that bound during construction and training-data resets.

## Validation

- The initial regression tests produced 11 expected failures on the old code.
- All 42 Windows CUDA/source/version tests passed, including quantized learning quality, numerical and categorical depth limits, unlimited depth, configuration resets, existing NaN/histogram regressions, native cubin and forced PTX JIT execution.
- CUDA memcheck passed four quantized training cases (512 and 1,025 rows, single and double precision), with **0 errors**. Before repair, it reported out-of-bounds writes in gradient discretization and then in partition offset preparation.
- Version tests cover stable, development and release-candidate versions.
- A 32-case probe passed all 24 supported parameter combinations; the other eight correctly rejected categorical quantized CUDA training.

## Local artifact

- DLL: `.builds/three-bug-fixes-20260929/artifacts/lib_lightgbm.dll`
- SHA256: `87482314eafce8f00910ce62df4e2666072ee9b299bfeca92be31c5e467a5026`
- Release build: CUDA 13.2, `USE_CUDA=ON`, `USE_NCCL=OFF`, `CMAKE_CUDA_ARCHITECTURES=89-real;89-virtual`.
- Logs: `.builds/three-bug-fixes-20260929-tests-final.log` and `.builds/three-bug-fixes-20260929-memcheck-final.log`.

The root baseline DLL and previously packaged wheels are not replaced by this build. Distribute the new DLL with its matching Python wrapper to deliver these fixes.

## Re-run memory checks

```powershell
$env:LIGHTGBM_CUDA_DLL = (Resolve-Path .builds/three-bug-fixes-20260929/artifacts/lib_lightgbm.dll).Path
& "$env:CUDA_PATH/compute-sanitizer/compute-sanitizer.exe" `
  --tool memcheck --target-processes all --error-exitcode 99 `
  .venv/Scripts/python.exe -m pytest -q `
  tests/win_native_cuda_tests/test_cuda_training_limits.py -k quantized_training
```

`--target-processes all` includes the isolated Python subprocesses that load the candidate DLL. Version synchronization tests require `sh`, `head` and `sed` on PATH (for example, from Git for Windows).
