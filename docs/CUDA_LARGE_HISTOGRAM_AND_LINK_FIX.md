# CUDA incremental linking and large histogram fixes — 2026-09-29

## Reproduced failures

- Visual Studio incrementally rebuilt a CUDA object library without relinking its device code. A minimal program still returned `7` after its device function was changed to return `42`, despite a successful build. The real LightGBM build could also fail with mismatched CUDA registration symbols.
- Quantized CUDA training with 511 bins returned a constant model because the large-histogram dispatch branch did not launch a split search.
- Histogram repair considered only the first 512 bins and could request a reduction over more threads than had actually launched. The 1023-bin regressions exposed incorrect counts and learning failures.
- At 16,384 rows with four gradient quantization bins, histograms switch to wider packed entries. Split search still read them through a 32-bit pointer instead of a 64-bit pointer. The 16,383-row control passed while 16,384 and 32,768 rows failed.
- Ordinary CUDA large-histogram searches used doubled offsets into undersized scratch buffers and shared scratch space between concurrent forward/reverse tasks. Multi-feature regression cases failed with access violations.

## Repairs

- The shared Windows device-link helper adds the build directory to the final target's library search paths. NVIDIA's MSBuild `ResolvePaths` task can then discover the relative object-library inputs and track their timestamps. Both the DLL and CLI use the helper.
- Quantized split search scans block-sized tiles with a carried packed prefix total, comparing candidates across all tiles. It retains missing-value, extra-tree, regularization and feature-mask handling.
- Histogram repair accumulates all bins with a block-stride loop and reduces only the actual thread block. Reused shared memory is synchronized between reductions.
- Wide quantized histogram entries are read with their actual 64-bit storage type.
- Each large-histogram task receives an independent, sufficiently sized scratch range. Global prefix sums complete before other warps read their output.

## Validation

- **96 Windows CUDA/source/version/build tests passed**, including a real incremental CUDA build regression. It checks header changes, source changes, updated executable output, and that a no-op build leaves the device-link object unchanged.
- Large-bin tests cover 511/1023 bins, NaN and zero missing values, ordinary and extra trees, score consistency, and independent leaf counts/Hessian weights.
- Width-transition tests cover 16,383, 16,384 and 32,768 rows with 31/511 bins.
- Ordinary CUDA histograms are also checked at 1023 bins across two seeds.
- **40 regression bodies passed Compute Sanitizer, with 0 errors**, using the direct interpreter and `--target-processes application-only`.
- Applicable pre-commit checks passed. Real LightGBM rebuilds after CUDA source/header edits no longer required deleting the generated device-link object.

To run the build regression directly, set `LIGHTGBM_TEST_CMAKE` to `cmake.exe` and run `test_cuda_incremental_build.py`. The normal PowerShell build/test runner supplies this automatically.

## Local artifacts

- DLL: `.builds/cuda-large-bins-20260929/artifacts/lib_lightgbm.dll`
- SHA256: `ba6044996e8c5176332f75a7f6c0c64f7374f5976f795673934c21349acf57bb`
- Full suite: `.builds/large-bins-runtime-final.log`
- Memory check: `.builds/large-bins-memcheck.log`
- Local memory-check harness: `.builds/cuda-large-bins-20260929/memcheck_regressions.py`
- Initial incremental-link reproduction: `.builds/cuda-link-probe/`

Earlier documented binaries and installed packages were preserved. This DLL includes the preceding CUDA fixes.
