# CUDA percentile and small-data fixes — 2026-09-29

## Reproduced failures

- All 18 initial percentile probes disagreed with an independent NumPy calculation, while their CPU controls passed. The CUDA unweighted position formula used the wrong sample-count offset.
- Weighted prefix scans discarded the exclusive prefix result, read shared memory by thread index instead of using the returned value, and failed to accumulate all values assigned to a thread. The multi-block scan also left its leading zero uninitialized.
- Weighted selection used a different ordering/interpolation convention from the CPU implementation. Endpoint handling used unsorted identities and could fall through to a negative-index access.
- Compute Sanitizer reported 124 CUDA API errors from zero-grid initialization launches after constant features were filtered out, even though functional predictions passed.
- A one-row Dataset was incorrectly rejected by a partition block-size assertion.

## Repairs

- Use the CPU-compatible unweighted percentile position and ascending weighted cumulative distribution, including the existing CPU rule for weight intervals below one.
- Form complete per-thread and cross-block cumulative weights, initializing the block-prefix origin explicitly.
- Share weighted selection between initialization and leaf renewal; return sorted endpoints safely and terminate the single-value path immediately.
- Skip empty category/random initialization launches and accept the valid one-row partition size.

## Validation

- 62 percentile cases cover initialization and L1/quantile leaf renewal, integer/fractional weights, endpoint identities, constant features with extra trees, one-row data, and a 1,049,601-row multi-block scan.
- The complete 203-test Windows suite passed; the two subsequently added one-row cases passed on the final build (**205 distinct cases**).
- Compute Sanitizer passed 147 regression bodies with zero errors after the empty-launch fix. The two one-row cases then passed on the final build with zero errors (**149 distinct cases**).
- Applicable pre-commit checks passed.

## Local artifacts

- DLL: `.builds/cuda-percentiles-20260929/artifacts/lib_lightgbm.dll`
- SHA256: `a2ed1a4e436fc350ea0879af3be438e4ec4eae7eff6c367c565795def677a8d9`
- Full suite: `.builds/percentile-runtime-final.log`
- Single-row tests: `.builds/percentile-single-row-after.log`
- Memory checks: `.builds/percentile-memcheck-final.log` and `.builds/percentile-single-row-memcheck.log`
- Local runners: `.builds/cuda-percentiles-20260929/memcheck_regressions.py` and `memcheck_single_row.py`.

The DLL includes preceding fixes. Earlier binaries and installed packages were preserved.
