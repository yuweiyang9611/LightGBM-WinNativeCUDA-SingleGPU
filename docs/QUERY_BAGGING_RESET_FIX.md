# Query bagging scores and resets — 2026-09-29

## Reproduced failures

- Query sampling expanded only selected queries into row indices. Non-subset
  score updates also need the unselected rows, so cached training scores drifted
  from model prediction on CUDA and the CPU non-subset path.
- Disabling sampling retained an old query list for gradient computation.
- Changing `bagging_by_query` incorrectly used the unchanged-config fast path,
  retaining row/query buffers with incompatible dimensions and causing crashes.
- Dataset replacement retained old query counts/boundaries. Increasing the
  thread count retained undersized prefix-sum scratch and also caused crashes.

## Repairs

- Expand the complete query permutation, keeping the selected query prefix to
  identify the training rows and the remaining rows for score updates.
- Return null sampled-query indices when sampling is disabled, requesting all
  query gradients. Initialize and clear the sampled-query count explicitly.
- Include query mode in reset change detection.
- Refresh query metadata and thread count during reset and resize thread scratch.

## Validation

- **18 focused CPU/CUDA cases passed**: full-data controls, score consistency at
  two sampling fractions, both query-mode transitions, disabling sampling,
  Dataset growth/shrinkage and a one-to-eight-thread transition.
- Before the fixes, 15 cases failed across the targeted reproduction runs;
  three full-data/subset controls passed.
- **18 memcheck bodies passed with 0 errors and 0 leaked bytes**.
- **38 C++ tests passed** after rebuilding without CUDA.
- **102 Python basic tests passed**, with 13 expected skips.
- The full Windows CUDA/source/version/build suite passed **322 tests**.
- Applicable pre-commit checks passed.

## Local artifacts

- DLL: `.builds/query-bagging-reset-20260929/artifacts/lib_lightgbm.dll`
- SHA256: `cf1f61563278f5ac7e45ac4ea7c7e342a468f1e1b47102d006a35c9cc2e70a72`
- Reproductions: `.builds/query-bagging-before.log`, `.builds/query-bagging-precondition.log`, `.builds/query-bagging-metadata-before.log`, `.builds/query-bagging-threads-before.log`
- Focused tests: `.builds/query-bagging-after.log`
- Memory check: `.builds/query-bagging-memcheck.log`
- C++ suite: `.builds/query-bagging-cpp.log`
- Python basic suite: `.builds/query-bagging-basic.log`
- Full Windows suite: `.builds/query-bagging-runtime.log`
- Local sanitizer runner: `.builds/query-bagging-reset-20260929/memcheck_regressions.py`

Earlier validated DLL artifacts are preserved.
