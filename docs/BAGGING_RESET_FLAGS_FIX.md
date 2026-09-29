# Bagging reset flags and seeds — 2026-09-29

## Reproduced failures

- A second parameter reset cleared `need_re_bagging_` before deciding that the
  sampling configuration was unchanged. Splitting a configuration batch into
  two resets therefore skipped a required sample refresh when `bagging_freq`
  did not divide the current iteration. This affected both enabling bagging
  and changing an existing sampling fraction.
- Restoring the positive/negative fractions to one left balanced sampling
  enabled. The next tree used every row instead of the ordinary bagging fraction.
- Updating `bagging_seed` was accepted but ignored by the unchanged-config fast
  path; the previous random generators remained in use.

## Repairs

Preserve a pending sample refresh through no-op sampling resets, assign the
balanced flag from the current condition, and include the random seed in the
configuration comparison so a seed change rebuilds the generators.

## Validation

- Eight new CPU/CUDA cases reproduced the three failures. Pending refreshes
  compare split and combined configuration batches. Balanced-mode exit and seed
  changes compare against independently initialized continuation.
- **16 focused tests passed**, including the preceding eight ownership cases.
- **16 memcheck bodies passed with 0 errors and 0 leaked bytes**.
- **38 C++ tests passed** after rebuilding without CUDA.
- **102 Python basic tests passed**, with 13 expected skips.
- The full Windows CUDA/source/version/build suite passed **304 tests**.
- Applicable pre-commit checks passed.

## Local artifacts

- DLL: `.builds/bagging-reset-flags-20260929/artifacts/lib_lightgbm.dll`
- SHA256: `095f5642235c5cd1bf29421f90fb4f5812cd54b5af3e73d6490e6e788cc00a60`
- Before-fix cases: `.builds/bagging-flags-before.log` and `.builds/bagging-seed-before.log`
- Focused tests: `.builds/bagging-flags-focused.log`
- Memory check: `.builds/bagging-flags-memcheck.log`
- C++ suite: `.builds/bagging-flags-cpp.log`
- Python basic suite: `.builds/bagging-flags-basic.log`
- Full Windows suite: `.builds/bagging-flags-runtime.log`
- Local sanitizer runner: `.builds/bagging-reset-flags-20260929/memcheck_regressions.py`

Earlier validated DLL artifacts are preserved.
