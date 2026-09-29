# Query sampling with custom objectives — 2026-09-30

## Reproduced failure

With `bagging_by_query=true`, built-in objectives perform sampling in
`GBDT::Boosting()` before computing gradients. Custom gradients skip that method,
and the later sampling call also excluded query mode. Query sampling therefore
never ran for custom objectives, despite an active sampling fraction.

CPU and CUDA reproductions failed at both 0.3 and 0.8 fractions: the custom
objective did not match equivalent built-in regression and could fit all rows
instead of sampled queries.

## Repair

The training loop performs the later sampling step for custom objectives even
in query mode. Built-in query objectives keep their existing earlier sampling
step, so they are not sampled twice.

## Validation

- Four new CPU/CUDA cases passed after failing on the preceding DLL. Custom
  squared-error gradients match built-in regression with average initialization
  disabled; tree counts, sampled root counts and cached scores are checked.
- **22 regression bodies passed memcheck with 0 errors and 0 leaked bytes**,
  including the previous built-in query-sampling cases.
- **38 C++ tests passed** after rebuilding without CUDA.
- **102 Python basic tests passed**, with 13 expected skips.
- The full Windows CUDA/source/version/build suite passed **342 tests**.
- Applicable pre-commit checks passed.

## Local artifacts

- DLL: `.builds/custom-query-bagging-20260930/artifacts/lib_lightgbm.dll`
- SHA256: `21e0f4e315f45afb1d35ad2bea8847d12e33f7d45ba0cbed1065da7716e3ed4c`
- Before-fix cases: `.builds/custom-query-before.log`
- Focused tests: `.builds/custom-query-after.log`
- Memory check: `.builds/custom-query-memcheck.log`
- C++ suite: `.builds/custom-query-cpp.log`
- Python basic suite: `.builds/custom-query-basic.log`
- Full Windows suite: `.builds/custom-query-runtime.log`
- Local sanitizer runner: `.builds/custom-query-bagging-20260930/memcheck_regressions.py`

Previous validated DLL artifacts are preserved.
