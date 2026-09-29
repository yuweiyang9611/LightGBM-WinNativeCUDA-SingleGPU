# Query bagging input boundaries — 2026-09-30

## Reproduced failures

Active query bagging accepted data without query metadata during construction,
parameter reset and Dataset replacement. It then sampled zero queries instead
of rejecting an unusable configuration.

Zero-size groups are accepted by Dataset metadata, so the query count can exceed
the row count. Query indices and random-generator blocks were sized by rows;
enough empty groups caused out-of-bounds access and process crashes.

## Repairs

- Validate sampling against the prospective Dataset and objective before
  changing training state. Active ordinary or balanced query sampling requires
  query metadata; inactive sampling remains valid without groups.
- Use the same validation in configuration reset, Dataset reset and sampler
  initialization. Rejected updates preserve the previous Booster.
- Size query-index storage and random-generator blocks by the number of queries.
  Accepted empty groups remain supported.

## Validation

- All **16 new CPU/CUDA cases passed**: ordinary/balanced activation, construction
  and reset errors, inactive controls, 2,080 groups over 128 rows, and recovery
  after an ungrouped Dataset replacement.
- The preceding DLL failed **12 cases** and passed four inactive controls.
- **34 memcheck bodies passed with 0 errors and 0 leaked bytes**, including the
  preceding query-reset, metadata and thread-change tests.
- **38 C++ tests passed** after rebuilding without CUDA.
- **102 Python basic tests passed**, with 13 expected skips.
- The full Windows CUDA/source/version/build suite passed **338 tests**.
- Applicable pre-commit checks passed.

## Local artifacts

- DLL: `.builds/query-input-validation-20260929/artifacts/lib_lightgbm.dll`
- SHA256: `f511119adbe7446b744c491842970b8054313c7e465e344d2abd0b96deb1216f`
- Before-fix cases: `.builds/query-inputs-before-final.log`
- Focused tests: `.builds/query-inputs-after.log`
- Memory check: `.builds/query-inputs-memcheck.log`
- C++ suite: `.builds/query-inputs-cpp.log`
- Python basic suite: `.builds/query-inputs-basic.log`
- Full Windows suite: `.builds/query-inputs-runtime.log`
- Local sanitizer runner: `.builds/query-input-validation-20260929/memcheck_regressions.py`

Earlier validated DLL artifacts are preserved.
