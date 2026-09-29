# Dataset metadata refresh and reset validation — 2026-09-29

## Reproduced failures

- Native training-data reset ignored metadata changes when the Dataset pointer was unchanged. Adding/removing weights or changing initial scores produced different training results from an equivalent Dataset replacement.
- Python setters did not forward `None` for weights and initial scores, and clearing a field did not increment the Dataset version. Unit-weight Python lists were not normalized consistently with NumPy arrays.
- Device weight caches and query-level weights could retain cleared values.
- Invalid replacement labels or initial-score dimensions could leave the Booster with a dangling objective. A rejected multiclass reset changed subsequent predictions from probabilities into raw scores or crashed later operations.

## Repairs

- Explicit Dataset reset refreshes objectives, metrics, training scores and learner state even for the same object. The internal reset-state flag is also forwarded by random forests.
- Optional setters clear their native fields and field clearing advances the version. Row/query weight caches and initial-score device storage are cleared consistently.
- Replacement objectives and metrics are initialized in temporary ownership. Objective compatibility and initial-score shape are validated before the boosting state is changed.

## Validation

- 36 focused cases cover CPU/CUDA, ordinary/quantized training, ranking, random forests, weight addition/removal, unit weights, labels, initial-score addition/removal, and recovery after invalid labels or initial-score shapes.
- **241 Windows CUDA/source/version/build tests and all 38 C++ tests passed.**
- **13 newly selected generic CPU regressions passed.** The complete Python basic suite passed **102 tests**, with **13 expected CUDA/dependency skips**; some new cases are included in that basic-suite count.
- **185 regression bodies passed Compute Sanitizer with 0 errors**, using direct-interpreter application-only tracking.
- Applicable pre-commit checks passed.

## Local artifacts

- DLL: `.builds/cuda-metadata-20260929/artifacts/lib_lightgbm.dll`
- SHA256: `f1f112c24016ef9215c7faaa563d4277948133384808f0f25f979afc0eea12ae`
- Full suite: `.builds/metadata-runtime.log`
- Generic CPU checks: `.builds/metadata-generic.log` and `.builds/metadata-basic.log`
- C++ suite: `.builds/metadata-cpp.log`
- Memory check: `.builds/metadata-memcheck.log`
- Local runner: `.builds/cuda-metadata-20260929/memcheck_regressions.py`

Earlier DLLs and installed packages were preserved. This build contains the preceding fixes.
