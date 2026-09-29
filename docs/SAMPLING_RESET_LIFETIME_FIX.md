# Sampling reset ownership — 2026-09-29

## Reproduced failures

- GOSS replaced its temporary CPU Dataset during every configuration reset,
  even though the learner still referenced it. A second reset that changed the
  leaf budget dereferenced the freed Dataset and crashed.
- Ordinary CPU bagging retained its subset Dataset after switching to a larger
  sampling fraction or disabling sampling, producing invalid splits or counts.
- Disabling CUDA bagging freed its device indices but left the learner using
  that buffer on the next training iteration. The disabled sampling strategy
  also retained a pointer to the previous configuration.

## Repairs

- Reuse GOSS subset ownership across parameter-only resets. Replace its feature
  mapping only on a real Dataset change, after the learner has detached.
- Track whether ordinary bagging installed a learner subset and restore the
  full Dataset before leaving that mode.
- Explicitly clear learner sampling when bagging is disabled, and retain the
  current configuration while resetting the pending/balanced bagging flags.

## Validation

- Eight new cases cover CPU/CUDA, ordinary/quantized GOSS, consecutive parameter
  resets, bagging subset exit, disabling and re-enabling sampling. Results match
  combined resets or an explicit Dataset refresh; cached scores match prediction.
- The initial six-case reproduction failed four cases and passed two controls.
- **16 focused tests passed**, including the eight prior GOSS transition cases.
- **16 memcheck bodies passed with 0 errors and 0 leaked bytes**.
- **38 C++ tests passed** after rebuilding without CUDA.
- **102 Python basic tests passed**, with 13 expected skips.
- The full Windows CUDA/source/version/build suite passed **296 tests**.
- Applicable pre-commit checks passed.

## Local artifacts

- DLL: `.builds/sampling-lifetime-20260929/artifacts/lib_lightgbm.dll`
- SHA256: `0d9c694bffbfd90e04778f57b2fd6f0624378702e1b9307fd42895d1446c0e3b`
- Before-fix cases: `.builds/sampling-lifetime-before.log`
- Focused tests: `.builds/sampling-lifetime-focused.log`
- Memory check: `.builds/sampling-lifetime-memcheck.log`
- C++ suite: `.builds/sampling-lifetime-cpp.log`
- Python basic suite: `.builds/sampling-lifetime-basic.log`
- Full Windows suite: `.builds/sampling-lifetime-runtime.log`
- Local sanitizer runner: `.builds/sampling-lifetime-20260929/memcheck_regressions.py`

Earlier validated DLL artifacts are preserved.
