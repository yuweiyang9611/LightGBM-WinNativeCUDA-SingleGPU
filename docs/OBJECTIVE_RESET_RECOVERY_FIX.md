# Objective reset error recovery — 2026-09-29

## Reproduced failures

`Booster::ResetConfig()` wrote the new configuration and destroyed its existing
objective before initializing the replacement. When negative labels invalidated
Poisson or gamma training, or ranking data lacked query metadata, the boosting
engine retained a pointer to the destroyed objective. Subsequent prediction or
training crashed on both CPU and CUDA. A rejected unknown name also polluted
the configuration used by later Dataset replacement.

Separately, the CUDA objective factory fell off the end for unknown names,
returning an invalid pointer instead of raising the error used by the CPU
factory. This affected initial construction as well as runtime resets.

## Repairs

- Merge reset parameters into a temporary copy of the current configuration.
- Own the replacement objective locally through initialization and the boosting
  compatibility checks. Keep the old objective alive until those checks pass.
- Commit the replacement configuration after the boosting reset returns.
- Raise `Unknown objective type name` explicitly from the CUDA factory.

This repairs failures during objective construction, initialization and the
initial compatibility checks. Follow-up validation of tree-learner and sampling
configuration is documented in [the late validation fixes](LATE_CONFIG_VALIDATION_FIX.md).

## Validation

- Ten regression cases cover CPU/CUDA, invalid Poisson/gamma/ranking objectives,
  unknown names, construction failure and reset recovery. They verify unchanged
  predictions, continued training and metrics, subsequent valid parameter
  updates and Dataset replacement against an unaffected control model.
- The preceding DLL failed **9 cases** and passed the CPU construction control.
  The repaired DLL passed **all 10**.
- Compute Sanitizer memcheck with full leak checking passed all ten cases:
  **0 errors and 0 bytes leaked**.
- The Python basic suite passed **102 tests**, with **13 expected skips**.
- The complete Windows CUDA/source/version/build suite passed **267 tests**.
- Applicable pre-commit checks passed.

## Local artifacts

- DLL: `.builds/reset-config-recovery-20260929/artifacts/lib_lightgbm.dll`
- SHA256: `573effe6b490ab7b9a0d7a0cd1bd3293c15bcb7a9f88348e5a1f2d5921e03a57`
- Before-fix cases: `.builds/reset-config-before-final.log`
- After-fix cases: `.builds/reset-config-after.log`
- Memory check: `.builds/reset-config-memcheck.log`
- Python basic suite: `.builds/reset-config-basic.log`
- Full Windows suite: `.builds/reset-config-runtime.log`
- Local sanitizer runner: `.builds/reset-config-recovery-20260929/memcheck_regressions.py`

Earlier validated binaries and installed packages were preserved.
