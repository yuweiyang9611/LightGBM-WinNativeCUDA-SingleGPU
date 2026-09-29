# Late configuration validation and CUDA GOSS — 2026-09-29

## Reproduced failures

Several reset errors were detected only after part of a configuration batch had
been applied. A valid new objective combined with invalid feature contributions,
monotone-constraint dimensions, GOSS rates/bagging or random-forest sampling
changed predictions despite returning an error. A rejected CUDA device change
left the learner pointing at temporary configuration storage and changed
subsequent training results.

The recovery tests also exposed an independent GOSS failure: CUDA sampling
indices were never allocated on the device. Training failed when the warmup
ended, with or without the CPU subset optimization. That optimization also
compacted gradients despite the CUDA learner retaining original Dataset rows.

## Repairs

- Add read-only validation hooks to boosting, tree learners and sampling
  strategies. The C API checks the prospective objective/configuration before
  replacing the objective or touching training state.
- Share the existing feature-vector, GOSS, random-forest and CUDA-device checks
  through those hooks. Direct resets also validate before mutation.
- Use the stored feature count for vector validation, avoiding dependence on a
  training Dataset pointer when inspecting a loaded model.
- Allocate CUDA GOSS row-index storage during sampling setup/reset. Keep CUDA
  gradients in original row order and use device sampling indices instead of
  the CPU subset optimization.

These changes cover the reproduced validation failures. They do not promise
rollback after an allocation or device failure while applying an accepted
configuration.

## Validation

- Eleven configuration-rejection cases cover CPU/CUDA, feature-vector lengths,
  GOSS settings, random-forest settings and CUDA device changes. Each checks
  immediate prediction, continued training and a later valid reset against an
  unaffected model. Each reproduced a failure before the fix.
- Two additional tests train CUDA GOSS past warmup at sampling rates on both
  sides of the CPU subset threshold, checking loss and cached training scores.
  Both failed on the preceding DLL.
- These 13 cases plus the preceding 10 objective-recovery cases passed:
  **23 focused tests**, and **23 memcheck bodies with 0 errors and 0 leaked bytes**.
- **38 C++ tests passed** after rebuilding without CUDA.
- **102 Python basic tests passed**, with 13 expected skips.
- The complete Windows CUDA/source/version/build suite passed **280 tests**.
- Applicable pre-commit checks passed.

## Local artifacts

- DLL: `.builds/late-config-validation-20260929/artifacts/lib_lightgbm.dll`
- SHA256: `d8885f457e0548bd4d4c94145ecb4309b206f6a170cf52540642462a9f5b2cb1`
- Before-fix cases: `.builds/late-config-before.log`, `.builds/late-config-device-before.log`, `.builds/late-config-rf-before.log`, `.builds/goss-warmup-before.log`
- Focused tests: `.builds/late-config-focused.log`
- Memory check: `.builds/late-config-memcheck.log`
- C++ suite: `.builds/late-config-cpp.log`
- Python basic suite: `.builds/late-config-basic.log`
- Full Windows suite: `.builds/late-config-runtime.log`
- Local sanitizer runner: `.builds/late-config-validation-20260929/memcheck_regressions.py`

Earlier validated DLL artifacts and installed packages were preserved.
