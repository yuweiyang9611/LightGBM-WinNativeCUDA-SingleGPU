# GOSS warmup and subset transitions — 2026-09-29

## Reproduced failures

Lowering the learning rate can put GOSS back into its initial full-data warmup.
The sampler returned early without clearing the learner's previous sample.
CPU and CUDA consequently trained the next tree on fewer than the 2,048 input
rows, despite reporting a full-data iteration.

Changing GOSS sampling ratios also exposed both CPU subset transition errors:
leaving subset mode kept the old subset Dataset in the learner; entering it kept
the preceding original-row indices, which could cause invalid splits or heap
corruption when indexing the smaller Dataset.

## Repairs

- Track whether GOSS installed a subset in its learner. Restore the original
  Dataset before returning to warmup or switching to indexed full-data sampling.
- Clear sampled indices on every warmup iteration. CUDA data partitions now
  accept null indices as the same full-data reset supported by CPU partitions.
- Clear original-row indices when a serial learner enters compact subset mode.

## Validation

- Eight new cases cover CPU/CUDA, both sampling ratios, two full-data warmup
  updates, resumption of sampling and both directions of subset-mode changes.
  Warmup results match independently initialized continuation; mode changes
  match an explicit Dataset replacement. Cached scores agree with prediction.
- All four warmup cases failed before the fix. Both CPU mode transitions failed
  before the corresponding repair; both CUDA mode controls passed.
- **21 focused tests passed**, including the preceding configuration-recovery
  and GOSS sampling cases.
- **21 regression bodies passed memcheck with 0 errors and 0 leaked bytes**.
- **38 C++ tests passed** after a CPU rebuild.
- **102 Python basic tests passed**, with 13 expected skips.
- The full Windows CUDA/source/version/build suite passed **288 tests**.
- Applicable pre-commit checks passed.

## Local artifacts

- DLL: `.builds/goss-warmup-reset-20260929/artifacts/lib_lightgbm.dll`
- SHA256: `128e1c16c433c614c44e4bedf2eaf7ae5bdd8055f104fc06fc783120a6b559aa`
- Before-fix cases: `.builds/goss-reset-before.log` and `.builds/goss-mode-before.log`
- Focused tests: `.builds/goss-reset-focused.log`
- Memory check: `.builds/goss-reset-memcheck.log`
- C++ suite: `.builds/goss-reset-cpp.log`
- Python basic suite: `.builds/goss-reset-basic.log`
- Full Windows suite: `.builds/goss-reset-runtime.log`
- Local sanitizer runner: `.builds/goss-warmup-reset-20260929/memcheck_regressions.py`

Previous validated DLL artifacts are preserved.
