# CUDA categorical search settings — 2026-09-29

## Reproduced failures

CUDA grouped categorical search ignored `min_data_per_group` in both the
shared-memory and global-memory paths. It could split even when this minimum
exceeded the entire Dataset size, and selected boundaries that the configured
group interval should exclude.

The global-memory path generated a random threshold for `extra_trees` but did
not use it to filter candidates. Eight different seeds all selected the same
unique optimum. Both grouped paths also let every thread update the shared
random threshold and the same random-number generator state.

## Repair

- Build a candidate mask using the ordered categorical counts, leaf minima,
  Hessian minima and minimum group size. Reset the current group's count only
  at an eligible boundary, before random-threshold filtering, matching the CPU
  search. Reuse the discarded sorting-score buffer for this mask.
- Apply random-threshold filtering to both directions of global-memory search.
- Generate each grouped random threshold in thread zero and synchronize before
  reading it.

## Regression coverage

Ten cases exercise 64, 512 and 1,024 categories, oversized group minima,
three-category group intervals, nontrivial leaf minima, randomized thresholds
and repeated seeds. The group-boundary checks enumerate allowed subsets and
independently compute their squared loss, accounting for categories merged
into the default bin. CPU results provide an additional control.

Against the preceding DLL, **8 cases failed and 2 passed**. All ten passed after
the repair, including under Compute Sanitizer memcheck with **0 errors**.
Racecheck also passed all ten cases with **0 errors and 0 warnings**.
The complete Windows CUDA/source/version/build suite passed **257 tests**.
Applicable pre-commit checks passed.

## Local artifacts

- DLL: `.builds/cuda-categorical-settings-20260929/artifacts/lib_lightgbm.dll`
- SHA256: `58c6588bf9712fa4b2589685350fd11d143c787ca0d515a45ee907aa84258dfc`
- Before-fix tests: `.builds/categorical-settings-before-final.log`
- Focused tests: `.builds/categorical-settings-after.log`
- Full suite: `.builds/categorical-settings-runtime.log`
- Memory check: `.builds/categorical-settings-memcheck.log`
- Race check: `.builds/categorical-settings-racecheck.log`
- Local sanitizer runner: `.builds/cuda-categorical-settings-20260929/memcheck_regressions.py`

Use the base Python interpreter and application-only tracking with the staged
package for sanitizer runs. Earlier validated DLL artifacts are preserved.
