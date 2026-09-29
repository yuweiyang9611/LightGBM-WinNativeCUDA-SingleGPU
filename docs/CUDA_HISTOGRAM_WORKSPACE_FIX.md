# CUDA histogram workspace resizing — 2026-09-29

## Reproduced failure

Replacing a training Dataset can increase the histogram kernel's grid size.
`ResetTrainingData()` rebuilt row data and leaf histograms but retained the
global scratch buffer allocated for the original Dataset. Large-bin histogram
kernels then wrote beyond that buffer.

The reproduction grows an aligned Dataset from 12,000 to 800,000 rows, with one
feature containing more than 6,144 bins and 63 low-cardinality features.
Compute Sanitizer reported **311 errors** before the fix, including an invalid
8-byte write from the sparse global-memory histogram kernel beyond its
21,132,800-byte allocation.

## Repair

Initialization and Dataset replacement now share the workspace sizing helper.
After rebuilding row data, it derives the required global scratch size from
the new grid dimensions and precision mode, and refreshes the quantized
histogram conversion workspace. It clears global scratch when no large-bin
partition needs it.

## Validation

- Six regression cases cover double precision, single precision and quantized
  training, each with binary and 16-value auxiliary features. They grow and
  shrink the training Dataset, check continued tree growth and loss reduction,
  and compare cached training scores with independent prediction.
- All six cases passed Compute Sanitizer with **0 errors**. Ordinary functional
  assertions alone do not reliably detect this overwrite; the memory check is
  the decisive regression signal.
- The full Windows CUDA/source/version/build suite passed **247 tests**.
- Applicable pre-commit checks passed.

## Local artifacts

- DLL: `.builds/cuda-hist-workspace-20260929/artifacts/lib_lightgbm.dll`
- SHA256: `518b3bea3d3e5f3abca51e89a9f8640ceac1edcdb01a6d2210d83e5be8692a63`
- Focused tests: `.builds/hist-workspace-extended.log`
- Full suite: `.builds/hist-workspace-runtime.log`
- Before-fix memory check: `.builds/hist-workspace-memcheck-repro.log`
- After-fix memory check: `.builds/hist-workspace-memcheck-fixed.log`
- Memory-check runner: `.builds/hist_workspace_memcheck.py`

The memory-check runner executes the regression bodies directly in the staged
Python package. Use the base Python interpreter with Compute Sanitizer's
`--target-processes application-only --error-exitcode 99`, avoiding the Windows
virtual-environment launcher. Earlier DLL artifacts are preserved.
