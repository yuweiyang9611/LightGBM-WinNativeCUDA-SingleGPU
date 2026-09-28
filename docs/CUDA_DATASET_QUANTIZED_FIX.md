# CUDA Dataset initialization and quantized split fixes — 2026-09-29

## Corrected behavior

- Direct Python `Booster` construction prepares a lazy Dataset for CUDA before creating its native handle. Explicit Dataset binning parameters retain precedence. A constructed CPU Dataset can be rebuilt when its raw data was retained; otherwise it produces a clear error. CUDA storage inherited from a reference Dataset is recognized.
- The native Booster constructor, training-data replacement, and validation-data attachment reject unprepared CPU Datasets before dereferencing CUDA metadata. Failed attachments leave the existing Booster usable.
- Quantized CUDA split search now receives per-node feature masks, enforcing interaction constraints and `feature_fraction_bynode`, including changes applied through `reset_parameter()`.
- Packed gradient/Hessian totals are copied with split candidates and written into both child-leaf states. Root reduction retains the integer representation. The quantized histogram pool uses the same slot stride whichever child is smaller. These repairs prevent stale parent statistics, empty branches, and overlapping histogram slots.

## Evidence

- Six initial Dataset initialization cases failed with access violations before repair. Nine cases now pass, including device aliases, retained raw data, CUDA references, and native error recovery.
- Four quantized feature-mask regressions failed before repair. All pass after repair.
- Four quantized histogram regressions initially produced empty or inconsistent leaves. They now agree with independent prediction-derived row counts and unit-Hessian weights across two seeds and 31/255 bins.
- All **73** Windows CUDA/source/version tests pass, including the previous NaN routing, depth, interaction, native cubin and forced PTX regressions.
- Python basic tests: **97 passed, 13 skipped**, with the project's `TASK=cuda` marker. Without that marker, three tests exercise `position` metadata that this CUDA build explicitly does not support; the same failures reproduce with the pre-change DLL.
- All **38** C++ tests and applicable pre-commit checks pass.
- Compute Sanitizer: **20 regression bodies passed, 0 errors**. The bodies were run in one interpreter with `--target-processes application-only`. On this Windows setup, earlier all-process tracking runs failed to return after the tested Python process had exited; those runs were not counted as successful checks and were cleaned up.

## Local artifacts

- DLL: `.builds/cuda-dataset-init-20260929/artifacts/lib_lightgbm.dll`
- SHA256: `067a5e17d84b13a840b77507f8154fca14c0807b784ee68fa943f02ec0aa479b`
- Windows suite: `.builds/dataset-quantized-runtime.log`
- Python basic suite: `.builds/dataset-init-basic-cuda.log`
- C++ suite: `.builds/dataset-init-cpp.log`
- Memory check: `.builds/dataset-quantized-memcheck-single-process.log`
- Local diagnostic harness: `.builds/memcheck_constraints_direct.py`

The DLL includes the preceding CUDA fixes. Installed libraries and release packages were not replaced. For native API callers, construct the Dataset with `device_type=cuda` before passing it to a CUDA Booster.
