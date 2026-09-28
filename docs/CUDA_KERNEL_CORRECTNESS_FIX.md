# CUDA kernel correctness fixes — 2026-09-28

Two correctness defects were fixed in the native Windows, single-GPU CUDA implementation. The fixes retain the existing public API, categorical routing and Zero-missing behavior.

## Changes

- `src/io/cuda/cuda_tree.cu`: numerical NaN routing compares feature-local bins with a feature-local maximum bin. Previously a group-level NaN code such as255 became254 after conversion, but was still compared with255. Both indexed out-of-bag scoring and non-indexed validation traversal share this corrected kernel.
- `src/treelearner/cuda/cuda_histogram_constructor.cpp`: `SubtractHistogramForLeaf()` synchronizes its producer stream after histogram repair/subtraction, before independent split-search streams can read the output. This covers both ordinary and quantized-gradient dispatch. The synchronization waits for the producer stream, not the entire device.

The default `gpu_use_dp` setting was not changed. The real-input repeatability validation below explicitly uses double precision; these fixes are not a general bitwise-determinism guarantee for single-precision atomic accumulation.

## Regression results

The new NaN routing tests failed against the original DLL: maximum training/validation score differences were12.10708590416565 and3.470860028944162. Both passed with the fixed DLL.

Twenty-two source/runtime tests passed, including:

- training and validation score equality with the represented ensemble;
- indexed out-of-bag and non-indexed validation NaN traversal, with an asserted NaN/default-left split;
- frequent-zero histogram bins under two seeds and31/255 bins;
- saved leaf/internal counts and Hessian weights checked against independently predicted leaf membership, rather than only against each other;
- native cubin and forced PTX JIT execution;
- rejecting unsupported multi-GPU execution without NCCL.

The existing PowerShell test runner now collects the whole Windows CUDA test directory so these regressions are included in normal future validation.

## Frozen real-input validation

A local, development-only fixture contained7,560 rows,4,251 features and378 prediction rows from six stocks. No full research workflow, holdout evaluation or trading was run.

`validate_frozen_input.py` loaded the new DLL with a staged matching wrapper, using only existing local NumPy/SciPy dependencies. It ran two independent300-tree fits, a220-tree cache warmup with another parameter set, and a300-tree fit reusing that Dataset. Every training iteration checked the internal accumulated scores against ensemble inference.

Results:

- all1,120 checked iterations had maximum training-score difference **0**;
- independent/reused300-tree prediction arrays had maximum difference **0**;
- the three300-tree model files had identical SHA256:
  `cae77cccfc94551895749d1a8704c0192b68d68449529dc0d4f593849da5430f`.

These runs validate the repaired library on the captured failure case. Their timings include expensive per-iteration verification and should not be presented as pure training benchmarks.

## Local build and artifacts

- Build directory: `.builds/cuda-correctness-20260928`
- Candidate DLL: `.builds/cuda-correctness-20260928/artifacts/lib_lightgbm.dll`
- SHA256: `5418087761dcbed57f021ec69d43b5ea41e4af6d05d2bee58847e1266c8b4de8`
- Build: Release, `USE_CUDA=ON`, `USE_NCCL=OFF`, `BUILD_CLI=OFF`, `BUILD_CPP_TEST=OFF`, `CMAKE_CUDA_ARCHITECTURES=89-real;89-virtual`.
- Binary inspection:1 cubin image and16 relocatable PTX images.
- Local results: `runtime-tests.xml`, `histogram-tests.xml`, `real-validation/result.json` beneath the build directory.

The root baseline DLL and StockResearch's installed DLL were preserved. No package was downloaded, installed or replaced. Distribution requires an explicitly identified new local build; the earlier wheel still contains the old DLL.

## Re-run tests without rebuilding

```powershell
$env:LIGHTGBM_CUDA_DLL = (Resolve-Path .builds/cuda-correctness-20260928/artifacts/lib_lightgbm.dll).Path
.venv/Scripts/python.exe -m pytest -ra tests/win_native_cuda_tests
```

Optional captured-input validation accepts a directory containing `train.npy`, `target.npy`, `predict.npy`, `weights.npy` and `metadata.json`; the metadata records feature/category names, original parameter sets and file hashes. Use a new output directory for each run:

```powershell
.venv/Scripts/python.exe tests/win_native_cuda_tests/validate_frozen_input.py --dll .builds/cuda-correctness-20260928/artifacts/lib_lightgbm.dll --fixture <local-fixture-directory> --output .builds/cuda-correctness-20260928/real-validation-repeat
```
