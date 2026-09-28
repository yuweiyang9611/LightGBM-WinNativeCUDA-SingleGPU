# Windows native single-GPU CUDA tests

The fast source-contract tests do not require a CUDA Toolkit:

```powershell
.\.venv\Scripts\python.exe -m pytest -ra tests/win_native_cuda_tests/test_source_contract.py
```

The end-to-end runner configures LightGBM with `USE_CUDA=ON`,
`USE_NCCL=OFF`, and explicit `89-real;89-virtual` code generation. It builds
`lib_lightgbm.dll`, verifies that the DLL contains both cubin and relocatable
PTX, stages the matching Python wrapper, tests CUDA training through both the
native cubin and forced PTX JIT paths, and verifies that `num_gpu=2` is rejected:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install "narwhals>=1.15" numpy scipy pytest
tests/win_native_cuda_tests/run_windows_cuda_tests.ps1
```

If CMake is only available through CLion, pass its full path:

```powershell
tests/win_native_cuda_tests/run_windows_cuda_tests.ps1 `
  -CMakeExecutable "D:\Programs\JetBrains\CLion\bin\cmake\win\x64\bin\cmake.exe"
```

To test an already-built DLL without rebuilding it:

```powershell
$env:LIGHTGBM_CUDA_DLL = "D:\path\to\lib_lightgbm.dll"
.\.venv\Scripts\python.exe -m pytest -ra tests/win_native_cuda_tests/test_cuda_runtime.py
```

The offline bundle places the LightGBM wheel and its Windows CPython runtime
dependency wheels in one directory. From that directory, install without an
index or network access with:

```powershell
.\install_offline_wheel.ps1 -PythonExecutable "C:\path\to\python.exe"
```

That wheelhouse still requires Python and the Microsoft C++ runtime on the
target machine. For a no-install test machine, build the self-contained
[portable bundle](../../packaging/windows_cuda_portable/README.md), which
includes an isolated CPython runtime and app-local MSVC/OpenMP DLLs.

When producing a precompiled wheel, an optional PEP 427 build tag can identify
the CUDA variant without changing LightGBM's Python package version:

```powershell
.\.venv\Scripts\python.exe -m pip install build wheel
$env:LIGHTGBM_WHEEL_BUILD_TAG = "1cuda132sm89ptx"
```

## Kernel correctness regressions

Repeated training Dataset replacement, quantization buffer resizing and rejection recovery are covered on CPU and CUDA. See [the Dataset replacement fixes](../../docs/TRAINING_DATASET_REPLACEMENT_FIX.md).

Quantized continuation, changing leaf budgets and multi-block root statistics have dedicated regressions. See [the continuation and root-statistics fixes](../../docs/CUDA_QUANTIZED_CONTINUATION_FIX.md).

Known optimal numerical and categorical stumps check split quality as well as routing and counts. See [the split-search optimality fixes](../../docs/CUDA_SPLIT_OPTIMALITY_FIX.md).

Large-bin and histogram-width boundaries, plus real Visual Studio incremental device linking, are covered too. See [the large histogram and build fixes](../../docs/CUDA_LARGE_HISTOGRAM_AND_LINK_FIX.md). The build regression runs when `LIGHTGBM_TEST_CMAKE` is set; the PowerShell build/test runner sets it automatically.

Direct CUDA Dataset initialization, native error recovery, quantized feature masks and quantized histogram counts are covered by the suite. See [the Dataset and quantization fixes](../../docs/CUDA_DATASET_QUANTIZED_FIX.md).

Categorical interaction constraints and runtime feature-selection updates are covered as well. See [the constraint fixes and validation record](../../docs/CUDA_FEATURE_CONSTRAINTS_FIX.md).

The suite also covers quantized training, numerical/categorical depth limits, and configuration resets. See [the training fixes and memory-check instructions](../../docs/CUDA_TRAINING_LIMITS_FIX.md). Version synchronization tests require a POSIX shell and its utilities on PATH.

The runner also checks NaN routing for training and validation, and verifies histogram counts against independently predicted leaf membership. See [the correctness fix and local validation record](../../docs/CUDA_KERNEL_CORRECTNESS_FIX.md). Optional `validate_frozen_input.py` tests a separately exported local fixture without installing packages or starting a full research workflow.
