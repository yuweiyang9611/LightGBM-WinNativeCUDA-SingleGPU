# CI recovery — 2026-09-30

The pushed `aa0b226d` revision failed several independent jobs in the
[Python workflow](https://github.com/yuweiyang9611/LightGBM-WinNativeCUDA-SingleGPU/actions/runs/36594126754)
and [R workflow](https://github.com/yuweiyang9611/LightGBM-WinNativeCUDA-SingleGPU/actions/runs/36594126725).

## Causes and repairs

- **sdist checks:** the required `cmake/WindowsCudaDeviceLink.cmake` module added
  one file. The actual archive contains 808 files, while the check allowed 807.
  Update the bound to 808, retaining size/content checks.
- **macOS quantized metadata tests:** constructing another Dataset reset the
  process-wide thread default. A Booster configured with four threads rebuilt
  quantization state using the machine default instead. Restore the Booster's
  thread setting before Dataset reset and each built-in/custom training step.
- **OpenCL weight removal:** changing from variable to constant Hessians changed
  kernel argument types, but Dataset reset reused the old compiled kernels.
  Recompile when the Hessian mode changes before resetting GPU arguments.
- **Linux R setup:** the CMake download returned an HTML page that was executed
  as a shell installer. Check HTTP failures, retry invalid bodies, and require
  the installer shebang before invoking it.
- **macOS R setup:** Homebrew could not link `openssl@3` because the image retained
  an `openssl@1.1` executable symlink. Unlink the legacy formula before installing
  R system dependencies, keeping its installed libraries intact.

## Local validation

- The exact four quantized metadata failures reproduced with `OMP_NUM_THREADS=2`
  and model `num_threads=4`. After repair, all eight metadata cases passed with
  the original strict numerical assertions.
- Two CPU/CUDA thread-default regressions and three mocked download-response
  cases passed. Download tests use mock executables and do not access the network.
- A real sdist contains 808 files and the required device-link module. The old
  bound fails with `too-many-files`; the corrected bound passes pydistcheck.
- Applicable pre-commit checks passed.

The OpenCL and macOS dependency fixes require confirmation on the corresponding
GitHub runners. No tests or required workflow jobs were disabled.
