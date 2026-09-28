# Quantized continuation and root statistics fixes — 2026-09-29

## Reproduced failures

- Continuing CUDA quantized training beyond the initial iteration budget read beyond a precomputed array of random offsets and produced different predictions from uninterrupted training. Multiclass training also needs more quantization calls than its number of boosting rounds.
- Increasing `num_leaves` left quantization metadata at its original capacity. Both CPU and CUDA regressions failed; the CPU process reported heap corruption. CUDA leaf-renewal buffers also needed to follow the new leaf budget.
- Root gradient reduction wrote the full total into the device leaf structure but left only the first block's total in the buffer copied to the host. On a 4,224-row fixture with mean target 0.5, the root's reported value was 0 instead of 0.5. The 512-row controls passed.

## Repairs

- Draw one scalar random offset per quantization call from the existing seeded generator and pass it directly to the kernel. This preserves the original sequence without a fixed iteration-indexed device array.
- Share leaf-state resizing between initialization and parameter resets, applying it to CPU and CUDA quantizers and CUDA renewal buffers.
- Store the complete reduced gradient in the host-transfer buffer for both ordinary and quantized root initialization.

## Validation

- **12 new runtime cases pass:** uninterrupted versus continued regression/multiclass training with both rounding modes; CPU/CUDA leaf-budget changes `2 → 16 → 4 → 31 → 2`, with and without renewal; and ordinary/quantized root statistics across one and multiple reduction blocks.
- Continuation comparisons use live training objects on both sides. Serialized `split_gain` values have different rounding, so serialization is not mixed into that equality check.
- **131 Windows CUDA/source/version/build tests passed.**
- **Two general CPU regression tests and all 38 C++ tests passed.**
- **75 regression bodies passed Compute Sanitizer with 0 errors**, using direct-interpreter application-only tracking.
- Applicable pre-commit checks passed.

## Local artifacts

- DLL: `.builds/cuda-continuation-20260929/artifacts/lib_lightgbm.dll`
- SHA256: `dfdd6bdf23401c352ba047481f0f2fc087025f6855dd00c81d44cb4212521b00`
- Full suite: `.builds/quantized-continuation-runtime.log`
- CPU regression: `.builds/quantized-leaf-engine.log`
- C++ suite: `.builds/quantized-continuation-cpp.log`
- Memory check: `.builds/quantized-continuation-memcheck.log`
- Local memory-check runner: `.builds/cuda-continuation-20260929/memcheck_regressions.py`

Previously documented DLLs and installed packages were preserved. This build contains the preceding fixes.
