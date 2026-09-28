# Feature constraints and parameter reset fixes — 2026-09-29

## Reproduced failures

- A categorical CUDA split recorded its new right branch after incrementing the leaf count. This targeted the following slot instead of the new leaf: at leaf capacity it accessed invalid host memory, and before capacity it lost the right branch's feature history. With `interaction_constraints=[[0], [1]]`, a path could incorrectly use both features.
- CPU and CUDA feature samplers retained their construction-time interaction constraints after `reset_parameter()`. CUDA additionally retained the old per-node sampling switch and did not allocate masks when that feature was enabled later.
- Resetting `interaction_constraints=[]` serialized an empty value which `GetString()` ignored, leaving the previous constraint configuration in place.

## Fixes

- Record categorical branch metadata before incrementing the leaf count, matching numerical splits.
- Refresh the shared column sampler's constraint sets when configuration changes.
- Refresh the CUDA per-node sampling flag and allocate its masks when needed.
- Treat an explicit empty interaction-constraint value as a request to clear it.

## Validation

- Four categorical CUDA regressions failed before the fix and pass afterwards. They cover the final split at capacities 2 and 8, categorical set splits, one-hot splits, and root-to-leaf interaction invariants.
- Seven of ten feature-selection reset cases failed before repair; all ten now pass on CPU/CUDA, covering enabling, replacing and clearing constraints and toggling per-node feature sampling.
- Three general Python regressions fail before repair and pass afterwards; they change constraints after two training iterations. The existing interaction-constraint test also passes.
- All 56 Windows CUDA/source/version tests and all 38 C++ tests pass.
- The categorical leaf-capacity and mask enable/disable cases subsequently passed Compute Sanitizer on the successor build, with zero errors. See [the follow-up validation](CUDA_DATASET_QUANTIZED_FIX.md), including the Windows process-tracking workaround.

## Local artifacts

- DLL: `.builds/cuda-interaction-20260929/artifacts/lib_lightgbm.dll`
- SHA256: `3f2898d5920614cf445bad3d523db22ac3298d9dde5d54c02edd7293e7cc6c41`
- Runtime log: `.builds/interaction-runtime.log`
- Python engine log: `.builds/interaction-engine-after.log`
- C++ log: `.builds/interaction-cpp-tests-final.log`
- Memory-check log: `.builds/interaction-memcheck.log`

The DLL contains the earlier training-limit and quantization fixes. Existing installed DLLs and packaged releases were not replaced.
