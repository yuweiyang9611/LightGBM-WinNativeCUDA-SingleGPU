# CUDA split-search optimality fixes — 2026-09-29

## Reproduced failures

- Global-memory numerical search replaced a thread's best gain with later, weaker candidates. A one-tree regression with an exact threshold at `899.5` instead selected `255.5`; the CPU control fitted the target exactly.
- Reverse scans included the NaN bin in the wrong child totals. Known optimal stumps with missing values assigned to the left failed.
- The exclusive prefix scan stored warp totals by lane index and read the previous total using the lane index. This made all warp-ending lanes write the same slot and warp-starting lanes read the wrong shared-memory location.
- Large categorical search repeatedly initialized the same bins from every thread, counted valid categories incorrectly, and used a local sorting depth exceeding the 256-thread shared-memory tile. Category identity and split statistics were corrupted.
- One-hot/global categorical candidates also lost earlier best gains. Categorical outputs unnecessarily allocated device-heap arrays instead of using their preallocated buffers, and the one-hot output omitted the most-frequent-bin offset.

## Fixes

- Retain the strongest candidate per thread throughout numerical and categorical scans.
- Exclude NaN bins correctly in reverse numerical scans.
- Use warp IDs for prefix-scan carry storage and synchronize before reusing that storage.
- Partition categorical initialization by thread, preserve bin identities, count all valid bins, and use the correct sorting depth.
- Scan only the populated categorical prefix range and reuse the allocated category-output buffers.

## Validation

- **23 new optimality cases pass:** 15 numerical cases with known thresholds across early/middle/late histogram ranges and NaN/zero missing-value directions; eight categorical cases cover low/high category IDs, grouped/one-hot search, and a dominant missing bin.
- The tests check exact-fit prediction error, internal score agreement, missing-value routing, and categorical leaf counts. The original numerical suite had 12 failures; the initial four categorical cases had three failures.
- **119 Windows CUDA/source/version/build tests passed.**
- **63 regression bodies passed Compute Sanitizer with 0 errors**, using the direct interpreter and application-only tracking.
- Applicable pre-commit checks passed.

## Local artifacts

- DLL: `.builds/cuda-split-search-20260929/artifacts/lib_lightgbm.dll`
- SHA256: `b5e7e6ad8744fdeab074839f7c5f75bbdea4624328ec137014436b03ef2672fb`
- Full suite: `.builds/split-search-runtime.log`
- Memory check: `.builds/split-search-memcheck.log`
- Local memory-check runner: `.builds/cuda-split-search-20260929/memcheck_regressions.py`

This build includes the preceding fixes. Previously documented binaries and installed packages were preserved.
