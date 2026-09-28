"""A successful incremental CUDA build must contain the updated device code."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_visual_studio_relinks_cuda_object_library(tmp_path: Path) -> None:
    """Exercise source/header updates and a no-op build through the shared helper."""
    cmake = os.environ.get("LIGHTGBM_TEST_CMAKE")
    if sys.platform != "win32" or not cmake:
        pytest.skip("Set LIGHTGBM_TEST_CMAKE to run the Windows CUDA build regression")
    helper = (REPO_ROOT / "cmake/WindowsCudaDeviceLink.cmake").as_posix()
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.28)\n"
        "project(cuda_incremental LANGUAGES CXX CUDA)\n"
        "add_library(parts OBJECT part.cu)\n"
        "set_target_properties(parts PROPERTIES CUDA_SEPARABLE_COMPILATION ON)\n"
        "add_executable(probe main.cu $<TARGET_OBJECTS:parts>)\n"
        "set_target_properties(probe PROPERTIES CUDA_SEPARABLE_COMPILATION ON "
        "CUDA_RESOLVE_DEVICE_SYMBOLS ON CUDA_RUNTIME_LIBRARY Shared)\n"
        f'include("{helper}")\n'
        "configure_windows_cuda_device_link(probe)\n",
        encoding="utf-8",
    )
    part = tmp_path / "part.cu"
    header = tmp_path / "value.h"
    header.write_text("#define PROBE_VALUE 7\n", encoding="utf-8")
    part.write_text('#include "value.h"\n__device__ int device_value() { return PROBE_VALUE; }\n', encoding="utf-8")
    (tmp_path / "main.cu").write_text(
        "#include <cuda_runtime.h>\n#include <cstdio>\n"
        "__device__ int device_value();\n"
        "__global__ void evaluate(int* value) { *value = device_value(); }\n"
        "int main() {\n"
        "  int* device = nullptr; int value = -1;\n"
        "  if (cudaMalloc(&device, sizeof(int)) != cudaSuccess) return 1;\n"
        "  evaluate<<<1, 1>>>(device);\n"
        "  if (cudaMemcpy(&value, device, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return 2;\n"
        '  cudaFree(device); std::printf("%d\\n", value); return 0;\n}\n',
        encoding="utf-8",
    )

    def run(*args: str) -> str:
        result = subprocess.run(
            args, cwd=tmp_path, text=True, encoding="utf-8", errors="replace", capture_output=True, check=False
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout.strip()

    build = tmp_path / "build"
    run(cmake, "-S", str(tmp_path), "-B", str(build), "-A", "x64", "-DCMAKE_CUDA_ARCHITECTURES=89-real;89-virtual")
    executable = build / "Release/probe.exe"

    def build_and_expect(value: int) -> None:
        run(cmake, "--build", str(build), "--config", "Release", "--parallel", "2")
        assert run(str(executable)) == str(value)

    build_and_expect(7)
    header.write_text("#define PROBE_VALUE 42\n", encoding="utf-8")
    build_and_expect(42)
    part.write_text('#include "value.h"\n__device__ int device_value() { return PROBE_VALUE + 1; }\n', encoding="utf-8")
    build_and_expect(43)
    device_link = build / "probe.dir/Release/probe.device-link.obj"
    timestamp = device_link.stat().st_mtime_ns
    build_and_expect(43)
    assert device_link.stat().st_mtime_ns == timestamp
