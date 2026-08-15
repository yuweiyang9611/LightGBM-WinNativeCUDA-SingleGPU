"""Source-level contracts for the Windows native, single-GPU CUDA build."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_nccl_is_an_explicit_and_conditional_dependency() -> None:
    cmake = _read("CMakeLists.txt")

    assert 'option(USE_NCCL "Enable NCCL/RCCL multi-GPU communication for CUDA/ROCm" ON)' in cmake
    assert re.search(r"if\(USE_NCCL\)\s+find_package\(NCCL REQUIRED\)", cmake)
    assert re.search(
        r"if\(USE_CUDA AND USE_NCCL\)\s+"
        r"target_link_libraries\(lightgbm_objs PUBLIC NCCL::NCCL\)\s+endif\(\)",
        cmake,
    )
    assert cmake.count("find_package(NCCL REQUIRED)") == 1
    assert cmake.count("NCCL::NCCL") == 1


def test_no_nccl_build_does_not_include_nccl_headers() -> None:
    cuda_utils = _read("include/LightGBM/cuda/cuda_utils.hu")

    assert re.search(r"#if defined\(USE_NCCL\)\s+#include <nccl\.h>\s+#endif", cuda_utils)
    assert re.search(r"#if defined\(USE_NCCL\)\s+#include <rccl/rccl\.h>\s+#endif", cuda_utils)
    assert "#if !defined(USE_NCCL)" in cuda_utils
    assert "using ncclComm_t = void*;" in cuda_utils


def test_nccl_only_translation_units_are_excluded() -> None:
    cmake = _read("CMakeLists.txt")
    nccl_sources = (
        "include/LightGBM/cuda/cuda_nccl_topology.hpp",
        "src/boosting/cuda/nccl_gbdt.cpp",
        "src/boosting/cuda/nccl_gbdt.hpp",
        "src/boosting/cuda/nccl_gbdt_component.hpp",
    )

    assert re.search(
        r"if\(USE_NCCL\)\s+"
        r"list\(APPEND LGBM_CUDA_SOURCES src/boosting/cuda/nccl_gbdt\.cpp\)\s+endif\(\)",
        cmake,
    )
    for relative_path in nccl_sources:
        assert "#if defined(USE_CUDA) && defined(USE_NCCL)" in _read(relative_path)


def test_use_nccl_macro_does_not_collide_with_template_parameters() -> None:
    paths = list((REPO_ROOT / "include").rglob("*")) + list((REPO_ROOT / "src").rglob("*"))
    collision = re.compile(r"template\s*<[^>]*\bUSE_NCCL\b")

    offenders = [path for path in paths if path.is_file() and collision.search(path.read_text(encoding="utf-8"))]
    assert offenders == []


def test_msvc_cuda_flags_do_not_receive_unix_only_options() -> None:
    cmake = _read("CMakeLists.txt")

    expected = """if(MSVC)
      # nvcc forwards these options to cl.exe. Unix-only flags such as -fPIC
      # and -Wall make native Windows CUDA builds fail during host compilation.
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/utf-8 -Xcompiler=/bigobj")
      add_definitions(-DNOMINMAX)
    else()
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC -Xcompiler=-Wall")
    endif()"""
    assert expected in cmake


def test_fmt_rounding_thresholds_are_nvcc_compatible() -> None:
    cmake = _read("CMakeLists.txt")
    fmt_patch = _read("cmake/WindowsCudaFmtPatch.cmake")

    # NVCC 13.2 rejects the previous UTF-32 escape sequence because its values
    # are outside the Unicode scalar range. Two UTF-16 halves preserve the
    # uint32_t lookup-table values without modifying the fmt submodule itself.
    assert "include(cmake/WindowsCudaFmtPatch.cmake)" in cmake
    assert "prepare_fmt_for_msvc_cuda(FMT_INCLUDE_DIR)" in cmake
    assert 'U"\\x9999999a' in fmt_patch
    assert 'uint32_t(u"\\x9999\\x828f\\x8041\\x8006' in fmt_patch
    assert 'uint32_t(u"\\x999a\\x5c29\\x8938\\x8db9' in fmt_patch
    assert re.search(
        r"if\(USE_CUDA AND MSVC\)\s+"
        r".*?target_compile_definitions\(\s*"
        r"lightgbm_objs PRIVATE \$<\$<COMPILE_LANGUAGE:CUDA>:FMT_UNICODE=0>\s*"
        r"\)\s+endif\(\)",
        cmake,
        re.DOTALL,
    )


def test_windows_size_t_does_not_mix_with_unsigned_long() -> None:
    tree_learner = _read("src/treelearner/cuda/cuda_single_gpu_tree_learner.cpp")

    assert "std::max(static_cast<size_t>((max_cat_value + 31) / 32), size_t{1})" in tree_learner
    assert "std::max(static_cast<size_t>((max_cat_num_bin + 31) / 32), size_t{1})" in tree_learner


def test_cuda_visible_numeric_constants_are_constexpr() -> None:
    meta = _read("include/LightGBM/meta.h")

    assert "constexpr score_t kMinScore" in meta
    assert "constexpr score_t kMaxScore" in meta
    assert "constexpr score_t kEpsilon" in meta
    assert "constexpr double kZeroThreshold" in meta
    assert "constexpr int kAlignedSize" in meta


def test_visual_studio_final_targets_trigger_cuda_device_link() -> None:
    cmake = _read("CMakeLists.txt")
    cuda_link_anchor = _read("src/cuda/cuda_link.cu")

    assert re.search(
        r"set_target_properties\(\s*_lightgbm\s+PROPERTIES\s+"
        r"CUDA_SEPARABLE_COMPILATION ON\s+CUDA_RESOLVE_DEVICE_SYMBOLS ON\s*\)",
        cmake,
    )
    assert re.search(
        r'if\(MSVC AND CMAKE_GENERATOR MATCHES "\^Visual Studio"\)\s+'
        r"target_sources\(_lightgbm PRIVATE src/cuda/cuda_link\.cu\)\s+endif\(\)",
        cmake,
    )
    assert "set_property(TARGET _lightgbm PROPERTY CUDA_RUNTIME_LIBRARY Shared)" in cmake
    assert "intentionally empty CUDA translation unit" in cuda_link_anchor


def test_windows_runner_stages_dll_outputs_and_checks_python_dependencies() -> None:
    runner = _read("tests/win_native_cuda_tests/run_windows_cuda_tests.ps1")

    assert 'import narwhals, numpy, pytest, scipy' in runner
    assert 'Join-Path $RepoRoot ".venv\\Scripts\\python.exe"' in runner
    assert '[string]$CudaArchitectures = "89-real;89-virtual"' in runner
    assert '"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$ArtifactDirectory"' in runner
    assert '"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE=$ArtifactDirectory"' in runner
    assert 'Join-Path $ArtifactDirectory "lib_lightgbm.dll"' in runner
    assert '--all-fatbin --list-ptx $BuiltDll.FullName' in runner


def test_precompiled_windows_wheel_receives_a_platform_tag() -> None:
    build_script = _read("build-python.sh")

    assert "test -f ./lightgbm/lib/lib_lightgbm.dll" in build_script
    assert "sysconfig.get_platform()" in build_script
    assert "python -m wheel tags" in build_script
    assert '--platform-tag="${PYTHON_PLATFORM_TAG}"' in build_script
    assert '--build="${LIGHTGBM_WHEEL_BUILD_TAG}"' in build_script
    assert "Root-Is-Purelib: false" in build_script
    assert "python -m wheel pack" in build_script
    assert "python -m wheel version" in build_script
    assert "--remove" in build_script


def test_offline_installer_requires_checksums_and_reinstalls_the_binary() -> None:
    installer = _read("tests/win_native_cuda_tests/install_offline_wheel.ps1")

    assert "Offline package checksum manifest is missing" in installer
    assert "--no-index" in installer
    assert "--force-reinstall" in installer


def test_no_nccl_build_rejects_multi_gpu_at_runtime() -> None:
    boosting = _read("src/boosting/boosting.cpp")

    assert "#if defined(USE_CUDA) && !defined(USE_NCCL)" in boosting
    assert 'device_type == std::string("cuda") && num_gpu > 1' in boosting
    assert "CUDA multi-GPU training requires an NCCL-enabled build" in boosting
