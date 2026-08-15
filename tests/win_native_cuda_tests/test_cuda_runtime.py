"""Runtime checks for a prebuilt Windows CUDA DLL without NCCL.

Set ``LIGHTGBM_CUDA_DLL`` to the DLL produced by a ``USE_CUDA=ON`` and
``USE_NCCL=OFF`` build. The tests skip when no candidate DLL is available.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
DLL_ENV_VAR = "LIGHTGBM_CUDA_DLL"


def _find_cuda_dll() -> Path | None:
    configured = os.environ.get(DLL_ENV_VAR)
    if configured:
        path = Path(configured).expanduser().resolve()
        if not path.is_file():
            pytest.fail(f"{DLL_ENV_VAR} does not point to a file: {path}")
        return path

    candidates = (
        REPO_ROOT / ".builds" / "win-native-cuda" / "artifacts" / "lib_lightgbm.dll",
        REPO_ROOT / ".builds" / "win-native-cuda" / "artifacts" / "Release" / "lib_lightgbm.dll",
    )
    return next((path.resolve() for path in candidates if path.is_file()), None)


@pytest.fixture(scope="module")
def staged_lightgbm(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    if sys.platform != "win32":
        pytest.skip("The native Windows CUDA DLL test only runs on Windows")

    dll_path = _find_cuda_dll()
    if dll_path is None:
        pytest.skip(f"Set {DLL_ENV_VAR} to a built Windows CUDA lib_lightgbm.dll")

    stage_root = tmp_path_factory.mktemp("win_native_cuda_package")
    package_dir = stage_root / "lightgbm"
    shutil.copytree(REPO_ROOT / "python-package" / "lightgbm", package_dir)
    package_lib_dir = package_dir / "lib"
    package_lib_dir.mkdir(exist_ok=True)
    staged_dll = package_lib_dir / "lib_lightgbm.dll"
    shutil.copy2(dll_path, staged_dll)
    return stage_root, staged_dll


def _run_with_staged_package(
    stage_root: Path,
    staged_dll: Path,
    code: str,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(stage_root)
    env["PYTHONNOUSERSITE"] = "1"
    env["EXPECTED_LIGHTGBM_DLL"] = str(staged_dll)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=stage_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _assert_subprocess_passed(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"


@pytest.mark.parametrize("force_ptx_jit", (False, True), ids=("native-cubin", "forced-ptx-jit"))
def test_single_gpu_cuda_training(
    staged_lightgbm: tuple[Path, Path],
    force_ptx_jit: bool,
) -> None:
    stage_root, staged_dll = staged_lightgbm
    result = _run_with_staged_package(
        stage_root,
        staged_dll,
        """
        import os
        from pathlib import Path

        import lightgbm as lgb
        import numpy as np
        from lightgbm.libpath import _find_lib_path

        loaded_dll = Path(_find_lib_path()[0]).resolve()
        expected_dll = Path(os.environ["EXPECTED_LIGHTGBM_DLL"]).resolve()
        assert loaded_dll == expected_dll, (loaded_dll, expected_dll)

        rng = np.random.default_rng(20260815)
        features = rng.normal(size=(4096, 24)).astype(np.float32)
        labels = (features[:, 0] * 1.7 - features[:, 1] * 0.8 + features[:, 2] * 0.3).astype(np.float32)
        train_set = lgb.Dataset(features, label=labels)
        model = lgb.train(
            {
                "objective": "regression",
                "device_type": "cuda",
                "gpu_device_id": 0,
                "num_gpu": 1,
                "num_leaves": 15,
                "min_data_in_leaf": 5,
                "verbosity": 1,
                "seed": 7,
            },
            train_set,
            num_boost_round=8,
        )
        predictions = model.predict(features[:64])
        assert predictions.shape == (64,)
        assert np.isfinite(predictions).all()
        assert model.num_trees() == 8
        model_mse = np.mean((predictions - labels[:64]) ** 2)
        constant_baseline_mse = np.mean((labels.mean() - labels[:64]) ** 2)
        assert model_mse < 0.5 * constant_baseline_mse, (model_mse, constant_baseline_mse)
        """,
        extra_env={"CUDA_FORCE_PTX_JIT": "1", "CUDA_CACHE_DISABLE": "1"} if force_ptx_jit else None,
    )
    _assert_subprocess_passed(result)


def test_no_nccl_build_rejects_multiple_gpus(staged_lightgbm: tuple[Path, Path]) -> None:
    stage_root, staged_dll = staged_lightgbm
    result = _run_with_staged_package(
        stage_root,
        staged_dll,
        """
        import lightgbm as lgb
        import numpy as np
        from lightgbm.basic import LightGBMError

        features = np.arange(256, dtype=np.float32).reshape(64, 4)
        labels = np.arange(64, dtype=np.float32)
        try:
            lgb.train(
                {
                    "objective": "regression",
                    "device_type": "cuda",
                    "num_gpu": 2,
                    "verbosity": -1,
                },
                lgb.Dataset(features, label=labels),
                num_boost_round=1,
            )
        except LightGBMError as error:
            assert "CUDA multi-GPU training requires an NCCL-enabled build" in str(error)
        else:
            raise AssertionError("A USE_NCCL=OFF build accepted num_gpu=2")
        """,
    )
    _assert_subprocess_passed(result)
