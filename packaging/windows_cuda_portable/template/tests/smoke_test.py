"""Exercise the published LightGBM Windows native CUDA backend and write a JSON result."""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def query_nvidia_gpus() -> list[dict[str, Any]]:
    """Return the NVIDIA GPUs reported by ``nvidia-smi``."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,compute_cap,driver_version",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    gpus: list[dict[str, Any]] = []
    for row in csv.reader(completed.stdout.splitlines(), skipinitialspace=True):
        if len(row) != 4:
            raise RuntimeError(f"Unexpected nvidia-smi output row: {row!r}")
        major, minor = (int(part) for part in row[2].strip().split(".", maxsplit=1))
        driver = row[3].strip()
        gpus.append(
            {
                "index": int(row[0].strip()),
                "name": row[1].strip(),
                "compute_capability": f"{major}.{minor}",
                "compute_capability_tuple": [major, minor],
                "driver_version": driver,
                "driver_branch": int(driver.split(".", maxsplit=1)[0]),
            }
        )
    if not gpus:
        raise RuntimeError("nvidia-smi returned no NVIDIA GPU")
    return gpus


def validate_release_compatibility(gpu: dict[str, Any], force_ptx_jit: bool) -> str:
    """Validate the selected GPU and return its expected CUDA code path."""
    capability = tuple(gpu["compute_capability_tuple"])
    driver_branch = int(gpu["driver_branch"])
    if capability < (8, 9):
        raise RuntimeError(
            "This release contains sm_89 cubin plus compute_89 PTX and cannot run on "
            f"compute capability {gpu['compute_capability']} ({gpu['name']})."
        )
    if capability == (8, 9) and not force_ptx_jit:
        if driver_branch < 580:
            raise RuntimeError(f"sm_89 requires an R580-or-newer driver; found {gpu['driver_version']}")
        return "sm_89 native cubin (expected)"
    if driver_branch < 595:
        raise RuntimeError(
            f"The compute_89 PTX JIT path requires an R595-or-newer driver; found {gpu['driver_version']}"
        )
    return "compute_89 PTX JIT (expected)"


def write_result(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write the portable verification result as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_path, path)


def main() -> int:
    """Run the portable CUDA smoke test and return a process exit code."""
    result_path_text = os.environ.get("LIGHTGBM_TEST_RESULT")
    if not result_path_text:
        raise RuntimeError("LIGHTGBM_TEST_RESULT was not set by the portable launcher")
    result_path = Path(result_path_text)
    gpu_device_id = int(os.environ.get("LIGHTGBM_TEST_GPU_DEVICE_ID", "0"))
    force_ptx_jit = os.environ.get("CUDA_FORCE_PTX_JIT") == "1"
    started = time.perf_counter()
    result: dict[str, Any] = {
        "success": False,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "machine": platform.node(),
        "os": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "gpu_device_id": gpu_device_id,
        "force_ptx_jit": force_ptx_jit,
    }

    try:
        if sys.maxsize <= 2**32 or platform.machine().lower() not in {"amd64", "x86_64"}:
            raise RuntimeError("This package requires 64-bit Windows on x86-64")

        gpus = query_nvidia_gpus()
        result["gpus"] = gpus
        selected_gpu = next((gpu for gpu in gpus if gpu["index"] == gpu_device_id), None)
        if selected_gpu is None:
            raise RuntimeError(f"GPU device ID {gpu_device_id} was not reported by nvidia-smi")
        result["expected_cuda_code_path"] = validate_release_compatibility(selected_gpu, force_ptx_jit)

        import lightgbm as lgb  # noqa: I001, PLC0415
        import narwhals  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        import scipy  # noqa: PLC0415
        from lightgbm.libpath import _find_lib_path  # noqa: PLC0415

        result["versions"] = {
            "lightgbm": lgb.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "narwhals": narwhals.__version__,
        }
        result["lightgbm_library"] = [str(path) for path in _find_lib_path()]

        rng = np.random.default_rng(20260816)
        features = rng.normal(size=(32768, 64)).astype(np.float32)
        labels = (
            features[:, 0] * 1.7
            - features[:, 1] * 0.8
            + features[:, 2] * 0.3
            + rng.normal(scale=0.02, size=features.shape[0])
        ).astype(np.float32)
        train_set = lgb.Dataset(features, label=labels, free_raw_data=False)

        training_started = time.perf_counter()
        model = lgb.train(
            {
                "objective": "regression",
                "metric": "l2",
                "device_type": "cuda",
                "gpu_device_id": gpu_device_id,
                "num_gpu": 1,
                "num_leaves": 31,
                "min_data_in_leaf": 20,
                "max_bin": 63,
                "verbosity": 1,
                "seed": 7,
            },
            train_set,
            num_boost_round=40,
        )
        training_seconds = time.perf_counter() - training_started

        predictions = model.predict(features[:512])
        if predictions.shape != (512,) or not np.isfinite(predictions).all() or model.num_trees() != 40:
            raise RuntimeError("CUDA training returned an invalid model or predictions")
        model_mse = float(np.mean((predictions - labels[:512]) ** 2))
        baseline_mse = float(np.mean((labels.mean() - labels[:512]) ** 2))
        if model_mse >= 0.5 * baseline_mse:
            raise RuntimeError(
                f"CUDA model did not beat the constant baseline: model={model_mse}, baseline={baseline_mse}"
            )

        result.update(
            {
                "success": True,
                "training_seconds": training_seconds,
                "num_rows": int(features.shape[0]),
                "num_features": int(features.shape[1]),
                "num_trees": model.num_trees(),
                "model_mse": model_mse,
                "constant_baseline_mse": baseline_mse,
            }
        )
        sys.stdout.write(
            f"GPU: {selected_gpu['name']} (CC {selected_gpu['compute_capability']})\n"
            f"Driver: {selected_gpu['driver_version']}\n"
            f"Expected code path: {result['expected_cuda_code_path']}\n"
            f"LightGBM: {lgb.__version__}; trees={model.num_trees()}; MSE={model_mse:.6f}\n"
            "Windows native CUDA LightGBM verification passed\n"
        )
        return 0
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        traceback.print_exc()
        return 1
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_result(result_path, result)


if __name__ == "__main__":
    raise SystemExit(main())
