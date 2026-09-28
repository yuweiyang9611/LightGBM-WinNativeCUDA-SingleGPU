"""Bounded validation of a locally built CUDA DLL against exported frozen arrays.

No package installation, database access, strategy evaluation, or CPU training.
The candidate is loaded with the matching wrapper in an isolated staging folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def sha(path):
    with Path(path).open("rb") as stream:
        result = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
        return result.hexdigest()


def child(fixture, output, expected_dll):
    # Only the isolated child may import the candidate native library.
    import lightgbm as lgb  # noqa: PLC0415

    assert Path(lgb.basic._LIB._name).resolve() == expected_dll.resolve()
    meta = json.loads((fixture / "metadata.json").read_text(encoding="utf-8"))
    for name, value in meta["files"].items():
        assert sha(fixture / name) == value, f"Changed fixture: {name}"
    x, y, test, weights = [
        np.load(fixture / f"{n}.npy", mmap_mode="r") for n in ["train", "target", "predict", "weights"]
    ]
    names, categories = meta["featureNames"], meta["categorical"]
    records = {}
    prediction = {}
    parameters = {name: dict(value, gpu_use_dp=True) for name, value in meta["params"].items()}

    def native(values):
        result = values.copy()
        result.pop("n_estimators")
        result["num_threads"] = result.pop("n_jobs")
        result["seed"] = result.pop("random_state")
        return result

    def dataset(params):
        return lgb.Dataset(
            x,
            label=y,
            weight=weights,
            feature_name=names,
            categorical_feature=categories,
            params=native(params),
            free_raw_data=False,
        ).construct()

    def fit(name, params, data):
        print("START", name, flush=True)
        started = time.perf_counter()
        checks = []

        def check(env):
            # Every iteration: both in-bag and out-of-bag training scores must
            # agree with the represented ensemble, not only repeat each other.
            internal = env.model._Booster__inner_predict(data_idx=0).copy()
            external = env.model.predict(x, num_threads=8, num_iteration=env.iteration + 1)
            error = float(np.max(np.abs(internal - external)))
            checks.append(error)
            if error >= 1e-9:
                raise AssertionError(f"{name}: score mismatch at iteration {env.iteration}: {error}")

        check.order = 30
        booster = lgb.train(
            native(params), data, num_boost_round=params["n_estimators"], valid_sets=[data], callbacks=[check]
        )
        values = booster.predict(test, num_threads=8) / 100.0
        assert np.isfinite(values).all()
        path = output / f"{name}.txt"
        booster.save_model(str(path))
        text = path.read_text(encoding="utf-8")
        assert "[device_type: cuda]" in text
        assert "[gpu_use_dp: 1]" in text
        np.save(output / f"{name}.npy", values)
        prediction[name] = values
        records[name] = {
            "seconds": time.perf_counter() - started,
            "iterations": booster.current_iteration(),
            "maximumTrainingScoreDifference": max(checks),
            "modelSha256": sha(path),
        }
        print("DONE", name, json.dumps(records[name]), flush=True)

    target = parameters["trial-0029"]
    prior = parameters["trial-0028"]
    fit("fresh_a", target, dataset(target))
    fit("fresh_b", target, dataset(target))
    reused = dataset(prior)
    fit("cache_warmup", prior, reused)
    fit("cache_reused", target, reused)
    differences = {
        name: float(np.max(np.abs(prediction["fresh_a"] - prediction[name]))) for name in ["fresh_b", "cache_reused"]
    }
    result = {
        "status": "passed" if all(v <= 1e-9 for v in differences.values()) else "failed_repeatability",
        "dll": str(expected_dll),
        "dllSha256": sha(expected_dll),
        "package": str(Path(lgb.__file__).resolve()),
        "fixtureShape": list(x.shape),
        "predictionRows": len(test),
        "sourcePanelSha256": meta["sourcePanelSha256"],
        "doublePrecision": True,
        "forcedLaunchBlocking": False,
        "fits": records,
        "predictionDifferences": differences,
        "holdoutEvaluated": False,
        "fullTrainingRun": False,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    assert result["status"] == "passed", differences
    print("FROZEN_INPUT_VALIDATION_PASSED", json.dumps(differences), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        child(args.fixture, args.output, args.dll)
        return
    repo = Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True, exist_ok=False)
    stage = args.output / "python-stage"
    shutil.copytree(repo / "python-package/lightgbm", stage / "lightgbm", ignore=shutil.ignore_patterns("__pycache__"))
    lib = stage / "lightgbm/lib"
    lib.mkdir(exist_ok=True)
    staged = lib / "lib_lightgbm.dll"
    shutil.copy2(args.dll, staged)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(stage.resolve()), PYTHONNOUSERSITE="1", PYTHONIOENCODING="utf-8")
    env.pop("CUDA_LAUNCH_BLOCKING", None)
    env.pop("CUDA_FORCE_PTX_JIT", None)
    result = subprocess.run(
        [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--child",
            "--dll",
            str(staged.resolve()),
            "--fixture",
            str(args.fixture.resolve()),
            "--output",
            str(args.output.resolve()),
        ],
        env=env,
        cwd=stage,
        check=False,
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
