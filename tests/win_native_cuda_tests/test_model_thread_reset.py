"""Dataset construction must not override a Booster's configured thread count."""

from pathlib import Path

import pytest
from test_cuda_runtime import _assert_subprocess_passed, _run_with_staged_package

pytest_plugins = ["test_cuda_runtime"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_metadata_reset_restores_booster_threads(staged_lightgbm: tuple[Path, Path], device: str) -> None:
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np
        rng = np.random.default_rng(1729)
        x = rng.normal(size=(2048,4))
        y = 3*x[:,0] + x[:,1]
        params = dict(objective='regression', device_type='{device}', use_quantized_grad=True,
            gpu_use_dp=True, num_threads=4, num_leaves=7, max_bin=31, seed=1729, verbosity=-1)
        original = lgb.Dataset(x,label=y)
        reference = lgb.Dataset(x.copy(),label=y.copy())
        actual = lgb.train(params,original,num_boost_round=2,keep_training_booster=True)
        expected = lgb.train(params,reference,num_boost_round=2,keep_training_booster=True)
        weights = (.5 + np.abs(x[:,2])).astype(np.float32)
        original.set_weight(weights)
        actual.update()
        expected.update(train_set=lgb.Dataset(x,label=y,weight=weights,reference=reference))
        np.testing.assert_allclose(actual.predict(x,num_threads=4),expected.predict(x,num_threads=4),rtol=0,atol=1e-10)
        """,
        extra_env={"OMP_NUM_THREADS": "2"},
    )
    _assert_subprocess_passed(result)
