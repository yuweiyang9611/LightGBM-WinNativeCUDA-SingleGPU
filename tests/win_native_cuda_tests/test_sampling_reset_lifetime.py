"""Sampling resets must retain valid Dataset and row-index ownership."""

from pathlib import Path

import pytest
from test_cuda_runtime import _assert_subprocess_passed, _run_with_staged_package

pytest_plugins = ["test_cuda_runtime"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("quantized", [False, True])
def test_repeated_goss_resets(staged_lightgbm: tuple[Path, Path], device: str, quantized: bool) -> None:
    """Two resets without an update match the same configuration applied together."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np
        rng = np.random.default_rng(1729)
        x = rng.normal(size=(2048, 4))
        y = 3*x[:, 0] - x[:, 1]
        params = dict(objective='regression', device_type='{device}', data_sample_strategy='goss',
            use_quantized_grad={quantized}, stochastic_rounding=False,
            top_rate=.2, other_rate=.1, learning_rate=.3, gpu_use_dp=True,
            num_threads=4, num_leaves=7, max_bin=31, verbosity=-1, seed=1729)
        actual = lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=8, keep_training_booster=True)
        expected = lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=8, keep_training_booster=True)
        actual.reset_parameter(dict(learning_rate=.2))
        actual.reset_parameter(dict(num_leaves=15))
        expected.reset_parameter(dict(learning_rate=.2, num_leaves=15))
        actual.update()
        expected.update()
        np.testing.assert_allclose(actual.predict(x, num_threads=4),
            expected.predict(x, num_threads=4), rtol=0, atol=1e-10)
        """,
    )
    _assert_subprocess_passed(result)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("fraction", [0.8, 1.0])
def test_bagging_leaves_subset_mode(staged_lightgbm: tuple[Path, Path], device: str, fraction: float) -> None:
    """Leaving a subset or disabling bagging matches an explicit Dataset refresh."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np
        rng = np.random.default_rng(1729)
        x = rng.normal(size=(2048, 4))
        y = 3*x[:, 0] - x[:, 1]
        params = dict(objective='regression', device_type='{device}', bagging_fraction=.3,
            bagging_freq=1, learning_rate=.2, gpu_use_dp=True, num_threads=4,
            num_leaves=7, max_bin=31, verbosity=-1, seed=1729)
        actual = lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=3, keep_training_booster=True)
        expected = lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=3, keep_training_booster=True)
        actual.reset_parameter(dict(bagging_fraction={fraction}))
        expected.reset_parameter(dict(bagging_fraction={fraction}))
        actual.update()
        expected.update(train_set=lgb.Dataset(x, label=y, reference=expected.train_set))
        if {fraction} == 1.:
            assert actual.dump_model()['tree_info'][-1]['tree_structure']['internal_count'] == len(x)
        np.testing.assert_allclose(actual.predict(x, num_threads=4),
            expected.predict(x, num_threads=4), rtol=0, atol=1e-10)
        np.testing.assert_allclose(actual._Booster__inner_predict(data_idx=0),
            actual.predict(x, num_threads=4), rtol=0, atol=1e-10)
        for model in [actual, expected]:
            model.reset_parameter(dict(bagging_fraction=.3))
            model.update()
        np.testing.assert_allclose(actual.predict(x, num_threads=4),
            expected.predict(x, num_threads=4), rtol=0, atol=1e-10)
        """,
    )
    _assert_subprocess_passed(result)
