"""Repeated bagging resets must preserve pending work and select the right sampler."""

from pathlib import Path

import pytest
from test_cuda_runtime import _assert_subprocess_passed, _run_with_staged_package

pytest_plugins = ["test_cuda_runtime"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_bagging_seed_reset(staged_lightgbm: tuple[Path, Path], device: str) -> None:
    """An accepted seed update must match independently initialized sampling with that seed."""
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
        params = dict(objective='regression', device_type='{device}', bagging_fraction=.7,
            bagging_freq=1, gpu_use_dp=True, num_threads=4, num_leaves=7, max_bin=31,
            verbosity=-1, seed=1729)
        actual = lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=3, keep_training_booster=True)
        snapshot = actual.model_to_string()
        actual.reset_parameter(dict(bagging_seed=77))
        actual.update()
        expected = lgb.train(dict(params, bagging_seed=77), lgb.Dataset(x, label=y),
            num_boost_round=1, init_model=lgb.Booster(model_str=snapshot), keep_training_booster=True)
        np.testing.assert_allclose(actual.predict(x, num_threads=4),
            expected.predict(x, num_threads=4), rtol=0, atol=1e-10)
        """,
    )
    _assert_subprocess_passed(result)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("initial_fraction", [0.3, 1.0])
def test_pending_rebag_survives_second_reset(
    staged_lightgbm: tuple[Path, Path], device: str, initial_fraction: float
) -> None:
    """Splitting one configuration batch must not cancel its pending sample refresh."""
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
        params = dict(objective='regression', device_type='{device}', bagging_fraction={initial_fraction},
            bagging_freq=3, gpu_use_dp=True, num_threads=4, num_leaves=7, max_bin=31,
            verbosity=-1, seed=1729)
        actual = lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=2, keep_training_booster=True)
        expected = lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=2, keep_training_booster=True)
        actual.reset_parameter(dict(bagging_fraction=.8))
        actual.reset_parameter(dict(learning_rate=.2))
        expected.reset_parameter(dict(bagging_fraction=.8, learning_rate=.2))
        actual.update()
        expected.update()
        np.testing.assert_allclose(actual.predict(x, num_threads=4),
            expected.predict(x, num_threads=4), rtol=0, atol=1e-10)
        assert actual.dump_model()['tree_info'][-1]['tree_structure']['internal_count'] == \
            expected.dump_model()['tree_info'][-1]['tree_structure']['internal_count']
        """,
    )
    _assert_subprocess_passed(result)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_balanced_bagging_returns_to_uniform(staged_lightgbm: tuple[Path, Path], device: str) -> None:
    """Restoring class-specific fractions to one must honor the ordinary fraction."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np
        rng = np.random.default_rng(1729)
        x = rng.normal(size=(2048, 4))
        y = (x[:, 0] > .5).astype(float)
        params = dict(objective='binary', device_type='{device}', bagging_fraction=.5,
            pos_bagging_fraction=.2, neg_bagging_fraction=.8, bagging_freq=1,
            gpu_use_dp=True, num_threads=4, num_leaves=7, max_bin=31, verbosity=-1, seed=1729)
        actual = lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=3, keep_training_booster=True)
        snapshot = actual.model_to_string()
        actual.reset_parameter(dict(pos_bagging_fraction=1., neg_bagging_fraction=1.))
        actual.update()
        tree = actual.dump_model()['tree_info'][-1]['tree_structure']
        assert tree['internal_count'] < len(x), tree
        expected = lgb.train(dict(params, pos_bagging_fraction=1., neg_bagging_fraction=1.),
            lgb.Dataset(x, label=y), num_boost_round=1, init_model=lgb.Booster(model_str=snapshot),
            keep_training_booster=True)
        np.testing.assert_allclose(actual.predict(x, num_threads=4),
            expected.predict(x, num_threads=4), rtol=0, atol=1e-10)
        """,
    )
    _assert_subprocess_passed(result)
