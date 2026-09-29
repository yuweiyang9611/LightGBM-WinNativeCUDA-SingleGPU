"""Query bagging must validate metadata and size buffers by query count."""

from pathlib import Path

import pytest
from test_cuda_runtime import _assert_subprocess_passed, _run_with_staged_package

pytest_plugins = ["test_cuda_runtime"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("balanced", [False, True])
def test_query_bagging_requires_groups(
    staged_lightgbm: tuple[Path, Path], device: str, reset: bool, balanced: bool
) -> None:
    """Missing query metadata is rejected before changing an existing model."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np
        rng = np.random.default_rng(1729)
        x = rng.normal(size=(512, 4))
        y = 3*x[:, 0] - x[:, 1]
        params = dict(objective='regression', device_type='{device}', num_threads=4,
            gpu_use_dp=True, num_leaves=7, max_bin=31, seed=1729, verbosity=-1)
        changes = dict(bagging_by_query=True, bagging_fraction=.5, bagging_freq=1)
        if {balanced}:
            y = (x[:, 0] > 0).astype(float)
            changes.update(objective='binary', bagging_fraction=1., pos_bagging_fraction=.5)
        if {reset}:
            actual = lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=2, keep_training_booster=True)
            expected = lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=2, keep_training_booster=True)
        try:
            if {reset}:
                actual.reset_parameter(dict(changes, learning_rate=.7))
            else:
                lgb.train(dict(params, **changes), lgb.Dataset(x, label=y), num_boost_round=1)
        except lgb.basic.LightGBMError as error:
            assert 'query' in str(error).lower(), error
        else:
            raise AssertionError('Query sampling without groups was accepted')
        if {reset}:
            actual.update()
            expected.update()
            np.testing.assert_allclose(actual.predict(x, num_threads=4),
                expected.predict(x, num_threads=4), rtol=0, atol=1e-10)
        """,
    )
    _assert_subprocess_passed(result)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("disabled", ["frequency", "fraction"])
def test_inactive_query_bagging_without_groups(staged_lightgbm: tuple[Path, Path], device: str, disabled: str) -> None:
    """A query flag needs no metadata when sampling is inactive."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np
        rng = np.random.default_rng(1729)
        x = rng.normal(size=(512, 4))
        y = 3*x[:, 0] - x[:, 1]
        params = dict(objective='regression', device_type='{device}', num_threads=4,
            bagging_fraction={0.5 if disabled == "frequency" else 1.0},
            bagging_freq={0 if disabled == "frequency" else 1},
            gpu_use_dp=True, num_leaves=7, max_bin=31, verbosity=-1, seed=1729)
        actual = lgb.train(dict(params, bagging_by_query=True), lgb.Dataset(x, label=y), num_boost_round=3)
        expected = lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=3)
        np.testing.assert_allclose(actual.predict(x, num_threads=4),
            expected.predict(x, num_threads=4), rtol=0, atol=1e-10)
        """,
    )
    _assert_subprocess_passed(result)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_query_bagging_empty_groups(staged_lightgbm: tuple[Path, Path], device: str) -> None:
    """Accepted zero-size groups must not overflow the sampled query-index buffer."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np
        rng = np.random.default_rng(1729)
        x = rng.normal(size=(128, 4))
        y = 3*x[:, 0] - x[:, 1]
        groups = np.tile([0]*64 + [4], 32)
        params = dict(objective='regression', device_type='{device}', bagging_by_query=True,
            bagging_fraction=.8, bagging_freq=1, num_threads=4, gpu_use_dp=True,
            min_data_in_leaf=2, num_leaves=7, max_bin=31, verbosity=-1, seed=1729)
        model = lgb.train(params, lgb.Dataset(x, label=y, group=groups),
            num_boost_round=3, keep_training_booster=True)
        prediction = model.predict(x, num_threads=4)
        assert np.isfinite(prediction).all()
        assert np.mean((prediction-y)**2) < np.var(y)
        np.testing.assert_allclose(model._Booster__inner_predict(data_idx=0),
            prediction, rtol=0, atol=1e-10)
        """,
    )
    _assert_subprocess_passed(result)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_query_metadata_rejection_preserves_dataset(staged_lightgbm: tuple[Path, Path], device: str) -> None:
    """Replacing grouped training data with ungrouped data must fail without mutation."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np
        rng = np.random.default_rng(1729)
        x = rng.normal(size=(512, 4))
        y = 3*x[:, 0] - x[:, 1]
        params = dict(objective='regression', device_type='{device}', num_threads=4,
            bagging_by_query=True, bagging_fraction=.8, bagging_freq=1,
            gpu_use_dp=True, num_leaves=7, max_bin=31, verbosity=-1, seed=1729)
        actual = lgb.train(params, lgb.Dataset(x, label=y, group=[8]*64), num_boost_round=2, keep_training_booster=True)
        expected = lgb.train(params, lgb.Dataset(x, label=y, group=[8]*64), num_boost_round=2, keep_training_booster=True)
        try:
            actual.update(train_set=lgb.Dataset(x, label=y, reference=actual.train_set))
        except lgb.basic.LightGBMError as error:
            assert 'query' in str(error).lower(), error
        else:
            raise AssertionError('Ungrouped replacement accepted')
        actual.update()
        expected.update()
        np.testing.assert_allclose(actual.predict(x, num_threads=4),
            expected.predict(x, num_threads=4), rtol=0, atol=1e-10)
        """,
    )
    _assert_subprocess_passed(result)
