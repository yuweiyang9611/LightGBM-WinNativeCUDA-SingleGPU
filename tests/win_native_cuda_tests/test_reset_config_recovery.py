"""A rejected runtime configuration must leave the existing Booster usable."""

from pathlib import Path

import pytest
from test_cuda_runtime import _assert_subprocess_passed, _run_with_staged_package

pytest_plugins = ["test_cuda_runtime"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_unknown_objective_rejected_during_construction(staged_lightgbm: tuple[Path, Path], device: str) -> None:
    """Unknown names must raise a normal error instead of returning an invalid pointer."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np
        x = np.arange(128).reshape(-1, 1)
        data = lgb.Dataset(x, label=x[:, 0])
        params = dict(objective='not_an_objective', device_type='{device}', num_threads=4, verbosity=-1)
        try:
            lgb.train(params, data, num_boost_round=1)
        except lgb.basic.LightGBMError as error:
            assert 'Unknown objective type name' in str(error)
        else:
            raise AssertionError('Unknown objective was accepted')
        model = lgb.train(dict(params, objective='regression'), data, num_boost_round=1)
        assert np.isfinite(model.predict(x, num_threads=4)).all()
        """,
    )
    _assert_subprocess_passed(result)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("objective", ["poisson", "gamma", "lambdarank", "not_an_objective"])
def test_rejected_objective_preserves_booster(staged_lightgbm: tuple[Path, Path], device: str, objective: str) -> None:
    """Prediction and continued training match a model that never received the bad reset."""
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
        params = dict(objective='regression', metric='l2', device_type='{device}',
            num_threads=4, gpu_use_dp=True, num_leaves=7, max_bin=31,
            learning_rate=.1, verbosity=-1, seed=1729)
        actual = lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=2, keep_training_booster=True)
        expected = lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=2, keep_training_booster=True)
        before = actual.predict(x, num_threads=4)
        try:
            actual.reset_parameter(dict(objective='{objective}', learning_rate=.7))
        except lgb.basic.LightGBMError:
            pass
        else:
            raise AssertionError('Invalid objective reset was accepted')
        np.testing.assert_array_equal(actual.predict(x, num_threads=4), before)
        actual.update()
        expected.update()
        np.testing.assert_allclose(actual.predict(x, num_threads=4),
            expected.predict(x, num_threads=4), rtol=0, atol=1e-10)
        np.testing.assert_allclose([v[2] for v in actual.eval_train()],
            [v[2] for v in expected.eval_train()], rtol=0, atol=1e-10)
        for model in [actual, expected]:
            model.reset_parameter(dict(learning_rate=.2))
            model.update(train_set=lgb.Dataset(x, label=y, reference=model.train_set))
        np.testing.assert_allclose(actual.predict(x, num_threads=4),
            expected.predict(x, num_threads=4), rtol=0, atol=1e-10)
        """,
    )
    _assert_subprocess_passed(result)
