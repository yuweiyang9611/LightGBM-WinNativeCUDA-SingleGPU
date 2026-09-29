"""Late configuration validation must not partially update a Booster."""

from pathlib import Path

import pytest
from test_cuda_runtime import _assert_subprocess_passed, _run_with_staged_package

pytest_plugins = ["test_cuda_runtime"]


@pytest.mark.parametrize("top_rate", [0.2, 0.6])
def test_cuda_goss_beyond_warmup(staged_lightgbm: tuple[Path, Path], top_rate: float) -> None:
    """GOSS must keep training once sampling starts, with and without a CPU subset."""
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
        params = dict(objective='regression', device_type='cuda', data_sample_strategy='goss',
            top_rate={top_rate}, other_rate=.1, learning_rate=.3, gpu_use_dp=True,
            num_threads=4, num_leaves=7, max_bin=31, verbosity=-1, seed=1729)
        model = lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=8,
            keep_training_booster=True)
        prediction = model.predict(x, num_threads=4)
        assert model.num_trees() == 8
        assert np.isfinite(prediction).all()
        assert np.mean((prediction-y)**2) < np.var(y)*.2
        np.testing.assert_allclose(model._Booster__inner_predict(data_idx=0), prediction, rtol=0, atol=1e-10)
        """,
    )
    _assert_subprocess_passed(result)


@pytest.mark.parametrize(
    ("device", "case"),
    [("cuda", "device")]
    + [
        (device, case)
        for device in ["cpu", "cuda"]
        for case in ["contribution", "monotone", "rates", "bagging", "rf_sampling"]
    ],
)
def test_rejected_late_configuration(staged_lightgbm: tuple[Path, Path], device: str, case: str) -> None:
    """A rejected batch containing an objective change must preserve all old behavior."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np
        rng = np.random.default_rng(1729)
        x = rng.normal(size=(512, 4))
        y = (x[:, 0] + x[:, 1] > 0).astype(float)
        params = dict(objective='regression', metric='l2', device_type='{device}',
            gpu_device_id=0, gpu_use_dp=True, num_threads=4, num_leaves=7,
            max_bin=31, learning_rate=.1, verbosity=-1, seed=1729)
        if '{case}' in ['rates', 'bagging']:
            params['data_sample_strategy'] = 'goss'
        if '{case}' == 'rf_sampling':
            params.update(boosting='rf', bagging_fraction=.8, bagging_freq=1)
        changes = dict(
            device=dict(gpu_device_id=999),
            contribution=dict(feature_contri=[1, 1]),
            monotone=dict(monotone_constraints=[1, 0]),
            rates=dict(top_rate=.8, other_rate=.8),
            bagging=dict(bagging_fraction=.8, bagging_freq=1),
            rf_sampling=dict(bagging_fraction=1., bagging_freq=0))['{case}']
        changes['learning_rate'] = .7
        if '{case}' != 'device':
            changes['objective'] = 'binary'
        actual = lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=2, keep_training_booster=True)
        expected = lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=2, keep_training_booster=True)
        before = actual.predict(x, num_threads=4)
        try:
            actual.reset_parameter(changes)
        except lgb.basic.LightGBMError:
            pass
        else:
            raise AssertionError('Invalid configuration accepted')
        np.testing.assert_array_equal(actual.predict(x, num_threads=4), before)
        for _ in range(3):
            actual.update()
            expected.update()
        np.testing.assert_allclose(actual.predict(x, num_threads=4),
            expected.predict(x, num_threads=4), rtol=0, atol=1e-10)
        for model in [actual, expected]:
            model.reset_parameter(dict(learning_rate=.2))
            model.update()
        np.testing.assert_allclose(actual.predict(x, num_threads=4),
            expected.predict(x, num_threads=4), rtol=0, atol=1e-10)
        """,
    )
    _assert_subprocess_passed(result)
