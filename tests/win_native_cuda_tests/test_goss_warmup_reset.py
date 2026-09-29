"""Returning to GOSS warmup must restore the full training Dataset."""

from pathlib import Path

import pytest
from test_cuda_runtime import _assert_subprocess_passed, _run_with_staged_package

pytest_plugins = ["test_cuda_runtime"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("top_rate", [0.2, 0.6])
def test_goss_changes_subset_mode(staged_lightgbm: tuple[Path, Path], device: str, top_rate: float) -> None:
    """Changing the sampling ratio matches a reset that explicitly restores the Dataset."""
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
            top_rate={top_rate}, other_rate=.1, learning_rate=.3, gpu_use_dp=True,
            num_threads=4, num_leaves=7, max_bin=31, verbosity=-1, seed=1729)
        actual = lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=8, keep_training_booster=True)
        expected = lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=8, keep_training_booster=True)
        new_top_rate = {0.6 if top_rate == 0.2 else 0.2}
        actual.reset_parameter(dict(top_rate=new_top_rate))
        expected.reset_parameter(dict(top_rate=new_top_rate))
        actual.update()
        expected.update(train_set=lgb.Dataset(x, label=y, reference=expected.train_set))
        np.testing.assert_allclose(actual.predict(x, num_threads=4),
            expected.predict(x, num_threads=4), rtol=0, atol=1e-10)
        np.testing.assert_allclose(actual._Booster__inner_predict(data_idx=0),
            actual.predict(x, num_threads=4), rtol=0, atol=1e-10)
        """,
    )
    _assert_subprocess_passed(result)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("top_rate", [0.2, 0.6])
def test_goss_reenters_warmup(staged_lightgbm: tuple[Path, Path], device: str, top_rate: float) -> None:
    """A warmup update uses every row, matching continuation with a fresh learner."""
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
            top_rate={top_rate}, other_rate=.1, learning_rate=.3, gpu_use_dp=True,
            num_threads=4, num_leaves=7, max_bin=31, verbosity=-1, seed=1729)
        actual = lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=8,
            keep_training_booster=True)
        snapshot = actual.model_to_string()
        actual.reset_parameter(dict(learning_rate=.01))
        actual.update()
        tree = actual.dump_model()['tree_info'][-1]['tree_structure']
        assert tree['internal_count'] == len(x), tree
        actual.update()
        assert actual.dump_model()['tree_info'][-1]['tree_structure']['internal_count'] == len(x)
        expected = lgb.train(dict(params, learning_rate=.01), lgb.Dataset(x, label=y),
            num_boost_round=2, init_model=lgb.Booster(model_str=snapshot), keep_training_booster=True)
        np.testing.assert_allclose(actual.predict(x, num_threads=4),
            expected.predict(x, num_threads=4), rtol=0, atol=1e-10)
        np.testing.assert_allclose(actual._Booster__inner_predict(data_idx=0),
            actual.predict(x, num_threads=4), rtol=0, atol=1e-10)
        actual.reset_parameter(dict(learning_rate=.3))
        actual.update()
        assert actual.dump_model()['tree_info'][-1]['tree_structure']['internal_count'] < len(x)
        np.testing.assert_allclose(actual._Booster__inner_predict(data_idx=0),
            actual.predict(x, num_threads=4), rtol=0, atol=1e-10)
        """,
    )
    _assert_subprocess_passed(result)
