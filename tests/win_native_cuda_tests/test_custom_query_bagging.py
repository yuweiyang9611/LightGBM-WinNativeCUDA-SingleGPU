"""Query sampling must also run when gradients come from a custom objective."""

from pathlib import Path

import pytest
from test_cuda_runtime import _assert_subprocess_passed, _run_with_staged_package

pytest_plugins = ["test_cuda_runtime"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("fraction", [0.3, 0.8])
def test_custom_objective_query_sampling(staged_lightgbm: tuple[Path, Path], device: str, fraction: float) -> None:
    """Custom squared-error gradients must match built-in regression on sampled queries."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np
        rng = np.random.default_rng(1729)
        groups = np.tile([4, 12], 64)
        x = rng.normal(size=(sum(groups), 4))
        y = 3*x[:, 0] - x[:, 1]
        def squared_error(prediction, data):
            return prediction-data.get_label(), np.ones_like(prediction)
        params = dict(device_type='{device}', bagging_by_query=True, bagging_fraction={fraction},
            bagging_freq=1, boost_from_average=False, gpu_use_dp=True, num_threads=4,
            num_leaves=7, max_bin=31, seed=1729, verbosity=-1)
        actual = lgb.train(dict(params, objective=squared_error),
            lgb.Dataset(x, label=y, group=groups), num_boost_round=3, keep_training_booster=True)
        expected = lgb.train(dict(params, objective='regression'),
            lgb.Dataset(x, label=y, group=groups), num_boost_round=3, keep_training_booster=True)
        assert actual.num_trees() == 3
        for tree in actual.dump_model()['tree_info']:
            assert tree['tree_structure']['internal_count'] < len(x), tree
        np.testing.assert_allclose(actual.predict(x, num_threads=4),
            expected.predict(x, num_threads=4), rtol=0, atol=1e-7)
        np.testing.assert_allclose(actual._Booster__inner_predict(data_idx=0),
            actual.predict(x, num_threads=4), rtol=0, atol=1e-10)
        """,
    )
    _assert_subprocess_passed(result)
