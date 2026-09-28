"""Runtime parameter resets must refresh feature selection on both backends."""

from pathlib import Path

import pytest
from test_cuda_runtime import _assert_subprocess_passed, _run_with_staged_package

pytest_plugins = ["test_cuda_runtime"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    ("initial", "updated"),
    [
        ({}, {"interaction_constraints": [[1], [2], [3]]}),
        ({"interaction_constraints": [[1], [2], [3]]}, {"interaction_constraints": []}),
        ({"interaction_constraints": [[0], [1]]}, {"interaction_constraints": [[2], [3]]}),
        ({"feature_fraction_bynode": 1.0}, {"feature_fraction_bynode": 0.25}),
        ({"feature_fraction_bynode": 0.25}, {"feature_fraction_bynode": 1.0}),
    ],
)
def test_reset_feature_selection_matches_initial_config(
    staged_lightgbm: tuple[Path, Path], device: str, initial: dict, updated: dict
) -> None:
    """Compare a reset before the first tree with the same config set at creation."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np

        rng = np.random.default_rng(1729)
        x = rng.normal(size=(512, 4))
        y = 10 * x[:, 0] + x[:, 1] - 2 * x[:, 2] + x[:, 3]
        params = dict(objective='regression', device_type='{device}', num_threads=4,
            num_leaves=4, max_bin=31, gpu_use_dp=True, verbosity=-1, feature_fraction_seed=3)
        initial = {initial!r}
        updated = {updated!r}
        model = lgb.Booster(dict(params, **initial), lgb.Dataset(x, label=y, params=params))
        model.reset_parameter(updated)
        model.update()
        expected = lgb.Booster(dict(params, **updated), lgb.Dataset(x, label=y, params=params))
        expected.update()
        np.testing.assert_allclose(model.predict(x), expected.predict(x), rtol=0, atol=1e-12)
        assert model.dump_model()['tree_info'] == expected.dump_model()['tree_info']
        """,
    )
    _assert_subprocess_passed(result)
