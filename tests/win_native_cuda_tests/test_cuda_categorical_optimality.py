"""Large categorical split searches must retain actual category identities."""

from pathlib import Path

import pytest
from test_cuda_runtime import _assert_subprocess_passed, _run_with_staged_package

pytest_plugins = ["test_cuda_runtime"]


@pytest.mark.parametrize("target_category", [40, 400])
@pytest.mark.parametrize("one_hot", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_large_categorical_histogram_finds_exact_stump(
    staged_lightgbm: tuple[Path, Path], target_category: int, one_hot: bool, missing: bool
) -> None:
    """One category has a different target, so its exact one-leaf split is known."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np

        x = np.tile(np.arange(512), 16).reshape(-1, 1).astype(float)
        if {missing}:
            x = np.concatenate([x, np.full((4096, 1), -1.)])
        y = (x[:, 0] == {target_category}).astype(float)
        data = lgb.Dataset(x, label=y, categorical_feature=[0])
        model = lgb.train(dict(objective='regression', device_type='cuda', num_threads=4,
            gpu_use_dp=True, min_data_in_leaf=1, min_data_per_group=1, cat_smooth=0,
            cat_l2=0, max_cat_to_onehot={1024 if one_hot else 4}, num_leaves=2,
            learning_rate=1, verbosity=-1), data, num_boost_round=1, keep_training_booster=True)
        assert data.feature_num_bin(0) > 256
        tree = model.dump_model()['tree_info'][0]['tree_structure']
        assert tree['decision_type'] == '=='
        prediction = model.predict(x, num_threads=4)
        assert np.mean((prediction-y)**2) < 1e-12, tree
        np.testing.assert_allclose(model._Booster__inner_predict(data_idx=0), prediction, rtol=0, atol=1e-10)
        leaves = model.predict(x, pred_leaf=True, num_threads=4).reshape(-1)
        counts = np.bincount(leaves.astype(int), minlength=2)
        assert counts[tree['left_child']['leaf_index']] == tree['left_child']['leaf_count']
        assert counts[tree['right_child']['leaf_index']] == tree['right_child']['leaf_count']
        """,
    )
    _assert_subprocess_passed(result)
