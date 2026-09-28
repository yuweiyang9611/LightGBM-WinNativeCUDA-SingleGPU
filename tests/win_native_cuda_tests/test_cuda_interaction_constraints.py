"""Categorical CUDA splits must retain the branch's interaction constraints."""

from pathlib import Path

import pytest
from test_cuda_runtime import _assert_subprocess_passed, _run_with_staged_package

pytest_plugins = ["test_cuda_runtime"]


@pytest.mark.parametrize("num_leaves", [2, 8])
def test_categorical_split_at_leaf_capacity(staged_lightgbm: tuple[Path, Path], num_leaves: int) -> None:
    """The final categorical split must not write beyond branch metadata."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np

        x = (np.arange(1024) % {num_leaves}).reshape(-1, 1).astype(float)
        y = x[:, 0] ** 2
        model = lgb.train(dict(objective='regression', device_type='cuda',
            gpu_use_dp=True, num_threads=4, verbosity=-1, num_leaves={num_leaves},
            min_data_in_leaf=5, cat_smooth=0, cat_l2=0, min_data_per_group=5,
            interaction_constraints=[[0]], seed=1729),
            lgb.Dataset(x, label=y, categorical_feature=[0]), num_boost_round=3)
        trees = model.dump_model()['tree_info']
        assert len(trees) == 3
        assert all(t['num_leaves'] == {num_leaves} for t in trees)
        assert np.isfinite(model.predict(x)).all()
        """,
    )
    _assert_subprocess_passed(result)


@pytest.mark.parametrize("one_hot", [False, True])
def test_categorical_splits_preserve_interaction_groups(staged_lightgbm: tuple[Path, Path], one_hot: bool) -> None:
    """Every root-to-leaf path must stay inside an allowed feature group."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np

        grid = np.indices((8, 8)).reshape(2, -1).T
        x = np.tile(grid, (32, 1)).astype(float)
        y = 2 * x[:, 0] ** 2 + x[:, 1] ** 2
        model = lgb.train(dict(objective='regression', device_type='cuda',
            gpu_use_dp=True, num_threads=4, verbosity=-1, num_leaves=16, max_depth=3,
            min_data_in_leaf=5, cat_smooth=0, cat_l2=0, min_data_per_group=5,
            max_cat_to_onehot={16 if one_hot else 4},
            interaction_constraints=[[0], [1]], seed=1729),
            lgb.Dataset(x, label=y, categorical_feature=[0, 1]), num_boost_round=8)
        def check(node, features):
            if 'split_index' not in node:
                return
            features = features | {{node['split_feature']}}
            assert len(features) == 1, features
            assert node['decision_type'] == '=='
            check(node['left_child'], features)
            check(node['right_child'], features)
        trees = model.dump_model()['tree_info']
        assert all(t['num_leaves'] >= 3 for t in trees)
        for tree in trees:
            check(tree['tree_structure'], set())
        """,
    )
    _assert_subprocess_passed(result)
