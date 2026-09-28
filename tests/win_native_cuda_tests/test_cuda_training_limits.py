"""Regression checks for CUDA quantized training and tree depth limits."""

from pathlib import Path

import pytest
from test_cuda_runtime import _assert_subprocess_passed, _run_with_staged_package

pytest_plugins = ["test_cuda_runtime"]


@pytest.mark.parametrize("num_rows", [512, 1025])
@pytest.mark.parametrize("gpu_use_dp", [False, True])
def test_quantized_training_learns_nonconstant_target(
    staged_lightgbm: tuple[Path, Path], num_rows: int, gpu_use_dp: bool
) -> None:
    """Exercise packed gradients across CUDA block boundaries."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np

        x = np.linspace(-1, 1, {num_rows}).reshape(-1, 1)
        y = x[:, 0] ** 2 + x[:, 0]
        model = lgb.train(dict(objective='regression', device_type='cuda',
            gpu_use_dp={gpu_use_dp}, use_quantized_grad=True, num_threads=4,
            verbosity=-1, num_leaves=8, min_data_in_leaf=5, max_bin=31, seed=1729),
            lgb.Dataset(x, label=y), num_boost_round=3)
        predictions = model.predict(x, num_threads=4)
        assert np.isfinite(predictions).all()
        assert model.num_trees() == 3
        # Quantization can stop splitting before reaching the leaf budget.
        assert all(tree['num_leaves'] > 1 for tree in model.dump_model()['tree_info'])
        assert np.std(predictions) > 0.1
        assert np.mean((predictions-y)**2) < 0.7 * np.var(y)
        """,
    )
    _assert_subprocess_passed(result)


@pytest.mark.parametrize(("categorical", "quantized"), [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("max_depth", [1, 2, 0, -1])
def test_cuda_respects_max_depth(
    staged_lightgbm: tuple[Path, Path], categorical: bool, quantized: bool, max_depth: int
) -> None:
    """Check actual tree topology for numerical and categorical splits."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np

        x = np.linspace(-1, 1, 512).reshape(-1, 1)
        categorical = {categorical}
        if categorical:
            x = (np.arange(512) % 16).reshape(-1, 1).astype(float)
        y = x[:, 0] ** 2 + x[:, 0]
        model = lgb.train(dict(objective='regression', device_type='cuda',
            gpu_use_dp=True, use_quantized_grad={quantized}, num_threads=4, verbosity=-1, num_leaves=8,
            max_depth={max_depth}, min_data_in_leaf=5, max_bin=31, seed=1729,
            cat_smooth=0, cat_l2=0, min_data_per_group=5),
            lgb.Dataset(x, label=y, categorical_feature=[0] if categorical else []),
            num_boost_round=3)
        def depth(node):
            if 'split_index' not in node:
                return 0
            return 1 + max(depth(node['left_child']), depth(node['right_child']))
        trees = model.dump_model()['tree_info']
        actual_depths = [depth(tree['tree_structure']) for tree in trees]
        assert len(trees) == 3
        if {max_depth} > 0:
            assert max(actual_depths) <= {max_depth}, actual_depths
            assert max(actual_depths) == {max_depth}, actual_depths
        else:
            assert max(actual_depths) >= (2 if {quantized} else 3), actual_depths
        if categorical:
            assert all(tree['tree_structure']['decision_type'] == '==' for tree in trees)
        """,
    )
    _assert_subprocess_passed(result)


def test_cuda_depth_limit_after_reset_config(staged_lightgbm: tuple[Path, Path]) -> None:
    """Changing leaf capacity and depth must also constrain subsequent trees."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        """
        import lightgbm as lgb
        import numpy as np

        x = np.linspace(-1, 1, 512).reshape(-1, 1)
        y = x[:, 0] ** 2 + x[:, 0]
        model = lgb.train(dict(objective='regression', device_type='cuda',
            gpu_use_dp=True, num_threads=4, verbosity=-1, num_leaves=4,
            max_depth=1, min_data_in_leaf=5, max_bin=31),
            lgb.Dataset(x, label=y), num_boost_round=2, keep_training_booster=True)
        model.reset_parameter(dict(max_depth=2, num_leaves=8))
        model.update()
        model.update()
        model.reset_parameter(dict(max_depth=1))
        model.update()
        def depth(node):
            if 'split_index' not in node:
                return 0
            return 1 + max(depth(node['left_child']), depth(node['right_child']))
        actual = [depth(t['tree_structure']) for t in model.dump_model()['tree_info']]
        assert actual == [1, 1, 2, 2, 1], actual
        """,
    )
    _assert_subprocess_passed(result)
