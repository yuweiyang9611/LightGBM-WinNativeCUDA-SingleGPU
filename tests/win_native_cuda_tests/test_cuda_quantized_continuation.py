"""CUDA quantization state must grow with continued training and leaf budgets."""

from pathlib import Path

import pytest
from test_cuda_runtime import _assert_subprocess_passed, _run_with_staged_package

pytest_plugins = ["test_cuda_runtime"]


@pytest.mark.parametrize("objective", ["regression", "multiclass"])
@pytest.mark.parametrize("stochastic_rounding", [False, True])
def test_quantized_continuation_matches_uninterrupted_training(
    staged_lightgbm: tuple[Path, Path], objective: str, stochastic_rounding: bool
) -> None:
    """Continuing beyond the initial iteration budget preserves the random sequence."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np
        rng = np.random.default_rng(1729)
        x = rng.normal(size=(4096, 8))
        y = x[:, 0]*3 - x[:, 1] + x[:, 2]**2
        if '{objective}' == 'multiclass':
            y = np.argmax(np.stack([2*x[:, 0], -x[:, 1], x[:, 2]], axis=1), axis=1)
        params = dict(objective='{objective}', device_type='cuda', use_quantized_grad=True,
            stochastic_rounding={stochastic_rounding},
            gpu_use_dp=True, num_threads=4, num_leaves=8, max_bin=31, seed=1729, verbosity=-1)
        if '{objective}' == 'multiclass':
            params['num_class'] = 3
        continued = lgb.train(params, lgb.Dataset(x,label=y), num_boost_round=2, keep_training_booster=True)
        for _ in range(6):
            continued.update()
        expected = lgb.train(params, lgb.Dataset(x,label=y), num_boost_round=8, keep_training_booster=True)
        assert continued.num_trees() == 8 * (3 if '{objective}' == 'multiclass' else 1)
        np.testing.assert_allclose(continued.predict(x), expected.predict(x), rtol=0, atol=1e-12)
        assert continued.dump_model()['tree_info'] == expected.dump_model()['tree_info']
        """,
    )
    _assert_subprocess_passed(result)


@pytest.mark.parametrize("renew_leaf", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_quantized_leaf_budget_can_grow_and_shrink(
    staged_lightgbm: tuple[Path, Path], renew_leaf: bool, device: str
) -> None:
    """All host/device quantization leaf buffers must follow configuration resets."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np
        rng = np.random.default_rng(1729)
        x = rng.normal(size=(4096, 8))
        y = x[:, 0]*3 - x[:, 1] + x[:, 2]**2
        params = dict(objective='regression', device_type='{device}', use_quantized_grad=True,
            quant_train_renew_leaf={renew_leaf}, gpu_use_dp=True, num_threads=4,
            num_leaves=2, num_iterations=20, max_bin=31, seed=1729, verbosity=-1)
        model = lgb.Booster(params, lgb.Dataset(x,label=y,params=params))
        model.update()
        expected_leaves = [2]
        for leaves in [16, 4, 31, 2]:
            model.reset_parameter(dict(num_leaves=leaves))
            model.update()
            expected_leaves.append(leaves)
        trees = model.dump_model()['tree_info']
        assert [t['num_leaves'] for t in trees] == expected_leaves
        assert np.isfinite(model.predict(x)).all()
        internal = model._Booster__inner_predict(data_idx=0).copy()
        np.testing.assert_allclose(internal, model.predict(x), rtol=0, atol=1e-9)
        """,
    )
    _assert_subprocess_passed(result)


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("num_rows", [512, 4224])
def test_root_statistics_use_all_rows(staged_lightgbm: tuple[Path, Path], quantized: bool, num_rows: int) -> None:
    """Root output should use every reduction block, not just the first one."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np
        x = np.arange({num_rows}).reshape(-1,1).astype(float)
        y = (x[:, 0] >= {num_rows}//2).astype(float)
        model = lgb.train(dict(objective='regression',device_type='cuda',use_quantized_grad={quantized},
            boost_from_average=False,learning_rate=1,num_threads=4,gpu_use_dp=True,
            num_leaves=2,max_bin=31,verbosity=-1),lgb.Dataset(x,label=y),num_boost_round=1)
        root = model.dump_model()['tree_info'][0]['tree_structure']
        assert abs(root['internal_value']-y.mean()) < 1e-6, root
        assert root['internal_count'] == len(y)
        assert abs(root['internal_weight']-len(y)) < 1e-8
        """,
    )
    _assert_subprocess_passed(result)
