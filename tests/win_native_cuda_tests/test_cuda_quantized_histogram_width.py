"""Packed histograms must retain their values across the 16/32-bit boundary."""

from pathlib import Path

import pytest
from test_cuda_runtime import _assert_subprocess_passed, _run_with_staged_package

pytest_plugins = ["test_cuda_runtime"]


@pytest.mark.parametrize("num_rows", [16383, 16384, 32768])
@pytest.mark.parametrize("max_bin", [31, 511])
def test_quantized_histogram_width_transitions(staged_lightgbm: tuple[Path, Path], num_rows: int, max_bin: int) -> None:
    """Exercise narrow roots, wide roots, and wide-to-narrow child histograms."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np

        rng = np.random.default_rng(1729)
        x = rng.normal(size=({num_rows}, 4))
        x[rng.random(x.shape) < .3] = 0
        y = 3*x[:, 0] - x[:, 1] + x[:, 2]**2
        data = lgb.Dataset(x, label=y)
        errors = []
        def check(env):
            expected = env.model.predict(x, num_iteration=env.iteration+1, num_threads=4)
            actual = env.model._Booster__inner_predict(data_idx=0).copy()
            errors.append(float(np.max(np.abs(expected-actual))))
        check.order = 30
        model = lgb.train(dict(objective='regression',device_type='cuda',use_quantized_grad=True,
            num_grad_quant_bins=4,gpu_use_dp=True,num_threads=4,num_leaves=8,max_bin={max_bin},
            verbosity=-1,seed=1729), data, num_boost_round=4,valid_sets=[data],callbacks=[check])
        assert model.num_trees() == 4
        assert max(errors) < 1e-9, errors
        assert np.mean((model.predict(x)-y)**2) < .75*np.var(y)
        leaves = model.predict(x,pred_leaf=True,num_threads=4)
        for index, tree in enumerate(model.dump_model()['tree_info']):
            counts = np.bincount(leaves[:, index].astype(int), minlength=tree['num_leaves'])
            def check_counts(node):
                if 'leaf_index' in node:
                    count = int(counts[node['leaf_index']])
                    assert count > 0
                    assert node['leaf_count'] == count, node
                    assert abs(node['leaf_weight']-count) < 1e-8, node
                    return count
                count = check_counts(node['left_child']) + check_counts(node['right_child'])
                assert node['internal_count'] == count, node
                assert abs(node['internal_weight']-count) < 1e-8, node
                return count
            assert check_counts(tree['tree_structure']) == len(x)
        """,
    )
    _assert_subprocess_passed(result)
