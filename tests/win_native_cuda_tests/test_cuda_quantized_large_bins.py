"""Quantized CUDA split search must handle more bins than one thread block."""

from pathlib import Path

import pytest
from test_cuda_runtime import _assert_subprocess_passed, _run_with_staged_package

pytest_plugins = ["test_cuda_runtime"]


@pytest.mark.parametrize("max_bin", [511, 1023])
@pytest.mark.parametrize("missing", ["none", "nan", "zero"])
@pytest.mark.parametrize("extra_trees", [False, True])
def test_quantized_large_bin_training(
    staged_lightgbm: tuple[Path, Path], max_bin: int, missing: str, extra_trees: bool
) -> None:
    """Check learning, score consistency, and independent leaf counts."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np

        rng = np.random.default_rng(1729)
        x = rng.normal(size=(8192, 3))
        if '{missing}' == 'nan':
            x[::7, 0] = np.nan
        elif '{missing}' == 'zero':
            x[::7, 0] = 0
        y = 3*np.nan_to_num(x[:, 0]) - x[:, 1] + 2*np.isnan(x[:, 0])
        data = lgb.Dataset(x, label=y)
        errors = []
        def check(env):
            internal = env.model._Booster__inner_predict(data_idx=0).copy()
            predicted = env.model.predict(x, num_iteration=env.iteration+1, num_threads=4)
            errors.append(float(np.max(np.abs(internal-predicted))))
        check.order = 30
        model = lgb.train(dict(objective='regression', device_type='cuda', use_quantized_grad=True,
            gpu_use_dp=True, num_threads=4, verbosity=-1, max_bin={max_bin}, num_leaves=8,
            zero_as_missing={"True" if missing == "zero" else "False"}, extra_trees={extra_trees}, seed=1729),
            data, num_boost_round=8, valid_sets=[data], callbacks=[check], keep_training_booster=True)
        assert data.feature_num_bin(1) > 256
        assert model.num_trees() == 8
        prediction = model.predict(x, num_threads=4)
        assert np.mean((prediction-y)**2) < .75*np.var(y)
        assert max(errors) < 1e-9, errors
        assignments = model.predict(x, pred_leaf=True, num_threads=4)
        for index, tree in enumerate(model.dump_model()['tree_info']):
            counts = np.bincount(assignments[:, index].astype(int), minlength=tree['num_leaves'])
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
