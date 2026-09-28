"""CUDA training/validation scores must agree with the ensemble's routing."""

from __future__ import annotations

import pytest
from test_cuda_runtime import _assert_subprocess_passed, _run_with_staged_package

pytest_plugins = ["test_cuda_runtime"]


@pytest.mark.parametrize("bagging", [1.0, 0.7], ids=["full-data", "out-of-bag"])
def test_nan_routing_training_and_validation_scores(staged_lightgbm, bagging):
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np
        from pathlib import Path
        import os

        assert Path(lgb.basic._LIB._name).resolve() == Path(os.environ['EXPECTED_LIGHTGBM_DLL']).resolve()
        rng = np.random.default_rng(29)
        x = rng.normal(size=(4096, 8)).astype(np.float64)
        x[::5, 0] = np.nan
        x[1::13, 0] = 0.0
        y = np.where(np.isnan(x[:, 0]), -9.0, 2.0*np.nan_to_num(x[:, 0])) + 0.1*x[:, 1]
        xv = rng.normal(size=(257, 8)).astype(np.float64)
        xv[::4, 0] = np.nan
        yv = np.where(np.isnan(xv[:, 0]), -9.0, 2.0*np.nan_to_num(xv[:, 0])) + 0.1*xv[:, 1]
        train = lgb.Dataset(x, label=y, free_raw_data=False)
        valid = lgb.Dataset(xv, label=yv, reference=train, free_raw_data=False)
        deviations = []
        def check(env):
            for index, values in [(0,x),(1,xv)]:
                internal = env.model._Booster__inner_predict(data_idx=index).copy()
                predicted = env.model.predict(values, num_threads=4, num_iteration=env.iteration+1)
                deviations.append(float(np.max(np.abs(internal-predicted))))
        check.order = 30
        model = lgb.train(dict(objective='regression', device_type='cuda', gpu_device_id=0, num_gpu=1,
            gpu_use_dp=True, seed=1729, verbosity=-1, num_threads=4, max_bin=31,
            num_leaves=7, min_data_in_leaf=20, bagging_fraction={bagging}, bagging_freq=1),
            train, num_boost_round=24, valid_sets=[train,valid], callbacks=[check])
        def has_nan_left(node):
            if 'split_index' not in node: return False
            return (node['split_feature']==0 and node['default_left'] and node['missing_type']=='NaN') or has_nan_left(node['left_child']) or has_nan_left(node['right_child'])
        assert any(has_nan_left(t['tree_structure']) for t in model.dump_model()['tree_info']), 'Fixture must exercise NaN default-left'
        assert max(deviations) < 1e-9, ('CUDA training/validation scores drift from ensemble', max(deviations))
    """,
    )
    _assert_subprocess_passed(result)
