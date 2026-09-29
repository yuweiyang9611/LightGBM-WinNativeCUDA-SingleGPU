"""Large-bin workspaces must grow when replacement data needs a larger grid."""

from pathlib import Path

import pytest
from test_cuda_runtime import _assert_subprocess_passed, _run_with_staged_package

pytest_plugins = ["test_cuda_runtime"]


@pytest.mark.parametrize(("quantized", "double_precision"), [(False, True), (False, False), (True, True)])
@pytest.mark.parametrize("values_per_feature", [2, 16])
def test_histogram_workspace_grows_on_dataset_reset(
    staged_lightgbm: tuple[Path, Path], quantized: bool, double_precision: bool, values_per_feature: int
) -> None:
    """A large-bin feature and low-cardinality columns force grid growth on replacement."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np
        rng = np.random.default_rng(1729)
        def generate(rows):
            x = rng.integers(0,{values_per_feature},size=(rows,64),dtype=np.int8).astype(np.float32)
            x[:,0] = np.linspace(-1,1,rows,dtype=np.float32)
            return x, 3*x[:,0] + .5*x[:,1]
        params = dict(objective='regression',device_type='cuda',gpu_use_dp={double_precision},
            use_quantized_grad={quantized},num_threads=4,num_leaves=4,max_bin=8191,
            min_data_in_bin=1,enable_bundle=False,is_enable_sparse=False,seed=1729,verbosity=-1)
        x, y = generate(12000)
        reference = lgb.Dataset(x,label=y,params=params)
        model = lgb.train(params,reference,num_boost_round=1,keep_training_booster=True)
        assert reference.feature_num_bin(0) > 6144
        for iteration, rows in enumerate([800000,12000], start=2):
            x, y = generate(rows)
            before = model.predict(x,num_threads=4)
            model.update(train_set=lgb.Dataset(x,label=y,reference=reference,params=params))
            prediction = model.predict(x,num_threads=4)
            assert model.num_trees() == iteration
            assert np.isfinite(prediction).all()
            assert np.mean((prediction-y)**2) < np.mean((before-y)**2)
            np.testing.assert_allclose(model._Booster__inner_predict(data_idx=0),prediction,rtol=0,atol=1e-9)
        """,
    )
    _assert_subprocess_passed(result)
