"""Training Dataset replacement must preserve lifetime and quantization state."""

from pathlib import Path

import pytest
from test_cuda_runtime import _assert_subprocess_passed, _run_with_staged_package

pytest_plugins = ["test_cuda_runtime"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_incompatible_dataset_reset_preserves_booster(staged_lightgbm: tuple[Path, Path], device: str) -> None:
    """Rejected bin mappers must not replace the objective or Python Dataset."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np

        rng = np.random.default_rng(1729)
        x = rng.normal(size=(512, 4))
        y = 3*x[:, 0] - x[:, 1]
        params = dict(objective='regression',device_type='{device}',use_quantized_grad=True,
            gpu_use_dp=True,num_threads=4,num_leaves=7,max_bin=31,verbosity=-1)
        original = lgb.Dataset(x,label=y,params=params)
        model = lgb.train(params,original,num_boost_round=2,keep_training_booster=True)
        before = model.predict(x)
        bad_params = dict(params,max_bin=15)
        bad = lgb.Dataset(x,label=y,params=bad_params)
        try:
            model.update(train_set=bad)
        except lgb.basic.LightGBMError as error:
            assert 'different bin mappers' in str(error), str(error)
        else:
            raise AssertionError('Expected incompatible Dataset rejection')
        assert model.train_set is original
        np.testing.assert_array_equal(before,model.predict(x))
        model.update()
        assert model.num_trees() == 3
        """,
    )
    _assert_subprocess_passed(result)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("change_weights", [False, True])
@pytest.mark.parametrize("quantized", [False, True])
def test_training_dataset_replacement(
    staged_lightgbm: tuple[Path, Path], device: str, change_weights: bool, quantized: bool
) -> None:
    """Grow and shrink rows while retaining bin alignment and existing trees."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np

        rng = np.random.default_rng(1729)
        def generate(rows):
            x = rng.normal(size=(rows, 4))
            y = 3*x[:, 0] - x[:, 1] + .2*x[:, 2]**2
            return x, y
        params = dict(objective='regression', device_type='{device}',use_quantized_grad={quantized},
            num_threads=4,gpu_use_dp=True,num_leaves=7,max_bin=31,verbosity=-1,seed=1729)
        x, y = generate(256)
        reference = lgb.Dataset(x,label=y,params=params,free_raw_data=False)
        model = lgb.train(params,reference,num_boost_round=2,keep_training_booster=True)
        for index, rows in enumerate([4096,128,8192]):
            x, y = generate(rows)
            weights = rng.uniform(.5,2,rows) if {change_weights} and index % 2 == 0 else None
            data = lgb.Dataset(x,label=y,weight=weights,reference=reference,params=params)
            before = model.predict(x,num_threads=4)
            model.update(train_set=data)
            prediction = model.predict(x,num_threads=4)
            assert model.num_trees() == index + 3
            assert np.isfinite(prediction).all()
            np.testing.assert_allclose(model._Booster__inner_predict(data_idx=0),prediction,rtol=0,atol=1e-9)
            before_loss = np.average((before-y)**2,weights=weights)
            after_loss = np.average((prediction-y)**2,weights=weights)
            assert after_loss < before_loss, (index, rows, before_loss, after_loss)
            root = model.dump_model()['tree_info'][-1]['tree_structure']
            assert root['internal_count'] == rows
            if weights is not None:
                # Each stochastically rounded Hessian differs by less than one bin.
                assert abs(root['internal_weight']-weights.sum()) <= rows*weights.max()/4 + 1e-5
        """,
    )
    _assert_subprocess_passed(result)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_identical_dataset_replacement_preserves_random_sequence(
    staged_lightgbm: tuple[Path, Path], device: str
) -> None:
    """Reattaching identical rows must not restart quantization's RNG."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np

        rng = np.random.default_rng(1729)
        x = rng.normal(size=(4096, 4))
        y = 3*x[:, 0] - x[:, 1] + .2*x[:, 2]**2
        params = dict(objective='regression',device_type='{device}',use_quantized_grad=True,
            gpu_use_dp=True,num_threads=4,num_leaves=7,max_bin=31,seed=1729,verbosity=-1)
        reference = lgb.Dataset(x,label=y,params=params)
        changed = lgb.train(params,reference,num_boost_round=2,keep_training_booster=True)
        changed.update(train_set=lgb.Dataset(x.copy(),label=y.copy(),reference=reference,params=params))
        changed.update()
        expected = lgb.train(params,lgb.Dataset(x,label=y),num_boost_round=4,keep_training_booster=True)
        np.testing.assert_allclose(changed.predict(x),expected.predict(x),rtol=0,atol=1e-12)
        assert changed.dump_model()['tree_info'] == expected.dump_model()['tree_info']
        """,
    )
    _assert_subprocess_passed(result)
