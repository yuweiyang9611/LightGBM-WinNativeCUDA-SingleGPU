"""In-place metadata updates must match an equivalent Dataset replacement."""

from pathlib import Path

import pytest
from test_cuda_runtime import _assert_subprocess_passed, _run_with_staged_package

pytest_plugins = ["test_cuda_runtime"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("invalid", ["labels", "init_score"])
def test_rejected_labels_preserve_training_objective(
    staged_lightgbm: tuple[Path, Path], device: str, invalid: str
) -> None:
    """Validate a replacement objective before releasing the current objective."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np
        rng = np.random.default_rng(1729)
        x = rng.normal(size=(512,4))
        y = np.arange(len(x)) % 3
        params = dict(objective='multiclass',num_class=3,device_type='{device}',
            num_threads=4,num_leaves=7,max_bin=31,verbosity=-1)
        original = lgb.Dataset(x,label=y,params=params)
        model = lgb.train(params,original,num_boost_round=2,keep_training_booster=True)
        prediction = model.predict(x)
        bad_labels = np.full(len(x),99) if '{invalid}' == 'labels' else y
        bad_score = np.zeros(len(x)*2) if '{invalid}' == 'init_score' else None
        bad = lgb.Dataset(x,label=bad_labels,init_score=bad_score,reference=original,params=params)
        try:
            model.update(train_set=bad)
        except lgb.basic.LightGBMError as error:
            message = 'Label must be in' if '{invalid}' == 'labels' else 'Number of class for initial score error'
            assert message in str(error), str(error)
        else:
            raise AssertionError('Invalid multiclass labels were accepted')
        assert model.train_set is original
        np.testing.assert_array_equal(model.predict(x),prediction)
        model.update()
        assert model.num_trees() == 9
        """,
    )
    _assert_subprocess_passed(result)


def _check_metadata_update(
    staged_lightgbm: tuple[Path, Path], device: str, quantized: bool, change: str, boosting: str = "gbdt"
) -> None:
    """Refresh cached objectives, metrics and initial scores on version changes."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np
        rng = np.random.default_rng(1729)
        x = rng.normal(size=(2048,4))
        y = 3*x[:,0] + x[:,1]
        weights = (.5 + np.abs(x[:,2])).astype(np.float32)
        old_weights = weights if '{change}' in ['remove_weights','unit_weights'] else None
        new_weights = weights if '{change}' == 'add_weights' else None
        params = dict(objective='regression',device_type='{device}',use_quantized_grad={quantized},
            gpu_use_dp=True,num_threads=4,num_leaves=7,max_bin=31,seed=1729,verbosity=-1)
        if '{boosting}' == 'rf':
            params.update(boosting='rf',bagging_fraction=.8,bagging_freq=1)
        old_score = .7*x[:,0] if '{change}' == 'clear_score' else None
        original = lgb.Dataset(x,label=y,weight=old_weights,init_score=old_score,params=params)
        reference = lgb.Dataset(x.copy(),label=y.copy(),weight=old_weights,init_score=old_score,params=params)
        actual = lgb.train(params,original,num_boost_round=2,keep_training_booster=True)
        expected = lgb.train(params,reference,num_boost_round=2,keep_training_booster=True)
        new_y = -2*y + .3*x[:,3] if '{change}' == 'labels' else y
        init_score = .7*x[:,0] if '{change}' == 'init_score' else None
        if '{change}' in ['add_weights','remove_weights']:
            original.set_weight(new_weights)
        elif '{change}' == 'unit_weights':
            original.set_weight(np.ones(len(x)))
        elif '{change}' == 'labels':
            original.set_label(new_y)
        else:
            original.set_init_score(init_score)
        replacement = lgb.Dataset(x,label=new_y,weight=new_weights,init_score=init_score,reference=reference,params=params)
        actual.update()
        expected.update(train_set=replacement)
        np.testing.assert_allclose(actual.predict(x),expected.predict(x),rtol=0,atol=1e-10)
        np.testing.assert_allclose(actual._Booster__inner_predict(data_idx=0),expected._Booster__inner_predict(data_idx=0),rtol=0,atol=1e-10)
        np.testing.assert_allclose([item[2] for item in actual.eval_train()], [item[2] for item in expected.eval_train()],rtol=0,atol=1e-10)
        """,
    )
    _assert_subprocess_passed(result)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize(
    "change", ["add_weights", "remove_weights", "unit_weights", "labels", "init_score", "clear_score"]
)
def test_dataset_metadata_update(staged_lightgbm: tuple[Path, Path], device: str, quantized: bool, change: str) -> None:
    """Changing metadata in place matches replacing the Dataset."""
    _check_metadata_update(staged_lightgbm, device, quantized, change)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("change", ["add_weights", "remove_weights"])
def test_random_forest_weight_update(staged_lightgbm: tuple[Path, Path], device: str, change: str) -> None:
    """Random forest score averaging must survive the forced refresh."""
    _check_metadata_update(staged_lightgbm, device, False, change, boosting="rf")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("remove", [False, True])
def test_ranking_weight_metadata_update(staged_lightgbm: tuple[Path, Path], device: str, remove: bool) -> None:
    """Query-level metric weights must follow row-weight changes as well."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np
        rng = np.random.default_rng(1729)
        x = rng.normal(size=(256,4))
        y = np.tile(np.arange(8) % 4,32)
        groups = [8]*32
        weights = np.repeat(np.linspace(.5,3,32),8).astype(np.float32)
        old_weights = weights if {remove} else None
        new_weights = None if {remove} else weights
        params = dict(objective='lambdarank',device_type='{device}',gpu_use_dp=True,
            num_threads=4,num_leaves=7,max_bin=31,seed=1729,verbosity=-1)
        original = lgb.Dataset(x,label=y,group=groups,weight=old_weights,params=params)
        reference = lgb.Dataset(x.copy(),label=y.copy(),group=groups,weight=old_weights,params=params)
        actual = lgb.train(params,original,num_boost_round=2,keep_training_booster=True)
        expected = lgb.train(params,reference,num_boost_round=2,keep_training_booster=True)
        original.set_weight(new_weights)
        actual.update()
        expected.update(train_set=lgb.Dataset(x,label=y,group=groups,weight=new_weights,reference=reference,params=params))
        np.testing.assert_allclose(actual.predict(x),expected.predict(x),rtol=0,atol=1e-6)
        np.testing.assert_allclose([item[2] for item in actual.eval_train()], [item[2] for item in expected.eval_train()],rtol=0,atol=1e-10)
        """,
    )
    _assert_subprocess_passed(result)
