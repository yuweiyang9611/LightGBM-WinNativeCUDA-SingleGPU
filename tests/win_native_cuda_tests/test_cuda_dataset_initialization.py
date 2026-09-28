"""CUDA Booster creation must prepare its Dataset or reject it safely."""

from pathlib import Path

import pytest
from test_cuda_runtime import _assert_subprocess_passed, _run_with_staged_package

pytest_plugins = ["test_cuda_runtime"]


def test_direct_booster_accepts_constructed_cuda_reference(staged_lightgbm: tuple[Path, Path]) -> None:
    """A Dataset inherits its reference's storage even without explicit params."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        """
        import lightgbm as lgb
        import numpy as np

        x = np.linspace(-1, 1, 512).reshape(-1, 1)
        y = x[:, 0] ** 2 + x[:, 0]
        params = dict(device_type='cuda', objective='regression', num_threads=4, verbosity=-1)
        reference = lgb.Dataset(x, label=y, params=params).construct()
        data = lgb.Dataset(x, label=y, reference=reference).construct()
        assert data.data is None
        model = lgb.Booster(params, data)
        model.update()
        assert model.num_trees() == 1
        assert np.std(model.predict(x)) > 0
        """,
    )
    _assert_subprocess_passed(result)


@pytest.mark.parametrize("preconstruct", [False, True])
@pytest.mark.parametrize("device_key", ["device_type", "device"])
def test_direct_booster_prepares_cuda_dataset(
    staged_lightgbm: tuple[Path, Path], preconstruct: bool, device_key: str
) -> None:
    """Lazy and reconstructable CPU datasets should both support CUDA training."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np

        x = np.linspace(-1, 1, 512).reshape(-1, 1)
        y = x[:, 0] ** 2 + x[:, 0]
        dataset = lgb.Dataset(x, label=y, params=dict(max_bin=63, verbosity=-1), free_raw_data=False)
        if {preconstruct}:
            dataset.construct()
        params = dict(objective='regression', num_threads=4, verbosity=-1,
            gpu_use_dp=True, num_leaves=8, max_bin=31)
        params['{device_key}'] = 'cuda'
        model = lgb.Booster(params=params, train_set=dataset)
        for _ in range(3):
            model.update()
        assert model.num_trees() == 3
        assert dataset.get_params()['max_bin'] == 63
        assert np.mean((model.predict(x)-y)**2) < 0.7 * np.var(y)
        """,
    )
    _assert_subprocess_passed(result)


@pytest.mark.parametrize("operation", ["add_valid", "reset_training"])
def test_cuda_rejects_cpu_dataset_without_changing_booster(staged_lightgbm: tuple[Path, Path], operation: str) -> None:
    """Failed Dataset attachment must leave a live Booster usable."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np

        x = np.linspace(-1, 1, 512).reshape(-1, 1)
        y = x[:, 0] ** 2 + x[:, 0]
        cpu_dataset = lgb.Dataset(x, label=y, params=dict(verbosity=-1)).construct()
        model = lgb.train(dict(objective='regression', device_type='cuda',
            num_threads=4, verbosity=-1, gpu_use_dp=True),
            lgb.Dataset(x, label=y), num_boost_round=2, keep_training_booster=True)
        expected = model.predict(x)
        try:
            if '{operation}' == 'add_valid':
                ret = lgb.basic._LIB.LGBM_BoosterAddValidData(model._handle, cpu_dataset._handle)
            else:
                ret = lgb.basic._LIB.LGBM_BoosterResetTrainingData(model._handle, cpu_dataset._handle)
            lgb.basic._safe_call(ret)
        except lgb.basic.LightGBMError as error:
            assert 'Dataset' in str(error) and 'cuda' in str(error).lower(), str(error)
        else:
            raise AssertionError('Expected a clear Dataset/CUDA initialization error')
        np.testing.assert_array_equal(model.predict(x), expected)
        model.update()
        assert model.num_trees() == 3
        """,
    )
    _assert_subprocess_passed(result)


@pytest.mark.parametrize("native", [False, True])
def test_cuda_rejects_unprepared_dataset_without_access_violation(
    staged_lightgbm: tuple[Path, Path], native: bool
) -> None:
    """A constructed CPU Dataset without raw data must produce a safe error."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import ctypes
        import lightgbm as lgb
        import numpy as np

        x = np.arange(512).reshape(-1, 1)
        dataset = lgb.Dataset(x, label=x[:, 0], params=dict(verbosity=-1)).construct()
        assert dataset.data is None
        try:
            if {native}:
                handle = ctypes.c_void_p()
                lgb.basic._safe_call(lgb.basic._LIB.LGBM_BoosterCreate(dataset._handle,
                    lgb.basic._c_str('device_type=cuda objective=regression verbosity=-1'),
                    ctypes.byref(handle)))
            else:
                lgb.Booster(dict(device_type='cuda', objective='regression', verbosity=-1), dataset)
        except lgb.basic.LightGBMError as error:
            assert 'Dataset' in str(error) and 'cuda' in str(error).lower(), str(error)
        else:
            raise AssertionError('Expected a clear Dataset/CUDA initialization error')
        """,
    )
    _assert_subprocess_passed(result)
