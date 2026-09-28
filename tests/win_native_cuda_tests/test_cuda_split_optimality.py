"""Large numerical histograms must retain the best threshold across all bins."""

from pathlib import Path

import pytest
from test_cuda_runtime import _assert_subprocess_passed, _run_with_staged_package

pytest_plugins = ["test_cuda_runtime"]


@pytest.mark.parametrize("cut", [100, 500, 900])
@pytest.mark.parametrize("missing", ["none", "nan-left", "nan-right", "zero-left", "zero-right"])
def test_large_histogram_finds_exact_stump(staged_lightgbm: tuple[Path, Path], cut: int, missing: str) -> None:
    """A single known threshold can represent every target exactly."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np

        x = np.tile(np.arange(1, 1025), 4).reshape(-1, 1).astype(float)
        y = (x[:, 0] >= {cut}).astype(float)
        mode = '{missing}'
        if mode != 'none':
            value = np.nan if mode.startswith('nan') else 0.0
            x = np.concatenate([x, np.full((128, 1), value)])
            y = np.concatenate([y, np.full(128, float(mode.endswith('right')))])
        data = lgb.Dataset(x, label=y)
        model = lgb.train(dict(objective='regression', device_type='cuda', num_threads=4,
            gpu_use_dp=True, max_bin=2048, min_data_in_bin=1, min_data_in_leaf=1,
            num_leaves=2, learning_rate=1, verbosity=-1, zero_as_missing=mode.startswith('zero')),
            data, num_boost_round=1, keep_training_booster=True)
        assert data.feature_num_bin(0) > 256
        tree = model.dump_model()['tree_info'][0]['tree_structure']
        assert np.isclose(tree['threshold'], {cut}-.5, rtol=0, atol=1e-10), tree
        predictions = model.predict(x, num_threads=4)
        assert np.mean((predictions-y)**2) < 1e-12, tree
        internal = model._Booster__inner_predict(data_idx=0).copy()
        np.testing.assert_allclose(internal, predictions, rtol=0, atol=1e-10)
        if mode != 'none':
            assert tree['default_left'] == mode.endswith('left'), tree
        """,
    )
    _assert_subprocess_passed(result)
