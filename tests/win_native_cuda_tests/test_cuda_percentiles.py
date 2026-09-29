"""CUDA percentiles must agree with independent order-statistic calculations."""

from pathlib import Path

import numpy as np
import pytest
from test_cuda_runtime import _assert_subprocess_passed, _run_with_staged_package

pytest_plugins = ["test_cuda_runtime"]


def _oracle(values: np.ndarray, weights: np.ndarray | None, alpha: float) -> float:
    order = np.argsort(values, kind="stable")
    values = values[order].astype(float)
    if weights is None:
        return float(np.quantile(values, alpha))
    cdf = np.cumsum(weights[order], dtype=float)
    threshold = cdf[-1] * alpha
    pos = min(int(np.searchsorted(cdf, threshold, side="right")), len(values) - 1)
    if pos == 0 or pos == len(values) - 1:
        return float(values[pos])
    span = cdf[pos] - cdf[pos - 1]
    if span < 1:
        return float(values[pos - 1])
    return float(values[pos - 1] + (threshold - cdf[pos - 1]) / span * (values[pos] - values[pos - 1]))


def _weights(rows: int, mode: str) -> np.ndarray | None:
    if mode == "none":
        return None
    result = (1 + np.arange(rows) % 7).astype(np.float32)
    return result * 0.125 if mode == "fractional" else result


def _check_initial_percentile(
    staged_lightgbm: tuple[Path, Path], rows: int, weight_mode: str, alpha: float, extra_trees: bool = False
) -> None:
    """Constant features isolate the initial score from split/leaf renewal."""
    stage, dll = staged_lightgbm
    values = np.random.default_rng(1729).permutation(rows).astype(np.float32)
    expected = float(np.float32(_oracle(values, _weights(rows, weight_mode), alpha)))
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np
        rng = np.random.default_rng(1729)
        y = rng.permutation({rows}).astype(np.float32)
        x = np.zeros((len(y), 1))
        weights = (1 + np.arange(len(y)) % 7).astype(np.float32) if '{weight_mode}' != 'none' else None
        if '{weight_mode}' == 'fractional':
            weights *= .125
        expected = {expected!r}
        for device in ['cpu', 'cuda']:
            model = lgb.train(dict(objective='quantile',alpha={alpha},device_type=device,extra_trees={extra_trees},
                gpu_use_dp=True,num_threads=4,verbosity=-1),lgb.Dataset(x,label=y,weight=weights),num_boost_round=1)
            actual = model.predict(x[:1])[0]
            assert abs(actual-expected) < 2e-4, (device, actual, expected)
        """,
    )
    _assert_subprocess_passed(result)


@pytest.mark.parametrize("rows", [33, 513, 2049])
@pytest.mark.parametrize("weight_mode", ["none", "integer", "fractional"])
@pytest.mark.parametrize("alpha", [0.2, 0.5, 0.8])
def test_cuda_initial_percentile(staged_lightgbm: tuple[Path, Path], rows: int, weight_mode: str, alpha: float) -> None:
    """Constant-feature training returns the independently computed percentile."""
    _check_initial_percentile(staged_lightgbm, rows, weight_mode, alpha)


@pytest.mark.parametrize("rows", [33, 513, 2049])
@pytest.mark.parametrize("weight_mode", ["none", "integer", "fractional"])
@pytest.mark.parametrize(("objective", "alpha"), [("regression_l1", 0.5), ("quantile", 0.2), ("quantile", 0.8)])
def test_cuda_leaf_percentile(
    staged_lightgbm: tuple[Path, Path], rows: int, weight_mode: str, objective: str, alpha: float
) -> None:
    """A known two-leaf split isolates percentile-based leaf output renewal."""
    rng = np.random.default_rng(1729)
    left = -10 - rng.permutation(rows).astype(np.float32)
    right = 10 + rng.permutation(rows).astype(np.float32)
    weights = _weights(2 * rows, weight_mode)
    expected = [
        _oracle(left, None if weights is None else weights[:rows], alpha),
        _oracle(right, None if weights is None else weights[rows:], alpha),
    ]
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np
        rng = np.random.default_rng(1729)
        left = -10 - rng.permutation({rows}).astype(np.float32)
        right = 10 + rng.permutation({rows}).astype(np.float32)
        y = np.concatenate([left, right])
        x = np.repeat([0., 1.], {rows}).reshape(-1,1)
        weights = (1 + np.arange(len(y)) % 7).astype(np.float32) if '{weight_mode}' != 'none' else None
        if '{weight_mode}' == 'fractional':
            weights *= .125
        for device in ['cpu', 'cuda']:
            model = lgb.train(dict(objective='{objective}',alpha={alpha},device_type=device,
                boost_from_average=False,learning_rate=1,num_leaves=2,min_data_in_leaf=1,
                gpu_use_dp=True,num_threads=4,verbosity=-1),lgb.Dataset(x,label=y,weight=weights),num_boost_round=1)
            assert model.dump_model()['tree_info'][0]['num_leaves'] == 2
            np.testing.assert_allclose(model.predict(np.array([[0.],[1.]])),{expected!r},rtol=0,atol=2e-4)
        """,
    )
    _assert_subprocess_passed(result)


@pytest.mark.parametrize("heavy_first", [False, True])
@pytest.mark.parametrize("alpha", [0.1, 0.9])
def test_cuda_weighted_percentile_endpoints(
    staged_lightgbm: tuple[Path, Path], heavy_first: bool, alpha: float
) -> None:
    """Endpoint selection uses sorted identities and never accesses index -1."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np
        y = np.array([7., 2.])
        weights = np.array([100., 1.]) if {heavy_first} else np.array([1., 100.])
        model = lgb.train(dict(objective='quantile',alpha={alpha},device_type='cuda',
            num_threads=4,verbosity=-1),lgb.Dataset(np.zeros((2,1)),label=y,weight=weights),num_boost_round=1)
        assert model.predict(np.zeros((1,1)))[0] == (7. if {heavy_first} else 2.)
        """,
    )
    _assert_subprocess_passed(result)


def test_cuda_weighted_percentile_spans_many_blocks(staged_lightgbm: tuple[Path, Path]) -> None:
    """Exercise multiple block totals per thread in the second scan stage."""
    test_cuda_initial_percentile(staged_lightgbm, 1049601, "integer", 0.8)


def test_cuda_constant_features_with_extra_trees(staged_lightgbm: tuple[Path, Path]) -> None:
    """An empty split-task list must not launch zero-sized CUDA kernels."""
    _check_initial_percentile(staged_lightgbm, 33, "integer", 0.5, extra_trees=True)


@pytest.mark.parametrize("weighted", [False, True])
def test_cuda_single_row_percentile(staged_lightgbm: tuple[Path, Path], weighted: bool) -> None:
    """The one-value percentile is the value itself, including small weights."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np
        weights = np.array([.25]) if {weighted} else None
        model = lgb.train(dict(objective='quantile',alpha=.8,device_type='cuda',
            num_threads=4,verbosity=-1),lgb.Dataset(np.zeros((1,1)),label=[17.],weight=weights),num_boost_round=1)
        assert model.predict(np.zeros((1,1)))[0] == 17.
        """,
    )
    _assert_subprocess_passed(result)
