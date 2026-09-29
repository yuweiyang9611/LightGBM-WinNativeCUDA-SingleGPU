"""Query sampling must follow the current configuration and Dataset metadata."""

from pathlib import Path

import pytest
from test_cuda_runtime import _assert_subprocess_passed, _run_with_staged_package

pytest_plugins = ["test_cuda_runtime"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_query_bagging_thread_reset(staged_lightgbm: tuple[Path, Path], device: str) -> None:
    """Query prefix-sum scratch must accommodate newly enabled worker threads."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np
        rng = np.random.default_rng(1729)
        x = rng.normal(size=(8192, 4))
        y = rng.integers(0, 4, size=len(x))
        params = dict(objective='lambdarank', device_type='{device}',
            bagging_by_query=True, bagging_fraction=.7, bagging_freq=1, gpu_use_dp=True,
            num_threads=1, num_leaves=7, max_bin=31, verbosity=-1, seed=1729)
        actual = lgb.train(params, lgb.Dataset(x, label=y, group=[4]*2048),
            num_boost_round=3, keep_training_booster=True)
        snapshot = actual.model_to_string()
        actual.reset_parameter(dict(num_threads=8, bagging_seed=77))
        actual.update()
        expected = lgb.train(dict(params, num_threads=8, bagging_seed=77),
            lgb.Dataset(x, label=y, group=[4]*2048), num_boost_round=1,
            init_model=lgb.Booster(model_str=snapshot), keep_training_booster=True)
        np.testing.assert_allclose(actual.predict(x, num_threads=8),
            expected.predict(x, num_threads=8), rtol=0, atol=1e-10)
        """,
    )
    _assert_subprocess_passed(result)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("fraction", [0.3, 0.8])
def test_query_bagging_updates_all_scores(staged_lightgbm: tuple[Path, Path], device: str, fraction: float) -> None:
    """Out-of-bag query rows still need the tree's prediction added to their scores."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np
        rng = np.random.default_rng(1729)
        groups = np.tile([4, 8, 12], 128)
        x = rng.normal(size=(sum(groups), 4))
        y = rng.integers(0, 4, size=len(x))
        model = lgb.train(dict(objective='lambdarank', device_type='{device}',
            bagging_by_query=True, bagging_fraction={fraction}, bagging_freq=1,
            gpu_use_dp=True, num_threads=4, num_leaves=7, max_bin=31, verbosity=-1, seed=1729),
            lgb.Dataset(x, label=y, group=groups), num_boost_round=3, keep_training_booster=True)
        np.testing.assert_allclose(model._Booster__inner_predict(data_idx=0),
            model.predict(x, num_threads=4), rtol=0, atol=1e-10)
        """,
    )
    _assert_subprocess_passed(result)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(("queries", "rows_per_query"), [(96, 8), (192, 12)])
def test_query_bagging_dataset_replacement(
    staged_lightgbm: tuple[Path, Path], device: str, queries: int, rows_per_query: int
) -> None:
    """Replacement query counts and boundaries match a fresh learner on aligned bins."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np
        rng = np.random.default_rng(1729)
        x = rng.normal(size=(1024, 4))
        y = rng.integers(0, 4, size=len(x))
        params = dict(objective='lambdarank', device_type='{device}',
            bagging_by_query=True, bagging_fraction=.6, bagging_freq=1,
            gpu_use_dp=True, num_threads=4, num_leaves=7, max_bin=31, verbosity=-1, seed=1729)
        reference = lgb.Dataset(x, label=y, group=[8]*128)
        actual = lgb.train(params, reference, num_boost_round=3, keep_training_booster=True)
        snapshot = actual.model_to_string()
        x = rng.normal(size=({queries * rows_per_query}, 4))
        y = rng.integers(0, 4, size=len(x))
        actual.update(train_set=lgb.Dataset(x, label=y, group=[{rows_per_query}]*{queries}, reference=reference))
        expected = lgb.train(params,
            lgb.Dataset(x, label=y, group=[{rows_per_query}]*{queries}, reference=reference),
            num_boost_round=1, init_model=lgb.Booster(model_str=snapshot), keep_training_booster=True)
        np.testing.assert_allclose(actual.predict(x, num_threads=4),
            expected.predict(x, num_threads=4), rtol=0, atol=1e-10)
        np.testing.assert_allclose(actual._Booster__inner_predict(data_idx=0),
            actual.predict(x, num_threads=4), rtol=0, atol=1e-10)
        """,
    )
    _assert_subprocess_passed(result)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("case", ["initial_full", "disable", "enable_query", "disable_query"])
def test_query_sampling_configuration(staged_lightgbm: tuple[Path, Path], device: str, case: str) -> None:
    """Query-mode toggles and full-data training agree with fresh continuation."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np
        rng = np.random.default_rng(1729)
        x = rng.normal(size=(1024, 4))
        y = rng.integers(0, 4, size=len(x))
        groups = [8]*(len(x)//8)
        params = dict(objective='lambdarank', device_type='{device}', bagging_freq=1,
            bagging_fraction={1.0 if case == "initial_full" else 0.5},
            bagging_by_query={case != "enable_query"}, gpu_use_dp=True,
            num_threads=4, num_leaves=7, max_bin=31, verbosity=-1, seed=1729)
        actual = lgb.train(params, lgb.Dataset(x, label=y, group=groups),
            num_boost_round=3, keep_training_booster=True)
        np.testing.assert_allclose(actual._Booster__inner_predict(data_idx=0),
            actual.predict(x, num_threads=4), rtol=0, atol=1e-10, err_msg='before reset')
        if '{case}' == 'initial_full':
            expected = lgb.train(dict(params, bagging_by_query=False),
                lgb.Dataset(x, label=y, group=groups), num_boost_round=3, keep_training_booster=True)
        else:
            snapshot = actual.model_to_string()
            changes = dict(disable=dict(bagging_fraction=1.),
                enable_query=dict(bagging_by_query=True),
                disable_query=dict(bagging_by_query=False))['{case}']
            actual.reset_parameter(changes)
            actual.update()
            expected_params = dict(params, **changes)
            if '{case}' == 'disable':
                expected_params['bagging_by_query'] = False
            expected = lgb.train(expected_params, lgb.Dataset(x, label=y, group=groups),
                num_boost_round=1, init_model=lgb.Booster(model_str=snapshot), keep_training_booster=True)
        np.testing.assert_allclose(actual.predict(x, num_threads=4),
            expected.predict(x, num_threads=4), rtol=0, atol=1e-10)
        np.testing.assert_allclose(actual._Booster__inner_predict(data_idx=0),
            actual.predict(x, num_threads=4), rtol=0, atol=1e-10)
        """,
    )
    _assert_subprocess_passed(result)
