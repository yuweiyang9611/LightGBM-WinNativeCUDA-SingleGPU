"""Categorical split search must honor grouping and randomized thresholds."""

from pathlib import Path

import pytest
from test_cuda_runtime import _assert_subprocess_passed, _run_with_staged_package

pytest_plugins = ["test_cuda_runtime"]


@pytest.mark.parametrize("categories", [64, 512])
def test_categorical_minimum_group_size(staged_lightgbm: tuple[Path, Path], categories: int) -> None:
    """An oversized minimum categorical group prevents every possible split."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np
        x = np.tile(np.arange({categories}), 16).reshape(-1, 1).astype(float)
        y = (x[:, 0] == {40 if categories == 64 else 400}).astype(float)
        for device in ['cpu', 'cuda']:
            data = lgb.Dataset(x, label=y, categorical_feature=[0])
            model = lgb.train(dict(objective='regression', device_type=device, num_threads=4,
                gpu_use_dp=True, min_data_in_leaf=1, min_data_per_group=len(x)+1,
                cat_smooth=0, cat_l2=0, max_cat_to_onehot=4, num_leaves=2,
                learning_rate=1, verbosity=-1), data, num_boost_round=1)
            tree = model.dump_model()['tree_info'][0]
            assert tree['num_leaves'] == 1, (device, tree)
        """,
    )
    _assert_subprocess_passed(result)


@pytest.mark.parametrize("categories", [64, 512, 1024])
@pytest.mark.parametrize("min_leaf", [1, 70])
def test_categorical_group_boundaries(staged_lightgbm: tuple[Path, Path], categories: int, min_leaf: int) -> None:
    """Check the best permissible split against an independent squared-error calculation."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np
        categories = {categories}
        per_category = 16
        x = np.tile(np.arange(categories), per_category).reshape(-1, 1).astype(float)
        targets = (np.arange(categories) / categories)**1.3
        y = targets[x[:, 0].astype(int)]
        group_step = 3  # At least 33 samples require three 16-sample categories.
        first_rank = max(group_step, ({min_leaf}+per_category-1)//per_category)
        for device in ['cpu', 'cuda']:
            data = lgb.Dataset(x, label=y, categorical_feature=[0])
            model = lgb.train(dict(objective='regression', device_type=device, num_threads=4,
                gpu_use_dp=True, min_data_in_leaf={min_leaf}, min_data_per_group=33,
                max_bin=2047, min_data_in_bin=1, cat_smooth=0, cat_l2=0,
                max_cat_to_onehot=4, max_cat_threshold=categories//2, num_leaves=2,
                learning_rate=1, verbosity=-1), data, num_boost_round=1)
            # Binning may merge rare categories into the default bin. Enumerate
            # the represented categories, keeping the default bin on the right.
            values = model.dump_model()['feature_infos']['Column_0']['values']
            represented = np.array(sorted(v for v in values if v >= 0))
            losses = []
            for rank in range(first_rank, (len(represented)+1)//2+1, group_step):
                for subset in [represented[:rank], represented[-rank:]]:
                    left = np.isin(np.arange(categories), subset)
                    prediction = np.where(left, targets[left].mean(), targets[~left].mean())
                    losses.append(np.mean((prediction-targets)**2))
            actual_loss = np.mean((model.predict(x, num_threads=4)-y)**2)
            np.testing.assert_allclose(actual_loss, min(losses), rtol=1e-9, atol=1e-12,
                err_msg=device)
        """,
    )
    _assert_subprocess_passed(result)


@pytest.mark.parametrize("categories", [64, 512])
def test_categorical_extra_trees_randomizes_threshold(staged_lightgbm: tuple[Path, Path], categories: int) -> None:
    """Random threshold ranks must not always select the unique optimal category."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np
        x = np.tile(np.arange({categories}), 16).reshape(-1, 1).astype(float)
        y = (x[:, 0] == {40 if categories == 64 else 400}).astype(float)
        params = dict(objective='regression', device_type='cuda', num_threads=4,
            gpu_use_dp=True, min_data_in_leaf=1, min_data_per_group=1,
            cat_smooth=0, cat_l2=0, max_cat_to_onehot=4, max_cat_threshold=32,
            num_leaves=2, learning_rate=1, verbosity=-1)
        data = lgb.Dataset(x, label=y, categorical_feature=[0])
        optimal = lgb.train(params, data, num_boost_round=1)
        assert np.mean((optimal.predict(x, num_threads=4)-y)**2) < 1e-12
        errors = []
        for seed in range(8):
            model = lgb.train(dict(params, extra_trees=True, extra_seed=seed), data, num_boost_round=1)
            prediction = model.predict(x, num_threads=4)
            repeated = lgb.train(dict(params, extra_trees=True, extra_seed=seed), data, num_boost_round=1)
            np.testing.assert_array_equal(prediction, repeated.predict(x, num_threads=4))
            errors.append(np.mean((prediction-y)**2))
        assert max(errors) > 1e-6, errors
        """,
    )
    _assert_subprocess_passed(result)
