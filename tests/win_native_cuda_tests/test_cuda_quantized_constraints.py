"""Quantized CUDA split search must use the current per-node feature masks."""

from pathlib import Path

import pytest
from test_cuda_runtime import _assert_subprocess_passed, _run_with_staged_package

pytest_plugins = ["test_cuda_runtime"]


@pytest.mark.parametrize("reset", [False, True])
def test_quantized_training_excludes_disallowed_features(staged_lightgbm: tuple[Path, Path], reset: bool) -> None:
    """A strong feature excluded by constraints must never be selected."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        f"""
        import lightgbm as lgb
        import numpy as np

        rng = np.random.default_rng(1729)
        x = rng.normal(size=(2048, 4))
        y = 10*x[:, 0] + x[:, 1] - 2*x[:, 2] + x[:, 3]
        params = dict(objective='regression',device_type='cuda',use_quantized_grad=True,
            gpu_use_dp=True,num_threads=4,num_leaves=4,verbosity=-1,max_bin=31)
        constraints = [[1], [2], [3]]
        if not {reset}:
            params['interaction_constraints'] = constraints
        model = lgb.Booster(params, lgb.Dataset(x, label=y, params=params))
        if {reset}:
            model.reset_parameter(dict(interaction_constraints=constraints))
        for _ in range(3):
            model.update()
        def check(node, features):
            if 'split_index' not in node:
                return
            features = features | {{node['split_feature']}}
            assert 0 not in features, features
            assert len(features) == 1, features
            check(node['left_child'], features)
            check(node['right_child'], features)
        trees = model.dump_model()['tree_info']
        assert all(t['num_leaves'] > 1 for t in trees)
        for tree in trees:
            check(tree['tree_structure'], set())
        """,
    )
    _assert_subprocess_passed(result)


def test_quantized_training_keeps_child_features_in_same_group(staged_lightgbm: tuple[Path, Path]) -> None:
    """The right and left child masks must retain the root's feature group."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        """
        import lightgbm as lgb
        import numpy as np

        rng = np.random.default_rng(1729)
        x = rng.normal(size=(2048, 2))
        x[:, 0] = np.arange(len(x)) % 2
        y = 10*x[:, 0] + x[:, 1]
        model = lgb.train(dict(objective='regression',device_type='cuda',use_quantized_grad=True,
            gpu_use_dp=True,num_threads=4,num_leaves=8,verbosity=-1,max_bin=31,
            interaction_constraints=[[0], [1]]), lgb.Dataset(x,label=y), num_boost_round=1)
        tree = model.dump_model()['tree_info'][0]
        assert tree['tree_structure']['split_feature'] == 0
        assert tree['num_leaves'] == 2, tree
        """,
    )
    _assert_subprocess_passed(result)


def test_quantized_training_applies_feature_fraction_bynode(staged_lightgbm: tuple[Path, Path]) -> None:
    """Single-feature sampling must change the dominant root split for this seed."""
    stage, dll = staged_lightgbm
    result = _run_with_staged_package(
        stage,
        dll,
        """
        import lightgbm as lgb
        import numpy as np

        rng = np.random.default_rng(1729)
        x = rng.normal(size=(2048, 4))
        y = 10*x[:, 0] + x[:, 1] - 2*x[:, 2] + x[:, 3]
        roots = []
        for fraction in [1.0, 0.25]:
            model = lgb.train(dict(objective='regression',device_type='cuda',use_quantized_grad=True,
                gpu_use_dp=True,num_threads=4,num_leaves=2,verbosity=-1,max_bin=31,
                feature_fraction_seed=3,feature_fraction_bynode=fraction),
                lgb.Dataset(x,label=y), num_boost_round=1)
            roots.append(model.dump_model()['tree_info'][0]['tree_structure']['split_feature'])
        assert roots[0] == 0
        assert roots[1] != roots[0], roots
        """,
    )
    _assert_subprocess_passed(result)
