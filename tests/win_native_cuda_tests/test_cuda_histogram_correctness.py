"""Check CUDA histogram counts against independently predicted leaf assignments.

Uses the isolated candidate DLL staged by ``test_cuda_runtime.py``.
These finite-data cases target histogram repair/subtraction independently of
missing-value routing. A race need not reproduce deterministically on the old
DLL; the original saved-model count discrepancy remains the historical witness.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from test_cuda_runtime import (
    _assert_subprocess_passed,
    _run_with_staged_package,
)

pytest_plugins = ["test_cuda_runtime"]


@pytest.mark.parametrize("seed", [29, 1729])
@pytest.mark.parametrize("max_bin", [31, 255])
def test_histogram_counts_match_independent_routing(
    staged_lightgbm: tuple[Path, Path], seed: int, max_bin: int
) -> None:
    """Exercise 64 trees in total, retaining normal asynchronous CUDA launches."""
    stage_root, staged_dll = staged_lightgbm
    result = _run_with_staged_package(
        stage_root,
        staged_dll,
        """
        import os
        from pathlib import Path

        import lightgbm as lgb
        import numpy as np
        from lightgbm.libpath import _find_lib_path

        loaded_dll = Path(_find_lib_path()[0]).resolve()
        expected_dll = Path(os.environ["EXPECTED_LIGHTGBM_DLL"]).resolve()
        assert loaded_dll == expected_dll, (loaded_dll, expected_dll)
        seed = int(os.environ["HISTOGRAM_TEST_SEED"])
        max_bin = int(os.environ["HISTOGRAM_TEST_MAX_BIN"])
        rng = np.random.default_rng(seed)
        features = rng.normal(size=(4096, 24))
        features[rng.random(features.shape) < 0.55] = 0.0
        # Exact zeros dominate an interior bin, with occupied bins on both sides.
        assert np.isfinite(features).all()
        assert np.all(np.mean(features == 0.0, axis=0) > 0.50)
        assert np.all(np.any(features < 0.0, axis=0))
        assert np.all(np.any(features > 0.0, axis=0))
        labels = (
            3.0 * (features[:, 0] > 0.0)
            - 2.0 * (features[:, 0] < 0.0)
            + 2.5 * (features[:, 1] == 0.0)
            + 0.8 * features[:, 2]
            - 0.6 * features[:, 3]
            + 0.4 * features[:, 4] * features[:, 5]
            + rng.normal(scale=0.05, size=features.shape[0])
        )
        assert np.isfinite(labels).all()
        rounds = 16
        model = lgb.train(
            {
                "objective": "regression",
                "device_type": "cuda",
                "gpu_device_id": 0,
                "num_gpu": 1,
                "gpu_use_dp": True,
                "max_bin": max_bin,
                "num_leaves": 15,
                "min_data_in_leaf": 20,
                "learning_rate": 0.1,
                "bagging_fraction": 1.0,
                "bagging_freq": 0,
                "feature_fraction": 1.0,
                "feature_fraction_bynode": 1.0,
                "zero_as_missing": False,
                "seed": seed,
                "data_random_seed": seed,
                "num_threads": 4,
                "verbosity": -1,
            },
            # Unweighted squared-error regression has exactly unit Hessians.
            lgb.Dataset(features, label=labels),
            num_boost_round=rounds,
        )
        trees = model.dump_model()["tree_info"]
        assert len(trees) == rounds, (seed, max_bin, len(trees))
        leaf_assignments = model.predict(
            features, pred_leaf=True, num_iteration=rounds, num_threads=4
        )
        assert leaf_assignments.shape == (features.shape[0], rounds)

        for tree_index, tree in enumerate(trees):
            context = (seed, max_bin, tree_index)
            # Two leaves alone would not exercise sibling histogram subtraction.
            assert tree["num_leaves"] >= 3, (context, tree["num_leaves"])
            assignments = leaf_assignments[:, tree_index]
            assert np.all(assignments >= 0), context
            assert np.all(assignments < tree["num_leaves"]), context
            independently_routed = np.bincount(
                assignments.astype(np.int64), minlength=tree["num_leaves"]
            )
            assert independently_routed.sum() == features.shape[0], context
            visited_leaf_indices = set()

            def check_node(node):
                if "leaf_index" in node:
                    leaf_index = node["leaf_index"]
                    assert leaf_index not in visited_leaf_indices, context
                    visited_leaf_indices.add(leaf_index)
                    count = int(independently_routed[leaf_index])
                    assert count > 0, (context, leaf_index, count)
                    count_key, weight_key = "leaf_count", "leaf_weight"
                    node_id = ("leaf", leaf_index)
                else:
                    # Aggregate prediction-derived counts, never stored counts.
                    count = check_node(node["left_child"]) + check_node(node["right_child"])
                    count_key, weight_key = "internal_count", "internal_weight"
                    node_id = ("split", node["split_index"])
                assert node[count_key] == count, (
                    context, node_id, count_key, node[count_key], count
                )
                # With unit Hessians, the independent count is also the weight.
                assert np.isclose(node[weight_key], count, rtol=0.0, atol=1e-8), (
                    context, node_id, weight_key, node[weight_key], count
                )
                return count

            assert check_node(tree["tree_structure"]) == features.shape[0], context
            assert visited_leaf_indices == set(range(tree["num_leaves"])), context
        """,
        extra_env={
            "HISTOGRAM_TEST_SEED": str(seed),
            "HISTOGRAM_TEST_MAX_BIN": str(max_bin),
            "CUDA_LAUNCH_BLOCKING": "0",
        },
    )
    _assert_subprocess_passed(result)
