"""Verify that an installed LightGBM wheel can train through its CUDA backend."""

from __future__ import annotations

import numpy as np

import lightgbm as lgb

rng = np.random.default_rng(20260816)
features = rng.normal(size=(4096, 24)).astype(np.float32)
labels = (features[:, 0] * 1.7 - features[:, 1] * 0.8 + features[:, 2] * 0.3).astype(np.float32)
train_set = lgb.Dataset(features, label=labels)
model = lgb.train(
    {
        "objective": "regression",
        "device_type": "cuda",
        "gpu_device_id": 0,
        "num_gpu": 1,
        "num_leaves": 15,
        "min_data_in_leaf": 5,
        "verbosity": 1,
        "seed": 7,
    },
    train_set,
    num_boost_round=8,
)

predictions = model.predict(features[:64])
if predictions.shape != (64,) or not np.isfinite(predictions).all() or model.num_trees() != 8:
    raise RuntimeError("CUDA training returned an invalid model or predictions")

model_mse = float(np.mean((predictions - labels[:64]) ** 2))
constant_baseline_mse = float(np.mean((labels.mean() - labels[:64]) ** 2))
if model_mse >= 0.5 * constant_baseline_mse:
    raise RuntimeError(
        f"CUDA model did not beat the constant baseline: model={model_mse}, baseline={constant_baseline_mse}"
    )

print(f"LightGBM {lgb.__version__}; trees={model.num_trees()}; MSE={model_mse:.6f}")
print("Windows native CUDA LightGBM verification passed")
