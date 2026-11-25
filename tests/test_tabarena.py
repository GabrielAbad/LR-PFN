"""Unit tests for TabArena feature mapping and ranking helpers."""

from __future__ import annotations

import numpy as np

from long_range_pfn.tabarena_benchmark import average_ranks
from long_range_pfn.tabarena_data import fit_feature_map, standardize_targets


def test_fit_feature_map_pads_and_reduces():
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(40, 3))
    x_test = rng.normal(size=(10, 3))
    tr, te = fit_feature_map(x_train, x_test, num_features=5)
    assert tr.shape == (40, 5)
    assert te.shape == (10, 5)
    assert np.allclose(tr[:, 3:], 0.0)

    x_train_wide = rng.normal(size=(40, 12))
    x_test_wide = rng.normal(size=(10, 12))
    tr2, te2 = fit_feature_map(x_train_wide, x_test_wide, num_features=5)
    assert tr2.shape == (40, 5)
    assert te2.shape == (10, 5)


def test_standardize_targets_unit_train():
    y_train = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    y_test = np.array([2.5, 3.5], dtype=np.float64)
    yt, ye, mean, std = standardize_targets(y_train, y_test)
    assert abs(float(yt.mean())) < 1e-6
    assert abs(float(yt.std()) - 1.0) < 1e-5
    assert mean == 2.5
    assert std > 0
    assert ye.shape == (2,)


def test_average_ranks_prefers_lower_rmse():
    obs = [
        {"uniform": 1.0, "crumb": 0.5, "longrange": 0.2},
        {"uniform": 0.9, "crumb": 0.4, "longrange": 0.3},
    ]
    ranks = average_ranks(obs, ["uniform", "crumb", "longrange"])
    assert ranks["longrange"] < ranks["crumb"] < ranks["uniform"]
