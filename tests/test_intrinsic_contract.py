from __future__ import annotations

import numpy as np
import pytest

from principal_manifold.geometry import _initialize_intrinsic_coordinates_on_first_k_pcs


def test_invalid_k_gt_d_rejected() -> None:
    X = np.random.default_rng(0).normal(size=(12, 3))
    with pytest.raises(ValueError, match=r"k must be <= n_features"):
        _initialize_intrinsic_coordinates_on_first_k_pcs(X, 4)
