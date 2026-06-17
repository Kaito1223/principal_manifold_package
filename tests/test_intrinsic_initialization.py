from __future__ import annotations

import numpy as np
import pytest

from principal_manifold.geometry import _initialize_intrinsic_coordinates_on_first_k_pcs


def test_intrinsic_initialization_shape_and_finite() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 4))

    intrinsic = _initialize_intrinsic_coordinates_on_first_k_pcs(X, k=3)

    assert intrinsic.shape == (20, 3)
    assert np.all(np.isfinite(intrinsic))


def test_intrinsic_initialization_is_deterministic() -> None:
    rng = np.random.default_rng(123)
    X = rng.normal(size=(20, 4))

    intrinsic_first = _initialize_intrinsic_coordinates_on_first_k_pcs(X, k=3)
    intrinsic_second = _initialize_intrinsic_coordinates_on_first_k_pcs(X, k=3)

    np.testing.assert_allclose(intrinsic_first, intrinsic_second, rtol=0.0, atol=0.0)


def test_invalid_k_rejected() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(20, 4))

    with pytest.raises(ValueError):
        _initialize_intrinsic_coordinates_on_first_k_pcs(X, k=5)
