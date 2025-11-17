import math

from real_simple_stats.normal_distributions import (
    area_between_0_and_z,
    area_between_z_scores,
    area_left_of_z,
    area_outside_range,
    area_right_of_z,
    chebyshev_theorem,
    z_score,
    z_score_standard_error,
)


def test_z_score_basic():
    assert z_score(85, 80, 5) == 1.0


def test_z_score_standard_error():
    val = z_score_standard_error(84, 80, 10, 100)
    assert math.isclose(val, (84 - 80) / (10 / math.sqrt(100)))


def test_area_functions_symmetry():
    # Standard normal properties
    assert math.isclose(area_between_0_and_z(0), 0.0, abs_tol=1e-12)
    assert area_between_0_and_z(1.96) > 0.0
    # Left/right complement
    z = 1.23
    left = area_left_of_z(z)
    right = area_right_of_z(z)
    assert math.isclose(left + right, 1.0, rel_tol=1e-12)


def test_area_between_and_outside():
    z1, z2 = -1.96, 1.96
    between = area_between_z_scores(z1, z2)
    outside = area_outside_range(z1, z2)
    assert between > 0.0 and outside > 0.0
    assert math.isclose(between + outside, 1.0, rel_tol=1e-12)


def test_chebyshev_theorem():
    # For k=2, at least 1 - 1/4 = 0.75 inside
    assert math.isclose(chebyshev_theorem(2), 0.75)


def test_chebyshev_invalid_k():
    import pytest

    with pytest.raises(ValueError):
        chebyshev_theorem(1)
