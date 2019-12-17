from pleip.core import distance_periodic, distance_pointline
import numpy as np
import pytest

np.random.seed(1)


def test_distance_periodic():
    point0 = np.zeros(2)
    point1 = np.zeros(2)

    dist = distance_periodic(point0, point1)
    assert np.isclose(dist, 0.0)

    point0 = (1 - 1e-10) * np.ones(2)
    point1 = np.zeros(2)
    dist = distance_periodic(point0, point1)
    assert dist < 1e-8

    point0 = (1 - 1e-10) * np.ones(2)
    point0[0] = 0
    point1 = np.zeros(2)
    dist = distance_periodic(point0, point1)
    assert dist < 1e-8

    point0 = (1 - 1e-10) * np.ones(2)
    point0[1] = 0
    point1 = np.zeros(2)
    dist = distance_periodic(point0, point1)
    assert dist < 1e-8

    point0 = np.random.rand(2)
    point1 = np.random.rand(2)
    dist01 = distance_periodic(point0, point1)
    dist10 = distance_periodic(point1, point0)
    assert np.isclose(dist01, dist10)

    point0 = np.random.rand(2) + 5 * np.ones(2)
    point1 = np.random.rand(2) + 5 * np.ones(2)
    dist01 = distance_periodic(point0, point1, 5.0, 10.0, 5.0, 10.0)
    dist10 = distance_periodic(point1, point0, 5.0, 10.0, 5.0, 10.0)
    assert np.isclose(dist01, dist10)

    with pytest.raises(ValueError):
        point0 = np.ones(2)  # can't be one
        point1 = np.zeros(2)
        dist = distance_periodic(point0, point1)
    with pytest.raises(ValueError):
        point0 = np.ones(2)  # can't be one
        point1 = np.zeros(2)
        dist = distance_periodic(point1, point0)
    with pytest.raises(ValueError):
        point0 = 0.5 * np.ones(2)
        point1 = np.zeros(2)
        dist = distance_periodic(point0, point1, 1.0, 0.0, 1.0, 0.0)


def test_distance_pointline():
    point0 = np.zeros(2)
    point1 = np.zeros(2)
    omega = np.ones(2)
    d, s = distance_pointline(point0, omega, point1)
    assert np.isclose(d, 0.0)
    assert np.isclose(s, 0.0)

    point0 = np.array([0.5, 0.5])
    point1 = np.array([0.5, 0.0])
    omega = np.array([0.0, 1.0])
    d, s = distance_pointline(point0, omega, point1)
    assert np.isclose(d, 0.0)
    assert np.isclose(s, 0.5)
