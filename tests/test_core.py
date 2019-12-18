from pleip.core import distance_periodic, distance_pointline, customnorm
import numpy as np
import pytest

np.random.seed(1)


def test_distance_periodic():
    # same point has distance zero
    point0 = np.zeros(2)
    point1 = np.zeros(2)
    dist = distance_periodic(point0, point1)
    assert np.isclose(dist, 0.0)

    # see if periodicity works
    point0 = (1 - 1e-10) * np.ones(2)
    point1 = np.zeros(2)
    dist = distance_periodic(point0, point1)
    assert dist < 1e-8

    # see again
    point0 = (1 - 1e-10) * np.ones(2)
    point0[0] = 0
    point1 = np.zeros(2)
    dist = distance_periodic(point0, point1)
    assert dist < 1e-8

    # see again
    point0 = (1 - 1e-10) * np.ones(2)
    point0[1] = 0
    point1 = np.zeros(2)
    dist = distance_periodic(point0, point1)
    assert dist < 1e-8

    # see if everything commutes
    point0 = np.random.rand(2)
    point1 = np.random.rand(2)
    dist01 = distance_periodic(point0, point1)
    dist10 = distance_periodic(point1, point0)
    assert np.isclose(dist01, dist10)

    # see for points outside
    point0 = np.random.rand(2) + 5 * np.ones(2)
    point1 = np.random.rand(2) + 5 * np.ones(2)
    dist01 = distance_periodic(point0, point1, 5.0, 10.0, 5.0, 10.0)
    dist10 = distance_periodic(point1, point0, 5.0, 10.0, 5.0, 10.0)
    assert np.isclose(dist01, dist10)

    # see if error raised when x0>x1
    with pytest.raises(ValueError):
        point0 = 0.5 * np.ones(2)
        point1 = np.zeros(2)
        dist = distance_periodic(point0, point1, 1.0, 0.0, 1.0, 0.0)


def test_distance_pointline():
    # same point twice should be close
    point0 = np.zeros(2)
    point1 = np.zeros(2)
    omega = np.ones(2)
    d, s = distance_pointline(point0, omega, point1)
    assert np.isclose(d, 0.0)
    assert np.isclose(s, 0.0)

    # point along the line
    point0 = np.array([0.5, 0.5])
    point1 = np.array([0.5, 0.0])
    omega = np.array([0.0, 1.0])
    d, s = distance_pointline(point0, omega, point1)
    assert np.isclose(d, 0.0)
    assert np.isclose(s, 0.5)

    # close to the edge should be close with the periodic norm
    point0 = np.array([0.0, 0.0])
    point1 = np.array([0.9999999, 0.5])
    omega = np.array([0.0, 1.0])
    d, s = distance_pointline(point0, omega, point1)
    assert (d - 0.0) ** 2 < 1e-8
    assert (s - 0.5) ** 2 < 1e-8

    # test once with omega and once with 2*omega, result should be the same
    point0 = np.random.rand(2)
    point1 = np.random.rand(2)
    omega = np.array([0.5, 0.5])
    d1, s1 = distance_pointline(point0, omega, point1)
    d2, s2 = distance_pointline(point0, 2 * omega, point1)
    print(d1, s1, d2, s2)
    assert np.isclose(d1, d2)
    assert np.isclose(s1, s2)  # since we ignore normalization


def test_norm():
    # norm(a-b) should be distance(a,b)
    point0 = np.random.rand(2)
    point1 = np.random.rand(2)
    A = distance_periodic(point0, point1)
    B = customnorm(point0 - point1, distance_periodic)
    assert np.isclose(A, B)
