from pleip.traverse import nextcell
import numpy as np
import pytest


def test_nextcell_horiverti():
    pos = np.array([0.5, 0.5])

    velocity = np.array([0.0, 1.0])
    x0, x1, y0, y1 = 0.0, 1.0, 0.0, 1.0
    nexti, nextj, dist = nextcell(pos, velocity, x0, x1, y0, y1)
    assert nexti == 0
    assert nextj == 1
    assert np.isclose(dist, 0.5)

    velocity = np.array([1.0, 0.0])
    x0, x1, y0, y1 = 0.0, 1.0, 0.0, 1.0
    nexti, nextj, dist = nextcell(pos, velocity, x0, x1, y0, y1)
    assert nexti == 1
    assert nextj == 0
    assert np.isclose(dist, 0.5)

    velocity = np.array([0.0, -1.0])
    x0, x1, y0, y1 = 0.0, 1.0, 0.0, 1.0
    nexti, nextj, dist = nextcell(pos, velocity, x0, x1, y0, y1)
    assert nexti == 0
    assert nextj == -1
    assert np.isclose(dist, 0.5)

    velocity = np.array([-1.0, 0.0])
    x0, x1, y0, y1 = 0.0, 1.0, 0.0, 1.0
    nexti, nextj, dist = nextcell(pos, velocity, x0, x1, y0, y1)
    assert nexti == -1
    assert nextj == 0
    assert np.isclose(dist, 0.5)


def test_nextcell_diagonal():
    pos = np.array([0.5, 0.5])
    sq2 = np.sqrt(2)
    velocity = np.array([1.0, 1.0]) / sq2
    x0, x1, y0, y1 = 0.0, 1.0, 0.0, 1.0
    nexti, nextj, dist = nextcell(pos, velocity, x0, x1, y0, y1)
    assert nexti == 1
    assert nextj == 1
    assert np.isclose(dist, 1 / sq2)

    velocity = np.array([1.0, -1.0]) / sq2
    x0, x1, y0, y1 = 0.0, 1.0, 0.0, 1.0
    nexti, nextj, dist = nextcell(pos, velocity, x0, x1, y0, y1)
    assert nexti == 1
    assert nextj == -1
    assert np.isclose(dist, 1 / sq2)

    velocity = np.array([-1.0, 1.0]) / sq2
    x0, x1, y0, y1 = 0.0, 1.0, 0.0, 1.0
    nexti, nextj, dist = nextcell(pos, velocity, x0, x1, y0, y1)
    assert nexti == -1
    assert nextj == 1
    assert np.isclose(dist, 1 / sq2)

    velocity = np.array([-1.0, -1.0]) / sq2
    x0, x1, y0, y1 = 0.0, 1.0, 0.0, 1.0
    nexti, nextj, dist = nextcell(pos, velocity, x0, x1, y0, y1)
    assert nexti == -1
    assert nextj == -1
    assert np.isclose(dist, 1 / sq2)


def test_nextcell_rais():
    pos = np.array([2.5, 0.5])
    sq2 = np.sqrt(2)
    velocity = np.array([1.0, 1.0]) / sq2
    x0, x1, y0, y1 = 0.0, 1.0, 0.0, 1.0
    with pytest.raises(ValueError):
        nexti, nextj, dist = nextcell(pos, velocity, x0, x1, y0, y1)
    pos = np.array([0.5, 0.5])
    velocity = np.array([0.0, 0.0]) / sq2
    x0, x1, y0, y1 = 0.0, 1.0, 0.0, 1.0
    with pytest.raises(ValueError):
        nexti, nextj, dist = nextcell(pos, velocity, x0, x1, y0, y1)
