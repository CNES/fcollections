from __future__ import annotations

import typing as tp

import numpy as np
import pytest

from fcollections.geometry._distances import (
    _great_circle_distance_along_axis,
    _spheroid_distances_along_axis,
    distances_along_axis,
)

if tp.TYPE_CHECKING:
    import numpy.typing as np_t


@pytest.fixture(scope="session")
def positions() -> list[tuple[float, float]]:
    # ((2, 21, 7.9199999999994475, 'E'), (48, 51, 23.7600000000009, 'N'))
    paris = 2.3522, 48.8566
    # ((139, 41, 30.11999999999034, 'E'), (35, 41, 22.2000000000088, 'N'))
    tokyo = 139.6917, 35.6895
    # ((70, 40, 9.480000000024802, 'W'), (33, 26, 56.04000000000667, 'S'))
    santiago = -70.6693, -33.4489

    return [paris, tokyo, santiago]


def fractional_to_degree_minute_second(lon, lat):
    # Useful to convert to a format compatible with online calculators
    lat_direction = "N" if lon > 0 else "S"
    lon_direction = "E" if lon > 0 else "W"
    lon, lat = abs(lon), abs(lat)

    def to_dms(x):
        degrees = int(x)
        minutes_fractional = (x - degrees) * 60
        minutes = int(minutes_fractional)
        seconds = (minutes_fractional - minutes) * 60
        return degrees, minutes, seconds

    return (*to_dms(lon), lon_direction), (*to_dms(lat), lat_direction)


@pytest.fixture(scope="session")
def latitudes(positions: list[tuple[float, float]]) -> np_t.NDArray[np.float64]:
    return np.array([p[1] for p in positions])


@pytest.fixture(scope="session")
def longitudes(positions: list[tuple[float, float]]) -> np_t.NDArray[np.float64]:
    return np.array([p[0] for p in positions])


@pytest.fixture(scope="session")
def great_circle_distances() -> np_t.NDArray[np.float64]:
    paris_tokyo_great_circle = (
        9712084.52079216  # Found 9712.07, 9712, 9711.68 on 2 online calculators
    )
    tokyo_santiago_great_circle = 17235088.14500951
    return np.array([paris_tokyo_great_circle, tokyo_santiago_great_circle])


def test_great_circle_distance(
    longitudes: np_t.NDArray[np.float64],
    latitudes: np_t.NDArray[np.float64],
    great_circle_distances: np_t.NDArray[np.float64],
):
    computed = _great_circle_distance_along_axis(longitudes, latitudes)
    assert np.allclose(great_circle_distances, computed)


def test_great_circle_distances_axis(
    longitudes: np_t.NDArray[np.float64],
    latitudes: np_t.NDArray[np.float64],
):
    longitudes_2d = np.broadcast_to(longitudes[:, None], (longitudes.size, 10))
    latitudes_2d = np.broadcast_to(latitudes[:, None], (longitudes.size, 10))

    distances = _great_circle_distance_along_axis(longitudes, latitudes)
    computed = _great_circle_distance_along_axis(longitudes_2d, latitudes_2d)
    assert computed.shape, (longitudes.size - 1, 10)
    assert np.array_equal(computed[:, 0], distances)
    assert np.any(computed != 0)

    computed0 = _great_circle_distance_along_axis(longitudes_2d, latitudes_2d, axis=1)
    assert computed0.shape, (longitudes.size, 9)
    assert np.all(computed0 == 0)

    computedT = _great_circle_distance_along_axis(
        longitudes_2d.T, latitudes_2d.T, axis=1
    )
    assert np.array_equal(computedT.T, computed)


def test_spheroid_distances_along_axis_axis(
    longitudes: np_t.NDArray[np.float64],
    latitudes: np_t.NDArray[np.float64],
):
    longitudes_2d = np.broadcast_to(longitudes[:, None], (longitudes.size, 10))
    latitudes_2d = np.broadcast_to(latitudes[:, None], (longitudes.size, 10))

    distances = _spheroid_distances_along_axis(longitudes, latitudes)
    computed = _spheroid_distances_along_axis(longitudes_2d, latitudes_2d)
    assert computed.shape, (longitudes.size - 1, 10)
    assert np.array_equal(computed[:, 0], distances)
    assert np.any(computed != 0)

    computed0 = _spheroid_distances_along_axis(longitudes_2d, latitudes_2d, axis=1)
    assert computed0.shape, (longitudes.size, 9)
    assert np.all(computed0 == 0)

    computedT = _spheroid_distances_along_axis(longitudes_2d.T, latitudes_2d.T, axis=1)
    assert np.array_equal(computedT.T, computed)


def test_distances_along_axis_no_padding(
    longitudes: np_t.NDArray[np.float64], latitudes: np_t.NDArray[np.float64]
):
    # Build more points
    longitudes = np.concatenate([longitudes, longitudes])
    latitudes = np.concatenate([latitudes, latitudes])

    distances_great_circle = distances_along_axis(
        longitudes, latitudes, return_full=False
    )
    distances_ellipsoid = distances_along_axis(
        longitudes, latitudes, return_full=False, spherical_approximation=False
    )
    assert distances_great_circle.size == 5

    # Less than 1% of difference between the two methods
    diff = (distances_ellipsoid - distances_great_circle) / distances_great_circle * 100
    assert np.all(diff < 1) and np.any(diff > 0)


def test_distances_along_axis_padding(
    longitudes: np_t.NDArray[np.float64], latitudes: np_t.NDArray[np.float64]
):
    # Build more points
    longitudes = np.concatenate([longitudes, longitudes])
    latitudes = np.concatenate([latitudes, latitudes])

    distances = distances_along_axis(longitudes, latitudes, return_full=True)
    assert distances.size == 6
    assert distances[0] == distances[1]
    assert distances[-2] == distances[-1]

    # Padding also introduce a smoothing to recenter the distance computation.
    # Padded and non-padded distances will not be strictly equal
    distances_no_pad = distances_along_axis(longitudes, latitudes, return_full=False)
    assert not np.array_equal(distances_no_pad, distances[1:])
    assert not np.array_equal(distances_no_pad, distances[:-1])
