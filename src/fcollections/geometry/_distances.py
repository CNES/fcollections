"""Adapts the distance computation to multiple data shapes."""

import numpy as np
from pyinterp.geometry.geographic import MultiPoint, Spheroid
from pyinterp.geometry.geographic.algorithms import for_each_point_pairwise_distance

from fcollections.utilities.reshape import slice_along_axis


def distances_along_axis(
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    axis: int = 0,
    return_full: bool = True,
    spherical_approximation: bool = True,
    **kwargs,
):
    """Compute the distances point to point along a given axis.

    In case the spherical approximation is used, the great circle distance is
    computed, with the earth radius being deduced from the spheroid model
    (mean_radius).

    Else, the distance will be computed using the ellipsoid model.

    Parameters
    ----------
    longitudes
        Longitudes in degrees
    latitudes
        Latitudes in degrees
    axis
        Axis along which the distance will be computed
    return_full
        True to return an array of the same shape as the input. If set to True,
        the distances vector is averaged with an offset of one element, then
        padded with edge values to have a the same number of elements as the
        input
    spherical_approximation
        Whether to use a spherical earth or an ellipsoid earth model

    Returns
    -------
    distances_along_axis
        Distance between points along the given axis in meters.
    """
    if spherical_approximation:
        distances_along_axis = _great_circle_distance_along_axis(
            longitudes, latitudes, axis=axis, **kwargs
        )
    else:
        distances_along_axis = _spheroid_distances_along_axis(
            longitudes, latitudes, axis=axis, **kwargs
        )

    # Average the distances before and after one point to smoothen the output.
    # Pad one element at the beginning and end of the axis dimension using the
    # edge value to get a full vector
    if return_full:
        append_shape = list(longitudes.shape)
        append_shape[axis] = 1

        selection_before = [
            slice(None),
        ] * distances_along_axis.ndim
        selection_before[axis] = slice(1, None)
        selection_after = [
            slice(None),
        ] * distances_along_axis.ndim
        selection_after[axis] = slice(0, -1)
        distances_along_axis = (
            distances_along_axis[*selection_before]
            + distances_along_axis[*selection_after]
        ) / 2

        padding = [
            (0, 0),
        ] * distances_along_axis.ndim
        padding[axis] = (1, 1)
        distances_along_axis = np.pad(distances_along_axis, padding, mode="edge")

    return distances_along_axis


def _spheroid_distances_along_axis(
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    axis: int = 0,
    wgs: Spheroid = Spheroid(),
):
    # Slice along axis to compute distance point to point
    lon0 = slice_along_axis(longitudes, axis, slice(0, -1))
    lon1 = slice_along_axis(longitudes, axis, slice(1, None))
    lat0 = slice_along_axis(latitudes, axis, slice(0, -1))
    lat1 = slice_along_axis(latitudes, axis, slice(1, None))

    # Compute distance on ellipsoid
    return for_each_point_pairwise_distance(
        MultiPoint(lon0.ravel(), lat0.ravel()),
        MultiPoint(lon1.ravel(), lat1.ravel()),
        spheroid=wgs,
    ).reshape(lon0.shape)


def _great_circle_distance_along_axis(
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    axis: int = 0,
    wgs: Spheroid = Spheroid(),
):

    longitudes = np.radians(longitudes)
    latitudes = np.radians(latitudes)

    # Mean radius is in meters
    earth_radius = wgs.mean_radius()

    delta_lat = np.abs(np.diff(latitudes, axis=axis))
    tmp = np.abs(np.diff(longitudes, axis=axis))
    delta_lon = np.minimum(tmp, 2 * np.pi - tmp)

    lat0 = slice_along_axis(latitudes, axis=axis, slice_along_axis=slice(0, -1))
    lat1 = slice_along_axis(latitudes, axis=axis, slice_along_axis=slice(1, None))

    # Haversine formulae
    tmp = np.sin(delta_lat / 2.0) ** 2 + np.sin(delta_lon / 2.0) ** 2 * np.cos(
        lat0
    ) * np.cos(lat1)

    return 2.0 * np.arcsin(np.sqrt(tmp)) * earth_radius
