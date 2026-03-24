from __future__ import annotations

import typing as tp

import numpy as np
from pyinterp.geometry.geographic import Spheroid

from fcollections.utilities.reshape import slice_along_axis

if tp.TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as np_t


def track_orientation(
    latitude: np.ndarray,
    longitude: np.ndarray,
    along_track_axis: int = 0,
    half_width: int = 1,
    spheroid: Spheroid = Spheroid(),
):
    """Determine angle of satellite track with respect the meridian passing the
    track.

    This method relies on the approximation of the track direction using neighbour points.
    The better the localisation, the better precision for the angle. When latitudes and longitudes
    are not very robust, it is possible to increase the half-width to smoothen the speed direction.

    SWOT remark: there will be a field computed by the ground segment in the L2 products (although it
    is computed for nadir only)

    Parameters
    ----------
    latitude
        Latitudes of the nadir track in degrees
    longitude
        Longitudes of the nadir track in degree
    along_track_axis
        Axis for the along track direction
    half_width
        Half-width of the finite difference calculation.  Set higher
        to smooth the signal if lats and lons are not smooth.
    spheroid
        Earth representation (defaults to WGS84)

    Returns
    -------
    angles_zonal_along: np.ndarray
        Angle between the equator and the along track direction (in radians)
        Positive angles follow the anti-clockwise direction
    """
    longitudes = np.radians(longitude)
    latitudes = np.radians(latitude)
    earth_radius = spheroid.mean_radius()

    delta_lon = slice_along_axis(
        longitudes, along_track_axis, slice(half_width, None)
    ) - slice_along_axis(longitudes, along_track_axis, slice(0, -half_width))

    # Normalizing the delta_lon between [-pi, pi] will ensure we take the shortest
    # of the two paths available for the distance computation
    # For retrograde orbit (lon goes from 0.5° to 359.5° -> delta_lon = +359° -> -1°)
    delta_lon[delta_lon > np.pi] = delta_lon[delta_lon > np.pi] - 2 * np.pi
    # For prograde orbit (lon goes from 359.5° to 0.5° -> delta_lon = -359° -> +1°)
    delta_lon[delta_lon < -np.pi] = delta_lon[delta_lon < -np.pi] + 2 * np.pi

    slice_after = slice_along_axis(latitudes, along_track_axis, slice(half_width, None))
    slice_before = slice_along_axis(latitudes, along_track_axis, slice(0, -half_width))
    delta_lat = slice_after - slice_before
    dy = earth_radius * delta_lat

    dx_before = earth_radius * delta_lon * np.cos(slice_after)
    dx_after = earth_radius * delta_lon * np.cos(slice_before)

    # return padded dx and dy
    padding = [(0, 0) for ii in range(latitudes.ndim)]
    padding_before = padding.copy()
    padding_before[along_track_axis] = (half_width, 0)
    padding_after = padding.copy()
    padding_after[along_track_axis] = (0, half_width)

    dx = np.pad(dx_before, pad_width=padding_before) + np.pad(
        dx_after, pad_width=padding_after
    )
    dy = np.pad(dy, pad_width=padding_before) + np.pad(dy, pad_width=padding_after)

    # This gives the angle relative to the equator. Arctan2 is needed to keep
    # the direction info (direction = sens in french)
    return np.arctan2(dy, dx)


def rotate_vector(
    v_I: float | np_t.NDArray[np.float64],
    v_J: float | np_t.NDArray[np.float64],
    angles_I_i: float | np_t.NDArray[np.float64],
) -> tuple[float | np_t.NDArray[np.float64], float | np_t.NDArray[np.float64]]:
    """Project a vector from (I, J) to (i, j) coordinates.

    The two frames must be direct.

    v_I
        Vector component over the I direction
    v_J
        Vector component over the J direction
    angles_I_i
        Angles between (I, i) (radians)

    Returns
    -------
    v_i
        Vector component over the i direction
    v_j
        Vector component over the j direction
    """
    # Apply the inverse rotation matrix to get the coordinates in the new frame
    v_i = v_I * np.cos(angles_I_i) + v_J * np.sin(angles_I_i)
    v_j = -v_I * np.sin(angles_I_i) + v_J * np.cos(angles_I_i)
    return v_i, v_j


def rotate_derivatives(
    dvX_dX: float | np_t.NDArray[np.float64],
    dvY_dY: float | np_t.NDArray[np.float64],
    dvX_dY: float | np_t.NDArray[np.float64],
    dvY_dX: float | np_t.NDArray[np.float64],
    angles_I_i: float | np_t.NDArray[np.float64],
) -> tuple[
    float | np_t.NDArray[np.float64],
    float | np_t.NDArray[np.float64],
    float | np_t.NDArray[np.float64],
    float | np_t.NDArray[np.float64],
]:
    """Given a vector v rotate its derivatives from the (I, J) from to the (i,
    j) frame.

    Let's note R the rotation between rx=(x, y) and rX=(X, Y): rx=R.rX
    drx = R.drX so drX/drx = R-1

    Let's note Ji the derivatives of vi in the (i, j) frame, and JI the
    derivatives of vI in the (I, J) frame. We want to return Ji, but the only
    available input is JI, the derivation of the vector vI in the (I, J) frame.
    Ji = dvi/dri = d(R.vI)/dri = R.dvI/dri = R.dvI/drI.drI/dr = R.JI.R-1

    Beware, the input vector vI should be expressed in the (I, J) frame, not
    (i, j)

    Parameters
    ----------
    dvX_dX
        X component of the derivative of VI along X direction
    dvY_dY
        Y component of the derivative of VI along Y direction
    dvX_dY
        X component of the derivative of VI along Y direction
    dvY_dX
        Y component of the derivative of VI along X direction
    angles_I_i
        The rotation angle of the R matrix: angle between the source frame
        (I, J) and the destination frame (I, J)

    Returns
    -------
    :
        The rotated derivatives dvx_dx, dvy_dy, dvy_dx, dvx_dy

    See Also
    --------
    rotate_vector: can rotate a vector to express it in the proper frame
    """
    cos2, sin2, cossin = (
        np.cos(angles_I_i) ** 2,
        np.sin(angles_I_i) ** 2,
        np.cos(angles_I_i) * np.sin(angles_I_i),
    )
    dvx_dx = dvX_dX * cos2 + dvY_dY * sin2 - dvX_dY * cossin - dvY_dX * cossin
    dvy_dy = dvX_dX * sin2 + dvY_dY * cos2 + dvX_dY * cossin + dvY_dX * cossin
    dvy_dx = dvX_dX * cossin - dvY_dY * cossin - dvX_dY * sin2 + dvY_dX * cos2
    dvx_dy = dvX_dX * cossin - dvY_dY * cossin + dvX_dY * cos2 - dvY_dX * sin2

    return dvx_dx, dvy_dy, dvx_dy, dvy_dx
