import numpy as np
import pyinterp as pyi
import pyinterp.geohash as pyi_geoh
import pyinterp.geometry.geographic as pyi_geom


def expand_box(box: pyi_geom.Box, precision: int) -> pyi_geom.Box:
    """Expand a geohash box with a given precision.

    The method looks for the geohashes of the input box's corners with a given
    precision. From these two geohashes, it computes the associated bounding
    boxes and find the bigger box that covers them both.

    Parameters
    ----------
    box: geographic.Box
        the box to expand
    precision: int
        the precision to expand the box to

    Returns
    -------
        a geographic.Box expanded to a lower precision
    """
    geohashes = pyi_geoh.encode(
        np.array([box.min_corner.lon, box.max_corner.lon]),
        np.array([box.min_corner.lat, box.max_corner.lat]),
        precision=precision,
    )

    min_lon, min_lat, max_lon, max_lat = (None,) * 4
    for g in geohashes:
        box = pyi_geoh.GeoHash.from_string(g.decode()).bounding_box()
        min_lon = (
            min(box.min_corner.lon, min_lon)
            if min_lon is not None
            else box.min_corner.lon
        )
        max_lon = (
            max(box.max_corner.lon, max_lon)
            if max_lon is not None
            else box.max_corner.lon
        )
        min_lat = (
            min(box.min_corner.lat, min_lat)
            if min_lat is not None
            else box.min_corner.lat
        )
        max_lat = (
            max(box.max_corner.lat, max_lat)
            if max_lat is not None
            else box.max_corner.lat
        )
    return pyi_geom.Box(min_corner=(min_lon, min_lat), max_corner=(max_lon, max_lat))
