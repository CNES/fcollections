"""Optional functionalities.

Optional functionalities include the geographical selection in the
FilesDatabase and IFileReader implementations. Because the dependencies
can be heavy, these functionalities are only enabled if we can import
shapely, pyinterp and geopandas.

In case this module import fails, the caller should fall back to the
default implementations.
"""

from __future__ import annotations

from ._area_selectors import (
    AreaSelector1D,
    AreaSelector2D,
    SwathAreaSelector,
    TemporalSerieAreaSelector,
)
from ._predicates import SwotGeometryFilterBuilder
from ._reader import (
    GeoOpenMfDataset,
    GeoSwotReaderL2LRSSH,
    GeoSwotReaderL3LRSSH,
    GeoSwotReaderL3WW,
)

__all__ = [
    "GeoOpenMfDataset",
    "GeoSwotReaderL3WW",
    "GeoSwotReaderL2LRSSH",
    "GeoSwotReaderL3LRSSH",
    "SwotGeometryFilterBuilder",
    "AreaSelector1D",
    "AreaSelector2D",
    "SwathAreaSelector",
    "TemporalSerieAreaSelector",
]
