from __future__ import annotations

import re

from fcollections.core import (
    FileNameConvention,
    FileNameFieldEnum,
    FileNameFieldInteger,
    FileNameFieldPeriod,
    FileNameFieldString,
    FilesDatabase,
    Layout,
    PeriodMixin,
    SubsetsUnmixer,
)

from ._definitions._constants import DESCRIPTIONS
from ._definitions._swot import ProductSubset
from ._l3_lr_ssh import AVISO_L3_LR_SSH_LAYOUT_V2
from ._predicates import SwotPhaseFilterBuilder
from ._readers import SwotReaderL3WW

SWOT_L3_LR_WINDWAVE_PATTERN = re.compile(
    r"SWOT_L3_LR_WIND_WAVE_(?P<subset>Extended){0,1}(_){0,1}(?P<cycle_number>\d{3})_(?P<pass_number>\d{3})_"
    r"(?P<time>\d{8}T\d{6}_\d{8}T\d{6})_v(?P<version>.*).nc"
)


class FileNameConventionSwotL3WW(FileNameConvention):
    """Swot L3_LR_WIND_WAVE product file names convention."""

    def __init__(self):
        super().__init__(
            regex=SWOT_L3_LR_WINDWAVE_PATTERN,
            fields=[
                FileNameFieldInteger(
                    "cycle_number", description=DESCRIPTIONS["cycle_number"]
                ),
                FileNameFieldInteger(
                    "pass_number", description=DESCRIPTIONS["pass_number"]
                ),
                FileNameFieldPeriod(
                    "time", "%Y%m%dT%H%M%S", "_", description=DESCRIPTIONS["time"]
                ),
                FileNameFieldEnum(
                    "subset",
                    ProductSubset,
                    default=ProductSubset.Light,
                    description=DESCRIPTIONS["subset"],
                ),
                FileNameFieldString("version", description=DESCRIPTIONS["version"]),
            ],
            generation_string="SWOT_L3_LR_WIND_WAVE_{subset!f}_{cycle_number:>03d}_{pass_number:>03d}_{time!f}_v{version}.nc",
        )


AVISO_L3_LR_WINDWAVE_LAYOUT = Layout(
    [
        *AVISO_L3_LR_SSH_LAYOUT_V2.conventions[:3],
        FileNameConventionSwotL3WW(),
    ]
)


class BasicNetcdfFilesDatabaseSwotLRWW(FilesDatabase, PeriodMixin):
    """Database mapping to explore and read the L3_LR_WIND_WAVE product.

    See Also
    --------
    fcollections.implementations.AVISO_L3_LR_WINDWAVE_LAYOUT
        Recommended layout for the database
    """

    layouts = [Layout([FileNameConventionSwotL3WW()]), AVISO_L3_LR_WINDWAVE_LAYOUT]
    reader = SwotReaderL3WW()
    sort_keys = "time"

    # These keys determines an homogeneous subset. We expect no duplicates in
    # an homogeneous subset
    unmixer = SubsetsUnmixer(
        partition_keys=["version", "subset"], auto_pick_last=("version",)
    )

    # Convert phase filter to cycle_numbers filter
    filter_builders = [SwotPhaseFilterBuilder]


try:
    from fcollections.implementations.optional import (
        GeoSwotReaderL3WW,
        SwotGeometryFilterBuilder,
    )

    class NetcdfFilesDatabaseSwotLRWW(BasicNetcdfFilesDatabaseSwotLRWW):
        reader = GeoSwotReaderL3WW()
        filter_builders = [SwotGeometryFilterBuilder, SwotPhaseFilterBuilder]

except ImportError:
    import logging

    from ._definitions._constants import MISSING_OPTIONAL_DEPENDENCIES_MESSAGE

    logger = logging.getLogger(__name__)
    logger.info(MISSING_OPTIONAL_DEPENDENCIES_MESSAGE)

    NetcdfFilesDatabaseSwotLRWW = BasicNetcdfFilesDatabaseSwotLRWW
