from __future__ import annotations

import re

from fcollections.core import (
    CaseType,
    FileNameConvention,
    FileNameFieldEnum,
    FileNameFieldInteger,
    FileNameFieldPeriod,
    FileNameFieldString,
    FilesDatabase,
    HalfOrbitMixin,
    Layout,
    PeriodMixin,
    SubsetsUnmixer,
)

from ._converters import SwotPhaseFilterBuilder
from ._definitions._constants import DESCRIPTIONS, ProductLevel
from ._definitions._swot import ProductSubset, Temporality
from ._readers import SwotReaderL3LRSSH

SWOT_L3_PATTERN = re.compile(
    r"SWOT_(?P<level>.*)_LR_SSH_(?P<subset>.*)_(?P<cycle_number>\d{3})_(?P<pass_number>\d{3})_"
    r"(?P<time>\d{8}T\d{6}_\d{8}T\d{6})_v(?P<version>.*).nc"
)


class FileNameConventionSwotL3(FileNameConvention):
    """Swot LR L3 datafiles parser."""

    def __init__(self):
        super().__init__(
            regex=SWOT_L3_PATTERN,
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
                    "level", ProductLevel, description=DESCRIPTIONS["level"]
                ),
                FileNameFieldEnum(
                    "subset", ProductSubset, description=DESCRIPTIONS["subset"]
                ),
                FileNameFieldString("version", description=DESCRIPTIONS["version"]),
            ],
            generation_string="SWOT_{level!f}_LR_SSH_{subset!f}_{cycle_number:>03d}_{pass_number:>03d}_{time!f}_v{version}.nc",
        )


class _FileNameFieldStringAdapter(FileNameFieldString):
    """Specific field for L3_LR_SSH version in folder versus file.

    L3_LR_SSH version is given as 1.0.0 in the files, but as 1_0_0 in
    the folder. In order to have consistent fields between the file name
    parsing and the folder layout parsing, we define this very specific
    adapter.
    """

    def decode(self, a: str) -> str:
        return super().decode(a.replace("_", "."))

    def encode(self, a: str) -> str:
        return super().encode(a.replace(".", "_"))


AVISO_L3_LR_SSH_LAYOUT_V2 = Layout(
    [
        FileNameConvention(
            re.compile(r"^v(?P<version>.*)$"),
            [_FileNameFieldStringAdapter("version")],
            "v{version!f}",
        ),
        FileNameConvention(
            re.compile(r"^(?P<subset>.*)$"),
            [FileNameConventionSwotL3().get_field("subset")],
            "{subset!f}",
        ),
        FileNameConvention(
            re.compile(r"^cycle_(?P<cycle_number>\d{3})$"),
            [FileNameConventionSwotL3().get_field("cycle_number")],
            "cycle_{cycle_number:0>3d}",
        ),
        FileNameConventionSwotL3(),
    ]
)

AVISO_L3_LR_SSH_LAYOUT_V3 = Layout(
    [
        *AVISO_L3_LR_SSH_LAYOUT_V2.conventions[:2],
        FileNameConvention(
            re.compile(r"^(?P<temporality>reproc|forward)$"),
            [
                FileNameFieldEnum(
                    "temporality",
                    Temporality,
                    case_type_encoded=CaseType.lower,
                    case_type_decoded=CaseType.upper,
                ),
            ],
            "{temporality!f}",
        ),
        AVISO_L3_LR_SSH_LAYOUT_V2.conventions[2],
        FileNameConventionSwotL3(),
    ]
)


class BasicNetcdfFilesDatabaseSwotLRL3(FilesDatabase, HalfOrbitMixin, PeriodMixin):
    """Database mapping to select and read Swot LR L3 Netcdf files in a local
    file system.

    Note
    ----
    HalfOrbitMixin overrides the temporal coverage access (it is usually faster)
    so it must be declared prior to the PeriodMixin
    """

    layouts = [
        Layout([FileNameConventionSwotL3()]),
        AVISO_L3_LR_SSH_LAYOUT_V2,
        AVISO_L3_LR_SSH_LAYOUT_V3,
    ]
    reader = SwotReaderL3LRSSH()
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
        GeoSwotReaderL3LRSSH,
        SwotGeometryFilterBuilder,
    )

    class NetcdfFilesDatabaseSwotLRL3(BasicNetcdfFilesDatabaseSwotLRL3):
        reader = GeoSwotReaderL3LRSSH()
        filter_builders = [SwotGeometryFilterBuilder, SwotPhaseFilterBuilder]

except ImportError:
    import logging

    from ._definitions._constants import MISSING_OPTIONAL_DEPENDENCIES_MESSAGE

    logger = logging.getLogger(__name__)
    logger.info(MISSING_OPTIONAL_DEPENDENCIES_MESSAGE)

    NetcdfFilesDatabaseSwotLRL3 = BasicNetcdfFilesDatabaseSwotLRL3
