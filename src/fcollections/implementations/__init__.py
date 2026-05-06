from fcollections.core import Layout

__all__ = []

from ._dac import (
    BasicNetcdfFilesDatabaseDAC,
    FileNameConventionDAC,
    NetcdfFilesDatabaseDAC,
)
from ._definitions._cmems import (
    Area,
    DataType,
    Group,
    Origin,
    ProductClass,
    Sensors,
    Thematic,
    Typology,
    Variable,
)
from ._definitions._constants import (
    Delay,
    ProductLevel,
)
from ._definitions._swot import ProductSubset, SwotPhases, Temporality
from ._era5 import FileNameConventionERA5, NetcdfFilesDatabaseERA5
from ._gridded_sla import AVISO_L4_SWOT_LAYOUT as _AVISO_L4_SWOT_LAYOUT
from ._gridded_sla import CMEMS_L4_SSHA_LAYOUT as _CMEMS_L4_SSHA_LAYOUT
from ._gridded_sla import (
    BasicNetcdfFilesDatabaseGriddedSLA,
    FileNameConventionGriddedSLA,
    FileNameConventionGriddedSLAInternal,
    NetcdfFilesDatabaseGriddedSLA,
)
from ._l2_lr_ssh import AVISO_L2_LR_SSH_LAYOUT as _AVISO_L2_LR_SSH_LAYOUT
from ._l2_lr_ssh import (
    BasicNetcdfFilesDatabaseSwotLRL2,
    FileNameConventionSwotL2,
    L2Version,
    L2VersionField,
    NetcdfFilesDatabaseSwotLRL2,
    Timeliness,
    build_version_parser,
)
from ._l2_nadir import (
    BasicNetcdfFilesDatabaseL2Nadir,
    FileNameConventionL2Nadir,
    NetcdfFilesDatabaseL2Nadir,
)
from ._l3_lr_ssh import AVISO_L3_LR_SSH_LAYOUT_V2 as _AVISO_L3_LR_SSH_LAYOUT_V2
from ._l3_lr_ssh import AVISO_L3_LR_SSH_LAYOUT_V3 as _AVISO_L3_LR_SSH_LAYOUT_V3
from ._l3_lr_ssh import (
    BasicNetcdfFilesDatabaseSwotLRL3,
    FileNameConventionSwotL3,
    NetcdfFilesDatabaseSwotLRL3,
)
from ._l3_lr_ww import AVISO_L3_LR_WINDWAVE_LAYOUT as _AVISO_L3_LR_WINDWAVE_LAYOUT
from ._l3_lr_ww import (
    BasicNetcdfFilesDatabaseSwotLRWW,
    FileNameConventionSwotL3WW,
    NetcdfFilesDatabaseSwotLRWW,
)
from ._l3_nadir import CMEMS_SSHA_L3_LAYOUT as _CMEMS_SSHA_L3_LAYOUT
from ._l3_nadir import (
    BasicNetcdfFilesDatabaseL3Nadir,
    FileNameConventionL3Nadir,
    NetcdfFilesDatabaseL3Nadir,
)
from ._mur import (
    BasicNetcdfFilesDatabaseMUR,
    FileNameConventionMUR,
    NetcdfFilesDatabaseMUR,
)
from ._ocean_color import CMEMS_OC_LAYOUT as _CMEMS_OC_LAYOUT
from ._ocean_color import (
    BasicNetcdfFilesDatabaseOC,
    FileNameConventionOC,
    NetcdfFilesDatabaseOC,
)
from ._ohc import (
    BasicNetcdfFilesDatabaseOHC,
    FileNameConventionOHC,
    NetcdfFilesDatabaseOHC,
)
from ._predicates import SwotPhaseFilterBuilder
from ._readers import (
    StackLevel,
    SwotReaderL2LRSSH,
    SwotReaderL3LRSSH,
    SwotReaderL3WW,
)
from ._s1aowi import (
    AcquisitionMode,
    FileNameConventionS1AOWI,
    NetcdfFilesDatabaseS1AOWI,
    S1AOWIProductType,
    S1AOWISlicePostProcessing,
)
from ._sst import CMEMS_SST_LAYOUT as _CMEMS_SST_LAYOUT
from ._sst import IFREMER_SST_LAYOUT as _IFREMER_SST_LAYOUT
from ._sst import (
    BasicNetcdfFilesDatabaseSST,
    FileNameConventionSST,
    NetcdfFilesDatabaseSST,
)
from ._swh import CMEMS_SWH_LAYOUT as _CMEMS_SWH_LAYOUT
from ._swh import (
    BasicNetcdfFilesDatabaseSWH,
    FileNameConventionSWH,
    NetcdfFilesDatabaseSWH,
)

# Realiased constant to be picked up by Sphinx to work around autodoc limitation
#: Layout on Aviso FTP, Aviso TDS for the L2_LR_SSH product
AVISO_L2_LR_SSH_LAYOUT: Layout = _AVISO_L2_LR_SSH_LAYOUT
#: Layout on Aviso FTP, Aviso TDS for the L3_LR_SSH product
AVISO_L3_LR_SSH_LAYOUT_V3: Layout = _AVISO_L3_LR_SSH_LAYOUT_V3
#: Layout on Aviso FTP, Aviso TDS for the L3_LR_SSH product
AVISO_L3_LR_SSH_LAYOUT_V2: Layout = _AVISO_L3_LR_SSH_LAYOUT_V2
#: Layout on Aviso FTP, Aviso TDS for the L3_LR_WindWave product
AVISO_L3_LR_WINDWAVE_LAYOUT: Layout = _AVISO_L3_LR_WINDWAVE_LAYOUT
#: Layout on Aviso FTP, Aviso TDS for the L4 Sea Level Anomaly experimental product including karin measurements
AVISO_L4_SWOT_LAYOUT: Layout = _AVISO_L4_SWOT_LAYOUT
#: Layout on CMEMS for the Level 3 SSHA nadir products
CMEMS_SSHA_L3_LAYOUT: Layout = _CMEMS_SSHA_L3_LAYOUT
#: Layout on CMEMS for the Level 4 SSHA gridded products
CMEMS_L4_SSHA_LAYOUT: Layout = _CMEMS_L4_SSHA_LAYOUT
#: Layout on CMEMS for the Level 3 and 4 ocean colour products
CMEMS_OC_LAYOUT: Layout = _CMEMS_OC_LAYOUT
#: Layout on CMEMS for the WAVE_GLO_PHY_SWH_L3_NRT_014_001 product
CMEMS_SWH_LAYOUT: Layout = _CMEMS_SWH_LAYOUT
#: Layout on CMEMS for the SST_GLO_SST_L3S_NRT_OBSERVATIONS_010_010 product
CMEMS_SST_LAYOUT: Layout = _CMEMS_SST_LAYOUT
#: Layout on CMEMS for the SST_GLO_SST_L3S_NRT_OBSERVATIONS_010_010 product
# (Ifremer special dataset_id not following the CMEMS convention)
IFREMER_SST_LAYOUT: Layout = _IFREMER_SST_LAYOUT

__all__ = [
    "BasicNetcdfFilesDatabaseDAC",
    "BasicNetcdfFilesDatabaseGriddedSLA",
    "BasicNetcdfFilesDatabaseSwotLRL2",
    "BasicNetcdfFilesDatabaseL2Nadir",
    "BasicNetcdfFilesDatabaseSwotLRL3",
    "BasicNetcdfFilesDatabaseSwotLRWW",
    "BasicNetcdfFilesDatabaseL3Nadir",
    "BasicNetcdfFilesDatabaseMUR",
    "BasicNetcdfFilesDatabaseOC",
    "BasicNetcdfFilesDatabaseOHC",
    "BasicNetcdfFilesDatabaseSST",
    "NetcdfFilesDatabaseSwotLRL2",
    "NetcdfFilesDatabaseSwotLRL3",
    "NetcdfFilesDatabaseGriddedSLA",
    "NetcdfFilesDatabaseSST",
    "NetcdfFilesDatabaseDAC",
    "NetcdfFilesDatabaseOC",
    "NetcdfFilesDatabaseSWH",
    "BasicNetcdfFilesDatabaseSWH",
    "NetcdfFilesDatabaseOHC",
    "NetcdfFilesDatabaseS1AOWI",
    "NetcdfFilesDatabaseMUR",
    "NetcdfFilesDatabaseERA5",
    "NetcdfFilesDatabaseL2Nadir",
    "NetcdfFilesDatabaseL3Nadir",
    "NetcdfFilesDatabaseSwotLRWW",
    # File name conventions
    "FileNameConventionERA5",
    "FileNameConventionOC",
    "FileNameConventionGriddedSLA",
    "FileNameConventionGriddedSLAInternal",
    "FileNameConventionSST",
    "FileNameConventionDAC",
    "FileNameConventionSwotL2",
    "FileNameConventionSwotL3",
    "FileNameConventionSwotL3WW",
    "FileNameConventionOHC",
    "FileNameConventionS1AOWI",
    "FileNameConventionMUR",
    "FileNameConventionSWH",
    "FileNameConventionL2Nadir",
    "FileNameConventionL3Nadir",
    # Layouts
    "AVISO_L2_LR_SSH_LAYOUT",
    "AVISO_L3_LR_SSH_LAYOUT_V2",
    "AVISO_L3_LR_SSH_LAYOUT_V3",
    "AVISO_L3_LR_WINDWAVE_LAYOUT",
    "AVISO_L4_SWOT_LAYOUT",
    "CMEMS_SSHA_L3_LAYOUT",
    "CMEMS_L4_SSHA_LAYOUT",
    "CMEMS_OC_LAYOUT",
    "CMEMS_SWH_LAYOUT",
    "CMEMS_SST_LAYOUT",
    "IFREMER_SST_LAYOUT",
    # Readers
    "SwotReaderL2LRSSH",
    "SwotReaderL3LRSSH",
    "SwotReaderL3WW",
    # Predicates
    "SwotPhaseFilterBuilder",
    # Common definitions
    "Delay",
    "ProductLevel",
    # Definitions from CMEMS
    "Origin",
    "Group",
    "ProductClass",
    "DataType",
    "Thematic",
    "Area",
    "ProductClass",
    "Variable",
    "Typology",
    "Sensors",
    # Definitions for SWOT mission
    "Temporality",
    "ProductSubset",
    "SwotPhases",
    # Definitions specific to one product
    "StackLevel",
    "ProductSubset",
    "ProductGroup",
    "Timeliness",
    "L2Version",
    "L2VersionField",
    "ProductType",
    "build_version_parser",
    "ProductLevel",
    "AcquisitionMode",
    "S1AOWIProductType",
    "Delay",
    "S1AOWISlicePostProcessing",
]
