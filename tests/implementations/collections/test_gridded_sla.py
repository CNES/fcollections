from __future__ import annotations

import os
import typing as tp

import numpy as np
import pytest
from fsspec.implementations.local import LocalFileSystem

from fcollections.core import DirNode, FileSystemMetadataCollector, LayoutMismatchError
from fcollections.implementations import (
    AVISO_L4_SWOT_LAYOUT,
    CMEMS_L4_SSHA_LAYOUT,
    Delay,
    FileNameConventionGriddedSLA,
    FileNameConventionGriddedSLAInternal,
    NetcdfFilesDatabaseGriddedSLA,
    SwotPhases,
)
from fcollections.implementations._definitions._cmems import (
    Area,
    DataType,
    Group,
    Origin,
    ProductClass,
    Thematic,
    Typology,
    Variable,
)
from fcollections.time import ISODuration, Period

if tp.TYPE_CHECKING:
    from pathlib import Path


class TestConvention:

    convention = FileNameConventionGriddedSLA()
    convention_internal = FileNameConventionGriddedSLAInternal()

    def test_internal_convention_generate(self):

        reference = "msla_oer_merged_h_27029.nc"
        period = Period(np.datetime64("2024-01-02"), np.datetime64("2024-01-03"))
        actual = self.convention_internal.generate(date=period)
        assert actual == reference

    def test_internal_convention_parse(self):
        filename = "msla_oer_merged_h_27029.nc"
        reference = (
            Period(
                np.datetime64("2024-01-02"),
                np.datetime64("2024-01-03"),
                include_stop=False,
            ),
        )
        actual = self.convention_internal.parse(
            self.convention_internal.match(filename)
        )
        assert actual == reference

    def test_convention_generate(self):

        reference = "dt_global_allsat_phy_l4_20230328_20250331.nc"
        actual = self.convention.generate(
            delay=Delay.DT,
            time=Period(
                np.datetime64("2023-03-28"),
                np.datetime64("2023-03-29"),
                include_stop=False,
            ),
            production_date=np.datetime64("2025-03-31"),
        )
        assert actual == reference

    def test_convention_parse(self):
        filename = "dt_global_allsat_phy_l4_20230328_20250331.nc.nc"
        reference = (
            Delay.DT,
            Period(
                np.datetime64("2023-03-28"),
                np.datetime64("2023-03-29"),
                include_stop=False,
            ),
            np.datetime64("2025-03-31"),
        )
        actual = self.convention.parse(self.convention.match(filename))
        assert actual == reference


class TestListing:

    @pytest.mark.parametrize(
        "params, result_size",
        [
            ({}, 3),
            ({"time": np.datetime64("2023-10-16")}, 2),
            ({"production_date": np.datetime64("2023-11-06")}, 1),
            ({"delay": Delay.NRT}, 2),
        ],
    )
    def test_list_gridded_sla(
        self, l4_ssha_dir_no_layout: Path, params: dict[str, tp.Any], result_size: int
    ):
        db = NetcdfFilesDatabaseGriddedSLA(l4_ssha_dir_no_layout)

        files = db.list_files(**params)
        assert files["filename"].size == result_size


class TestQuery:

    def test_bad_kwargs(self, l4_ssha_dir_no_layout: Path):
        db = NetcdfFilesDatabaseGriddedSLA(l4_ssha_dir_no_layout)
        with pytest.raises(ValueError):
            db.list_files(bad_arg="bad_arg")
        with pytest.raises(ValueError):
            db.query(bad_arg="bad_arg")

    @pytest.mark.without_geo_packages
    def test_query_bbox_disabled(self, l4_ssha_dir_no_layout: Path):
        db = NetcdfFilesDatabaseGriddedSLA(l4_ssha_dir_no_layout)
        with pytest.raises(ValueError):
            db.query(bbox=(-180, -90, 180, 90))


class TestLayout:

    @pytest.mark.parametrize(
        "dataset_id",
        [
            # SEALEVEL_GLO_PHY_L4_NRT_008_046
            "cmems_obs-sl_glo_phy-ssh_nrt_demo-allsat-swos-l4-duacs-0.125deg_P1D-i",
            "cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D",
            "cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.25deg_P1D",
            # SEALEVEL_GLO_PHY_L4_MY_008_047
            "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
            "cmems_obs-sl_glo_phy-ssh_my_allsat-demo-l4-duacs-0.125deg_P1D-i",
            "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1M-m",
            # Result from the get command (with timestamp)
            "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_202205",
            "cmems_obs-sl_glo_phy-ssh_my_allsat-demo-l4-duacs-0.125deg_P1D-i_202205",
        ],
    )
    def test_dataset_id_regex(self, dataset_id: str):
        """Check dataset id convention building for CMEMS."""
        actual = CMEMS_L4_SSHA_LAYOUT.conventions[0].match(dataset_id)
        assert actual is not None

    @pytest.mark.parametrize(
        "expected, data_type, blending, spatial_resolution, temporal_resolution, typology, version",
        [
            (
                "cmems_obs-sl_glo_phy-ssh_my_allsat-demo-l4-duacs-0.125deg_P1D-i_202205",
                DataType.MY,
                "allsat-demo",
                0.125,
                ISODuration(days=1),
                Typology.I,
                "202205",
            ),
            (
                "cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.25deg_PT12H",
                DataType.NRT,
                "allsat",
                0.25,
                ISODuration(hours=12),
                None,
                None,
            ),
        ],
    )
    def test_dataset_id_generate(
        self,
        expected: str,
        version: str | None,
        spatial_resolution: float,
        blending: str,
        temporal_resolution: ISODuration,
        data_type: DataType,
        typology: Typology | None,
    ):
        dataset_id = CMEMS_L4_SSHA_LAYOUT.conventions[0].generate(
            spatial_resolution=spatial_resolution,
            blending=blending,
            origin=Origin.CMEMS,
            group=Group.OBS,
            pc=ProductClass.SL,
            area=Area.GLO,
            thematic=Thematic.PHY,
            variable=Variable.SSH,
            type=data_type,
            temporal_resolution=temporal_resolution,
            typology=typology,
            version=version,
        )

        assert dataset_id == expected

    def test_generate_layout_aviso(self):
        path = AVISO_L4_SWOT_LAYOUT.generate(
            "/duacs-experimental/dt-phy-grids/l4_karin_nadir",
            method="miost",
            version="1.0",
            phase=SwotPhases.SCIENCE,
            delay=Delay.NRT,
            time=Period(
                np.datetime64("2025-03-31"),
                np.datetime64("2025-04-01"),
                include_stop=False,
            ),
            production_date=np.datetime64("2025-03-31"),
        )

        assert (
            path
            == "/duacs-experimental/dt-phy-grids/l4_karin_nadir/v1.0/miost/science/nrt_global_allsat_phy_l4_20250331_20250331.nc"
        )

    @pytest.mark.parametrize(
        "filters, expected",
        [
            ({}, [0, 1, 2, 3, 4, 5, 6]),
            ({"version": "0.3"}, [4, 5, 6]),
            ({"method": "4dvarnet"}, [4]),
            ({"phase": "SCIENCE", "version": "1.0"}, [0, 1]),
            ({"time": np.datetime64("2023-07-29")}, [1, 3, 4, 6]),
            ({"production_date": np.datetime64("2024-09-13")}, [4]),
            ({"delay": Delay.NRT}, [6]),
        ],
    )
    def test_list_layout_aviso(
        self,
        l4_ssha_dir_layout_aviso: Path,
        l4_ssha_files: list[str],
        expected: list[int],
        filters: dict[str, tp.Any],
    ):

        root_path_str = l4_ssha_dir_layout_aviso.as_posix()
        root_node = DirNode(
            root_path_str, {"name": root_path_str}, LocalFileSystem(), 0
        )

        collector = FileSystemMetadataCollector(
            NetcdfFilesDatabaseGriddedSLA.layouts, root_node
        )

        actual = {
            os.path.basename(f) for f in collector.to_dataframe(**filters).filename
        }
        expected = {os.path.basename(l4_ssha_files[ii]) for ii in expected}
        assert len(expected) > 0
        assert expected == actual

    def test_list_layout_aviso_incompatibility(
        self,
        l4_ssha_dir_layout_aviso: Path,
    ):
        root_path_str = l4_ssha_dir_layout_aviso.as_posix()
        root_node = DirNode(
            root_path_str, {"name": root_path_str}, LocalFileSystem(), 0
        )

        collector = FileSystemMetadataCollector(
            NetcdfFilesDatabaseGriddedSLA.layouts, root_node
        )

        with pytest.raises(LayoutMismatchError, match="layout-specific"):
            collector.to_dataframe(phase="SCIENCE")

    def test_list_layout_aviso_warning(
        self,
        l4_ssha_dir_layout_aviso: Path,
    ):
        db = NetcdfFilesDatabaseGriddedSLA(
            l4_ssha_dir_layout_aviso, enable_layouts=False
        )

        reference = db.list_files(sort=True)
        with pytest.warns(UserWarning, match="layout-specific"):
            actual = db.list_files(phase="SCIENCE", sort=True)

        assert len(reference) > 0
        assert reference.equals(actual)

    @pytest.mark.parametrize(
        "filters, expected",
        [
            ({}, [10, 11, 12, 13]),
            ({"delay": Delay.DT}, [0]),
            ({"type": DataType.MY}, [0]),
            ({"version": "202411"}, [13]),
            ({"time": np.datetime64("2023-07-28")}, [10, 13]),
            ({"production_date": np.datetime64("2024-09-13")}, [12]),
            ({"spatial_resolution": 0.5}, [12]),
            ({"temporal_resolution": "PT12H"}, [11]),
        ],
    )
    def test_list_layout_cmems(
        self,
        l4_ssha_dir_layout_cmems: Path,
        l4_ssha_files: list[str],
        expected: list[int],
        filters: dict[str, tp.Any],
    ):

        root_path_str = l4_ssha_dir_layout_cmems.as_posix()
        root_node = DirNode(
            root_path_str, {"name": root_path_str}, LocalFileSystem(), 0
        )

        collector = FileSystemMetadataCollector(
            NetcdfFilesDatabaseGriddedSLA.layouts, root_node
        )

        actual = {
            os.path.basename(f) for f in collector.to_dataframe(**filters).filename
        }
        expected = {os.path.basename(l4_ssha_files[ii]) for ii in expected}
        assert len(expected) > 0
        assert expected == actual
