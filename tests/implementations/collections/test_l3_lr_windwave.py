import os
import typing as tp
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from fsspec.implementations.local import LocalFileSystem
from utils import brute_force_geographical_selection

from fcollections.core import DirNode, FileSystemMetadataCollector, PerformanceWarning
from fcollections.implementations import (
    AVISO_L3_LR_WINDWAVE_LAYOUT,
    NetcdfFilesDatabaseSwotLRWW,
    ProductLevel,
    ProductSubset,
    SwotPhases,
    SwotReaderL3WW,
)
from fcollections.time import Period


class TestReader:

    @pytest.fixture(autouse=True)
    def setup(self):
        try:
            # Reader for test configuration with geo packages available
            from fcollections.implementations.optional import GeoSwotReaderL3WW

            self.reader = GeoSwotReaderL3WW()
        except ImportError:
            # Fall back reader
            self.reader = SwotReaderL3WW()

    def test_read_bad_subset(self):
        with pytest.raises(ValueError, match="Light or Extended"):
            self.reader.read(ProductSubset.Unsmoothed, ["dummy"])

    def test_read_light_no_files(self):
        # no files -> ValueError
        with pytest.raises(ValueError, match="least one"):
            self.reader.read(ProductSubset.Light, [])

    @pytest.mark.parametrize("tile, box", [(20, None), (None, 20), (10, 20)])
    def test_read_light_bad_arguments(self, tile: int, box: int):
        # tile and box should be None for Light subset
        with pytest.raises(ValueError, match="'tile' and 'box'"):
            self.reader.read(ProductSubset.Light, ["dummy"], tile=tile, box=box)

    def test_read_light_nominal(
        self, l3_lr_ww_light_files: list[Path], l3_lr_ww_light_dataset: xr.Dataset
    ):
        # Nominal case
        ds = self.reader.read(ProductSubset.Light, l3_lr_ww_light_files[:1])
        ds = ds.compute()
        xr.testing.assert_equal(l3_lr_ww_light_dataset, ds)

    def test_read_light_concatenated(self, l3_lr_ww_light_files: list[Path]):
        # Concatenated data
        ds = self.reader.read(ProductSubset.Light, l3_lr_ww_light_files[:2])
        ds_0 = self.reader.read(ProductSubset.Light, l3_lr_ww_light_files[:1])
        ds_1 = self.reader.read(ProductSubset.Light, l3_lr_ww_light_files[1:2])

        xr.testing.assert_identical(ds_0, ds.isel(n_box=slice(0, ds_0.sizes["n_box"])))
        xr.testing.assert_identical(
            ds_1, ds.isel(n_box=slice(ds_0.sizes["n_box"], None))
        )

    @pytest.mark.with_geo_packages
    def test_read_light_geographical_selection(
        self, l3_lr_ww_light_files: list[Path], l3_lr_ww_light_dataset: xr.Dataset
    ):
        # Cropped data
        bbox = (80, 70, 90, 90)
        reference = brute_force_geographical_selection(l3_lr_ww_light_dataset, *bbox)
        assert reference.sizes["n_box"] < l3_lr_ww_light_dataset.sizes["n_box"]
        assert reference.sizes["n_box"] > 0

        ds = self.reader.read(ProductSubset.Light, l3_lr_ww_light_files[:1], bbox=bbox)
        ds = ds.compute()

        xr.testing.assert_equal(reference, ds)

    @pytest.mark.without_geo_packages
    def test_read_light_geographical_selection_disabled(
        self,
        l3_lr_ww_light_files: list[Path],
    ):
        bbox = (80, 70, 90, 90)
        with pytest.raises(TypeError):
            self.reader.read(ProductSubset.Light, l3_lr_ww_light_files[:1], bbox=bbox)

    def test_read_light_variables_selection(self, l3_lr_ww_light_files: list[Path]):
        # Select variables
        requested_variables = {"longitude", "H18_model"}
        ds = self.reader.read(
            ProductSubset.Light,
            l3_lr_ww_light_files[:1],
            selected_variables=requested_variables,
        )
        assert set(ds.variables) == requested_variables

    def test_read_light_variables_selection_empty(
        self, l3_lr_ww_light_files: list[Path]
    ):
        # no valid variables -> empty dataset but with attributes
        ds = self.reader.read(
            ProductSubset.Light, l3_lr_ww_light_files[:1], selected_variables=["H19"]
        )
        assert len(ds) == 0
        assert ds.attrs != {}

    @pytest.mark.parametrize(
        "tile, box, selected_variables",
        [
            (None, None, None),
            (10, None, None),
            (None, 10, None),
            (-5, None, None),
            (10, -5, None),
            (None, 10, ["nfx"]),
            (-5, 10, ["nfx"]),
            (None, None, ["L18"]),
            (None, 5, ["L18"]),
            (10, None, ["L18"]),
            (-5, 10, ["L18"]),
            (10, -5, ["L18"]),
        ],
        ids=[
            "all_variables_missing_tile_box",
            "all_variables_missing_box",
            "all_variables_missing_tile",
            "all_variables_invalid_tile",
            "all_variables_invalid_box",
            "tile_variable_missing_tile",
            "tile_variable_invalid_tile",
            "box_variable_missing_tile_box",
            "box_variable_missing_tile",
            "box_variable_missing_box",
            "box_variable_invalid_box",
            "box_variable_invalid_tile",
        ],
    )
    def test_read_extended_bad_arguments(
        self,
        l3_lr_ww_extended_files: list[Path],
        tile: int | None,
        box: int | None,
        selected_variables: list[str] | None,
    ):
        # tile and box should be None for Light subset

        with pytest.raises(ValueError):
            self.reader.read(
                ProductSubset.Extended,
                l3_lr_ww_extended_files[:1],
                tile=tile,
                box=box,
                selected_variables=selected_variables,
            )

    def test_read_extended_nominal(
        self, l3_lr_ww_extended_files: list[Path], l3_lr_ww_extended_dataset: xr.Dataset
    ):
        # Open all variables
        ds = self.reader.read(
            ProductSubset.Extended, l3_lr_ww_extended_files[:1], tile=10, box=40
        )
        ds = ds.compute()
        xr.testing.assert_equal(l3_lr_ww_extended_dataset, ds)

    @pytest.mark.parametrize(
        "tile, box, requested_variables",
        [
            (None, None, set()),
            (10, None, {"filter_PTR"}),
            (10, 40, {"L18"}),
            (10, 40, {"filter_PTR", "L18"}),
        ],
        ids=["nothing", "tile_group", "box_group", "tile_box_groups"],
    )
    def test_read_extended_selected_variable_tile(
        self,
        tile: int,
        box: int | None,
        requested_variables: set[str],
        l3_lr_ww_extended_files: list[Path],
    ):
        # Open variable in tile group
        ds = self.reader.read(
            ProductSubset.Extended,
            l3_lr_ww_extended_files[:1],
            selected_variables=requested_variables,
            tile=tile,
            box=box,
        )
        assert set(ds.variables) == requested_variables

    def test_read_extended_concatenated(self, l3_lr_ww_extended_files: list[Path]):
        # Open multiple files
        ds = self.reader.read(
            ProductSubset.Extended, l3_lr_ww_extended_files[:2], tile=10, box=40
        )
        ds_0 = self.reader.read(
            ProductSubset.Extended, l3_lr_ww_extended_files[:1], tile=10, box=40
        )
        ds_1 = self.reader.read(
            ProductSubset.Extended, l3_lr_ww_extended_files[1:2], tile=10, box=40
        )

        xr.testing.assert_identical(ds_0, ds.isel(n_box=slice(0, ds_0.sizes["n_box"])))
        xr.testing.assert_identical(
            ds_1, ds.isel(n_box=slice(ds_0.sizes["n_box"], None))
        )

    def test_read_extended_concatenated_tile(self, l3_lr_ww_extended_files: list[Path]):
        # Open multiple files, tile group is constant
        ds = self.reader.read(
            ProductSubset.Extended,
            l3_lr_ww_extended_files[:2],
            selected_variables={"filter_PTR"},
            tile=10,
        )
        ds_0 = self.reader.read(
            ProductSubset.Extended,
            l3_lr_ww_extended_files[:1],
            selected_variables={"filter_PTR"},
            tile=10,
        )
        ds_1 = self.reader.read(
            ProductSubset.Extended,
            l3_lr_ww_extended_files[1:2],
            selected_variables={"filter_PTR"},
            tile=10,
        )

        xr.testing.assert_identical(ds_0, ds)
        xr.testing.assert_identical(ds_1, ds)

    @pytest.mark.with_geo_packages
    def test_read_extended_geographical_selection(
        self, l3_lr_ww_extended_files: list[Path], l3_lr_ww_extended_dataset: xr.Dataset
    ):
        bbox = (80, 70, 90, 90)
        reference = brute_force_geographical_selection(l3_lr_ww_extended_dataset, *bbox)
        assert reference.sizes["n_box"] < l3_lr_ww_extended_dataset.sizes["n_box"]
        assert reference.sizes["n_box"] > 0

        ds = self.reader.read(
            ProductSubset.Extended,
            l3_lr_ww_extended_files[:1],
            bbox=bbox,
            tile=10,
            box=40,
        )
        ds = ds.compute()

        xr.testing.assert_equal(reference, ds)

    @pytest.mark.without_geo_packages
    def test_read_extended_geographical_selection_disabled(
        self, l3_lr_ww_extended_files: list[Path]
    ):
        bbox = (80, 70, 90, 90)
        with pytest.raises(TypeError):
            self.reader.read(
                ProductSubset.Extended,
                l3_lr_ww_extended_files[:1],
                bbox=bbox,
                tile=10,
                box=40,
            )


class TestListing:

    @pytest.mark.parametrize(
        "query, half_orbits",
        [
            (
                {},
                [
                    (482, 11),
                    (482, 12),
                    (10, 10),
                ],
            ),
            (
                {"cycle_number": [482]},
                [(482, 11), (482, 12)],
            ),
            ({"pass_number": [10]}, [(10, 10)]),
            (
                {
                    "time": (
                        np.datetime64("2024-01-25T03"),
                        np.datetime64("2024-01-25T03:30"),
                    )
                },
                [(10, 10)],
            ),
            (
                {"subset": ProductSubset.Light},
                [(482, 11), (482, 12)],
            ),
            (
                {"version": "2.0"},
                [(482, 11), (482, 12)],
            ),
            ({"phase": SwotPhases.CALVAL}, [(482, 11), (482, 12)]),
            (
                {"phase": SwotPhases.SCIENCE},
                [
                    (10, 10),
                ],
            ),
        ],
    )
    def test_list(
        self,
        l3_lr_ww_dir_layout: Path,
        query: dict[str, tp.Any],
        half_orbits: list[tuple[int, int]],
    ):

        db = NetcdfFilesDatabaseSwotLRWW(l3_lr_ww_dir_layout)
        files = db.list_files(**query, sort=True)
        actual_half_orbits = sorted(
            [tuple(x) for x in files[["cycle_number", "pass_number"]].to_numpy()]
        )
        assert actual_half_orbits == sorted(half_orbits)


class TestQuery:

    @pytest.mark.without_geo_packages
    def test_query_bbox_disabled(self, l3_lr_ww_dir_layout: Path):
        db = NetcdfFilesDatabaseSwotLRWW(l3_lr_ww_dir_layout)
        with pytest.raises(ValueError):
            bbox = (260, 10, 300, 40)
            ds = db.query(subset="Light", bbox=bbox)


class TestLayout:

    def test_generate_layout(self):
        path = AVISO_L3_LR_WINDWAVE_LAYOUT.generate(
            "/swot_products/l3_karin/l3_lr_wind_wave",
            subset=ProductSubset.Extended,
            version="2.0",
            cycle_number=1,
            pass_number=10,
            time=Period(np.datetime64("2024-01-01"), np.datetime64("2024-01-02")),
            level=ProductLevel.L3,
        )

        assert (
            path
            == "/swot_products/l3_karin/l3_lr_wind_wave/v2_0/Extended/cycle_001/SWOT_L3_LR_WIND_WAVE_Extended_001_010_20240101T000000_20240102T000000_v2.0.nc"
        )

    def test_generate_layout_missing_field(self):
        with pytest.raises(ValueError):
            AVISO_L3_LR_WINDWAVE_LAYOUT.generate(
                "/swot_products/l3_karin/l3_lr_wind_wave",
                subset=ProductSubset.Light,
                cycle_number=1,
            )

    def test_generate_layout_bad_field(self):
        with pytest.raises(ValueError):
            AVISO_L3_LR_WINDWAVE_LAYOUT.generate(
                "/swot_products/l3_karin/l3_lr_wind_wave",
                subset=ProductSubset.Light,
                cycle_number="x",
                pass_number=10,
                time=Period(np.datetime64("2024-01-01"), np.datetime64("2024-01-02")),
                level=ProductLevel.L3,
            )

    @pytest.mark.parametrize(
        "filters, expected",
        [
            ({}, [0, 1, 2]),
            ({"version": "2.0"}, [0, 1]),
            ({"subset": "Extended"}, [2]),
            ({"cycle_number": slice(480, 490)}, [0, 1]),
            ({"pass_number": [10, 11]}, [0, 2]),
        ],
    )
    def test_list_layout(
        self,
        l3_lr_ww_dir_layout: Path,
        l3_lr_ww_files: list[str],
        expected: list[int],
        filters: dict[str, tp.Any],
    ):

        root_path_str = l3_lr_ww_dir_layout.as_posix()
        root_node = DirNode(
            root_path_str, {"name": root_path_str}, LocalFileSystem(), 0
        )

        collector = FileSystemMetadataCollector(
            NetcdfFilesDatabaseSwotLRWW.layouts, root_node
        )

        actual = {
            os.path.basename(f) for f in collector.to_dataframe(**filters).filename
        }
        expected = {os.path.basename(l3_lr_ww_files[ii]) for ii in expected}
        assert len(expected) > 0
        assert expected == actual


class TestHalfOrbitMixin:

    @pytest.mark.parametrize(
        "enable_layouts, filters, context, expected",
        [
            (True, {}, pytest.raises(ValueError, match="unmixed"), None),
            (False, {}, pytest.raises(ValueError, match="unmixed"), None),
            (True, {"subset": "Light"}, nullcontext(), ((482, 11), (482, 12))),
            (False, {"subset": "Light"}, nullcontext(), ((482, 11), (482, 12))),
            (
                True,
                {"subset": "Light", "pass_number": 11},
                nullcontext(),
                ((482, 11), (482, 11)),
            ),
            (
                False,
                {"subset": "Light", "pass_number": 11},
                nullcontext(),
                ((482, 11), (482, 11)),
            ),
            (
                True,
                {"subset": "Extended", "phase": "SCIENCE"},
                nullcontext(),
                ((10, 10), (10, 10)),
            ),
            (
                False,
                {"subset": "Extended", "phase": "SCIENCE"},
                nullcontext(),
                ((10, 10), (10, 10)),
            ),
            (True, {"subset": "Extended", "phase": "CALVAL"}, nullcontext(), None),
            (False, {"subset": "Extended", "phase": "CALVAL"}, nullcontext(), None),
        ],
        ids=[
            "missing_subset_key_layout",
            "missing_subset_key_no_layout",
            "auto_version_key_layout",
            "auto_version_key_no_layout_warns",
            "filtered_range_layout",
            "filtered_range_no_layout",
            "range_for_phase_layout",
            "range_for_phase_no_layout",
            "no_data_layout",
            "no_data_no_layout",
        ],
    )
    def test_half_orbit_range(
        self,
        l3_lr_ww_dir_layout: Path,
        enable_layouts: bool,
        filters: dict[str, tp.Any],
        context,
        expected: tuple[tuple[int, int], tuple[int, int]] | None,
    ):
        db = NetcdfFilesDatabaseSwotLRWW(
            l3_lr_ww_dir_layout, enable_layouts=enable_layouts
        )
        with context:
            assert db.half_orbit_range(**filters) == expected

    @pytest.mark.parametrize("enable_layouts", [True, False])
    def test_temporal_coverage(self, l3_lr_ww_dir_layout: Path, enable_layouts: bool):
        db = NetcdfFilesDatabaseSwotLRWW(
            l3_lr_ww_dir_layout, enable_layouts=enable_layouts
        )
        assert db.time_coverage(subset="Extended") == Period(
            np.datetime64("2024-01-25T02:53:52"), np.datetime64("2024-01-25T03:44:38")
        )

    @pytest.mark.parametrize(
        "enable_layouts, context",
        [(True, nullcontext()), (False, pytest.warns(PerformanceWarning))],
    )
    def test_cycle_range(
        self, l3_lr_ww_dir_layout: Path, enable_layouts: bool, context
    ):
        db = NetcdfFilesDatabaseSwotLRWW(
            l3_lr_ww_dir_layout, enable_layouts=enable_layouts
        )
        with context:
            assert db.cycle_range(subset="Light") == (482, 482)
