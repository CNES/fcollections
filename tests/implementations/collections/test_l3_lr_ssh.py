import itertools
import re
import typing as tp
from pathlib import Path

import fsspec
import fsspec.implementations
import fsspec.implementations.memory
import numpy as np
import pytest
import xarray as xr
from utils import brute_force_geographical_selection, extract_box_from_polygon

from fcollections.core import (
    FileNameConvention,
    Layout,
    LayoutMismatchError,
    PerformanceWarning,
)
from fcollections.implementations import (
    AVISO_L3_LR_SSH_LAYOUT_V2,
    AVISO_L3_LR_SSH_LAYOUT_V3,
    FileNameConventionSwotL3,
    NetcdfFilesDatabaseSwotLRL3,
    ProductLevel,
    ProductSubset,
    StackLevel,
    SwotPhases,
    SwotReaderL3LRSSH,
    Temporality,
)
from fcollections.time import Period


class TestReader:

    @pytest.fixture(autouse=True)
    def setup(self):
        try:
            # Reader for test configuration with geo packages available
            from fcollections.implementations.optional import (
                GeoSwotReaderL3LRSSH,
            )

            self.reader = GeoSwotReaderL3LRSSH()
        except ImportError:
            # Fall back reader
            self.reader = SwotReaderL3LRSSH()

    @pytest.mark.parametrize(
        "subset", [ProductSubset.WindWave, ProductSubset.Light, ProductSubset.Extended]
    )
    def test_invalid_subset(
        self, subset: ProductSubset, l3_lr_ssh_basic_files: list[str]
    ):
        with pytest.raises(ValueError):
            self.reader.read(subset, l3_lr_ssh_basic_files)

    def test_invalid_stack(self, l3_lr_ssh_basic_files: list[str]):
        with pytest.raises(ValueError):
            self.reader.read(ProductSubset.Basic, l3_lr_ssh_basic_files, stack=True)

    @pytest.mark.parametrize(
        "files, reference, half_orbit, subset",
        [
            (
                "l3_lr_ssh_basic_files",
                "l3_lr_ssh_basic_dataset",
                (531, 25),
                ProductSubset.Basic,
            ),
            (
                "l3_lr_ssh_expert_files",
                "l3_lr_ssh_expert_dataset",
                (532, 25),
                ProductSubset.Expert,
            ),
            (
                "l3_lr_ssh_unsmoothed_files",
                "l3_lr_ssh_unsmoothed_dataset",
                (10, 532),
                ProductSubset.Unsmoothed,
            ),
        ],
    )
    def test_all_read(
        self,
        files: list[str],
        reference: str,
        half_orbit: tuple[int, int],
        subset: ProductSubset,
        request: pytest.FixtureRequest,
    ):
        """Test nominal reading of Basic and expert files."""
        files = request.getfixturevalue(files)[:1]
        reference = request.getfixturevalue(reference)
        if "num_nadir" in reference.sizes:
            reference = reference.drop_dims(["num_nadir"])

        ds = self.reader.read(
            subset, files, nadir=True, swath=True, stack=StackLevel.NOSTACK
        )

        xr.testing.assert_equal(
            reference, ds.drop_vars(["cycle_number", "pass_number"]).compute()
        )
        assert np.all(ds.cycle_number == half_orbit[0])
        assert np.all(ds.pass_number == half_orbit[1])

    @pytest.mark.parametrize(
        "files, subset",
        [
            ("l3_lr_ssh_basic_files", ProductSubset.Basic),
            ("l3_lr_ssh_expert_files", ProductSubset.Expert),
            ("l3_lr_ssh_unsmoothed_files", ProductSubset.Unsmoothed),
        ],
    )
    def test_all_multi_read(
        self, files: list[str], subset: ProductSubset, request: pytest.FixtureRequest
    ):
        """Test nominal reading of Basic and expert files."""
        files = request.getfixturevalue(files)
        ds = self.reader.read(subset, files)
        index = 0
        for file in files:
            ds_pass = self.reader.read(subset, [file])
            reference_pass = ds.isel(
                num_lines=slice(index, index + ds_pass.num_lines.size)
            )
            xr.testing.assert_equal(reference_pass, ds_pass)
            index += ds_pass.num_lines.size

    @pytest.mark.with_geo_packages
    @pytest.mark.parametrize(
        "files", ["l3_lr_ssh_basic_files", "l3_lr_ssh_expert_files"]
    )
    @pytest.mark.parametrize("stack", StackLevel)
    def test_expert_bbox(self, files: str, stack: bool, request: pytest.FixtureRequest):
        pass_number = 25
        bbox = extract_box_from_polygon(pass_number)
        files = request.getfixturevalue(files)
        reference = self.reader.read(
            subset=ProductSubset.Basic, files=files[:1], stack=stack, nadir=False
        ).compute()
        # Set these coords to prevent the geographical_selection from reshaping
        # and casting these variables
        coords = {"cross_track_distance", "pass_number", "cycle_number"}
        coords &= set(reference.variables)
        reference = reference.set_coords(coords)

        reference_cropped = brute_force_geographical_selection(reference, *bbox)
        if stack == StackLevel.CYCLES_PASSES:
            coords.remove("cycle_number")
            coords.remove("pass_number")
        elif stack == StackLevel.CYCLES:
            coords.remove("cycle_number")
        reference_cropped = reference_cropped.reset_coords(coords)

        assert reference_cropped.num_lines.size > 0
        assert reference_cropped.num_lines.size < reference.num_lines.size

        actual_cropped = self.reader.read(
            subset=ProductSubset.Basic, files=files[:1], stack=stack, bbox=bbox
        ).compute()
        xr.testing.assert_equal(reference_cropped, actual_cropped)

    @pytest.mark.without_geo_packages
    def test_expert_bbox_disabled(self, l3_lr_ssh_basic_files: list[str]):
        with pytest.raises(TypeError):
            self.reader.read(
                ProductSubset.Basic, l3_lr_ssh_basic_files, bbox=(-180, -90, 180, 90)
            )

    @pytest.mark.with_geo_packages
    def test_expert_bbox_nadir_only(self, l3_lr_ssh_basic_files: list[str]):
        # A warning is emitted if a bad configuration is detected
        bbox = extract_box_from_polygon(26)
        reference = self.reader.read(
            subset=ProductSubset.Basic,
            files=l3_lr_ssh_basic_files,
            swath=False,
            nadir=True,
        )

        # Set these coords to prevent the geographical_selection from reshaping
        # and casting these variables
        coords = {"cross_track_distance", "pass_number", "cycle_number"}
        coords &= set(reference.variables)
        reference = reference.set_coords(coords)

        reference_cropped = brute_force_geographical_selection(reference, *bbox)
        reference_cropped = reference_cropped.reset_coords(coords)
        assert reference_cropped.num_nadir.size > 0
        assert reference_cropped.num_nadir.size < reference.num_nadir.size

        actual_cropped = self.reader.read(
            subset=ProductSubset.Basic,
            files=l3_lr_ssh_basic_files,
            bbox=bbox,
            swath=False,
            nadir=True,
        )
        xr.testing.assert_equal(reference_cropped, actual_cropped)

    @pytest.mark.parametrize(
        "files, subset, expected_half_orbits",
        [
            (
                "l3_lr_ssh_basic_files_unique",
                ProductSubset.Basic,
                [(531, 25), (531, 26), (532, 25), (532, 26)],
            ),
            (
                "l3_lr_ssh_expert_files_unique",
                ProductSubset.Expert,
                [(532, 25), (532, 26), (533, 25), (533, 26)],
            ),
        ],
    )
    def test_expert_multi_read_stack_cycles(
        self,
        files: list[list[str]],
        subset: ProductSubset,
        expected_half_orbits: list[tuple[int, int]],
        request: pytest.FixtureRequest,
    ):
        """Test cycle stacking."""
        files = request.getfixturevalue(files)
        expected_cycles = sorted({half_orbit[0] for half_orbit in expected_half_orbits})

        ds = self.reader.read(subset, files, stack="CYCLES")
        assert np.array_equal(ds.cycle_number.values, expected_cycles)

        pass_numbers = set(ds["pass_number"].values)

        ds_nostack = self.reader.read(subset, files, stack="NOSTACK")
        for cycle_number in expected_cycles:
            ds_cycle = ds.sel(cycle_number=cycle_number)

            valid_pass_numbers = {
                half_orbit[1]
                for half_orbit in expected_half_orbits
                if half_orbit[0] == cycle_number
            }
            missing_pass_numbers = pass_numbers - valid_pass_numbers

            reference_cycle = ds_nostack.set_coords("cycle_number").where(
                ds_nostack["cycle_number"] == cycle_number, drop=True
            )
            reference_cycle["cycle_number"] = np.unique(
                reference_cycle.cycle_number.values
            ).item()

            mask = xr.apply_ufunc(
                lambda p: p in valid_pass_numbers,
                ds_cycle["pass_number"],
                vectorize=True,
            )
            ds_valids = ds_cycle.where(mask, drop=True)
            xr.testing.assert_equal(reference_cycle, ds_valids)

            mask = xr.apply_ufunc(
                lambda p: p in missing_pass_numbers,
                ds_cycle["pass_number"],
                vectorize=True,
            )
            ds_invalids = ds_cycle.where(mask, drop=True)
            assert np.all(np.isnan(ds_invalids))

    @pytest.mark.parametrize(
        "files, subset, expected_half_orbits",
        [
            (
                "l3_lr_ssh_basic_files_unique",
                ProductSubset.Basic,
                [(531, 25), (531, 26), (532, 25), (532, 26)],
            ),
            (
                "l3_lr_ssh_expert_files_unique",
                ProductSubset.Expert,
                [(532, 25), (532, 26), (533, 25), (533, 26)],
            ),
        ],
    )
    def test_expert_multi_read_stack_cycles_passes(
        self,
        files: list[list[str]],
        subset: ProductSubset,
        expected_half_orbits: list[tuple[int, int]],
        request: pytest.FixtureRequest,
    ):
        """Test half orbit stacking."""
        files = request.getfixturevalue(files)
        ds = self.reader.read(subset, files, stack="CYCLES_PASSES")

        for file_half_orbit, (cycle_number, pass_number) in zip(
            files, expected_half_orbits, strict=True
        ):
            ds_half_orbit = self.reader.read(subset, [file_half_orbit])
            ds_half_orbit = ds_half_orbit.set_coords(["cycle_number", "pass_number"])

            ds_half_orbit["cycle_number"] = np.unique(
                ds_half_orbit.cycle_number.values
            ).item()
            ds_half_orbit["pass_number"] = np.unique(
                ds_half_orbit.pass_number.values
            ).item()

            reference_cycle = ds.sel(cycle_number=cycle_number, pass_number=pass_number)
            xr.testing.assert_equal(reference_cycle, ds_half_orbit)

        expected_cycles = sorted({half_orbit[0] for half_orbit in expected_half_orbits})
        expected_passes = sorted({half_orbit[1] for half_orbit in expected_half_orbits})
        assert np.array_equal(ds.cycle_number.values, expected_cycles)
        assert np.array_equal(ds.pass_number.values, expected_passes)

        # Missing half orbits should be filled with nans
        missing_half_orbits = set(
            itertools.product(expected_cycles, expected_passes)
        ) - set(expected_half_orbits)
        for cycle_number, pass_number in missing_half_orbits:
            assert np.all(
                np.isnan(ds.sel(cycle_number=cycle_number, pass_number=pass_number))
            )

    @pytest.mark.parametrize(
        "files, subset",
        [
            ("l3_lr_ssh_basic_files_unique", ProductSubset.Basic),
            ("l3_lr_ssh_expert_files_unique", ProductSubset.Expert),
        ],
    )
    @pytest.mark.parametrize("stack", [StackLevel.CYCLES, StackLevel.CYCLES_PASSES])
    def test_expert_multi_read_nadir_error(
        self,
        files: list[list[str]],
        subset: ProductSubset,
        stack: StackLevel,
        request: pytest.FixtureRequest,
    ):
        files = request.getfixturevalue(files)
        with pytest.raises(ValueError):
            # Stack with only nadir data is not possible
            self.reader.read(subset, files, stack=stack, nadir=True, swath=False)

    @pytest.mark.parametrize(
        "swath, nadir", [(True, True), (True, False), (False, True)]
    )
    def test_expert_selection_variables(
        self, l3_lr_ssh_expert_files: list[list[str]], swath: bool, nadir: bool
    ):
        """Tests we can select the read variables."""
        selected = ["longitude", "ssha_noiseless"]
        dropped = ["cross_track_distance", "latitude"]

        ds = self.reader.read(
            ProductSubset.Expert,
            l3_lr_ssh_expert_files[:1],
            nadir=nadir,
            swath=swath,
            selected_variables=selected,
        )

        assert all([s in ds for s in selected])
        assert all([s not in ds for s in dropped])

    def test_expert_nadir_only(self, l3_lr_ssh_basic_files: list[str]):
        ds = self.reader.read(
            ProductSubset.Basic, l3_lr_ssh_basic_files, nadir=True, swath=False
        )
        assert set(ds.sizes) == {"num_nadir"}

    def test_expert_nadir_only_memory(self, l3_lr_ssh_basic_files: list[str]):
        memory = fsspec.get_mapper("memory://")
        local = fsspec.get_mapper("local:///")

        # Copy one file to memory
        path = l3_lr_ssh_basic_files[0].as_posix()
        filename = l3_lr_ssh_basic_files[0].name
        memory[filename] = local[path]

        ds = self.reader.read(
            ProductSubset.Basic,
            [filename],
            nadir=True,
            swath=False,
            fs=fsspec.implementations.memory.MemoryFileSystem(),
        )
        assert set(ds.sizes) == {"num_nadir"}

    def test_expert_nadir_removal(self, l3_lr_ssh_basic_files: list[str]):
        ds = self.reader.read(
            ProductSubset.Basic, l3_lr_ssh_basic_files[:1], nadir=True, swath=True
        ).compute()
        assert not np.any(np.isnan(ds["ssha_noiseless"]))

        ds = self.reader.read(
            ProductSubset.Basic, l3_lr_ssh_basic_files[:1], nadir=False, swath=True
        ).compute()
        assert np.all(np.isnan(ds["ssha_noiseless"][::3, 34]))

    def test_expert_no_nadir_no_swath(self, l3_lr_ssh_basic_files: list[str]):
        with pytest.raises(ValueError):
            self.reader.read(
                ProductSubset.Basic, l3_lr_ssh_basic_files, swath=False, nadir=False
            )

    def test_unsmoothed_selected_variables(self, l3_lr_ssh_unsmoothed_files: list[str]):
        selected = ["longitude", "ssha_noiseless"]
        dropped = ["cross_track_distance", "latitude"]

        ds = self.reader.read(
            ProductSubset.Unsmoothed,
            l3_lr_ssh_unsmoothed_files,
            selected_variables=selected,
        )

        assert all([s in ds for s in selected])
        assert all([s not in ds for s in dropped])

    @pytest.mark.parametrize(
        "subset",
        [
            ProductSubset.Basic,
            ProductSubset.Expert,
            ProductSubset.Unsmoothed,
            ProductSubset.Technical,
        ],
    )
    def test_all_empty(self, subset: ProductSubset):
        with pytest.raises(ValueError):
            assert self.reader.read(subset, []).equals(xr.Dataset())


class TestListing:

    @pytest.mark.parametrize(
        "query, half_orbits",
        [
            (
                {},
                [
                    (531, 25),
                    (531, 26),
                    (532, 25),
                    (532, 26),
                    (532, 25),
                    (532, 25),
                    (532, 26),
                    (533, 25),
                    (533, 26),
                    (10, 532),
                ],
            ),
            (
                {"cycle_number": [532]},
                [(532, 25), (532, 26), (532, 25), (532, 25), (532, 26)],
            ),
            ({"pass_number": [532]}, [(10, 532)]),
            (
                {
                    "time": (
                        np.datetime64("2024-01-25T03"),
                        np.datetime64("2024-01-25T03:30"),
                    )
                },
                [(10, 532)],
            ),
            (
                {"subset": ProductSubset.Expert},
                [(532, 25), (532, 25), (532, 26), (533, 25), (533, 26)],
            ),
            (
                {"version": "2.0.1"},
                [(532, 25), (532, 26), (533, 25), (533, 26), (10, 532)],
            ),
            (
                {"phase": SwotPhases.CALVAL},
                [
                    (531, 25),
                    (531, 26),
                    (532, 25),
                    (532, 26),
                    (532, 25),
                    (532, 25),
                    (532, 26),
                    (533, 25),
                    (533, 26),
                ],
            ),
            (
                {"phase": SwotPhases.SCIENCE},
                [
                    (10, 532),
                ],
            ),
        ],
    )
    def test_list(
        self,
        l3_lr_ssh_dir_empty_files: Path,
        query: dict[str, tp.Any],
        half_orbits: list[tuple[int, int]],
    ):

        db = NetcdfFilesDatabaseSwotLRL3(l3_lr_ssh_dir_empty_files)
        files = db.list_files(**query, sort=True)
        actual_half_orbits = sorted(
            [tuple(x) for x in files[["cycle_number", "pass_number"]].to_numpy()]
        )
        assert actual_half_orbits == sorted(half_orbits)

    @pytest.mark.with_geo_packages
    @pytest.mark.parametrize(
        "bbox, pass_numbers",
        [
            ((-180, -90, 180, 90), [25, 26, 532]),
            ((0, -90, 360, 90), [25, 26, 532]),
        ],
    )
    def test_list_bbox_global(
        self,
        l3_lr_ssh_dir_empty_files: Path,
        bbox: tuple[int, int, int, int],
        pass_numbers: list[int],
    ):

        db = NetcdfFilesDatabaseSwotLRL3(l3_lr_ssh_dir_empty_files)

        df_pass_number = db.list_files(pass_number=pass_numbers, sort=True)
        assert not df_pass_number.empty
        df_bbox = db.list_files(bbox=bbox, sort=True)
        assert df_bbox.equals(df_pass_number)

    @pytest.mark.with_geo_packages
    @pytest.mark.parametrize(
        "pass_number, latitude, pass_numbers",
        [
            (25, 0, [25]),
            (26, 0, [26]),
            # Taking a box at highest latitude will also covers the next pass
            (25, 90, [25, 26]),
        ],
    )
    def test_list_bbox(
        self,
        l3_lr_ssh_dir_empty_files: Path,
        pass_number: int,
        latitude: float,
        pass_numbers: list[int],
    ):

        db = NetcdfFilesDatabaseSwotLRL3(l3_lr_ssh_dir_empty_files)

        # Call the box extraction inside the method, else the import will fail
        # We need pytest to setup the geometries fixture before this call can
        # work
        bbox = extract_box_from_polygon(pass_number, latitude=latitude)
        df_pass_number = db.list_files(pass_number=pass_numbers, sort=True)
        assert not df_pass_number.empty
        df_bbox = db.list_files(bbox=bbox, sort=True)
        assert df_bbox.equals(df_pass_number)

    @pytest.mark.without_geo_packages
    def test_list_bbox_disabled(self, l3_lr_ssh_dir_empty_files: Path):
        db = NetcdfFilesDatabaseSwotLRL3(l3_lr_ssh_dir_empty_files)
        with pytest.raises(ValueError):
            db.list_files(bbox=(-180, -90, 180, 90))

    def test_list_unmix_autopick_version(self, l3_lr_ssh_dir_empty_files: Path):
        # Test we have only one version in the dataset. No mixing between multiple
        # versions is expected for the L3_LR_SSH product
        # Test we actually have the latest versions
        db = NetcdfFilesDatabaseSwotLRL3(l3_lr_ssh_dir_empty_files)
        files_metadata = db.list_files(
            subset="Expert", cycle_number=532, pass_number=25
        )
        assert {str(x) for x in files_metadata["version"]} == {"1.0.2", "2.0.1"}

        # Autopick version
        files_metadata = db.list_files(
            subset="Expert", cycle_number=532, pass_number=25, unmix=True
        )
        assert len(files_metadata) == 1
        assert files_metadata["version"].values[0] == "2.0.1"

    def test_list_unmix_error(self, l3_lr_ssh_dir_empty_files: Path):
        # When duplication removal is enabled, the subset is expected to be unique
        # after duplication. If it is not filtered, it can trigger an error
        db = NetcdfFilesDatabaseSwotLRL3(l3_lr_ssh_dir_empty_files)

        with pytest.raises(ValueError):
            db.list_files(unmix=True)

    def test_list_unknown_filter(self, l3_lr_ssh_dir_empty_files: Path):
        db = NetcdfFilesDatabaseSwotLRL3(l3_lr_ssh_dir_empty_files)
        with pytest.raises(ValueError):
            db.list_files(bad_arg="bad_arg")

    def test_list_empty_result(self, l3_lr_ssh_dir_empty_files: Path):
        db = NetcdfFilesDatabaseSwotLRL3(l3_lr_ssh_dir_empty_files)
        assert db.list_files(cycle_number=399).empty


@pytest.fixture(scope="session", autouse=True)
def add_layout():
    """Additional layout needed to handle files in l3_lr_ssh_dir fixture."""
    NetcdfFilesDatabaseSwotLRL3.layouts.append(
        Layout(
            [
                FileNameConvention(re.compile("^basic|expert|unsmoothed$"), []),
                FileNameConventionSwotL3(),
            ]
        )
    )


class TestQuery:

    @pytest.mark.parametrize("stack", StackLevel)
    def test_query_empty_result(self, l3_lr_ssh_basic_dir: Path, stack: StackLevel):
        """Test limit case with no files matching filters."""
        db = NetcdfFilesDatabaseSwotLRL3(l3_lr_ssh_basic_dir)
        ds = db.query(cycle_number=-1, stack=stack)
        assert ds is None

    def test_query(self, l3_lr_ssh_dir: Path):
        """Tests high level query."""
        db = NetcdfFilesDatabaseSwotLRL3(l3_lr_ssh_dir)
        ds = db.query(cycle_number=532, pass_number=26, subset="Basic")
        assert ds.num_lines.size == 9860

    def test_query_stack(self, l3_lr_ssh_dir: Path):
        """Tests high level stacked query."""
        db = NetcdfFilesDatabaseSwotLRL3(l3_lr_ssh_dir)
        ds = db.query(pass_number=26, subset="Expert", stack="CYCLES")
        assert ds.cycle_number.size == 2

    @pytest.mark.with_geo_packages
    def test_query_bbox(self, l3_lr_ssh_dir: Path):
        """Tests high level stacked query."""
        db = NetcdfFilesDatabaseSwotLRL3(l3_lr_ssh_dir)
        bbox = (260, 10, 300, 40)
        ds = db.query(subset="Basic", bbox=bbox)
        # Real pass 20 is crossing the bbox but
        # result has 0 num_lines cause test data in swot_dir_geom has fake lon_lat
        assert ds == None

    @pytest.mark.without_geo_packages
    def test_query_bbox_disabled(self, l3_lr_ssh_dir: Path):
        db = NetcdfFilesDatabaseSwotLRL3(l3_lr_ssh_dir)
        with pytest.raises(ValueError):
            bbox = (260, 10, 300, 40)
            ds = db.query(subset="Basic", bbox=bbox)

    def test_query_unknown_filter(self, l3_lr_ssh_dir: Path):
        db = NetcdfFilesDatabaseSwotLRL3(l3_lr_ssh_dir)
        with pytest.raises(ValueError):
            db.query(bad_arg="bad_arg")


class TestLayout:

    def test_generate_string(self):
        conv = FileNameConventionSwotL3()
        t0 = np.datetime64("2024-09-25T00")
        t1 = np.datetime64("2024-09-30T00")
        assert (
            conv.generate(
                cycle_number=11,
                pass_number=15,
                level=ProductLevel.L3,
                subset=ProductSubset.Basic,
                time=Period(t0, t1),
                version="2_0",
            )
            == "SWOT_L3_LR_SSH_Basic_011_015_20240925T000000_20240930T000000_v2_0.nc"
        )

    def test_generate_layout_v2(self):
        path = AVISO_L3_LR_SSH_LAYOUT_V2.generate(
            "/swot_products/l3_karin_nadir/l3_lr_ssh",
            subset=ProductSubset.Expert,
            version="1.0.2",
            cycle_number=1,
            pass_number=10,
            time=Period(np.datetime64("2024-01-01"), np.datetime64("2024-01-02")),
            level=ProductLevel.L3,
        )

        assert (
            path
            == "/swot_products/l3_karin_nadir/l3_lr_ssh/v1_0_2/Expert/cycle_001/SWOT_L3_LR_SSH_Expert_001_010_20240101T000000_20240102T000000_v1.0.2.nc"
        )

    @pytest.mark.parametrize(
        "temporality, expected",
        [
            (
                Temporality.REPROC,
                "/swot_products/l3_karin_nadir/l3_lr_ssh/v1_0_2/Expert/reproc/cycle_001/SWOT_L3_LR_SSH_Expert_001_010_20240101T000000_20240102T000000_v1.0.2.nc",
            ),
            (
                Temporality.FORWARD,
                "/swot_products/l3_karin_nadir/l3_lr_ssh/v1_0_2/Expert/forward/cycle_001/SWOT_L3_LR_SSH_Expert_001_010_20240101T000000_20240102T000000_v1.0.2.nc",
            ),
        ],
    )
    def test_generate_layout_v3(self, temporality: Temporality, expected: str):
        path = AVISO_L3_LR_SSH_LAYOUT_V3.generate(
            "/swot_products/l3_karin_nadir/l3_lr_ssh",
            subset=ProductSubset.Expert,
            version="1.0.2",
            temporality=temporality,
            cycle_number=1,
            pass_number=10,
            time=Period(np.datetime64("2024-01-01"), np.datetime64("2024-01-02")),
            level=ProductLevel.L3,
        )

        assert path == expected

    def test_generate_layout_missing_field(self):
        with pytest.raises(ValueError):
            AVISO_L3_LR_SSH_LAYOUT_V3.generate(
                "/swot_products/l3_karin_nadir/l3_lr_ssh",
                version="1.0.2",
                cycle_number=1,
            )

    def test_generate_layout_bad_field(self):
        with pytest.raises(ValueError):
            AVISO_L3_LR_SSH_LAYOUT_V3.generate(
                "/swot_products/l3_karin_nadir/l3_lr_ssh",
                subset=ProductSubset.Expert,
                version="1.0.2",
                cycle_number="1",
            )

    @pytest.mark.parametrize(
        "filters",
        [
            {"version": "1.0.2"},
            {"subset": "Basic"},
            {"cycle_number": slice(532, 533)},
            {"pass_number": [25, 26]},
        ],
    )
    def test_list_swot_lr_l3_layout(
        self,
        l3_lr_ssh_dir_empty_files_layout: Path,
        l3_lr_ssh_dir_empty_files: Path,
        filters: dict[str, tp.Any],
    ):
        db = NetcdfFilesDatabaseSwotLRL3(l3_lr_ssh_dir_empty_files)
        db_no_layout = NetcdfFilesDatabaseSwotLRL3(l3_lr_ssh_dir_empty_files_layout)

        # Duplicates in the sort key (unmix set to False allows this). Need to
        # test the tuples because dataframe order is not ensured
        actual = db.list_files(**filters).drop(columns=["filename"])
        expected = db_no_layout.list_files(**filters).drop(columns=["filename"])

        assert len(expected) > 0
        assert set(map(tuple, actual.to_numpy())) == set(
            map(tuple, expected.to_numpy())
        )

    def test_list_layout_reproc(
        self,
        l3_lr_ssh_dir_empty_files_layout: Path,
    ):
        db = NetcdfFilesDatabaseSwotLRL3(l3_lr_ssh_dir_empty_files_layout)
        files = db.list_files(temporality="FORWARD", version="2.0.1")
        actual_half_orbits = sorted(
            [tuple(x) for x in files[["cycle_number", "pass_number"]].to_numpy()]
        )
        assert actual_half_orbits == [(10, 532)]

    def test_list_layout_reproc_incompatibility(
        self,
        l3_lr_ssh_dir_empty_files_layout: Path,
    ):
        db = NetcdfFilesDatabaseSwotLRL3(l3_lr_ssh_dir_empty_files_layout)
        with pytest.raises(LayoutMismatchError, match="layout-specific"):
            db.list_files(temporality="FORWARD")

    def test_list_layout_reproc_warning(
        self,
        l3_lr_ssh_dir_empty_files_layout: Path,
    ):
        db = NetcdfFilesDatabaseSwotLRL3(
            l3_lr_ssh_dir_empty_files_layout, enable_layouts=False
        )

        expected = db.list_files(sort=True)
        with pytest.warns(UserWarning, match="layout-specific"):
            actual = db.list_files(sort=True, temporality="FORWARD")
        assert expected.equals(actual)


def test_subsets(l3_lr_ssh_dir: Path):
    db = NetcdfFilesDatabaseSwotLRL3(l3_lr_ssh_dir)
    expected = [
        {
            "subset": ProductSubset.Unsmoothed,
            "version": "2.0.1",
        },
        {
            "subset": ProductSubset.Expert,
            "version": "2.0.1",
        },
        {
            "subset": ProductSubset.Expert,
            "version": "1.0.2",
        },
        {
            "subset": ProductSubset.Basic,
            "version": "1.0.2",
        },
    ]

    with pytest.warns(PerformanceWarning):
        assert len(db.subsets) == len(expected)
        assert all([subset in expected for subset in db.subsets])


class TestHalfOrbitMixin:

    def test_subsets_mixed(self, l3_lr_ssh_dir: Path):
        db = NetcdfFilesDatabaseSwotLRL3(l3_lr_ssh_dir)
        with pytest.raises(ValueError, match="unmixed"):
            db.half_orbit_range()

    def test_half_orbit_range(self, l3_lr_ssh_dir: Path):
        db = NetcdfFilesDatabaseSwotLRL3(l3_lr_ssh_dir)
        assert db.half_orbit_range(subset="Basic") == ((531, 25), (532, 26))

    def test_half_orbit_range_filtered(self, l3_lr_ssh_dir: Path):
        db = NetcdfFilesDatabaseSwotLRL3(l3_lr_ssh_dir)
        assert db.half_orbit_range(subset="Basic", pass_number=25) == (
            (531, 25),
            (532, 25),
        )

    def test_half_orbit_range_phase(self, l3_lr_ssh_dir: Path):
        db = NetcdfFilesDatabaseSwotLRL3(l3_lr_ssh_dir)
        assert db.half_orbit_range(subset="Basic", phase="CALVAL") == (
            (531, 25),
            (532, 26),
        )

    def test_half_orbit_range_phase_empty(self, l3_lr_ssh_dir: Path):
        db = NetcdfFilesDatabaseSwotLRL3(l3_lr_ssh_dir)
        assert db.half_orbit_range(subset="Basic", phase="SCIENCE") is None

    def test_half_orbit_range_phase_layout(
        self, l3_lr_ssh_dir_empty_files_layout: Path
    ):
        db = NetcdfFilesDatabaseSwotLRL3(l3_lr_ssh_dir_empty_files_layout)
        assert db.half_orbit_range(subset="Basic", phase="CALVAL") == (
            (531, 25),
            (532, 26),
        )

    def test_temporal_coverage(self, l3_lr_ssh_dir: Path):
        db = NetcdfFilesDatabaseSwotLRL3(l3_lr_ssh_dir)
        assert db.time_coverage(subset="Unsmoothed") == Period(
            np.datetime64("2024-01-25T02:53:52"), np.datetime64("2024-01-25T03:44:38")
        )

    def test_temporal_coverage_layout(self, l3_lr_ssh_dir_empty_files_layout: Path):
        db = NetcdfFilesDatabaseSwotLRL3(l3_lr_ssh_dir_empty_files_layout)
        assert db.time_coverage(subset="Unsmoothed") == Period(
            np.datetime64("2024-01-25T02:53:52"), np.datetime64("2024-01-25T03:44:38")
        )

    def test_cycle_range(self, l3_lr_ssh_dir: Path):
        db = NetcdfFilesDatabaseSwotLRL3(l3_lr_ssh_dir)
        with pytest.warns(PerformanceWarning):
            assert db.cycle_range(subset="Expert") == (532, 533)

    def test_cycle_range_layout(self, l3_lr_ssh_dir_empty_files_layout: Path):
        db = NetcdfFilesDatabaseSwotLRL3(l3_lr_ssh_dir_empty_files_layout)
        assert db.cycle_range(subset="Basic") == (531, 532)
