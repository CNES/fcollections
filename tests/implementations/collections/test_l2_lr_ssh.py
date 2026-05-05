import itertools
import typing as tp
from pathlib import Path

import fsspec.implementations.local as fs_loc
import numpy as np
import pytest
import xarray as xr
from utils import brute_force_geographical_selection, extract_box_from_polygon

from fcollections.implementations import (
    AVISO_L2_LR_SSH_LAYOUT,
    L2Version,
    L2VersionField,
    NetcdfFilesDatabaseSwotLRL2,
    ProductLevel,
    ProductSubset,
    StackLevel,
    SwotReaderL2LRSSH,
    Timeliness,
)
from fcollections.time import Period


class TestVersionField:

    def test_version_type(self):
        assert L2VersionField("version").type == L2Version

    @pytest.mark.parametrize(
        "input, sanitized",
        [
            (L2Version(Timeliness.G, "C", 0, 1), L2Version(Timeliness.G, "C", 0, 1)),
            ("PGC0_01", L2Version(Timeliness.G, "C", 0, 1)),
            ("PGC0", L2Version(Timeliness.G, "C", 0)),
            ("PGC?", L2Version(Timeliness.G, "C")),
            ("P?C?", L2Version(baseline="C")),
        ],
    )
    def test_version_field_sanitize(self, input: str | L2Version, sanitized: L2Version):
        field = L2VersionField("version")
        assert field.sanitize(input) == sanitized

    @pytest.mark.parametrize(
        "reference, result",
        [
            (L2Version(Timeliness.G), True),
            (L2Version(None, "C"), True),
            (L2Version(None, None, 0), True),
            (L2Version(None, None, None, 1), True),
            (L2Version(None, "C", None, 1), True),
            (L2Version(Timeliness.I), False),
            (L2Version(None, "A"), False),
            (L2Version(None, None, 1), False),
            (L2Version(None, None, None, 2), False),
            (L2Version(Timeliness.I, "C", 0, 2), False),
        ],
    )
    def test_version_field_test(self, reference, result):
        field = L2VersionField("version")
        tested = L2Version(Timeliness.G, "C", 0, 1)
        assert field.test(reference, tested) == result

    def test_version_field_repr(self):
        assert str(L2Version(Timeliness.G, "C", 0, 1)) == "PGC0_01"
        assert str(L2Version(Timeliness.G, "C", 0)) == "PGC0"
        assert str(L2Version()) == "P???"
        assert str(L2Version(None, "C", 0, 1)) == "P?C0_01"

    def test_version_field_hash(self):
        assert hash(L2Version(Timeliness.G, "C", 0, 1)) == hash("PGC0_01")


class TestReader:

    @pytest.fixture(autouse=True)
    def setup(self):
        try:
            # Reader for test configuration with geo packages available
            from fcollections.implementations.optional import (
                GeoSwotReaderL2LRSSH,
            )

            self.reader = GeoSwotReaderL2LRSSH()
        except ImportError:
            # Fall back reader
            self.reader = SwotReaderL2LRSSH()

    @pytest.mark.parametrize(
        "subset", [ProductSubset.Technical, ProductSubset.Light, ProductSubset.Extended]
    )
    def test_invalid_subset(
        self, subset: ProductSubset, l2_lr_ssh_expert_files: list[str]
    ):
        with pytest.raises(ValueError):
            self.reader.read(subset, l2_lr_ssh_expert_files)

    def test_invalid_stack(self, l2_lr_ssh_expert_files: list[str]):
        with pytest.raises(ValueError):
            self.reader.read(ProductSubset.Expert, l2_lr_ssh_expert_files, stack=True)

    @pytest.mark.parametrize(
        "files, subset",
        [
            ("l2_lr_ssh_basic_files", ProductSubset.Basic),
            ("l2_lr_ssh_expert_files", ProductSubset.Expert),
            ("l2_lr_ssh_unsmoothed_files", ProductSubset.Unsmoothed),
        ],
    )
    def test_all_multi_read_non_stack(
        self, files: list[str], subset: ProductSubset, request: pytest.FixtureRequest
    ):
        """Test nominal reading of Basic and expert files."""
        files = request.getfixturevalue(files)
        ds = self.reader.read(subset, files, stack="NOSTACK")
        index = 0
        for file in files:
            ds_pass = self.reader.read(subset, [file])
            reference_pass = ds.isel(
                num_lines=slice(index, index + ds_pass.num_lines.size)
            )
            xr.testing.assert_equal(reference_pass, ds_pass)
            index += ds_pass.num_lines.size

    @pytest.mark.parametrize(
        "files, subset, expected_half_orbits",
        [
            (
                "l2_lr_ssh_basic_files_unique",
                ProductSubset.Basic,
                [
                    (482, 11),
                    (482, 12),
                    (482, 25),
                    (482, 26),
                    (483, 25),
                    (483, 26),
                    (546, 11),
                ],
            ),
            (
                "l2_lr_ssh_expert_files_unique",
                ProductSubset.Expert,
                [(6, 11), (6, 532), (6, 533), (7, 532), (7, 533), (546, 18)],
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
                "l2_lr_ssh_basic_files_unique",
                ProductSubset.Basic,
                [
                    (482, 11),
                    (482, 12),
                    (482, 25),
                    (482, 26),
                    (483, 25),
                    (483, 26),
                    (546, 11),
                ],
            ),
            (
                "l2_lr_ssh_expert_files_unique",
                ProductSubset.Expert,
                [(6, 11), (6, 532), (6, 533), (7, 532), (7, 533), (546, 18)],
            ),
        ],
    )
    def test_expert_multi_read_stack_cycles_passes(
        self,
        files: list[list[str]],
        subset: ProductSubset,
        expected_half_orbits: list[int, int],
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
        "files", ["l2_lr_ssh_basic_files", "l2_lr_ssh_expert_files"]
    )
    @pytest.mark.parametrize("stack", StackLevel)
    @pytest.mark.with_geo_packages
    def test_expert_bbox(
        self, files: str, stack: StackLevel, request: pytest.FixtureRequest
    ):
        pass_number = 25
        bbox = extract_box_from_polygon(pass_number)
        files = request.getfixturevalue(files)
        reference = self.reader.read(
            subset=ProductSubset.Basic, files=files, stack=stack
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
            subset=ProductSubset.Basic, files=files, stack=stack, bbox=bbox
        ).compute()
        xr.testing.assert_equal(reference_cropped, actual_cropped)

    @pytest.mark.without_geo_packages
    def test_expert_bbox_disabled(self, l2_lr_ssh_basic_files: list[str]):
        with pytest.raises(TypeError):
            self.reader.read(
                ProductSubset.Basic, l2_lr_ssh_basic_files, bbox=(-180, -90, 180, 90)
            )

    def test_expert_selected_variables(self, l2_lr_ssh_expert_files: list[str]):
        """Tests we can select the read variables."""
        selected = ["longitude", "ssha_karin_2"]
        dropped = ["cross_track_distance", "latitude"]

        ds = self.reader.read(
            ProductSubset.Expert,
            l2_lr_ssh_expert_files[:1],
            selected_variables=selected,
        )

        assert all([s in ds for s in selected])
        assert all([s not in ds for s in dropped])

    def test_unsmoothed_read(
        self,
        l2_lr_ssh_unsmoothed_files: list[str],
        l2_lr_ssh_unsmoothed_dataset: xr.Dataset,
    ):
        ds = self.reader.read(
            ProductSubset.Unsmoothed,
            l2_lr_ssh_unsmoothed_files[:1],
            left_swath=True,
            right_swath=False,
        ).compute()
        xr.testing.assert_equal(
            ds.drop_vars(["cycle_number", "pass_number"]), l2_lr_ssh_unsmoothed_dataset
        )
        assert np.all(ds["cycle_number"] == 10)
        assert np.all(ds["pass_number"] == 4)

        ds = self.reader.read(
            ProductSubset.Unsmoothed,
            l2_lr_ssh_unsmoothed_files[:1],
            left_swath=False,
            right_swath=True,
        ).compute()
        xr.testing.assert_equal(
            ds.drop_vars(["cycle_number", "pass_number"]), l2_lr_ssh_unsmoothed_dataset
        )
        assert np.all(ds["cycle_number"] == 10)
        assert np.all(ds["pass_number"] == 4)

    def test_unsmoothed_read_complete_swath(
        self, l2_lr_ssh_unsmoothed_files: list[str]
    ):
        # cannot read complete swath yet
        reference = self.reader.read(
            ProductSubset.Unsmoothed,
            l2_lr_ssh_unsmoothed_files,
            left_swath=True,
            right_swath=False,
        )
        with pytest.warns(UserWarning):
            ds = self.reader.read(
                ProductSubset.Unsmoothed,
                l2_lr_ssh_unsmoothed_files,
                left_swath=True,
                right_swath=True,
            )
        xr.testing.assert_equal(reference, ds)

    def test_unsmoothed_read_nothing(self, l2_lr_ssh_unsmoothed_files: list[str]):
        reference = self.reader.read(
            ProductSubset.Unsmoothed,
            l2_lr_ssh_unsmoothed_files,
            left_swath=True,
            right_swath=False,
        )

        with pytest.warns(UserWarning):
            ds = self.reader.read(
                ProductSubset.Unsmoothed,
                l2_lr_ssh_unsmoothed_files,
                left_swath=False,
                right_swath=False,
            )

        xr.testing.assert_equal(reference, ds)

    def test_unsmoothed_selected_variables(self, l2_lr_ssh_unsmoothed_files: list[str]):
        selected = ["longitude", "ssha_karin_2"]
        dropped = ["cross_track_distance", "latitude"]

        ds = self.reader.read(
            ProductSubset.Unsmoothed,
            l2_lr_ssh_unsmoothed_files,
            selected_variables=selected,
        )

        assert all([s in ds for s in selected])
        assert all([s not in ds for s in dropped])

    def test_metadata(self, l2_lr_ssh_unsmoothed_files: list[str]):
        # Light test to check that swot reader effectively calls the sub-reader
        # method
        metadata = self.reader.metadata(
            l2_lr_ssh_unsmoothed_files[0], fs_loc.LocalFileSystem()
        )
        assert metadata is not None

    @pytest.mark.parametrize(
        "subset",
        [
            ProductSubset.Basic,
            ProductSubset.Expert,
            ProductSubset.Unsmoothed,
            ProductSubset.WindWave,
        ],
    )
    def test_empty(self, subset: ProductSubset):
        with pytest.raises(ValueError):
            assert self.reader.read(subset, []).equals(xr.Dataset())


class TestListing:

    @pytest.mark.parametrize(
        "query, half_orbits",
        [
            (
                {},
                [
                    (577, 18),
                    (546, 11),
                    (546, 18),
                    (577, 11),
                    (577, 18),
                    (6, 11),
                    (6, 532),
                    (6, 533),
                    (7, 532),
                    (7, 533),
                    (482, 11),
                    (482, 12),
                    (482, 25),
                    (482, 26),
                    (483, 25),
                    (483, 26),
                    (546, 18),
                    (546, 18),
                    (10, 4),
                ],
            ),
            ({"cycle_number": [577]}, [(577, 18), (577, 11), (577, 18)]),
            ({"pass_number": [11]}, [(546, 11), (577, 11), (6, 11), (482, 11)]),
            (
                {"time": (np.datetime64("2023-06-08"), np.datetime64("2023-06-09"))},
                [(546, 11)],
            ),
            (
                {"subset": ProductSubset.Basic},
                [
                    (546, 11),
                    (482, 11),
                    (482, 12),
                    (482, 25),
                    (482, 26),
                    (483, 25),
                    (483, 26),
                ],
            ),
            ({"subset": ProductSubset.Unsmoothed}, [(10, 4)]),
            ({"version": L2Version(baseline="A")}, [(577, 18)]),
        ],
    )
    def test_list(
        self,
        l2_lr_ssh_dir_empty_files: Path,
        query: dict[str, tp.Any],
        half_orbits: list[tuple[int, int]],
    ):

        db = NetcdfFilesDatabaseSwotLRL2(l2_lr_ssh_dir_empty_files)

        files = db.list_files(**query, sort=True)
        actual_half_orbits = sorted(
            [tuple(x) for x in files[["cycle_number", "pass_number"]].to_numpy()]
        )
        assert actual_half_orbits == sorted(half_orbits)

    @pytest.mark.parametrize(
        "bbox, pass_numbers",
        [
            ((-180, -90, 180, 90), [25, 26, 532]),
            ((0, -90, 360, 90), [25, 26, 532]),
        ],
    )
    @pytest.mark.with_geo_packages
    def test_list_bbox_global(
        self,
        l2_lr_ssh_dir_empty_files: Path,
        bbox: tuple[int, int, int, int],
        pass_numbers: list[int],
    ):

        db = NetcdfFilesDatabaseSwotLRL2(l2_lr_ssh_dir_empty_files)

        df_pass_number = db.list_files(pass_number=pass_numbers, sort=True)
        assert not df_pass_number.empty
        df_bbox = db.list_files(bbox=bbox, sort=True)
        assert df_bbox.equals(df_pass_number)

    @pytest.mark.parametrize(
        "pass_number, latitude, pass_numbers",
        [
            (25, 0, [25]),
            (26, 0, [26]),
            # Taking a box at highest latitude will also covers the next pass
            (25, 90, [25, 26]),
        ],
    )
    @pytest.mark.with_geo_packages
    def test_list_bbox(
        self,
        l2_lr_ssh_dir_empty_files: Path,
        pass_number: int,
        latitude: float,
        pass_numbers: list[int],
    ):

        db = NetcdfFilesDatabaseSwotLRL2(l2_lr_ssh_dir_empty_files)

        # Call the box extraction inside the method, else the import will fail
        # We need pytest to setup the geometries fixture before this call can
        # work
        bbox = extract_box_from_polygon(pass_number, latitude=latitude)
        df_pass_number = db.list_files(pass_number=pass_numbers, sort=True)
        assert not df_pass_number.empty
        df_bbox = db.list_files(bbox=bbox, sort=True)
        assert df_bbox.equals(df_pass_number)

    @pytest.mark.without_geo_packages
    def test_list_bbox_disabled(self, l2_lr_ssh_dir_empty_files: Path):
        db = NetcdfFilesDatabaseSwotLRL2(l2_lr_ssh_dir_empty_files)
        with pytest.raises(ValueError):
            db.list_files(bbox=(-180, -90, 180, 90))

    def test_list_deduplicate(self, l2_lr_ssh_dir_empty_files: Path):
        db = NetcdfFilesDatabaseSwotLRL2(l2_lr_ssh_dir_empty_files)

        # check there are some duplicates
        files = db.list_files(subset="Expert")
        half_orbits_duplicates = [
            tuple(x) for x in files[["cycle_number", "pass_number"]].to_numpy()
        ]
        assert len(set(half_orbits_duplicates)) < len(half_orbits_duplicates)

        # Duplicates have been removed
        files = db.list_files(deduplicate=True, unmix=True, sort=True, subset="Expert")
        half_orbits_no_duplicates = [
            tuple(x) for x in files[["cycle_number", "pass_number"]].to_numpy()
        ]

        assert sorted(half_orbits_no_duplicates) == sorted(set(half_orbits_duplicates))

    def test_list_deduplicate_autopick(self, l2_lr_ssh_dir_empty_files: Path):
        # Test we actually have the latest versions
        db = NetcdfFilesDatabaseSwotLRL2(l2_lr_ssh_dir_empty_files)
        files_metadata = db.list_files(cycle_number=577, pass_number=18)
        assert {str(x) for x in files_metadata["version"]} == {"PIA2_02", "PIB0_01"}

        # Autopick version
        files_metadata = db.list_files(
            cycle_number=577, pass_number=18, deduplicate=True
        )
        assert len(files_metadata) == 1
        assert str(files_metadata["version"][0]) == "PIB0_01"

    def test_list_unmix_error(self, l2_lr_ssh_dir_empty_files: Path):
        # When duplication removal is enabled, the subset is expected to be unique
        # after duplication. If it is not filtered, it can trigger an error
        db = NetcdfFilesDatabaseSwotLRL2(l2_lr_ssh_dir_empty_files)

        with pytest.raises(ValueError):
            db.list_files(unmix=True)

    def test_list_unknown_filter(self, l2_lr_ssh_dir_empty_files: Path):
        db = NetcdfFilesDatabaseSwotLRL2(l2_lr_ssh_dir_empty_files)
        with pytest.raises(ValueError):
            db.list_files(bad_arg="bad_arg")

    def test_list_empty_result(self, l2_lr_ssh_dir_empty_files: Path):
        db = NetcdfFilesDatabaseSwotLRL2(l2_lr_ssh_dir_empty_files)
        assert db.list_files(cycle_number=399).empty


class TestQuery:

    def test_query_unknown_filter(self, l2_lr_ssh_dir_empty_files: Path):
        db = NetcdfFilesDatabaseSwotLRL2(l2_lr_ssh_dir_empty_files)
        with pytest.raises(ValueError):
            db.query(bad_arg="bad_arg")

    def test_query_empty_result(self, l2_lr_ssh_dir_empty_files: Path):
        db = NetcdfFilesDatabaseSwotLRL2(l2_lr_ssh_dir_empty_files)
        assert db.query(cycle_number=399) is None

    @pytest.mark.without_geo_packages
    def test_query_bbox_disabled(self, l2_lr_ssh_dir_empty_files: Path):
        db = NetcdfFilesDatabaseSwotLRL2(l2_lr_ssh_dir_empty_files)
        with pytest.raises(ValueError):
            bbox = (260, 10, 300, 40)
            ds = db.query(subset="Basic", bbox=bbox)


class TestLayout:

    def test_generate_layout(self):
        t0 = np.datetime64("2024-09-25T00")
        t1 = np.datetime64("2024-09-30T00")
        path = AVISO_L2_LR_SSH_LAYOUT.generate(
            "/swot_products/l2_karin/l2_lr_ssh",
            subset=ProductSubset.Expert,
            version=L2Version(Timeliness.G, "C", 0, 1),
            cycle_number=1,
            pass_number=15,
            time=Period(t0, t1),
            level=ProductLevel.L2,
        )

        assert (
            path
            == "/swot_products/l2_karin/l2_lr_ssh/PGC0/Expert/cycle_001/SWOT_L2_LR_SSH_Expert_001_015_20240925T000000_20240930T000000_PGC0_01.nc"
        )

    def test_generate_layout_missing_field(self):
        with pytest.raises(ValueError):
            AVISO_L2_LR_SSH_LAYOUT.generate(
                "/swot_products/l3_karin_nadir/l3_lr_ssh",
                subset=ProductSubset.Expert,
                cycle_number=1,
            )

    def test_generate_layout_bad_field(self):
        with pytest.raises(ValueError):
            t0 = np.datetime64("2024-09-25T00")
            t1 = np.datetime64("2024-09-30T00")
            AVISO_L2_LR_SSH_LAYOUT.generate(
                "/swot_products/l3_karin_nadir/l3_lr_ssh",
                subset=ProductSubset.Expert,
                version=L2Version(Timeliness.G, "C", 0, 1),
                cycle_number="1",
                pass_number=15,
                time=Period(t0, t1),
                level=ProductLevel.L2,
            )

    @pytest.mark.parametrize(
        "filters",
        [
            {"version": L2Version(Timeliness.I, "C", 2, 1)},
            {"version": "P?C2"},
            {"version": "PIC2_01"},
            {"subset": "Basic"},
            {"cycle_number": slice(480, 490)},
            {"pass_number": [10, 11]},
        ],
    )
    def test_list_swot_lr_l2_layout(
        self,
        l2_lr_ssh_dir_empty_files: Path,
        l2_lr_ssh_dir_empty_files_layout: Path,
        filters: dict[str, tp.Any],
    ):
        db = NetcdfFilesDatabaseSwotLRL2(l2_lr_ssh_dir_empty_files_layout)
        db_no_layout = NetcdfFilesDatabaseSwotLRL2(l2_lr_ssh_dir_empty_files)

        # Duplicates in the sort key (unmix set to False allows this). Need to
        # test the tuples because dataframe order is not ensured
        actual = db.list_files(**filters).drop(columns=["filename"])
        expected = db_no_layout.list_files(**filters).drop(columns=["filename"])

        assert len(expected) > 0
        assert set(map(tuple, actual.to_numpy())) == set(
            map(tuple, expected.to_numpy())
        )


def test_subsets(l2_lr_ssh_dir_empty_files_layout: Path):

    expected = [
        {"subset": ProductSubset.Basic},
        {"subset": ProductSubset.Unsmoothed},
        {"subset": ProductSubset.Expert},
    ]

    db = NetcdfFilesDatabaseSwotLRL2(l2_lr_ssh_dir_empty_files_layout)
    assert len(db.subsets) == len(expected)
    assert all([x in expected for x in db.subsets])


def test_subsets_flat(l2_lr_ssh_dir_empty_files: Path):

    expected = [
        {"subset": ProductSubset.Basic},
        {"subset": ProductSubset.Unsmoothed},
        {"subset": ProductSubset.Expert},
    ]

    db = NetcdfFilesDatabaseSwotLRL2(l2_lr_ssh_dir_empty_files)
    assert len(db.subsets) == len(expected)
    assert all([x in expected for x in db.subsets])
