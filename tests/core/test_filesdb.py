from __future__ import annotations

import re
import sys
import typing as tp
from contextlib import nullcontext
from pathlib import Path

import dask
import fsspec.implementations.memory as fs_mem
import numpy as np
import pandas as pda
import pytest
import xarray as xr

from fcollections.core import (
    Deduplicator,
    FileNameConvention,
    FileNameField,
    FileNameFieldDatetime,
    FileNameFieldInteger,
    FileNameFieldString,
    FilesDatabase,
    IFilesReader,
    IFilterBuilder,
    Layout,
    LayoutMismatchError,
    NotExistingPathError,
    SubsetsUnmixer,
)

if tp.TYPE_CHECKING:
    import fsspec


class FileNameConventionTest(FileNameConvention):

    def __init__(self):
        super().__init__(
            regex=re.compile(r"a_file_(?P<a_number>\d{3})_(?P<time>\d{8})"),
            fields=[
                FileNameFieldDatetime("time", "%Y%m%d"),
                FileNameFieldInteger("a_number"),
            ],
        )


class FilesDatabaseTestInconsistentDeduplicator(FilesDatabase):
    layouts = [Layout([FileNameConventionTest()])]
    deduplicator = Deduplicator(("a1",), ("a2",))


class ReaderStub(IFilesReader):

    def read(
        self,
        files: list[str],
        fs: fsspec.AbstractFileSystem,
        selected_variables: list[str] | None = None,
        factor: int = 1,
        stack: bool = True,
    ) -> xr.Dataset:
        ds = xr.Dataset(
            data_vars=dict(a=("dim_0", np.ones(2)), b=("dim_0", np.ones(2)))
        )

        if selected_variables is not None:
            ds = ds[selected_variables]
        return ds * factor

    def metadata(self, files, fs=None):
        return f"metadata_from_reader_{fs.protocol}"


class FilesDatabaseTestNoUnmixer(FilesDatabase):
    layouts = [
        Layout([FileNameConventionTest()]),
        Layout(
            [
                FileNameConvention(
                    regex=re.compile(r"^a_(?P<a_number>\d{3})$"),
                    fields=[FileNameFieldInteger("a_number")],
                ),
                FileNameConvention(
                    regex=re.compile(r"^(?P<b_string>foo|bar)$"),
                    fields=[FileNameFieldString("b_string")],
                ),
                FileNameConventionTest(),
            ]
        ),
    ]
    reader = ReaderStub()


class FilesDatabaseTest(FilesDatabaseTestNoUnmixer):
    unmixer = SubsetsUnmixer(("a_number",))


class ModuloFilterBuilder(IFilterBuilder):

    @classmethod
    def build_predicate(
        cls, record_mapping: dict[str, int], b_number: int
    ) -> tp.Callable:
        index = record_mapping["a_number"]

        def _predicate(record: tuple[tp.Any, ...]) -> bool:
            return record[index] % b_number == 0

        return _predicate

    @classmethod
    def build_filter(cls, *args):
        raise NotImplementedError()

    @classmethod
    def parameter(cls) -> FileNameField:
        return FileNameFieldInteger("b_number")


class RangeFilterBuilder(IFilterBuilder):

    @classmethod
    def build_filter(cls, c_number: int) -> dict[str, list[int]]:
        return {"a_number": list(range(0, 100, c_number))}

    @classmethod
    def build_predicate(
        cls, record_mapping: dict[str, int], _number: int
    ) -> tp.Callable:
        raise NotImplementedError()

    @classmethod
    def parameter(cls) -> FileNameField:
        return FileNameFieldInteger("c_number")


class FilesDatabaseTestPredicate(FilesDatabaseTestNoUnmixer):
    filter_builders = [ModuloFilterBuilder, RangeFilterBuilder]


def test_bad_path():
    with pytest.raises(NotExistingPathError):
        FilesDatabaseTest(path="bad_path")


@pytest.fixture
def df_with_duplicates() -> pda.DataFrame:
    return pda.DataFrame.from_records(
        [
            (1, 2, "v1", "Expert", 20250302),
            (1, 2, "v2", "Expert", 20250302),
            (1, 2, "v1", "Unsmoothed", 20250302),
            (1, 2, "v1", "Unsmoothed", 20250303),
            (1, 2, "v1", "Unsmoothed", 20250304),
            (1, 3, "v1", "Expert", 20250302),
            (1, 3, "v1", "Unsmoothed", 20250303),
            (1, 3, "v1", "Unsmoothed", 20250304),
        ],
        columns=[
            "cycle_number",
            "pass_number",
            "version",
            "product",
            "production_date",
        ],
    ).sample(frac=1, random_state=4)


def test_deduplicator_inconsistent(tmpdir: Path):
    with pytest.raises(ValueError, match="Deduplicator"):
        FilesDatabaseTestInconsistentDeduplicator(tmpdir)


def test_deduplication(df_with_duplicates: pda.DataFrame):
    deduplicator = Deduplicator(
        auto_pick_last=("version", "production_date"),
        unique=("cycle_number", "pass_number"),
    )

    # Deduplication will remove the duplicate found in the Expert subset
    df_no_duplicates = deduplicator(
        df_with_duplicates[df_with_duplicates["product"] == "Expert"].copy()
    )
    assert (
        df_with_duplicates.iloc[[4, 5]].reset_index(drop=True).equals(df_no_duplicates)
    )


def test_deduplicator_empty():
    deduplicator = Deduplicator(
        auto_pick_last=("version", "production_date"),
        unique=("cycle_number", "pass_number"),
    )
    df = deduplicator(
        pda.DataFrame(
            columns=("cycle_number", "pass_number", "version", "production_date")
        )
    )
    assert len(df) == 0


def test_unmixer_inconsistent(tmpdir: Path):
    with pytest.raises(ValueError, match="are not partitioning"):
        SubsetsUnmixer(("a1",), ("a2",))


@pytest.mark.parametrize(
    "auto_pick, message",
    [
        (("version",), "fixed manually: {'product': ['Unsmoothed', 'Expert']}"),
        (("product",), "fixed manually: {'version': ['v2', 'v1']}"),
    ],
)
def test_unmixing_failed(
    df_with_duplicates: pda.DataFrame, auto_pick: tuple[str], message: str
):
    unmixer = SubsetsUnmixer(
        partition_keys=("version", "product"), auto_pick_last=auto_pick
    )

    with pytest.raises(ValueError) as exc_info:
        # Removing the duplicates will show that we have mixed dataset from
        # different products. An exception will be raised to show that we don't
        # tolerate non-unique values for columns not handled in the
        # deduplication
        unmixer(df_with_duplicates)
        assert message in exc_info.value


def test_unmixing_empty():
    unmixer = SubsetsUnmixer(partition_keys=("version", "product"))
    assert len(unmixer(pda.DataFrame(columns=("version", "product")))) == 0


@pytest.fixture(scope="session")
def subsets() -> list[dict[str, str]]:
    return [
        {"version": "v1", "product": "B"},
        {"version": "v1", "product": "C"},
        {"version": "v2", "product": "A"},
        {"version": "v2", "product": "B"},
    ]


@pytest.mark.parametrize(
    "context, auto_pick, subset_filters, expected",
    [
        (nullcontext(), ("version", "product"), {}, {"version": "v2", "product": "B"}),
        (nullcontext(), ("product", "version"), {}, {"version": "v1", "product": "C"}),
        (pytest.raises(ValueError, match="not be unmixed"), ("version",), {}, None),
        (
            nullcontext(),
            ("version",),
            {"product": "B"},
            {"version": "v2", "product": "B"},
        ),
    ],
)
def test_unmixing_auto_pick_subset(
    subsets: list[dict[str, str]],
    context,
    auto_pick: tuple[str, ...],
    subset_filters: dict[str, str],
    expected: dict[str, str],
):

    unmixer = SubsetsUnmixer(
        partition_keys=("version", "product"), auto_pick_last=auto_pick
    )

    with context:
        subset = unmixer.pick_subset(subsets, **subset_filters)
        assert subset == expected


def test_unmixing_auto_pick_subset_no_unmix(subsets: list[dict[str, str]]):

    unmixer = SubsetsUnmixer(
        partition_keys=("version", "product"), auto_pick_last=tuple()
    )
    assert unmixer.pick_subset(subsets[:1]) == subsets[0]


@pytest.mark.parametrize(
    "auto_pick, group_names",
    [
        (("product", "version"), ("v1", "Unsmoothed")),
        (("version", "product"), ("v2", "Expert")),
    ],
)
def test_unmixing_auto_pick_dataframe(
    df_with_duplicates: pda.DataFrame,
    auto_pick: tuple[str, str],
    group_names: tuple[str, str],
):
    unmixer = SubsetsUnmixer(
        partition_keys=("version", "product"), auto_pick_last=auto_pick
    )
    df = unmixer(df_with_duplicates)
    subset = (df_with_duplicates["version"] == group_names[0]) & (
        df_with_duplicates["product"] == group_names[1]
    )
    assert df.equals(df_with_duplicates[subset])


def test_unmixing_manual_pick(df_with_duplicates: pda.DataFrame):
    unmixer = SubsetsUnmixer(
        partition_keys=("version", "product"), auto_pick_last=("version",)
    )
    df = unmixer(df_with_duplicates[df_with_duplicates["product"] == "Expert"].copy())
    subset = (df_with_duplicates["version"] == "v2") & (
        df_with_duplicates["product"] == "Expert"
    )
    assert df.equals(df_with_duplicates[subset])


@pytest.fixture(scope="session")
def db_with_files() -> FilesDatabaseTest:
    fs = fs_mem.MemoryFileSystem()
    fs.touch("flat/a_file_001_20250101.nc")
    fs.touch("flat/a_file_002_20250101.nc")
    db = FilesDatabaseTest(path="/flat", fs=fs)
    return db


def test_metadata_nominal(db_with_files: FilesDatabaseTest):
    assert db_with_files.variables_info(a_number=1) == "metadata_from_reader_memory"


def test_metadata_ambiguous(db_with_files: FilesDatabaseTest):
    with pytest.raises(ValueError):
        db_with_files.variables_info()


def test_metadata_no_files(tmp_path: Path):
    db = FilesDatabaseTest(path=tmp_path)
    with pytest.warns(UserWarning):
        metadata = db.variables_info()
        assert metadata is None


def test_metadata_wrong_filters(tmp_path: Path):
    db = FilesDatabaseTest(path=tmp_path)
    with pytest.raises(TypeError):
        # time is a valid filter in other method but not in subset unmixing
        db.variables_info(time=("20220102", "20220103"))


def test_metadata_wrong_filters(tmp_path: Path):
    db = FilesDatabaseTestNoUnmixer(path=tmp_path)
    with pytest.raises(TypeError):
        # x is not a valid filter
        db.variables_info(x=("20220102", "20220103"))


def test_list_files(db_with_files: FilesDatabaseTest):
    expected = pda.DataFrame(
        [
            (np.datetime64("2025-01-01", "us"), 1, "/flat/a_file_001_20250101.nc"),
            (np.datetime64("2025-01-01", "us"), 2, "/flat/a_file_002_20250101.nc"),
        ],
        columns=["time", "a_number", "filename"],
    )
    assert expected.equals(db_with_files.list_files())


def test_list_files_filter(db_with_files: FilesDatabaseTest):
    expected = pda.DataFrame(
        [(np.datetime64("2025-01-01", "us"), 2, "/flat/a_file_002_20250101.nc")],
        columns=["time", "a_number", "filename"],
    )
    assert expected.equals(db_with_files.list_files(a_number=2))


def test_list_files_wrong_filter(db_with_files: FilesDatabaseTest):
    with pytest.raises(ValueError):
        db_with_files.list_files(x=1)


@pytest.fixture(scope="session")
def db_predicate_converter() -> FilesDatabaseTestPredicate:
    fs = fs_mem.MemoryFileSystem()
    fs.touch("predicate/a_file_001_20250101.nc")
    fs.touch("predicate/a_file_002_20250101.nc")
    fs.touch("predicate/a_file_003_20250101.nc")
    fs.touch("predicate/a_file_004_20250101.nc")
    db = FilesDatabaseTestPredicate(path="/predicate", fs=fs)
    return db


@pytest.mark.parametrize(
    "filters", [dict(b_number=2), dict(c_number=2)], ids=["predicate", "converter"]
)
def test_list_files_filter_builders(
    db_with_files: FilesDatabaseTest,
    db_predicate_converter: FilesDatabaseTestPredicate,
    filters: dict[str, int],
):
    expected = pda.DataFrame(
        [
            (np.datetime64("2025-01-01", "us"), 2, "/predicate/a_file_002_20250101.nc"),
            (np.datetime64("2025-01-01", "us"), 4, "/predicate/a_file_004_20250101.nc"),
        ],
        columns=["time", "a_number", "filename"],
    )

    with pytest.raises(ValueError):
        # Predicate parameter is unknown in DB not setup properly
        assert db_with_files.list_files(**filters)

    # We should have applied a 'modulo' filter using the b_number argument
    assert expected.equals(db_predicate_converter.list_files(**filters))

    # Auto predicate will not be built
    assert expected.equals(db_predicate_converter.list_files(a_number=[2, 4]))


def test_list_files_filter_builders_error(
    db_predicate_converter: FilesDatabaseTestPredicate,
):
    with pytest.raises(ValueError, match="Incompatible"):
        db_predicate_converter.list_files(a_number=[2, 4], c_number=2)


def test_query_empty(db_with_files: FilesDatabaseTest):
    assert db_with_files.query(a_number=10) is None


@pytest.mark.parametrize(
    "parameter, value",
    [("c_number", 10), ("unmix", False), ("deduplicate", False), ("sort", False)],
)
def test_query_wrong_parameter(
    db_with_files: FilesDatabaseTest, parameter: str, value: int | bool
):
    with pytest.raises(ValueError):
        db_with_files.query(**{parameter: value})


def test_query_mixed(db_with_files: FilesDatabaseTest):
    with pytest.raises(ValueError):
        # unmix defaults to True -> mixed subsets will trigger an error
        db_with_files.query()


def test_query(db_with_files: FilesDatabaseTest):
    ds = db_with_files.query(a_number=2)
    assert set(ds) == {"a", "b"}
    assert all(ds["a"].values == [1.0, 1.0])
    assert all(ds["b"].values == [1.0, 1.0])


def test_query_selected_variables(db_with_files: FilesDatabaseTest):
    ds = db_with_files.query(a_number=2, selected_variables=["a"])
    assert set(ds) == {"a"}
    assert all(ds["a"].values == [1.0, 1.0])


def test_query_reader_arg(db_with_files: FilesDatabaseTest):
    ds = db_with_files.query(a_number=2, factor=2)
    assert set(ds) == {"a", "b"}
    assert all(ds["a"].values == [2.0, 2.0])
    assert all(ds["b"].values == [2.0, 2.0])


class FilesDatabaseTestBadMetadataInjection(FilesDatabaseTest):
    metadata_injection = {"foo": ("dim_0",)}


def test_query_metadata_injection_unknown_field():
    with pytest.raises(ValueError, match="Metadata Injection"):
        FilesDatabaseTestBadMetadataInjection("path")


class FilesDatabaseTestBadDim(FilesDatabaseTest):
    metadata_injection = {"a_number": ("dim_0",)}


@pytest.fixture(scope="session")
def db_bad_dim() -> FilesDatabaseTestBadDim:
    fs = fs_mem.MemoryFileSystem()
    fs.touch("bad_dim/a_file_001_20250101.nc")
    db = FilesDatabaseTestBadDim(path="/bad_dim", fs=fs)
    return db


def test_query_metadata_injection_unknown_dim(db_bad_dim: FilesDatabaseTestBadDim):
    with pytest.raises(ValueError):
        # Will try to inject a vector of size 1 on a dimension of size 2
        db_bad_dim.query(a_number=1)


class FilesDatabaseTestGoodDim(FilesDatabaseTest):
    metadata_injection = {"a_number": ("dim_1",)}


@pytest.fixture(scope="session")
def db_good_dim() -> FilesDatabaseTestGoodDim:
    fs = fs_mem.MemoryFileSystem()
    fs.touch("good_dim/a_file_001_20250101.nc")
    fs.touch("good_dim/a_file_002_20250101.nc")
    db = FilesDatabaseTestGoodDim(path="/good_dim", fs=fs)
    return db


def test_query_metadata_injection(db_good_dim: FilesDatabaseTestGoodDim):
    # unknown dimensions are created in the dataset
    ds = db_good_dim.query(a_number=2)
    assert set(ds.variables) == {"a", "b", "a_number"}
    assert ds.sizes == {"dim_0": 2, "dim_1": 1}
    assert all(ds.a_number.values == 2)


@pytest.fixture(scope="session")
def db_with_files_bad_layout() -> FilesDatabaseTest:
    fs = fs_mem.MemoryFileSystem()
    fs.touch("/bad_layout/baz/a_001/foo/a_file_001_20250101.nc")
    fs.touch("/bad_layout/baz/a_001/bar/a_file_001_20250101.nc")
    fs.touch("/bad_layout/baz/a_002/bar/a_file_002_20250101.nc")
    db = FilesDatabaseTest(path="/bad_layout", fs=fs)
    return db


@pytest.fixture(scope="session")
def db_with_files_good_layout(
    db_with_files_bad_layout: FilesDatabaseTest,
) -> FilesDatabaseTest:
    fs = db_with_files_bad_layout.fs
    fixed_path = Path(db_with_files_bad_layout.path) / "baz"
    db = FilesDatabaseTest(fixed_path, fs, enable_layouts=True)
    return db


def test_query_bad_layout(db_with_files_bad_layout: FilesDatabaseTest):
    with pytest.raises(LayoutMismatchError):
        db_with_files_bad_layout.query()


def test_query_bad_layout_fallback(
    db_with_files_bad_layout: FilesDatabaseTest,
    db_with_files_good_layout: FilesDatabaseTest,
):
    reference = db_with_files_good_layout.query(a_number=1)
    assert reference is not None

    db = FilesDatabaseTest(
        db_with_files_bad_layout.path, db_with_files_bad_layout.fs, enable_layouts=False
    )
    actual = db.query(a_number=1)

    xr.testing.assert_equal(reference, actual)


def test_query_layout_parameter_not_known(db_with_files_good_layout: FilesDatabaseTest):
    # Low level interface knows of layout filters
    df = db_with_files_good_layout.discoverer.to_dataframe(b_string="foo")
    assert len(df) > 0

    # But higher level interface does not (yet)
    with pytest.raises(ValueError):
        db_with_files_good_layout.query(b_string="foo")


def test_map_no_dask(monkeypatch: pytest.MonkeyPatch, db_with_files: FilesDatabaseTest):
    monkeypatch.setitem(sys.modules, "dask.bag.core", None)
    with pytest.raises(NotImplementedError):
        db_with_files.map(lambda ds, record: None)


def test_map(db_with_files: FilesDatabaseTest):

    def func(ds: xr.Dataset, record: dict[str, tp.Any]):
        return record["a_number"], list(ds.a.values)

    with dask.config.set(scheduler="synchronous"):
        # Use synchronous scheduler to run in sequential and compute proper
        # coverage
        result = db_with_files.map(func, a_number=1).compute()
    assert result == [(1, [1.0, 1.0])]


@pytest.mark.parametrize(
    "parameter, value",
    [("c_number", 10), ("unmix", False), ("deduplicate", False), ("sort", False)],
)
def test_map_wrong_parameter(
    db_with_files: FilesDatabaseTest, parameter: str, value: int | bool
):
    with pytest.raises(ValueError):
        db_with_files.map(lambda x, y: None, **{parameter: value})


def test_map_empty(db_with_files: FilesDatabaseTest):
    assert db_with_files.map(lambda x, y: x, a_number=-1).compute() == []


def test_subsets_no_unmixer(db_with_files_good_layout: FilesDatabaseTest):
    db = FilesDatabaseTestNoUnmixer(
        db_with_files_good_layout.path, db_with_files_good_layout.fs
    )
    assert len(db.subsets) == 0


def test_subsets_empty_dir(db_with_files_good_layout: FilesDatabaseTest):
    db_with_files_good_layout.fs.mkdir("empty")
    db = FilesDatabaseTest("empty", db_with_files_good_layout.fs)
    assert len(db.subsets) == 0


def test_subsets(db_with_files_good_layout: FilesDatabaseTest):
    assert len(db_with_files_good_layout.subsets) == 2
    assert all(
        [
            subset in [{"a_number": 1}, {"a_number": 2}]
            for subset in db_with_files_good_layout.subsets
        ]
    )


def test_filters_value_full_scan_filter_not_in_layout(
    db_with_files_good_layout: FilesDatabaseTest,
):
    with pytest.warns(UserWarning, match="intermediate"):
        values = db_with_files_good_layout.filter_values("time", a_number=1)
    assert values == {np.datetime64("2025-01-01")}


def test_filters_value_layouts_disabled_unknown_field(
    db_with_files_bad_layout: FilesDatabaseTest,
):
    db = FilesDatabaseTest(
        db_with_files_bad_layout.path, db_with_files_bad_layout.fs, enable_layouts=False
    )
    with pytest.raises(ValueError, match="Unknown"):
        db.filter_values("b_string", a_number=1)


def test_filters_value_layouts_disabled_full_scan(
    db_with_files_bad_layout: FilesDatabaseTest,
):
    db = FilesDatabaseTest(
        db_with_files_bad_layout.path, db_with_files_bad_layout.fs, enable_layouts=False
    )
    with pytest.warns(UserWarning, match="enabled"):
        values = db.filter_values(
            "a_number",
        )
    assert values == {1, 2}


def test_filters_value_full_scan_flat(db_with_files: FilesDatabaseTest):
    values = db_with_files.filter_values("a_number")
    assert values == {1, 2}


def test_filters_value_full_scan_layouts_mismatch(
    db_with_files_bad_layout: FilesDatabaseTest,
):
    with pytest.raises(LayoutMismatchError):
        db_with_files_bad_layout.filter_values("b_string", a_number=1)


def test_filters_value_layout(db_with_files_good_layout: FilesDatabaseTest):
    values = db_with_files_good_layout.filter_values("b_string", a_number=1)
    assert values == {"foo", "bar"}


def test_filters_value_unknown(db_with_files_good_layout: FilesDatabaseTest):
    with pytest.raises(ValueError, match="Unknown filter"):
        db_with_files_good_layout.filter_values("c_unknown", a_number=1)


def test_filters_value_missing_subset_selection(
    db_with_files_good_layout: FilesDatabaseTest,
):
    with pytest.raises(ValueError, match="heterogenous datasets"):
        db_with_files_good_layout.filter_values("b_string")
