from __future__ import annotations

import re
import sys
import typing as tp
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
    FileNameFieldDatetime,
    FileNameFieldInteger,
    FileNameFieldString,
    FilesDatabase,
    IFilesReader,
    IPredicate,
    Layout,
    LayoutMismatchError,
    NotExistingPathError,
    SubsetsUnmixer,
)

if tp.TYPE_CHECKING:
    from pathlib import Path

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


class FilesDatabaseTestInconsistentUnmixer(FilesDatabase):
    layouts = [Layout([FileNameConventionTest()])]
    unmixer = SubsetsUnmixer(("a1",), ("a2",))


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


class ModuloPredicate(IPredicate):

    def __init__(self, indexes: tuple[int], b_number: int):
        self.index = indexes[0]
        self.b_number = b_number

    def __call__(self, record: tuple[tp.Any, ...]) -> bool:
        return record[self.index] % self.b_number == 0

    @classmethod
    def record_fields(cls) -> tuple[str, ...]:
        return ("a_number",)

    @classmethod
    def parameters(cls) -> tuple[str, ...]:
        return ("b_number",)


class FilesDatabaseTestPredicate(FilesDatabaseTestNoUnmixer):
    predicate_classes = [ModuloPredicate]


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
    with pytest.raises(ValueError, match="Subsets Unmixer"):
        FilesDatabaseTestInconsistentUnmixer(tmpdir)


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


@pytest.mark.parametrize(
    "auto_pick, group_names",
    [
        (("product", "version"), ("v1", "Unsmoothed")),
        (("version", "product"), ("v2", "Expert")),
    ],
)
def test_unmixing_auto_pick(
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


def test_unmixing_callable(df_with_duplicates: pda.DataFrame):
    """Use a callable to transform the columns prior to auto pick."""
    unmixer = SubsetsUnmixer(
        partition_keys=("version", "product"), auto_pick_last=("product", "version")
    )
    df = unmixer(df_with_duplicates)
    subset = (df_with_duplicates["version"] == "v1") & (
        df_with_duplicates["product"] == "Unsmoothed"
    )
    assert df.equals(df_with_duplicates[subset])

    # We reverse the product column sort internally -> selecting Expert instead
    # of Unsmoothed
    unmixer = SubsetsUnmixer(
        partition_keys={
            "version": None,
            "product": lambda x: 1 if x == "Expert" else 0,
        },
        auto_pick_last=("product", "version"),
    )
    df = unmixer(df_with_duplicates)
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
def db_predicate() -> FilesDatabaseTestPredicate:
    fs = fs_mem.MemoryFileSystem()
    fs.touch("predicate/a_file_001_20250101.nc")
    fs.touch("predicate/a_file_002_20250101.nc")
    fs.touch("predicate/a_file_003_20250101.nc")
    fs.touch("predicate/a_file_004_20250101.nc")
    db = FilesDatabaseTestPredicate(path="/predicate", fs=fs)
    return db


def test_list_files_predicate(
    db_with_files: FilesDatabaseTest, db_predicate: FilesDatabaseTestPredicate
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
        assert db_with_files.list_files(b_number=2)

    # We should have applied a 'modulo' filter using the b_number argument
    assert expected.equals(db_predicate.list_files(b_number=2))

    # Auto predicate will not be built
    assert expected.equals(db_predicate.list_files(a_number=[2, 4]))


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
    fs.touch("baz/foo/a_file_001_20250101.nc")
    fs.touch("baz/bar/a_file_002_20250101.nc")
    db = FilesDatabaseTest(path="/", fs=fs)
    return db


def test_query_bad_layout(db_with_files_bad_layout: FilesDatabaseTest):
    with pytest.raises(LayoutMismatchError):
        db_with_files_bad_layout.query()


def test_query_bad_layout_fallback(db_with_files_bad_layout: FilesDatabaseTest):
    fs = db_with_files_bad_layout.fs
    fixed_path = Path(db_with_files_bad_layout.path) / "baz"
    db = FilesDatabaseTest(fixed_path, fs, enable_layouts=True)
    reference = db.query(a_number=1)
    assert reference is not None

    db = FilesDatabaseTest(db_with_files_bad_layout.path, fs, enable_layouts=False)
    actual = db.query(a_number=1)

    xr.testing.assert_equal(reference, actual)


def test_query_layout_parameter_not_known(db_with_files_bad_layout: FilesDatabaseTest):
    fs = db_with_files_bad_layout.fs
    fixed_path = Path(db_with_files_bad_layout.path) / "baz"
    db = FilesDatabaseTest(fixed_path, fs, enable_layouts=True)

    # Low level interface knows of layout filters
    df = db.discoverer.to_dataframe(b_string="foo")
    assert len(df) > 0

    # But higher level interface does not (yet)
    with pytest.raises(ValueError):
        db.query(b_string="foo")


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


def test_map_empty(db_with_files: FileDatabaseTest):
    assert db_with_files.map(lambda x, y: x, a_number=-1).compute() == []
