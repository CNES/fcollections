import re
import socket
import threading
import typing as tp
from contextlib import nullcontext
from enum import Enum, auto
from pathlib import Path

import numpy as np
import pytest
from fsspec.implementations.ftp import FTPFileSystem
from fsspec.implementations.local import LocalFileSystem
from fsspec.implementations.memory import MemoryFileSystem
from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer

from fcollections.core import (
    FileNameConvention,
    FileNameFieldDateDelta,
    FileNameFieldDatetime,
    FileNameFieldEnum,
    FileNameFieldFloat,
    FileNameFieldInteger,
    FileNameFieldISODuration,
    FileNameFieldPeriod,
    FileNameFieldString,
    FileSystemMetadataCollector,
    Layout,
)
from fcollections.core._listing import (
    DirNode,
    FileListingError,
    FileNode,
    LayoutMismatchError,
    LayoutMismatchHandling,
    LayoutVisitor,
    NoLayoutVisitor,
    RecordFilter,
    StandardVisitor,
    VisitResult,
    walk,
)
from fcollections.time import ISODuration, Period


def test_file_node_no_children():
    node = FileNode("foo", {}, 0)
    assert len(node.children()) == 0


def test_dir_node_clear():
    node = DirNode("foo", {"name": "foo"}, MemoryFileSystem(), 0)
    node.children()
    assert node._children is not None
    node.clear()
    assert node._children is None


@pytest.fixture(scope="session")
def filepaths() -> list[str]:
    # The files that we will use to test the listing and filtering using the
    # file names will allow us
    return [
        "root/RED/LR_000/file_000_.25_foo-bar_20230201_RED_20121101_20130705_20010101_PT1S.txt",
        "root/BLUE/LR_001/file_001_.25_foo-bar_20230202_BLUE_20121101_20130705_20010101_PT1S.txt",
        "root/GREEN/LR_002/file_002_.25_foo-bar_20230203_GREEN_20121101_20130705_20010101_PT1S.txt",
        "root/RED/LR_003/file_003_1.75_foo-bar_20230204_RED_20121101_20130705_20010101_PT1S.txt",
        "root/BLUE/LR_004/file_004_1.75_foo-bar_20230205_BLUE_20121101_20130705_20010101_PT1S.txt",
        "root/GREEN/LR_005/file_005_1.75_foo-bar_20230206_GREEN_20121101_20130705_20010101_PT1S.txt",
        "root/RED/HR_006/file_006_5.6_baz_20230207_RED_20221101_20230705_19500101_PT1S.txt",
        "root/BLUE/HR_007/file_007_5.8_baz_20230208_BLUE_20221101_20230705_19500101_PT1S.txt",
        "root/GREEN/HR_008/file_008_7.4_baz_20230209_GREEN_20221101_20230705_19500101_PT1S.txt",
        "root/HR_009/file_009_5.6_baz_20230207_RED_20221101_20230705_19500101_PT1S.txt",
        "root/HR_010/file_010_5.8_baz_20230208_BLUE_20221101_20230705_19500101_PT2S.txt",
        "root/HR_011/file_011_7.4_baz_20230209_GREEN_20221101_20230705_19500101_PT2S.txt",
        "root/HR_011/KO.txt",
        "root/RED/dead_branch",
        "root/HR_011/dead_branch",
        "root/dead_branch",
        "clean/RED/LR_000/file_000_.25_foo-bar_20230201_RED_20121101_20130705_20010101_PT1S.txt",
        "clean/BLUE/LR_001/file_001_.25_foo-bar_20230202_BLUE_20121101_20130705_20010101_PT1S.txt",
        "clean/GREEN/LR_002/file_002_.25_foo-bar_20230203_GREEN_20121101_20130705_20010101_PT1S.txt",
        "clean/RED/LR_003/file_003_1.75_foo-bar_20230204_RED_20121101_20130705_20010101_PT1S.txt",
        "clean/BLUE/LR_004/file_004_1.75_foo-bar_20230205_BLUE_20121101_20130705_20010101_PT1S.txt",
        "clean/GREEN/LR_005/file_005_1.75_foo-bar_20230206_GREEN_20121101_20130705_20010101_PT1S.txt",
        "clean/RED/HR_006/file_006_5.6_baz_20230207_RED_20221101_20230705_19500101_PT1S.txt",
        "clean/BLUE/HR_007/file_007_5.8_baz_20230208_BLUE_20221101_20230705_19500101_PT1S.txt",
        "clean/GREEN/HR_008/file_008_7.4_baz_20230209_GREEN_20221101_20230705_19500101_PT1S.txt",
        "clean/HR_009/file_009_5.6_baz_20230207_RED_20221101_20230705_19500101_PT1S.txt",
        "clean/HR_010/file_010_5.8_baz_20230208_BLUE_20221101_20230705_19500101_PT2S.txt",
        "clean/HR_011/file_011_7.4_baz_20230209_GREEN_20221101_20230705_19500101_PT2S.txt",
    ]


@pytest.fixture(scope="session")
def memory_fs() -> MemoryFileSystem:
    return MemoryFileSystem()


@pytest.fixture(scope="session")
def memory_root(memory_fs: MemoryFileSystem, filepaths: list[str]) -> Path:
    root = Path("/myfc")
    for filepath in filepaths:
        filepath = root / filepath
        memory_fs.makedirs(filepath.parent, exist_ok=True)
        memory_fs.touch(filepath)
    return root


@pytest.fixture(scope="session")
def ftp_server(tmp_path_factory: pytest.TempPathFactory):
    # Create a temp directory to act as FTP root
    ftp_root = tmp_path_factory.mktemp("myfc")

    # Setup user
    authorizer = DummyAuthorizer()
    authorizer.add_user(
        username="user",
        password="12345",
        homedir=str(ftp_root),
        perm="elradfmw",
    )

    handler = FTPHandler
    handler.authorizer = authorizer

    # Bind to random free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        host, port = s.getsockname()

    server = FTPServer((host, port), handler)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield {
        "host": host,
        "port": port,
        "username": "user",
        "password": "12345",
        "root": ftp_root,
    }

    server.close_all()


@pytest.fixture(scope="session")
def ftp_fs(ftp_server: list[str]) -> FTPFileSystem:
    fs = FTPFileSystem(
        host=ftp_server["host"],
        port=ftp_server["port"],
        username=ftp_server["username"],
        password=ftp_server["password"],
    )
    return fs


@pytest.fixture(scope="session")
def ftp_root(
    ftp_fs: FTPFileSystem, ftp_server: list[str], filepaths: list[str]
) -> FTPFileSystem:
    root = ftp_server["root"]
    for filepath in filepaths:
        filepath = root / filepath
        ftp_fs.makedirs(filepath.parent.as_posix(), exist_ok=True)
        ftp_fs.touch(filepath.as_posix())
    return root


@pytest.fixture
def local_fs() -> LocalFileSystem:
    return LocalFileSystem()


@pytest.fixture(scope="session")
def links() -> list[tuple[str]]:
    return [
        ("links/RED", "../clean/RED"),
        ("links/BLUE", "../clean/BLUE"),
        ("links/GREEN", "../clean/GREEN"),
        ("links/HR_009", "../clean/HR_009"),
        ("links/HR_010", "../clean/HR_010"),
        ("links/HR_011", "../clean/HR_011"),
    ]


@pytest.fixture
def local_root(
    tmp_path_factory: pytest.TempPathFactory,
    filepaths: list[str],
    links: list[tuple[str]],
) -> Path:
    my_fc = tmp_path_factory.mktemp("myfc")
    for filepath in filepaths:
        (my_fc / filepath).parent.mkdir(parents=True, exist_ok=True)
        (my_fc / filepath).touch()

    for destination, source in links:
        (my_fc / destination).parent.mkdir(parents=True, exist_ok=True)
        (my_fc / destination).symlink_to(source)
    return my_fc


class MockFS:

    def _strip_protocol(self, name: str):
        return name

    def ls(self, path, detail=True):
        # parent itself appears in its own listing
        return [
            {"name": f"{path}/", "type": "directory"},
            {"name": f"{path}/file1.txt", "type": "file"},
        ]


def test_walk_parent_pruning():
    mock_fs = MockFS()
    root_node = DirNode("root", {"name": "root"}, mock_fs, 0)

    nodes_reimpl = list(walk(root_node, StandardVisitor()))
    assert nodes_reimpl == [("root", [], ["", "file1.txt"])]


class MockErrorFS:

    def _strip_protocol(self, name: str):
        return name

    def ls(self, path, detail=True):
        raise FileNotFoundError


def test_walk_error():
    mock_fs = MockErrorFS()
    root_node = DirNode("root", {"name": "root"}, mock_fs, 0)

    nodes_reimpl = list(walk(root_node, StandardVisitor()))
    assert nodes_reimpl == [("root", [], [])]


@pytest.mark.parametrize(
    "fs_fixture_name, path_fixture_name",
    [
        ("local_fs", "local_root"),
        ("memory_fs", "memory_root"),
        ("ftp_fs", "ftp_root"),
    ],
    ids=["local", "memory", "ftp"],
)
def test_walk(
    fs_fixture_name: str, path_fixture_name: str, request: pytest.FixtureRequest
):
    fs = request.getfixturevalue(fs_fixture_name)
    root_path = request.getfixturevalue(path_fixture_name)
    root_str = root_path.as_posix()
    root_node = DirNode(root_str, {"name": root_str}, fs, 0)

    nodes_reimpl = list(walk(root_node, StandardVisitor()))
    nodes_fsspec = list(fs.walk(root_str))
    assert nodes_reimpl == nodes_fsspec


def test_walk_no_symlinks(local_root: Path, local_fs: LocalFileSystem):
    """By defaults, file system should not follow symlinks."""
    root_str = (local_root / "clean").as_posix()
    root_node = DirNode(root_str, {"name": root_str}, local_fs, 0)
    result = list(walk(root_node, StandardVisitor()))
    assert len(result) > 1

    root_str = (local_root / "links").as_posix()
    root_node = DirNode(root_str, {"name": root_str}, local_fs, 0)
    result = list(walk(root_node, StandardVisitor()))
    assert len(result) == 1


class Color(Enum):
    RED = auto()
    GREEN = auto()
    BLUE = auto()
    gray = auto()


@pytest.fixture(scope="session")
def expected_record() -> tuple[tp.Any, ...]:
    return (
        8,
        7.4,
        "baz",
        np.datetime64("2023-02-09T00:00:00.000000"),
        Color.GREEN,
        Period(np.datetime64("2022-11-01"), np.datetime64("2023-07-05")),
        Period(
            np.datetime64("1950-01-01"),
            np.datetime64("1950-01-01T01"),
            include_stop=False,
        ),
        ISODuration(seconds=1),
    )


def test_record_filter(expected_record, convention):
    record_filter = RecordFilter(convention.fields, field_f=7.4, field_s="baz")
    assert record_filter.test(expected_record)


@pytest.mark.parametrize(
    "field, value, value_sanitized",
    [
        ("field_i", 2, 2),
        ("field_f", 0.25, 0.25),
        ("field_s", "foo-bar", "foo-bar"),
        ("field_enum", "BLUE", Color.BLUE),
        ("field_enum", ("BLUE", "RED"), (Color.BLUE, Color.RED)),
        ("field_enum", ["BLUE", "RED"], (Color.BLUE, Color.RED)),
        (
            "field_date",
            ("2023-01-01", None),
            Period(
                np.datetime64("2023-01-01"), np.datetime64("9999-12-31T23:59:59.999999")
            ),
        ),
        (
            "field_period",
            (None, "2023-01-01"),
            Period(np.datetime64("0001-01-01"), np.datetime64("2023-01-01")),
        ),
        ("field_date_delta", "2023-01-01", np.datetime64("2023-01-01")),
        ("field_iso_duration", "PT2S", ISODuration(seconds=2)),
    ],
)
def test_record_filter_sanitize(convention, field, value, value_sanitized):
    expected = RecordFilter(convention.fields, **{field: value})
    actual = RecordFilter(convention.fields, **{field: value_sanitized})

    assert expected.references == actual.references


def test_record_filter_bad_keys(convention):
    with pytest.raises(FileListingError):
        RecordFilter(convention.fields, field_f=0.25, fieldA="10")


@pytest.fixture(scope="session")
def layout(convention: FileNameConvention) -> Layout:
    return Layout(
        [
            FileNameConvention(
                re.compile(r"(?P<field_enum>\w+)"),
                [convention.get_field("field_enum")],
                "{field_enum!f}",
            ),
            FileNameConvention(
                re.compile(r"(?P<resolution>\w+)_(?P<field_i>\d{3})"),
                [FileNameFieldString("resolution"), FileNameFieldInteger("field_i")],
                "{resolution!f}_{field_i:>03d}",
            ),
        ]
    )


def test_layout_names(layout: Layout):
    assert layout.names == {"field_enum", "field_i", "resolution"}


def test_layout_generate(layout: Layout):
    actual = layout.generate("root", field_enum=Color.RED, field_i=12, resolution="HR")
    expected = "root/RED/HR_012"
    assert actual == expected


def test_layout_generate_missing_field(layout: Layout):
    with pytest.raises(ValueError):
        layout.generate("root", field_i=12, resolution="HR")


def test_layout_generate_bad_field(layout: Layout):
    with pytest.raises(ValueError):
        layout.generate("root", field_enum=Color.RED, field_i="12", resolution="HR")


@pytest.mark.parametrize(
    "filters, node, expected_record, test_result",
    [
        ({"field_i": [1, 2, 3]}, "LR_001", ("LR", 1), True),
        ({"field_i": [1, 2, 3]}, "LR_004", ("LR", 4), False),
        ({"field_X": "unknown_in_layout"}, "LR_004", ("LR", 4), True),
    ],
)
def test_layout_parse_test(
    layout: Layout,
    filters: dict[str, int | str],
    node: str,
    expected_record: tuple[str, int],
    test_result: bool,
):
    layout.set_filters(**filters)
    record = layout.parse_node(1, node)
    assert record == expected_record
    assert layout.test_record(1, record) is test_result


def test_layout_parse_empty():
    # If no match, ensure None is returned, even if the convention has no field
    layout = Layout([FileNameConvention(re.compile("foo|bar"), [])])
    result = layout.parse_node(0, "baz")
    assert result is None


@pytest.fixture(scope="session")
def convention():
    regex = re.compile(
        r"file_(?P<field_i>\d+)_(?P<field_f>[+-]?([0-9]*[.])?[0-9]+)_(?P<field_s>[a-zA-Z0-9.-]+)_(?P<field_date>\d{8})_(?P<field_enum>\w+)_(?P<field_period>\d{8}_\d{8})_(?P<field_date_delta>\d{8})_(?P<field_iso_duration>\w+).txt"
    )
    fields = [
        FileNameFieldInteger("field_i"),
        FileNameFieldFloat("field_f"),
        FileNameFieldString("field_s"),
        FileNameFieldDatetime("field_date", "%Y%m%d"),
        FileNameFieldEnum("field_enum", Color),
        FileNameFieldPeriod("field_period", "%Y%m%d"),
        FileNameFieldDateDelta("field_date_delta", "%Y%m%d", np.timedelta64(1, "h")),
        FileNameFieldISODuration("field_iso_duration"),
    ]
    generation_string = "file_{field_i:>03d}_{field_f}_{field_s}_{field_date!f}_{field_enum!f}_{field_period!f}_{field_date_delta!f}_{field_iso_duration!f}.txt"
    return FileNameConvention(regex, fields, generation_string)


@pytest.fixture(scope="session")
def layouts_v2(layout: Layout, convention: FileNameConvention) -> list[Layout]:
    return [
        Layout([*layout.conventions, convention]),
        Layout([layout.conventions[1], convention]),
    ]


@pytest.mark.parametrize(
    "path, level, expected_explore_next, layouts_selection",
    [
        ("root", 0, True, [0, 1]),
        ("root/RED", 1, True, [0]),
        ("root/HR_009", 1, True, [1]),
    ],
)
def test_layout_visit_dir(
    layouts_v2: list[Layout],
    memory_fs: MemoryFileSystem,
    memory_root: Path,
    path: str,
    level: int,
    expected_explore_next: bool,
    layouts_selection: list[int],
):

    path = memory_root / path
    node = DirNode(path.name, {"name": path.as_posix()}, memory_fs, level)

    visitor = LayoutVisitor(layouts_v2)
    result = visitor.visit_dir(node)
    assert result.explore_next == expected_explore_next
    assert result.payload is None
    assert result.surviving_layouts == [layouts_v2[ii] for ii in layouts_selection]


@pytest.fixture(scope="session")
def layouts_v3(layout: Layout) -> list[Layout]:
    return [
        Layout([layout.conventions[0]]),
        Layout([layout.conventions[1]]),
    ]


@pytest.mark.parametrize(
    "path, level, expected_explore_next, payload",
    [
        ("root", 0, True, None),
        ("root/RED", 1, False, (Color.RED,)),
        ("root/HR_009", 1, False, ("HR", 9)),
    ],
)
def test_layout_visit_dir_payload(
    layouts_v3: list[Layout],
    memory_fs: MemoryFileSystem,
    memory_root: Path,
    path: str,
    level: int,
    expected_explore_next: bool,
    payload: None | tuple[tp.Any, ...],
):

    path = memory_root / path
    node = DirNode(path.name, {"name": path.as_posix()}, memory_fs, level)

    visitor = LayoutVisitor(layouts_v3)
    result = visitor.visit_dir(node)
    assert result.explore_next == expected_explore_next
    assert result.payload == payload


@pytest.mark.parametrize(
    "path, context, on_mismatch",
    [
        ("root/outlier", nullcontext(), LayoutMismatchHandling.IGNORE),
        ("root/outlier", pytest.warns(UserWarning), LayoutMismatchHandling.WARN),
        (
            "root/outlier",
            pytest.raises(LayoutMismatchError),
            LayoutMismatchHandling.RAISE,
        ),
    ],
)
def test_layout_visit_dir_outlier(
    layouts_v2: list[Layout],
    memory_root: Path,
    memory_fs: MemoryFileSystem,
    context,
    path: Path,
    on_mismatch: LayoutMismatchHandling,
):
    path = memory_root / path
    node = DirNode(path.name, {"name": path.as_posix()}, memory_fs, 1)
    visitor = LayoutVisitor(layouts_v2, on_mismatch_directory=on_mismatch)

    with context:
        result = visitor.visit_dir(node)
        assert result.payload is None
        assert result.explore_next is False
        assert result.surviving_layouts == []


@pytest.mark.parametrize(
    "path, level, layouts_selection",
    [
        (
            "root/GREEN/HR_008/file_008_7.4_baz_20230209_GREEN_20221101_20230705_19500101_PT1S.txt",
            3,
            [0],
        ),
        (
            "root/HR_009/file_008_7.4_baz_20230209_GREEN_20221101_20230705_19500101_PT1S.txt",
            2,
            [1],
        ),
    ],
)
def test_layout_visit_file(
    layouts_v2: list[Layout],
    memory_root: Path,
    path: str,
    level: int,
    expected_record: tuple[tp.Any, ...],
    layouts_selection: list[int],
):
    path = memory_root / path
    node = FileNode(path.name, {"name": path.as_posix()}, level)

    # Simulate layout pruning
    visitor = LayoutVisitor([layouts_v2[ii] for ii in layouts_selection])

    result = visitor.visit_file(node)
    assert not result.explore_next
    assert result.surviving_layouts == []

    assert result.payload == (*expected_record, (memory_root / path).as_posix())


@pytest.mark.parametrize(
    "path, context, on_mismatch",
    [
        ("file_KO.txt", nullcontext(), LayoutMismatchHandling.IGNORE),
        ("file_KO.txt", pytest.warns(UserWarning), LayoutMismatchHandling.WARN),
        (
            "file_KO.txt",
            pytest.raises(LayoutMismatchError),
            LayoutMismatchHandling.RAISE,
        ),
    ],
)
def test_layout_visit_file_outlier(
    layouts_v2: list[Layout],
    memory_root: Path,
    memory_fs: MemoryFileSystem,
    context,
    path: Path,
    on_mismatch: LayoutMismatchHandling,
):
    path = memory_root / path
    node = DirNode(path.name, {"name": path.as_posix()}, memory_fs, 1)
    visitor = LayoutVisitor(layouts_v2, on_mismatch_file=on_mismatch)

    with context:
        result = visitor.visit_file(node)
        assert result.payload is None
        assert result.explore_next is False
        assert result.surviving_layouts == []


def test_layout_visit_file_stat_fields(
    layouts_v2: list[Layout],
    memory_root: Path,
    expected_record: tuple[tp.Any, ...],
):
    path = "root/GREEN/HR_008/file_008_7.4_baz_20230209_GREEN_20221101_20230705_19500101_PT1S.txt"
    path = memory_root / path
    node = FileNode(
        path.name,
        {"name": path.as_posix(), "size": 100, "type": "file", "created": 250},
        3,
    )

    # Simulate layout pruning
    visitor = LayoutVisitor(layouts_v2[:1], ("size", "type"))

    result = visitor.visit_file(node)
    assert result.payload == (
        *expected_record,
        (memory_root / path).as_posix(),
        100,
        "file",
    )


def test_layout_advance(layouts_v2: list[Layout]):
    visitor = LayoutVisitor(
        layouts_v2,
        on_mismatch_directory=LayoutMismatchHandling.WARN,
        on_mismatch_file=LayoutMismatchHandling.RAISE,
    )
    result = VisitResult(True, None, layouts_v2)
    new_visitor = visitor.advance(result)
    assert new_visitor is not visitor
    assert new_visitor.on_mismatch_directory == visitor.on_mismatch_directory
    assert new_visitor.on_mismatch_file == visitor.on_mismatch_file

    result = VisitResult(True, None, layouts_v2[:1])
    new_visitor = visitor.advance(result)
    assert len(new_visitor.layouts) == 1
    assert len(visitor.layouts) == 2


@pytest.mark.parametrize(
    "filters, record_index, expected, count",
    [({"field_enum": "BLUE"}, 4, Color.BLUE, 4), ({"field_f": 5.6}, 1, 5.6, 2)],
)
def test_walk_layout(
    layouts_v2: list[Layout],
    filters: dict[str, tp.Any],
    record_index: int,
    expected: tp.Any,
    count: int,
    memory_root: Path,
    memory_fs: MemoryFileSystem,
):
    for layout in layouts_v2:
        layout.set_filters(**filters)
    visitor = LayoutVisitor(
        layouts_v2,
        on_mismatch_directory=LayoutMismatchHandling.IGNORE,
        on_mismatch_file=LayoutMismatchHandling.IGNORE,
    )
    root_str = (memory_root / "root").as_posix()
    root_node = DirNode(root_str, {"name": root_str}, memory_fs, 0)

    records = list(walk(root_node, visitor))
    assert len(records) == count
    assert all([record[record_index] == expected for record in records])


@pytest.mark.parametrize(
    "path, level",
    [
        ("root", 0),
        ("root/RED", 1),
        ("root/outlier", 1),
    ],
)
def test_no_layout_visit_dir(
    convention: FileNameConvention,
    memory_fs: MemoryFileSystem,
    memory_root: Path,
    path: str,
    level: int,
):

    path = memory_root / path
    node = DirNode(path.name, {"name": path.as_posix()}, memory_fs, level)

    visitor = NoLayoutVisitor(convention, None)
    result = visitor.visit_dir(node)
    assert result.explore_next is True
    assert result.payload is None
    assert result.surviving_layouts == []


@pytest.mark.parametrize(
    "path",
    [
        "root/GREEN/HR_008/file_008_7.4_baz_20230209_GREEN_20221101_20230705_19500101_PT1S.txt",
        "root/HR_009/file_008_7.4_baz_20230209_GREEN_20221101_20230705_19500101_PT1S.txt",
    ],
)
def test_no_layout_visit_file(
    layouts_v2: list[Layout],
    memory_root: Path,
    path: str,
    expected_record: tuple[tp.Any, ...],
):
    path = memory_root / path
    node = FileNode(path.name, {"name": path.as_posix()}, 10)

    layout = layouts_v2[-1]
    layout.set_filters()

    # Retrieve RecordFilter automatically built in the Layout
    visitor = NoLayoutVisitor(layout.conventions[-1], layout.filters[-1])

    result = visitor.visit_file(node)
    assert not result.explore_next
    assert result.surviving_layouts == []

    assert result.payload == (*expected_record, (memory_root / path).as_posix())


def test_no_layout_advance():
    visitor = NoLayoutVisitor(None, None)
    assert visitor is visitor.advance(None)


@pytest.mark.parametrize(
    "filters, record_index, expected, count",
    [({"field_enum": "BLUE"}, 4, Color.BLUE, 4), ({"field_f": 5.6}, 1, 5.6, 2)],
)
def test_walk_no_layout(
    layouts_v2: list[Layout],
    filters: dict[str, tp.Any],
    record_index: int,
    expected: tp.Any,
    count: int,
    memory_root: Path,
    memory_fs: MemoryFileSystem,
):
    layout = layouts_v2[-1]
    layout.set_filters(**filters)

    visitor = NoLayoutVisitor(layout.conventions[-1], layout.filters[-1])
    root_str = (memory_root / "root").as_posix()
    root_node = DirNode(root_str, {"name": root_str}, memory_fs, 0)

    records = list(walk(root_node, visitor))
    assert len(records) == count
    assert all([record[record_index] == expected for record in records])


def test_walk_no_layout_symlinks(
    layouts_v2: list[Layout], local_fs: LocalFileSystem, local_root: Path
):
    layout = layouts_v2[-1]
    layout.set_filters()

    visitor = NoLayoutVisitor(layout.conventions[-1], layout.filters[-1])
    root_str = (local_root / "clean").as_posix()
    link_str = (local_root / "links").as_posix()

    root_node = DirNode(root_str, {"name": root_str}, local_fs, 0)
    link_node = DirNode(link_str, {"name": link_str}, local_fs, 0, follow_symlinks=True)

    records_clean = set(walk(root_node, visitor))
    records_link = set(walk(link_node, visitor))
    assert records_clean == records_link


@pytest.fixture(
    scope="session",
    params=[("clean", True), ("root", False)],
    ids=["clean_layout", "bad_layout"],
)
def collector_status(
    memory_fs: MemoryFileSystem,
    layouts_v2: list[Layout],
    memory_root: Path,
    request: pytest.FixtureRequest,
) -> tuple[FileSystemMetadataCollector, bool]:
    root_path_str = (memory_root / request.param[0]).as_posix()
    root_node = DirNode(root_path_str, {"name": root_path_str}, memory_fs, 0)
    return (
        FileSystemMetadataCollector(layouts_v2, root_node),
        request.param[1],
    )


@pytest.mark.parametrize(
    "filters, expected_selection_clean, expected_selection_bad",
    [
        (dict(field_enum=Color.RED, field_f=1.75), [3], [3]),
        (
            dict(
                field_period=Period(
                    np.datetime64("2011-01-01"), np.datetime64("2013-12-31")
                )
            ),
            list(range(6)),
            list(range(6)),
        ),
        (
            dict(
                field_date=Period(
                    np.datetime64("2023-02-03"), np.datetime64("2023-02-08")
                ),
                field_s="baz",
            ),
            [6, 7, 9, 10],
            [6, 7, 9, 10],
        ),
        (dict(field_f=0.25, field_date_delta=np.datetime64("1950-01-01")), [], []),
        (
            dict(resolution="HR", field_i=list(range(0, 10, 2))),
            [6, 8],
            list(range(0, 10, 2)),
        ),
        (
            dict(
                predicates=[
                    lambda record: record[0] % 2 == 0 and record[2] == "foo-bar"
                ]
            ),
            [0, 2, 4],
            [0, 2, 4],
        ),
        (
            dict(
                predicates=[
                    lambda record: record[0] % 2 == 0,
                    lambda record: record[2] == "foo-bar",
                ]
            ),
            [0, 2, 4],
            [0, 2, 4],
        ),
    ],
)
def test_collector(
    collector_status: tuple[FileSystemMetadataCollector, bool],
    filters: dict[str, tp.Any],
    expected_selection_clean: list[int],
    expected_selection_bad: list[int],
):

    collector, enable_layouts = collector_status
    df = collector.to_dataframe(enable_layouts=enable_layouts, **filters)

    # In case we gave layout filters not in the file name, it is possible to
    # have a difference between the LayoutVisitor and NoLayoutVisitor walk
    if enable_layouts:
        assert np.array_equal(sorted(df["field_i"].values), expected_selection_clean)
    else:
        assert np.array_equal(sorted(df["field_i"].values), expected_selection_bad)
    assert "filename" in df.columns


def test_collector_stat_fields(
    collector_status: tuple[FileSystemMetadataCollector, bool],
):
    collector, enable_layouts = collector_status
    df = collector.to_dataframe(
        enable_layouts=enable_layouts, stat_fields=("size", "type")
    )
    assert np.array_equal(df["size"].values, np.full(12, fill_value=0))
    assert np.array_equal(df["type"].values, np.full(12, fill_value="file"))


def test_collector_stat_fields_error(
    collector_status: tuple[FileSystemMetadataCollector, bool],
):
    with pytest.raises(KeyError):
        collector, enable_layouts = collector_status
        collector.to_dataframe(enable_layouts=enable_layouts, stat_fields=("foo",))
