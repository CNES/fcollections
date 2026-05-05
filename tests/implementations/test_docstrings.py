import pytest

from fcollections.implementations import (
    NetcdfFilesDatabaseL3Nadir,
    NetcdfFilesDatabaseSST,
    NetcdfFilesDatabaseSwotLRL2,
    NetcdfFilesDatabaseSwotLRL3,
)


@pytest.fixture
def swot_lr_l2_db(tmpdir, scope="module"):
    db = NetcdfFilesDatabaseSwotLRL2(tmpdir)
    return db


@pytest.fixture
def swot_lr_l3_db(tmpdir, scope="module"):
    db = NetcdfFilesDatabaseSwotLRL3(tmpdir)
    return db


@pytest.fixture
def nadir_db(tmpdir, scope="module"):
    db = NetcdfFilesDatabaseL3Nadir(tmpdir)
    return db


@pytest.fixture
def sst_db(tmpdir, scope="module"):
    db = NetcdfFilesDatabaseSST(tmpdir)
    return db


@pytest.mark.parametrize(
    "db, keywords",
    [
        (
            "swot_lr_l2_db",
            [
                "List the files matching the given criteria.",
                "sort",
                "cycle_number",
                "pass_number",
                "time",
                "subset",
            ],
        ),
        (
            "swot_lr_l3_db",
            [
                "List the files matching the given criteria.",
                "sort",
                "cycle_number",
                "pass_number",
                "time",
                "level",
                "subset",
            ],
        ),
        (
            "nadir_db",
            ["List the files matching the given criteria.", "sort", "time", "sensor"],
        ),
        ("sst_db", ["List the files matching the given criteria.", "sort", "time"]),
    ],
)
def test_list_files_docstring(db, keywords, request):
    db = request.getfixturevalue(db)

    for keyword in keywords:
        assert keyword in db.list_files.__doc__


@pytest.mark.parametrize(
    "db, keywords",
    [
        (
            "swot_lr_l2_db",
            [
                "Query a dataset by reading selected files in file system.",
                "stack",
                "left_swath",
                "selected_variables",
                "cycle_number",
                "pass_number",
                "time",
                "subset",
                "right_swath",
            ],
        ),
        (
            "swot_lr_l3_db",
            [
                "Query a dataset by reading selected files in file system.",
                "stack",
                "swath",
                "nadir",
                "selected_variables",
                "cycle_number",
                "pass_number",
                "time",
                "level",
                "subset",
            ],
        ),
        (
            "nadir_db",
            [
                "Query a dataset by reading selected files in file system.",
                "selected_variables",
                "time",
                "sensor",
            ],
        ),
        (
            "sst_db",
            [
                "Query a dataset by reading selected files in file system.",
                "selected_variables",
                "time",
            ],
        ),
    ],
)
def test_query_docstring(db, keywords, request):
    db = request.getfixturevalue(db)

    for keyword in keywords:
        assert keyword in db.query.__doc__


@pytest.mark.parametrize(
    "db, keywords",
    [
        ("swot_lr_l2_db", ["metadata describing the variables", "subset"]),
        ("swot_lr_l3_db", ["metadata describing the variables"]),
        ("nadir_db", ["metadata describing the variables", "resolution", "sensor"]),
        ("sst_db", ["metadata describing the variables"]),
    ],
)
def test_variables_info_docstring(db, keywords, request):
    db = request.getfixturevalue(db)

    for keyword in keywords:
        assert keyword in db.variables_info.__doc__
