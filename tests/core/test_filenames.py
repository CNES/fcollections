from __future__ import annotations

import re
import typing as tp
from enum import Enum, auto

import numpy as np
import pytest

from fcollections.core import (
    DecodingError,
    FileNameConvention,
    FileNameField,
    FileNameFieldDateDelta,
    FileNameFieldDateJulian,
    FileNameFieldDateJulianDelta,
    FileNameFieldDatetime,
    FileNameFieldEnum,
    FileNameFieldFloat,
    FileNameFieldGeoBox,
    FileNameFieldInteger,
    FileNameFieldISODuration,
    FileNameFieldPeriod,
    FileNameFieldString,
)
from fcollections.time import ISODuration, Period


class Color(Enum):
    RED = auto()
    GREEN = auto()
    BLUE = auto()
    gray = auto()
    PINK_SCARLET = auto()


def test_field_bad_init():
    with pytest.raises(ValueError):
        FileNameFieldDateJulian(name="foo", reference=None, julian_day_format="foo")


@pytest.mark.parametrize(
    "field, encoded, decoded",
    [
        (FileNameFieldDatetime("", "%Y%m%d"), "20231202", np.datetime64("2023-12-02")),
        (
            FileNameFieldDateDelta("", "%Y%m%d", np.timedelta64(1, "D")),
            "20231202",
            Period(
                np.datetime64("2023-12-02"),
                np.datetime64("2023-12-03"),
                include_stop=False,
            ),
        ),
        (
            FileNameFieldDatetime("", "%Y-%m-%dT%H:%M:%S.%fZ"),
            "2023-12-02T02:31:15.000000Z",
            np.datetime64("2023-12-02T02:31:15.000000"),
        ),
        (FileNameFieldEnum("", Color), "BLUE", Color.BLUE),
        (FileNameFieldEnum("", Color), "PINK_SCARLET", Color.PINK_SCARLET),
        (
            FileNameFieldEnum("", Color, underscore_encoded=False),
            "PINK-SCARLET",
            Color.PINK_SCARLET,
        ),
        (FileNameFieldEnum("", Color, "upper", "lower"), "blue", Color.BLUE),
        (FileNameFieldEnum("", Color, "lower", "upper"), "GRAY", Color.gray),
        (FileNameFieldInteger(""), "-2", -2),
        (FileNameFieldFloat(""), "15.2", 15.2),
        (FileNameFieldFloat(""), "10", 10),
        (FileNameFieldString(""), "random_string", "random_string"),
        (
            FileNameFieldPeriod("", "%Y%m%d", "_"),
            "20231202_20231203",
            Period(np.datetime64("2023-12-02"), np.datetime64("2023-12-03")),
        ),
        (
            FileNameFieldPeriod("", "%Y_%m%d", "_"),
            "2023_1202_2023_1203",
            Period(np.datetime64("2023-12-02"), np.datetime64("2023-12-03")),
        ),
        (
            FileNameFieldDateJulianDelta(
                "", np.timedelta64(1, "D"), np.datetime64("1950-01-01T00")
            ),
            "23831",
            Period(
                np.datetime64("2015-04-01T00"),
                np.datetime64("2015-04-02"),
                include_stop=False,
            ),
        ),
        (
            FileNameFieldDateJulian("", np.datetime64("1950-01-01T00")),
            "23831_06",
            np.datetime64("2015-04-01T06"),
        ),
        (
            FileNameFieldDateJulian(
                "", np.datetime64("1950-01-01T00"), julian_day_format="fractional"
            ),
            "23831.25",
            np.datetime64("2015-04-01T06"),
        ),
        (FileNameFieldISODuration(""), "PT15M", ISODuration(minutes=15)),
        (FileNameFieldISODuration(""), "P1D", ISODuration(days=1)),
        (FileNameFieldISODuration(""), "P1DT3H", ISODuration(days=1, hours=3)),
        (FileNameFieldISODuration(""), "P1W", ISODuration(weeks=1)),
        (
            FileNameFieldISODuration(""),
            "P1Y4M2DT1H1S",
            ISODuration(years=1, months=4, days=2, hours=1, seconds=1),
        ),
        (FileNameFieldISODuration(""), "PT0S", ISODuration()),
    ],
)
def test_fields_encode_decode_nominal(
    field: FileNameField, encoded: str, decoded: tp.Any
):
    assert field.decode(encoded) == decoded
    assert field.encode(decoded) == encoded


@pytest.mark.parametrize(
    "field, input_string",
    [
        (FileNameFieldDatetime("", "%Y%m%dT%H"), "20231202"),
        (FileNameFieldDatetime("", "%Y%m%d"), "20231302"),
        (FileNameFieldDatetime("", "invalid"), "20231302"),
        (FileNameFieldFloat(""), "pi"),
        (FileNameFieldInteger(""), "10.2"),
        (FileNameFieldInteger(""), "ten"),
        (FileNameFieldEnum("", Color), "red"),
        (FileNameFieldPeriod("", "%Y%m%d", "_"), "20231202T00_20231203"),
        (FileNameFieldPeriod("", "%Y%m%d", "_"), "20231202-20231203"),
        (
            FileNameFieldDateJulianDelta(
                "", np.timedelta64(1, "D"), np.datetime64("1950-01-01T00")
            ),
            "2023-12-02",
        ),
        (
            FileNameFieldDateJulian(
                "", np.datetime64("1950-01-01T00"), julian_day_format="days_hours"
            ),
            "17831",
        ),
        (
            FileNameFieldDateJulian(
                "", np.datetime64("1950-01-01T00"), julian_day_format="fractional"
            ),
            "17831-01",
        ),
        (
            FileNameFieldDateJulian(
                "", np.datetime64("1950-01-01T00"), julian_day_format="days"
            ),
            "17831.25",
        ),
        (FileNameFieldISODuration(""), "P0.2W"),
    ],
    ids=[
        "Non matching string with date format",
        "Invalid input date",
        "Invalid date format",
        "String instead of float number",
        "Float number instead of integer",
        "String instead of integer number",
        "Non matching enum",
        "Invalid dates in period",
        "Bad separator in period",
        "Not a julian day",
        "Not a julian day with hours",
        "Not a fractional julian day",
        "Not a julian day",
        "Not a vlid ISO8601 code",
    ],
)
def test_fields_decode_error(field: FileNameField, input_string: str):
    with pytest.raises(DecodingError):
        field.decode(input_string)


@pytest.mark.parametrize(
    "field, reference, tested, filtered",
    [
        (FileNameFieldInteger(""), 2, 2, True),
        (FileNameFieldInteger(""), [1, 2, 6], 2, True),
        (FileNameFieldInteger(""), slice(-3, 7), 2, True),
        (FileNameFieldInteger(""), 1, 2, False),
        (FileNameFieldInteger(""), [1, 6, 7], 2, False),
        (FileNameFieldInteger(""), slice(3, 7), 2, False),
        (FileNameFieldFloat(""), 10.5, 10.5, True),
        (FileNameFieldFloat(""), 10.5, 2.1, False),
        (FileNameFieldString(""), "ref_eq_tested", "ref_eq_tested", True),
        (FileNameFieldString(""), "reference", "tested", False),
        (FileNameFieldEnum("", Color), Color.RED, Color.RED, True),
        (FileNameFieldEnum("", Color), Color.RED, "red", False),
        (FileNameFieldEnum("", Color), [Color.BLUE, Color.RED], Color.RED, True),
        (FileNameFieldEnum("", Color), [Color.BLUE, Color.GREEN], Color.RED, False),
        (
            FileNameFieldDatetime("", ""),
            np.datetime64("2023-01-01"),
            np.datetime64("2023-01-01"),
            True,
        ),
        (
            FileNameFieldDatetime("", ""),
            Period(np.datetime64("2013-01-01"), np.datetime64("2033-01-01")),
            np.datetime64("2023-01-01"),
            True,
        ),
        (
            FileNameFieldDatetime("", ""),
            np.datetime64("2023-01-01"),
            np.datetime64("2023-01-02"),
            False,
        ),
        (
            FileNameFieldDatetime("", ""),
            Period(np.datetime64("1993-01-01"), np.datetime64("2003-01-01")),
            np.datetime64("2023-01-01"),
            False,
        ),
        (
            FileNameFieldPeriod("", ""),
            Period(np.datetime64("2001-01-01"), np.datetime64("2002-01-01")),
            Period(np.datetime64("1993-01-01"), np.datetime64("2003-01-01")),
            True,
        ),
        (
            FileNameFieldPeriod("", ""),
            np.datetime64("2000-01-01"),
            Period(np.datetime64("1993-01-01"), np.datetime64("2003-01-01")),
            True,
        ),
        (
            FileNameFieldPeriod("", ""),
            Period(np.datetime64("1900-01-01"), np.datetime64("1910-01-01")),
            Period(np.datetime64("1993-01-01"), np.datetime64("2003-01-01")),
            False,
        ),
        (
            FileNameFieldDateDelta("", "", np.timedelta64(1, "D")),
            np.datetime64("1900-01-01"),
            Period(np.datetime64("1993-01-01"), np.datetime64("2003-01-01")),
            False,
        ),
        (
            FileNameFieldDateDelta("", "", np.timedelta64(1, "D")),
            Period(np.datetime64("2001-01-01"), np.datetime64("2002-01-01")),
            Period(np.datetime64("1993-01-01"), np.datetime64("2003-01-01")),
            True,
        ),
        (
            FileNameFieldDateDelta("", "", np.timedelta64(1, "D")),
            np.datetime64("2000-01-01"),
            Period(np.datetime64("1993-01-01"), np.datetime64("2003-01-01")),
            True,
        ),
        (
            FileNameFieldDateDelta("", "", np.timedelta64(1, "D")),
            Period(np.datetime64("1900-01-01"), np.datetime64("1910-01-01")),
            Period(np.datetime64("1993-01-01"), np.datetime64("2003-01-01")),
            False,
        ),
        (
            FileNameFieldDateDelta("", "", np.timedelta64(1, "D")),
            np.datetime64("1900-01-01"),
            Period(np.datetime64("1993-01-01"), np.datetime64("2003-01-01")),
            False,
        ),
        (
            FileNameFieldDateJulianDelta(
                "", np.timedelta64(1, "D"), np.datetime64("1950-01-01T00")
            ),
            Period(np.datetime64("1900-01-01"), np.datetime64("1910-01-01")),
            Period(np.datetime64("1993-01-01"), np.datetime64("2003-01-01")),
            False,
        ),
        (
            FileNameFieldDateJulianDelta(
                "", np.timedelta64(1, "D"), np.datetime64("1950-01-01T00")
            ),
            np.datetime64("2000-01-01"),
            Period(np.datetime64("1993-01-01"), np.datetime64("2003-01-01")),
            True,
        ),
        (
            FileNameFieldDateJulian("", np.datetime64("1950-01-01T00")),
            np.datetime64("2023-01-01"),
            np.datetime64("2023-01-01"),
            True,
        ),
        (
            FileNameFieldDateJulian("", np.datetime64("1950-01-01T00")),
            Period(np.datetime64("2013-01-01"), np.datetime64("2033-01-01")),
            np.datetime64("2023-01-01"),
            True,
        ),
        (
            FileNameFieldDateJulian("", np.datetime64("1950-01-01T00")),
            np.datetime64("2023-01-01"),
            np.datetime64("2023-01-02"),
            False,
        ),
        (
            FileNameFieldDateJulian("", np.datetime64("1950-01-01T00")),
            Period(np.datetime64("1993-01-01"), np.datetime64("2003-01-01")),
            np.datetime64("2023-01-01"),
            False,
        ),
        (
            FileNameFieldGeoBox(""),
            (-60, -10, 60, 10),
            (-50, -8, 58, -6),
            True,
        ),
        (
            FileNameFieldGeoBox(""),
            (-60, -10, 60, 10),
            (48, 8, 52, 12),
            True,
        ),
        (
            FileNameFieldGeoBox(""),
            (-60, -10, 60, 10),
            (-80, -20, -59, -9),
            True,
        ),
        (
            FileNameFieldGeoBox(""),
            (-60, -10, 60, 10),
            (-50, 20, -40, 30),
            False,
        ),
        (
            FileNameFieldGeoBox(""),
            (-60, -10, 60, 10),
            (70, -5, 80, 5),
            False,
        ),
    ],
)
def test_field_test(field, reference, tested, filtered):
    assert field.test(reference, tested) == filtered


@pytest.mark.parametrize(
    "field, expected_type",
    [
        (FileNameFieldString(""), str),
        (FileNameFieldFloat(""), float),
        (FileNameFieldInteger(""), list[int] | slice | int),
        (FileNameFieldEnum("", Color), Color),
        (FileNameFieldDatetime("", ""), np.datetime64),
        (FileNameFieldDateJulian("", np.datetime64("1950-01-01T00")), np.datetime64),
        (FileNameFieldPeriod("", ""), Period),
        (
            FileNameFieldDateJulianDelta(
                "", np.timedelta64(1, "D"), np.datetime64("1950-01-01T00")
            ),
            Period,
        ),
        (FileNameFieldDateDelta("", "", np.timedelta64(1, "D")), Period),
        (FileNameFieldISODuration(""), ISODuration),
        (FileNameFieldGeoBox(""), tuple[float, float, float, float]),
    ],
)
def test_field_type(field, expected_type):
    assert field.type == expected_type


@pytest.mark.parametrize(
    "field, expected_type_name",
    [
        (FileNameFieldInteger(""), "list[int] | slice | int"),
        (FileNameFieldEnum("", Color), "Color"),
        (FileNameFieldPeriod("", ""), "Period"),
    ],
)
def test_field_type_name(field: FileNameField, expected_type_name: str):
    assert field.type_name == expected_type_name


@pytest.mark.parametrize(
    "field, elements",
    [
        (FileNameFieldInteger("ifield"), ["list", "slice", "integer"]),
        (FileNameFieldFloat("ffield"), ["float"]),
        (FileNameFieldString("sfield"), ["string"]),
        (FileNameFieldDatetime("dfield", ""), ["[%Y-%m-%dT%H:%M:%S]"]),
        (FileNameFieldPeriod("pfield", ""), ["[%Y-%m-%dT%H:%M:%S]"]),
        (FileNameFieldEnum("efield", Color), ["RED", "BLUE", "GREEN", "gray"]),
        (FileNameFieldISODuration("Ifield"), ["ISO8601"]),
        (FileNameFieldGeoBox("gfield"), ["GeoBox", "intersect"]),
    ],
)
def test_field_description(field: FileNameField, elements: list[str]):
    for x in elements:
        assert x in field.description
    assert field.test_description in field.description
    assert not field.description.startswith(" ")


def test_field_choices():
    fields = FileNameFieldEnum("efield", Color, underscore_encoded=False)
    assert fields.choices() == ["RED", "GREEN", "BLUE", "gray", "PINK-SCARLET"]


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


@pytest.fixture
def expected_record():
    return (
        2,
        0.25,
        "foo-bar",
        np.datetime64("2023-02-01"),
        Color.RED,
        Period(np.datetime64("2012-11-01"), np.datetime64("2013-07-05")),
        Period(
            np.datetime64("2001-01-01"),
            np.datetime64("2001-01-01T01"),
            include_stop=False,
        ),
        ISODuration(seconds=1),
    )


@pytest.fixture
def expected_filename():
    return "file_002_0.25_foo-bar_20230201_RED_20121101_20130705_20010101_PT1S.txt"


def test_filename_convention_get_field(convention):
    assert convention.get_field("field_f") == convention.fields[1]


def test_filename_convention_get_field_error(convention):
    with pytest.raises(KeyError):
        convention.get_field("dummy")


def test_filename_convention_match(convention, expected_filename):
    match = convention.match(expected_filename)
    assert match is not None
    assert isinstance(match, re.Match)


def test_filename_convention_match_error(convention):
    filename = "bad_filename.pp"
    assert not convention.match(filename)


def test_filename_convention_parse(convention, expected_record, expected_filename):
    record = convention.parse(convention.match(expected_filename))
    assert record == expected_record


def test_filename_convention_parse_default(convention, expected_record):

    # Adapt fields and regex to handle optional group
    new_fields = convention.fields
    new_fields[0].default = -127
    regex = re.compile(
        r"file_(?P<field_i>\d+)*(_)*(?P<field_f>[+-]?([0-9]*[.])?[0-9]+)_(?P<field_s>[a-zA-Z0-9.-]+)_(?P<field_date>\d{8})_(?P<field_enum>\w+)_(?P<field_period>\d{8}_\d{8})_(?P<field_date_delta>\d{8})_(?P<field_iso_duration>\w+).txt"
    )
    new_record = list(expected_record)
    new_record[0] = -127

    new_parser = FileNameConvention(regex, new_fields)
    filename = "file_.25_foo-bar_20230201_RED_20121101_20130705_20010101_PT1S.txt"
    record = new_parser.parse(new_parser.match(filename))
    assert all([r == e for r, e in zip(record, new_record)])


def test_filename_convention_parse_error(convention, expected_record):
    """Checks the error raised in case of a None filename parse error."""
    with pytest.raises(AttributeError):
        convention.parse(None)


def test_filename_convention_generate(convention, expected_record, expected_filename):
    assert expected_filename == convention.generate(
        **{field.name: v for field, v in zip(convention.fields, expected_record)}
    )


def test_filename_convention_generate_missing_variables(convention, expected_record):
    with pytest.raises(ValueError):
        convention.generate(
            **{
                field.name: v
                for field, v in zip(convention.fields[:-1], expected_record)
            }
        )


class TestFieldFormatter:

    @pytest.fixture(autouse=True)
    def _set_formatter(self, convention: FileNameConvention):
        self.fmt = convention._formatter
        self.date = np.datetime64("2023-02-01")

    def test_nominal(self):
        result = self.fmt.format("Hello {field_date!f}", field_date=self.date)
        assert result == "Hello 20230201"

    def test_default_conversion_unchanged(self):
        result = str.format("Hello {field_date}", field_date=self.date)
        assert result == "Hello 2023-02-01"

    def test_format_spec_applied_after_encoding(self):
        result = self.fmt.format("{field_date!f:>10}", field_date=self.date)
        assert result == "  20230201"

    def test_nested_format_spec(self):
        result = self.fmt.format(
            "{field_date!f:{width}}", field_date=self.date, width=11
        )
        assert result == "20230201   "

    def test_missing_field_encoder_raises(self):
        with pytest.raises(KeyError):
            self.fmt.format("{x!f}", x="test")

    def test_auto_numbering_not_compatible(self):
        with pytest.raises(KeyError):
            # Auto numbering functionality is broken by our patch
            self.fmt.format("{!f}", "x")

    def test_auto_numbering_default(self):
        result = self.fmt.format("{}", "x")
        assert result == "x"

    def test_max_string_recursion_exceeded(self):
        # Each ":" introduces another recursive _vformat call
        deep = "{x:" * 100 + "}" * 100
        with pytest.raises(ValueError, match="Max string recursion exceeded"):
            self.fmt.format(deep, x="test")

    def test_manual_then_automatic_raises(self):
        with pytest.raises(
            ValueError,
            match="cannot switch from manual field specification to automatic field numbering",
        ):
            self.fmt.format("{0} {}", "a", "b")

    def test_automatic_then_manual_raises(self):
        with pytest.raises(
            ValueError,
            match="cannot switch from manual field specification to automatic field numbering",
        ):
            self.fmt.format("{} {0}", "a", "b")


def test_filename_convention_generate_no_generation_string():
    convention = FileNameConvention(re.compile(""), [])
    with pytest.raises(NotImplementedError):
        convention.generate()


def test_filename_convention_bad_init():
    regex = re.compile(r"file_(?P<group_name>\w+).txt")
    with pytest.raises(ValueError, match="Regex"):
        # FileNameFields have a field that is not in the regex
        FileNameConvention(regex, fields=[FileNameFieldString("group_name2")])

    with pytest.raises(ValueError, match="convention"):
        # regex has a field that is not in FileNameFields
        FileNameConvention(regex, fields=[])

    with pytest.raises(ValueError, match="Generation string"):
        # FileNameFields have a field that is not in the generation string
        FileNameConvention(
            regex,
            fields=[FileNameFieldString("group_name")],
            generation_string="file.txt",
        )

    with pytest.raises(ValueError, match="generation string"):
        # generation_string has a field that is not in the generation string
        FileNameConvention(
            regex,
            fields=[FileNameFieldString("group_name")],
            generation_string="file_{group_name}_ {group_name2}.txt",
        )
