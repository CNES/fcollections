from __future__ import annotations

import dataclasses as dc
import re
import string
import typing as tp
from enum import Enum

import numpy as np

from fcollections.time import ISODuration, Period

from ._codecs import (
    CaseType,
    DateTimeCodec,
    EnumCodec,
    FloatCodec,
    ICodec,
    IntegerCodec,
    ISODurationCodec,
    JulianDayCodec,
    PeriodCodec,
    PeriodDeltaCodec,
    StringCodec,
)
from ._testers import (
    DateTimeTester,
    EnumTester,
    FloatTester,
    GeoBoxTester,
    IntegerTester,
    ITester,
    PeriodTester,
    StringTester,
)

T = tp.TypeVar("T")
U = tp.TypeVar("U")


class FileNameField(ICodec[T], ITester[U, T]):

    def __init__(self, name: str, default: T | None = None, description: str = ""):
        self.default = default
        self.name = name
        self.field_description = description

    @property
    def description(self) -> str:
        return (
            self.test_description
            if len(self.field_description) == 0
            else self.field_description + " " + self.test_description
        )


class FileNameFieldString(FileNameField, StringTester, StringCodec):
    pass


class FileNameFieldDatetime(FileNameField, DateTimeTester, DateTimeCodec):
    """Numpy datetime value.

    Attributes
    ----------
    name: str
        name of the field
    date_fmt: str|List
        date format
    """

    def __init__(
        self,
        name: str,
        date_fmt: str | list,
        default: np.timedelta64 | None = None,
        description: str = "",
    ):
        super().__init__(name, default, description)
        super(FileNameField, self).__init__(date_fmt)


class FileNameFieldDateDelta(
    FileNameField, PeriodTester, PeriodDeltaCodec, DateTimeCodec
):
    """Numpy datetime value.

    Attributes
    ----------
    name: str
        name of the field
    date_fmt: str
        date format
    delta: np.timedelta64
        time delta
    include_stop: bool
        Whether the delta is included or not, default to False
    """

    def __init__(
        self,
        name: str,
        date_fmt: str | list,
        delta: np.timedelta64,
        include_stop: bool = False,
        default: Period | None = None,
        description: str = "",
    ):
        super().__init__(name, default, description)
        DateTimeCodec.__init__(self, date_fmt)
        PeriodDeltaCodec.__init__(self, delta, include_stop)


class FileNameFieldDateJulianDelta(
    FileNameField, PeriodTester, PeriodDeltaCodec, JulianDayCodec
):
    """Datetime value given as a julian day.

    Attributes
    ----------
    name: str
        name of the field
    delta: np.timedelta64
        time delta
    reference
        Reference date for the given julian days
    include_stop: bool
        Whether the delta is included or not, default to False
    julian_day_format
        Whether the julian day is expected as 'days', 'days_hours' or
        'fractional'. For example 24000, 24000_06 or 24000.25
    """

    def __init__(
        self,
        name: str,
        delta: np.timedelta64,
        reference: np.datetime64,
        include_stop: bool = False,
        default: Period | None = None,
        description: str = "",
        julian_day_format: str = "days",
    ):
        super().__init__(name, default, description)
        JulianDayCodec.__init__(self, julian_day_format, reference)
        PeriodDeltaCodec.__init__(self, delta, include_stop)


class FileNameFieldDateJulian(FileNameField, DateTimeTester, JulianDayCodec):

    def __init__(
        self,
        name: str,
        reference: np.datetime64,
        default: Period | None = None,
        description: str = "",
        julian_day_format: str = "days_hours",
    ):
        super().__init__(name, default, description)
        super(FileNameField, self).__init__(julian_day_format, reference)


class FileNameFieldInteger(FileNameField, IntegerTester, IntegerCodec):
    """Integer value.

    Attributes
    ----------
    name: str
        name of the field
    """


class FileNameFieldFloat(FileNameField, FloatTester, FloatCodec):
    """Float value.

    Attributes
    ----------
    name: str
        name of the field
    """


class FileNameFieldEnum(FileNameField, EnumTester, EnumCodec):
    """Enum field for files selection.

    Attributes
    ----------
    name: str
        name of the field
    enum_cls:
        enum class
    """

    def __init__(
        self,
        name: str,
        enum_cls: type[Enum],
        case_type_decoded: CaseType | None = None,
        case_type_encoded: CaseType | None = None,
        underscore_encoded: bool = True,
        default: type[Enum] | None = None,
        description: str = "",
    ):
        super().__init__(name, default, description)
        EnumTester.__init__(self, enum_cls)
        EnumCodec.__init__(
            self, enum_cls, case_type_decoded, case_type_encoded, underscore_encoded
        )

    def choices(self) -> list[str]:
        return [self.encode(x) for x in self.enum_cls]


class FileNameFieldPeriod(FileNameField, PeriodTester, PeriodCodec):
    """Period value.

    Attributes
    ----------
    name: str
        name of the field
    date_fmt: str
        date format
    separator: str
        dates separator. Default: '-'
    """

    def __init__(
        self,
        name: str,
        date_fmt: str,
        separator="_",
        default: Period | None = None,
        description: str = "",
    ):
        super().__init__(name, default, description)
        super(FileNameField, self).__init__(date_fmt, separator)


class FileNameFieldISODuration(FileNameField, ISODurationCodec):
    """ISO8601 duration codes field (PT1D, P1W, ...)"""

    @property
    def test_description(self) -> str:
        description = (
            "ISO8601 duration field can be tested against an "
            "ISODuration object or its string representation "
            "(PT1S, ...)"
        )
        return description

    @property
    def type(self) -> type[ISODuration]:
        return ISODuration

    def sanitize(self, reference: str | ISODuration) -> ISODuration:
        if isinstance(reference, str):
            return self.decode(reference)
        return reference


class FileNameFieldGeoBox(FileNameField, StringCodec, GeoBoxTester):
    """Geobox value.

    Geobox is a tuple (lon_min, lat_min, lon_max, lat_max) in °.

    Attributes
    ----------
    name: str
        name of the field
    """


class FieldFormatter(string.Formatter):

    def __init__(self, fields: dict[str, FileNameField]):
        self._fields = fields
        super().__init__()

    def _vformat(
        self, format_string, args, kwargs, used_args, recursion_depth, auto_arg_index=0
    ):
        # Override of the _vformat method to intercept the object conversion
        # Taken from string.py
        if recursion_depth < 0:
            raise ValueError("Max string recursion exceeded")
        result = []
        for literal_text, field_name, format_spec, conversion in self.parse(
            format_string
        ):

            # output the literal text
            if literal_text:
                result.append(literal_text)

            # if there's a field, output it
            if field_name is not None:
                # this is some markup, find the object and do
                #  the formatting

                # handle arg indexing when empty field_names are given.
                if field_name == "":
                    if auto_arg_index is False:
                        raise ValueError(
                            "cannot switch from manual field "
                            "specification to automatic field "
                            "numbering"
                        )
                    field_name = str(auto_arg_index)
                    auto_arg_index += 1
                elif field_name.isdigit():
                    if auto_arg_index:
                        raise ValueError(
                            "cannot switch from manual field "
                            "specification to automatic field "
                            "numbering"
                        )
                    # disable auto arg incrementing, if it gets
                    # used later on, then an exception will be raised
                    auto_arg_index = False

                # given the field_name, find the object it references
                #  and the argument it came from
                obj, arg_used = self.get_field(field_name, args, kwargs)
                used_args.add(arg_used)

                # PATCH here
                if conversion == "f":
                    obj = self._fields[field_name].encode(obj)
                else:
                    obj = self.convert_field(obj, conversion)

                # expand the format spec, if needed
                format_spec, auto_arg_index = self._vformat(
                    format_spec,
                    args,
                    kwargs,
                    used_args,
                    recursion_depth - 1,
                    auto_arg_index=auto_arg_index,
                )

                # format the object and append to the result
                result.append(self.format_field(obj, format_spec))

        return "".join(result), auto_arg_index


@dc.dataclass
class FileNameConvention:
    """Parse or generate filenames with a convention definition.

    The convention is expressed as both a regex and a simple string to
    handle both parsing and generation. The generation string can be
    omitted and set to None if the convention is only used to parse
    files.
    """

    regex: re.Pattern
    """Pattern for filename matching."""
    fields: list[FileNameField]
    """List of fields, each field name must correspond to a group in the regex
    pattern."""
    generation_string: str | None = None
    """String that will be formatted with the input objects to generate a
    string.

    The string can use the formatting language described in
    help('FORMATTING'). In addition, the formatting can be delegated to each
    field.encode methods by specifying the field name `fn` spec. This allows
    handling more complex objects such as Period. For example with an
    FileNameFieldInteger and FileNameFieldPeriod defined:
    '{cycle_number:>03d}_{period!f}' -> '003_20230102_20240201
    """

    def __post_init__(self):
        self._formatter = FieldFormatter({f.name: f for f in self.fields})
        self._check_consistency()

    def match(self, filename: str) -> tp.Any:
        # Match the file name
        match_object = self.regex.search(filename)
        return match_object

    def parse(self, match_object: re.Match) -> tuple:
        return tuple(
            [
                (
                    f.decode(match_object.group(f.name))
                    if match_object.group(f.name) is not None
                    else f.default
                )
                for f in self.fields
            ]
        )

    def generate(self, **kwargs):
        if self.generation_string is None:
            msg = (
                "The current file name convention is only configured for "
                "parsing. Please specify a 'generation_string' to enable "
                "file name generation"
            )
            raise NotImplementedError(msg)
        try:
            return self._formatter.format(self.generation_string, **kwargs)
        except KeyError as exc:
            raise ValueError("Missing arguments to generate the file name") from exc

    def get_field(self, name: str) -> FileNameField:
        """Retrieve a field from its name.

        Only the first matching field is returned. It is assumed that the
        convention has fields with independant names.

        Parameters
        ----------
        name
            Name of the field to seek

        Returns
        -------
        :
            The requestion FileNameField

        Raises
        ------
        KeyError
            In case the requested field has no match in the convention
        """
        try:
            return next(filter(lambda f: f.name == name, self.fields))
        except StopIteration as exc:
            all_names = [f.name for f in self.fields]
            msg = f"Field {name} does not exists in convention (fields={all_names})"
            raise KeyError(msg) from exc

    def _check_consistency(self):
        self._check_consistency_regex()
        if self.generation_string is not None:
            self._check_consistency_generation_string()

    def _check_consistency_generation_string(self):
        field_names_generation_string = set(
            filter(
                lambda x: x is not None,
                map(lambda x: x[1], self._formatter.parse(self.generation_string)),
            )
        )
        fields_names = {f.name for f in self.fields}

        # Check that the pattern has the necessary groups. It will raise an
        # exception if there is a missing group
        missing_fields = fields_names - field_names_generation_string
        if len(missing_fields) > 0:
            raise ValueError(
                f"Generation string {self.generation_string} misses the "
                f"following fields: '{missing_fields}'"
            )

        # Check that the parser defines all of the regex groups. It will raise
        # an exception if there is a missing field definition
        missing_fields = field_names_generation_string - fields_names
        if len(missing_fields) > 0:
            raise ValueError(
                "The following fields are defined in the generation string but "
                f"not in the FileNameField list: '{missing_fields}'"
            )

    def _check_consistency_regex(self):
        regex_groups = set(self.regex.groupindex.keys())
        fields_names = {f.name for f in self.fields}

        # Check that the pattern has the necessary groups. It will raise an
        # exception if there is a missing group
        missing_regex_groups = fields_names - regex_groups
        if len(missing_regex_groups) > 0:
            raise ValueError(
                f"Regex misses '{self.regex.pattern}' the following fields: "
                f"'{missing_regex_groups}'"
            )

        # Check that the parser defines all of the regex groups. It will raise
        # an exception if there is a missing field definition
        missing_fields = regex_groups - fields_names
        if len(missing_fields) > 0:
            raise ValueError(
                f"Missing fields definition in convention: '{missing_fields}'"
            )
