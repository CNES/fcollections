import abc
import collections.abc
import datetime as dt
import typing as tp
from enum import Enum

import numpy as np

from fcollections.time import Period

T = tp.TypeVar("T")
U = tp.TypeVar("U")


class ITester(abc.ABC, tp.Generic[U, T]):
    """Compare two objects of types U and T.

    This interface can be used to define filters that needs to compare
    objects with different but close types. For example, an integer with
    another integer or a list of integers.

    In addition to the testing functionality, this interface also
    provides a way to cast an object to one of the expected U types.
    This is useful for sanitizing user inputs that are in the simplest
    possible types. Such example is the automatic building of a
    np.datetime64 from a string given by the user ('2024-01-01')
    """

    def sanitize(self, reference: tp.Any) -> U:
        """Cast to one of the types handled by this tester.

        Parameters
        ----------
        reference
            The reference object to cast

        Returns
        -------
        :
            The input cast to the proper type
        """
        return reference

    def test(self, reference: U, tested: T) -> bool:
        """Compare two objects of similar types.

        Parameters
        ----------
        reference
            The reference object
        tested
            The tested object

        Returns
        -------
        :
            True if the test is successful, False otherwise
        """
        return reference == tested

    @property
    @abc.abstractmethod
    def test_description(self) -> str:
        """User-friendly description of the possible types for the
        reference."""

    @property
    @abc.abstractmethod
    def type(self) -> type[T]:
        """Type of the tested field."""

    @property
    def type_name(self) -> str:
        """Type name of the tested field for signature parameters."""
        # Most tester will only accept one class as a reference value. The name
        # of this class can be used for defining the signature parameters
        return self.type.__name__


class StringTester(ITester[str, str]):

    @property
    def test_description(self) -> str:
        return (
            "As a String field, it can filtered by giving a reference "
            "string. The tested value from the file name will be filtered out if it"
            " is not equal to the reference value."
        )

    @property
    def type(self) -> type[str]:
        return str


class FloatTester(ITester[float, float]):

    @property
    def test_description(self) -> str:
        return (
            "As a Float field, it can be filtered by using a reference "
            "float value. The tested value found in the file name will be filtered "
            "out if it is not equal to the reference value."
        )

    @property
    def type(self) -> type[float]:
        return float


class IntegerTester(ITester[list[int] | slice | int, int]):

    def test(self, reference: list[int] | slice | int, tested: int) -> bool:
        if isinstance(reference, list):
            return tested in reference
        elif isinstance(reference, slice):
            return reference.start <= tested < reference.stop
        else:
            return tested == reference

    @property
    def test_description(self) -> str:
        return (
            "As a Integer field, it can be filtered by using a reference "
            "value. The reference value can either be a list, a slice or an integer"
            ". The tested value from the file name will be filtered out if it is "
            "outside the given list/slice or not equal to the integer value."
        )

    @property
    def type(self) -> type[int]:
        return list[int] | slice | int

    @property
    def type_name(self) -> str:
        return str(self.type)


class EnumTester(ITester[type[Enum] | list[type[Enum]], type[Enum]]):

    def __init__(self, enum_cls: type[Enum]):
        self.enum_cls = enum_cls

    @property
    def test_description(self) -> str:
        return (
            "As an Enum field, it can be filtered using a reference "
            f"{self.enum_cls} or its equivalent string. The tested value found in "
            "the file name will be filtered out if it is not equal to the given "
            f"enum field. Possible values are: {[e.name for e in self.enum_cls]}"
        )

    def test(
        self, reference: type[Enum] | list[type[Enum]], tested: type[Enum]
    ) -> bool:
        if hasattr(reference, "__iter__"):
            return tested in reference
        else:
            return tested == reference

    def sanitize(
        self, reference: str | type[Enum] | list[str] | list[type[Enum]]
    ) -> type[Enum] | tuple[type[Enum], ...]:
        if isinstance(reference, str):
            return self.enum_cls[reference]
        elif (
            isinstance(reference, collections.abc.Sequence)
            and len(reference) > 0
            and isinstance(reference[0], str)
        ):
            return tuple([self.enum_cls[nested] for nested in reference])
        else:
            return reference

    @property
    def type(self) -> type[Enum]:
        return self.enum_cls


class DateTimeTester(ITester[Period | np.datetime64, np.datetime64]):

    @property
    def test_description(self) -> str:
        return (
            "As a DateTime field, it can be filtered by giving a reference "
            "Period, datetime. The tested value from the file name will be "
            "filtered out if it is not included or not equal to the reference "
            "Period or datetime respectively. The reference value can be given "
            "as a string or tuple of string following with the numpy date "
            "formatting [%Y-%m-%dT%H:%M:%S])"
        )

    def test(self, reference: Period | np.datetime64, tested: np.datetime64) -> bool:
        if isinstance(reference, Period):
            return reference.intersects(tested)
        else:
            return reference == tested

    def sanitize(
        self,
        reference: (
            tuple[str | None | np.datetime64, str | None | np.datetime64]
            | Period
            | np.datetime64
            | str
        ),
    ) -> Period | np.datetime64:
        return _sanitize_time(reference)

    @property
    def type(self) -> type[np.datetime64]:
        return np.datetime64


class PeriodTester(ITester[Period | np.datetime64, Period]):

    def test(self, reference: Period | np.datetime64, tested: Period) -> bool:
        return tested.intersects(reference)

    @property
    def test_description(self) -> str:
        return (
            "As a Period field, it can be filtered by giving a reference "
            "Period or datetime. The tested value from the file name will be "
            "filtered out if it does not intersect the reference Period or does"
            " not contain the reference datetime. The reference value can be "
            "given as a string or tuple of string following the "
            "[%Y-%m-%dT%H:%M:%S] formatting"
        )

    def sanitize(
        self,
        reference: (
            tuple[str | None | np.datetime64, str | None | np.datetime64]
            | Period
            | np.datetime64
            | str
        ),
    ) -> Period | np.datetime64:
        return _sanitize_time(reference)

    @property
    def type(self) -> type[Period]:
        return Period


GeoBox: tp.TypeAlias = tuple[float, float, float, float]


class GeoBoxTester(ITester[GeoBox, GeoBox]):

    @property
    def test_description(self) -> str:
        return (
            "As a GeoBox field, it can be filtered by giving a reference "
            "GeoBox. A file will be filtered out if the GeoBox extracted from "
            "its file name does not intersect the reference GeoBox."
        )

    def test(self, reference: GeoBox, tested: GeoBox) -> bool:
        longitude_match = (reference[0] <= tested[0] <= reference[2]) or (
            reference[0] <= tested[2] <= reference[2]
        )

        latitude_match = (reference[1] <= tested[1] <= reference[3]) or (
            reference[1] <= tested[3] <= reference[3]
        )

        return longitude_match and latitude_match

    @property
    def type(self) -> type[GeoBox]:
        return GeoBox


def _sanitize_time(
    reference: (
        tuple[str | None | np.datetime64, str | None | np.datetime64]
        | Period
        | np.datetime64
        | str
    ),
) -> Period | np.datetime64:
    if isinstance(reference, tuple):
        start = reference[0] if reference[0] is not None else dt.datetime.min
        stop = reference[1] if reference[1] is not None else dt.datetime.max
        return Period(start=np.datetime64(start), stop=np.datetime64(stop))
    elif isinstance(reference, str):
        return np.datetime64(reference)
    else:
        return reference
