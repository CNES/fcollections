from __future__ import annotations

import typing as tp

import numpy as np
import pytest

from fcollections.time import Period

if tp.TYPE_CHECKING:
    import numpy.typing as np_t


def test_inconsistent_period():
    with pytest.raises(ValueError):
        Period(np.datetime64("2023-01-02"), np.datetime64("2023-01-01"))


@pytest.fixture
def reference() -> Period:
    return Period(np.datetime64("2023-01-01"), np.datetime64("2023-02-01"))


@pytest.mark.parametrize(
    "tested, expected",
    [
        (
            Period(np.datetime64("2023-01-01"), np.datetime64("2023-02-01")),
            (True, True, True, False, False, True, True, True, False, False),
        ),
        (
            Period(np.datetime64("2023-01-05"), np.datetime64("2023-01-21")),
            (False, True, True, True, True, False, False, False, False, False),
        ),
        (
            Period(np.datetime64("2022-12-25"), np.datetime64("2023-02-25")),
            (False, False, False, False, False, False, True, True, True, True),
        ),
        (
            Period(np.datetime64("2022-12-25"), np.datetime64("2023-01-01")),
            (False, True, False, True, False, False, False, True, False, True),
        ),
        (
            Period(np.datetime64("2022-12-25"), np.datetime64("2023-02-01")),
            (False, True, False, False, False, False, True, True, False, True),
        ),
        (
            Period(np.datetime64("2023-02-01"), np.datetime64("2023-02-25")),
            (False, False, True, False, True, False, True, False, True, False),
        ),
        (
            Period(np.datetime64("2023-01-01"), np.datetime64("2023-02-25")),
            (False, False, True, False, False, False, True, True, True, False),
        ),
    ],
    ids=[
        "same",
        "enclosed",
        "enclosing",
        "before",
        "before_overlap",
        "after",
        "after_overlap",
    ],
)
def test_comparisons(reference: Period, tested: Period, expected: tuple[bool, ...]):
    assert (reference == tested) == expected[0]
    assert (reference >= tested) == expected[1]
    assert (reference <= tested) == expected[2]
    assert (reference > tested) == expected[3]
    assert (reference < tested) == expected[4]

    # Test symetry
    assert (tested == reference) == expected[5]
    assert (tested >= reference) == expected[6]
    assert (tested <= reference) == expected[7]
    assert (tested > reference) == expected[8]
    assert (tested < reference) == expected[9]


@pytest.mark.parametrize(
    "tested, expected",
    [
        (
            Period(
                np.datetime64("2023-01-01"), np.datetime64("2023-02-01"), True, True
            ),
            True,
        ),
        (
            Period(
                np.datetime64("2023-01-01"), np.datetime64("2023-02-01"), False, True
            ),
            True,
        ),
        (
            Period(
                np.datetime64("2023-01-01"), np.datetime64("2023-02-01"), True, False
            ),
            True,
        ),
        (
            Period(
                np.datetime64("2023-01-01"), np.datetime64("2023-02-01"), False, False
            ),
            True,
        ),
    ],
)
def test_intersects_same_periods(reference: Period, tested: Period, expected: bool):
    assert reference.intersects(tested) == expected
    assert tested.intersects(reference) == expected


@pytest.mark.parametrize(
    "tested, expected",
    [
        (
            Period(
                np.datetime64("2022-01-01"), np.datetime64("2023-01-01"), True, True
            ),
            True,
        ),
        (
            Period(
                np.datetime64("2022-01-01"), np.datetime64("2023-01-01"), False, True
            ),
            True,
        ),
        (
            Period(
                np.datetime64("2022-01-01"), np.datetime64("2023-01-01"), True, False
            ),
            False,
        ),
        (
            Period(
                np.datetime64("2022-01-01"), np.datetime64("2023-01-01"), False, False
            ),
            False,
        ),
        (
            Period(
                np.datetime64("2023-02-01"), np.datetime64("2023-03-01"), True, True
            ),
            True,
        ),
        (
            Period(
                np.datetime64("2023-02-01"), np.datetime64("2023-03-01"), False, True
            ),
            False,
        ),
        (
            Period(
                np.datetime64("2023-02-01"), np.datetime64("2023-03-01"), True, False
            ),
            True,
        ),
        (
            Period(
                np.datetime64("2023-02-01"), np.datetime64("2023-03-01"), False, False
            ),
            False,
        ),
    ],
)
def test_intersects_common_bound_ext(
    reference: Period, tested: Period, expected: tuple
):
    assert reference.intersects(tested) == expected
    assert tested.intersects(reference) == expected


@pytest.mark.parametrize(
    "tested, expected",
    [
        (
            Period(
                np.datetime64("2023-01-01"), np.datetime64("2023-01-15"), True, True
            ),
            True,
        ),
        (
            Period(
                np.datetime64("2023-01-01"), np.datetime64("2023-01-15"), False, False
            ),
            True,
        ),
        (
            Period(
                np.datetime64("2023-01-01"), np.datetime64("2023-03-01"), True, True
            ),
            True,
        ),
        (
            Period(
                np.datetime64("2023-01-01"), np.datetime64("2023-03-01"), False, False
            ),
            True,
        ),
        (
            Period(
                np.datetime64("2023-01-15"), np.datetime64("2023-02-01"), True, True
            ),
            True,
        ),
        (
            Period(
                np.datetime64("2023-01-15"), np.datetime64("2023-02-01"), False, False
            ),
            True,
        ),
        (
            Period(
                np.datetime64("2022-01-01"), np.datetime64("2023-02-01"), True, True
            ),
            True,
        ),
        (
            Period(
                np.datetime64("2022-01-01"), np.datetime64("2023-02-01"), False, False
            ),
            True,
        ),
    ],
)
def test_intersects_common_bound_int(
    reference: Period, tested: Period, expected: tuple
):
    assert reference.intersects(tested) == expected
    assert tested.intersects(reference) == expected


@pytest.mark.parametrize(
    "tested, expected",
    [
        (
            Period(
                np.datetime64("2023-01-15"), np.datetime64("2023-01-16"), True, True
            ),
            True,
        ),
        (
            Period(
                np.datetime64("2023-01-15"), np.datetime64("2023-01-16"), False, False
            ),
            True,
        ),
        (
            Period(
                np.datetime64("2022-01-02"), np.datetime64("2023-03-01"), True, True
            ),
            True,
        ),
        (
            Period(
                np.datetime64("2022-01-02"), np.datetime64("2023-03-01"), False, False
            ),
            True,
        ),
    ],
)
def test_intersects_int_ext(reference: Period, tested: Period, expected: tuple):
    assert reference.intersects(tested) == expected
    assert tested.intersects(reference) == expected


@pytest.fixture
def periods() -> list[Period]:
    return [
        Period(np.datetime64("2023-01-01"), np.datetime64("2023-02-01")),
        Period(np.datetime64("2022-12-01"), np.datetime64("2023-01-01")),
        Period(np.datetime64("2023-02-01"), np.datetime64("2023-03-01")),
        Period(np.datetime64("2022-11-01"), np.datetime64("2022-12-01")),
    ]


def test_sort(periods: list[Period]):
    assert sorted(periods) == [periods[ii] for ii in [3, 1, 0, 2]]
    assert sorted(periods, reverse=True) == [periods[ii] for ii in [2, 0, 1, 3]]


def test_hashable(periods: list[Period]):
    """Using a frozen dataclass will automatically generate a hash function."""
    assert hash(periods[0]) is not None


def test_repr():
    period = Period("2023-01-01", "2023-02-01T01:02:03.004", include_stop=False)
    assert str(period) == "[2023-01-01, 2023-02-01T01:02:03.004["
    period = Period("2023-01-01", "2023-02-01T01:02:03.004", include_start=False)
    assert str(period) == "]2023-01-01, 2023-02-01T01:02:03.004]"


def test_center():
    assert Period(
        np.datetime64("2023-01-01"), np.datetime64("2023-01-02")
    ).center == np.datetime64("2023-01-01T12")


@pytest.mark.parametrize(
    "period1, period2, expected",
    [
        (
            ("2023-01-01", "2023-02-01", True, False),
            ("2023-01-01", "2023-02-01", False, True),
            ("2023-01-01", "2023-02-01", True, True),
        ),
        (
            ("2023-01-01", "2023-02-01", True, True),
            ("2022-01-01", "2025-02-01", False, False),
            ("2022-01-01", "2025-02-01", False, False),
        ),
        (
            ("2023-01-01", "2023-02-01", False, True),
            ("2024-01-01", "2024-02-01", False, True),
            ("2023-01-01", "2024-02-01", False, True),
        ),
    ],
)
def test_union(
    period1: tuple[str, str, bool, bool],
    period2: tuple[str, str, bool, bool],
    expected: tuple[str, str, bool, bool],
):
    period1 = Period(
        np.datetime64(period1[0]), np.datetime64(period1[1]), period1[2], period1[3]
    )
    period2 = Period(
        np.datetime64(period2[0]), np.datetime64(period2[1]), period2[2], period2[3]
    )
    expected = Period(
        np.datetime64(expected[0]), np.datetime64(expected[1]), expected[2], expected[3]
    )

    assert period1.union(period2) == expected
    assert period2.union(period1) == expected


@pytest.mark.parametrize(
    "period1, period2, expected",
    [
        (
            ("2023-01-01", "2023-02-01", True, False),
            ("2023-01-01", "2023-02-01", False, True),
            ("2023-01-01", "2023-02-01", False, False),
        ),
        (
            ("2023-01-01", "2023-02-01", True, True),
            ("2022-01-01", "2025-02-01", False, False),
            ("2023-01-01", "2023-02-01", True, True),
        ),
        (
            ("2023-01-01", "2023-02-01", False, True),
            ("2024-01-01", "2024-02-01", False, True),
            None,
        ),
        (
            ("2023-01-01", "2023-02-01", True, True),
            ("2023-02-01", "2024-02-01", True, True),
            ("2023-02-01", "2023-02-01", True, True),
        ),
        (
            ("2023-01-01", "2023-02-01", True, False),
            ("2023-02-01", "2024-02-01", True, True),
            None,
        ),
        (
            ("2023-01-01", "2023-02-01", True, True),
            ("2023-02-01", "2024-02-01", False, True),
            None,
        ),
    ],
)
def test_intersection(
    period1: tuple[str, str, bool, bool],
    period2: tuple[str, str, bool, bool],
    expected: tuple[str, str, bool, bool] | None,
):
    period1 = Period(
        np.datetime64(period1[0]), np.datetime64(period1[1]), period1[2], period1[3]
    )
    period2 = Period(
        np.datetime64(period2[0]), np.datetime64(period2[1]), period2[2], period2[3]
    )
    expected = (
        Period(
            np.datetime64(expected[0]),
            np.datetime64(expected[1]),
            expected[2],
            expected[3],
        )
        if expected is not None
        else None
    )

    assert period1.intersection(period2) == expected
    assert period2.intersection(period1) == expected


@pytest.mark.parametrize(
    "time, expected",
    [
        (np.datetime64("2023-01-01"), np.datetime64("2023-01-01")),
        (np.datetime64("2023-01-10"), np.datetime64("2023-01-10")),
        (np.datetime64("2023-02-01"), np.datetime64("2023-02-01")),
        (np.datetime64("2022-01-10"), None),
        (np.datetime64("2027-01-10"), None),
    ],
)
def test_intersection_time(
    reference: Period,
    time: np_t.NDArray[np.float64],
    expected: np_t.NDArray[np.float64] | None,
):
    assert reference.intersection(time) == expected
