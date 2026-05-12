import pyinterp.geometry.geographic as pyi_geom
import pytest

from fcollections.geometry import expand_box


@pytest.mark.parametrize(
    "box, expected",
    [
        (((80, 67), (86, 73)), ((78.75, 66.09375), (87.1875, 73.125))),
        (((-97.8, 25.3), (-96.8, 28.4)), ((-98.4375, 23.90625), (-95.625, 29.53125))),
        (((5.9, 4.2), (7.5, 4.8)), ((5.625, 2.8125), (8.4375, 5.625))),
    ],
    ids=["bigger_than_precision", "long_vertical_box", "smaller_than_precision"],
)
def test_expand_box(
    box: tuple[tuple[float, float], tuple[float, float]],
    expected: tuple[tuple[float, float], tuple[float, float]],
):
    """Test box expansion with a given precision."""
    box = pyi_geom.Box(min_corner=box[0], max_corner=box[1])
    expected = pyi_geom.Box(min_corner=expected[0], max_corner=expected[1])

    actual = expand_box(box, precision=3)
    assert (
        actual.min_corner.lon == expected.min_corner.lon
        and actual.min_corner.lat == expected.min_corner.lat
        and actual.max_corner.lon == expected.max_corner.lon
        and actual.max_corner.lat == expected.max_corner.lat
    )
