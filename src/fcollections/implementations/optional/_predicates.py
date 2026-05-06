from __future__ import annotations

import functools
import logging
import typing as tp

from fcollections.core import FileNameField, FileNameFieldGeoBox, IFilterBuilder
from fcollections.geometry import query_half_orbits_intersect
from fcollections.missions import PHASES, Missions

logger = logging.getLogger(__name__)


class SwotGeometryFilterBuilder(IFilterBuilder):
    """Predicate builder for swot karin footprints.

    This predicate builder can transform a box in a callable that can predict if
    a given half orbit crosses the box. It uses KaRIn reference footprints for
    one cycle.

    Parameters
    ----------
    indexes
        Indexes of the 'cycle_number' and 'pass_number' element in the input
        record of the predicate
    bbox
        Bounding box, given as lon_min, lat_min, lon_max, lat_max
    """

    @classmethod
    def build_predicate(
        cls, record_indexes: dict[str, int], bbox: tuple[float, float, float, float]
    ) -> tp.Callable[[tuple[tp.Any, ...]], bool]:

        def selected(
            cycle_number: int,
            pass_number: int,
            cycle_range: tuple[int, int | None],
            selected_pass_numbers: list[int],
        ) -> bool:
            return (
                (cycle_range[0] <= cycle_number)
                and (cycle_range[1] is None or cycle_number <= cycle_range[1])
                and (pass_number in selected_pass_numbers)
            )

        predicates = []
        for phase in PHASES[Missions.Swot]:
            pass_numbers_intersect = list(
                query_half_orbits_intersect(bbox, phase).pass_number
            )
            logger.info(
                "The bbox intersects with pass numbers (%s phase): %s",
                phase.short_name.lower(),
                pass_numbers_intersect,
            )

            predicates.append(
                functools.partial(
                    selected,
                    cycle_range=phase.cycles,
                    selected_pass_numbers=pass_numbers_intersect,
                )
            )

        cycle_number_index = record_indexes["cycle_number"]
        pass_number_index = record_indexes["pass_number"]

        def _predicate(record: tuple[tp.Any, ...]) -> bool:
            cycle_number, pass_number = (
                record[cycle_number_index],
                record[pass_number_index],
            )
            return functools.reduce(
                lambda x, y: x or y,
                [predicate(cycle_number, pass_number) for predicate in predicates],
            )

        return _predicate

    @classmethod
    def build_filter(cls):
        msg = "Swot Geometry Filter can only be built as a predicate for records."
        raise NotImplementedError(msg)

    @classmethod
    def parameter(cls) -> FileNameField:
        return FileNameFieldGeoBox(
            "bbox",
            description=(
                "The bounding box (lon_min, lat_min, lon_max, lat_max) used to "
                "select the data in a given area. Longitude coordinates can be "
                "provided in [-180, 180[ or [0, 360[ convention. If bbox's "
                "longitude crosses the circularity, it will be split in two "
                "subboxes to ensure a proper selection (e.g. longitude interval"
                ": [170, -170] -> data in [170, 180[ and [-180, -170] will be "
                "retrieved"
            ),
        )
