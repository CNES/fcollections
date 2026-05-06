import typing as tp

from fcollections.core import IPredicate
from fcollections.missions import MissionsPhases, Phase

from ._definitions._swot import SwotPhases


class SwotPhasePredicate(IPredicate):
    """Swot phases predicate.

    Converts a phase filter (science/calval orbit) to a range of valid cycle
    numbers.

    Parameters
    ----------
    indexes
        Indexes of the 'cycle_number' and 'pass_number' element in the input
        record of the predicate.
    phase
        SWOT mission phase (calval or science).
    """

    def __init__(self, indexes: tuple[int], phase: SwotPhases):
        self.cycle_number_index = indexes[0]

        phase: Phase = MissionsPhases[phase.name.lower()].value
        self.cycle_bounds = (phase.half_orbits[0][0], phase.half_orbits[1][0])

    def __call__(self, record: tuple[tp.Any, ...]) -> bool:
        cycle_number = record[self.cycle_number_index]
        return self.cycle_bounds[0] <= cycle_number <= self.cycle_bounds[1]

    @classmethod
    def record_fields(cls) -> tuple[str, ...]:
        return ("cycle_number",)

    @classmethod
    def parameters(cls) -> tuple[str, ...]:
        return ("phase",)
