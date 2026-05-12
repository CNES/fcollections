import typing as tp

from fcollections.core import CaseType, FileNameField, FileNameFieldEnum, IFilterBuilder
from fcollections.missions import MissionsPhases, Phase

from ._definitions._swot import SwotPhases


class SwotPhaseFilterBuilder(IFilterBuilder):
    """Swot phases filter builder.

    Converts a phase filter (science/calval orbit) to a range of valid
    cycle numbers.
    """

    @classmethod
    def build_filter(cls, phase: SwotPhases) -> dict[str, slice]:
        """Converts a ``phase`` filter to a ``cycle_number``.

        Parameters
        ----------
        phase
            SWOT mission phase (calval or science).

        Returns
        -------
        dict[str, tp.Any]
            Mapping associating the filter name to its authorized values.
        """
        phase: Phase = MissionsPhases[phase.name.lower()].value
        return {
            "cycle_number": slice(phase.half_orbits[0][0], phase.half_orbits[1][0] + 1)
        }

    @classmethod
    def build_predicate(self, record_mapping: dict[str, int], *args: tp.Any):
        msg = "SwotPhase filter can only be built as a simple filter."
        raise NotImplementedError(msg)

    @classmethod
    def parameter(cls) -> FileNameField:
        return FileNameFieldEnum(
            "phase",
            SwotPhases,
            description=(
                "Phase of the SWOT mission that can be used to select the "
                "associated cycle numbers range."
            ),
            case_type_decoded=CaseType.upper,
            case_type_encoded=CaseType.lower,
        )

    @classmethod
    def target_fields(cls) -> tuple[str, ...]:
        return ("cycle_number",)
