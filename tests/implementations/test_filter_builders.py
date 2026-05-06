import pytest

from fcollections.implementations import SwotPhaseFilterBuilder, SwotPhases
from fcollections.implementations.optional import SwotGeometryFilterBuilder


def test_geometry_filter_builder_no_filter():
    builder = SwotGeometryFilterBuilder()
    with pytest.raises(NotImplementedError):
        builder.build_filter()


def test_phase_filter_builder():
    builder = SwotPhaseFilterBuilder()
    actual = builder.build_filter(SwotPhases.CALVAL)
    assert actual == {"cycle_number": slice(402, 579)}


def test_phase_filter_builder_no_predicate():
    builder = SwotPhaseFilterBuilder()
    with pytest.raises(NotImplementedError):
        builder.build_predicate({})
