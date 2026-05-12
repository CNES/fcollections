import typing as tp
import warnings
from pathlib import Path
from unittest.mock import Mock

import fsspec.implementations.memory as fs_mem
import numpy as np
import pytest

from fcollections.core import (
    DiscreteTimesMixin,
    DownloadMixin,
    HalfOrbitMixin,
    PerformanceWarning,
    PeriodMixin,
)
from fcollections.time import Period


class PeriodMixinEmpty(PeriodMixin):

    def filter_values(self, filter_name: str, *args, **kwargs) -> set[tp.Any]:
        return set()


def test_period_mixin_empty():
    mixin = PeriodMixinEmpty()
    assert mixin.time_coverage() is None
    assert len(list(mixin.time_holes())) == 0


class PeriodMixinStub(PeriodMixin):

    def filter_values(self, field_name: str, *args, **kwargs) -> set[tp.Any]:
        return {
            Period(
                np.datetime64("2024-01-01"),
                np.datetime64("2024-01-02"),
                include_stop=False,
            ),
            Period(
                np.datetime64("2024-01-02"),
                np.datetime64("2024-01-03"),
                include_stop=False,
            ),
            Period(
                np.datetime64("2024-01-04"),
                np.datetime64("2024-01-05"),
                include_stop=False,
            ),
            Period(
                np.datetime64("2024-01-10"),
                np.datetime64("2024-01-20"),
                include_start=False,
                include_stop=False,
            ),
        }


def test_period_mixin():
    mixin = PeriodMixinStub()
    assert mixin.time_coverage() == Period(
        np.datetime64("2024-01-01"), np.datetime64("2024-01-20"), include_stop=False
    )
    assert list(mixin.time_holes()) == [
        Period(
            np.datetime64("2024-01-03"), np.datetime64("2024-01-04"), include_stop=False
        ),
        Period(np.datetime64("2024-01-05"), np.datetime64("2024-01-10")),
    ]


class DiscreteTimesEmpty(DiscreteTimesMixin):

    def filter_values(self, field_name: str, *args, **kwargs) -> set[tp.Any]:
        return set()


def test_discrete_times_mixin_empty():
    mixin = DiscreteTimesEmpty(np.timedelta64(1, "D"))
    assert mixin.time_coverage() is None
    assert len(list(mixin.time_holes())) == 0


class DiscreteTimesStub(DiscreteTimesMixin):

    def filter_values(self, *args, **kwargs) -> set[tp.Any]:
        return {
            np.datetime64("2024-01-01"),
            np.datetime64("2024-01-02"),
            np.datetime64("2024-01-04"),
            np.datetime64("2024-01-10"),
        }


def test_discrete_times_mixin():
    mixin = DiscreteTimesStub(np.timedelta64(1, "D"))
    assert mixin.time_coverage() == Period(
        np.datetime64("2024-01-01"), np.datetime64("2024-01-10")
    )
    assert list(mixin.time_holes()) == [
        Period(
            np.datetime64("2024-01-02"),
            np.datetime64("2024-01-04"),
            include_start=False,
            include_stop=False,
        ),
        Period(
            np.datetime64("2024-01-04"),
            np.datetime64("2024-01-10"),
            include_start=False,
            include_stop=False,
        ),
    ]


def test_discrete_times_mixin_no_sampling():
    mixin = DiscreteTimesStub()
    assert mixin.time_coverage() == Period(
        np.datetime64("2024-01-01"), np.datetime64("2024-01-10")
    )
    with pytest.warns(UserWarning):
        assert list(mixin.time_holes()) == []


class HalfOrbitMixinEmpty(HalfOrbitMixin, PeriodMixinStub):

    filter_builders = []

    def filter_values(self, field_name: str, *args, **kwargs) -> set[tp.Any]:
        return set()


def test_half_orbit_mixin_empty():
    mixin = HalfOrbitMixinEmpty()
    assert mixin.cycle_range() is None
    assert mixin.half_orbit_range() is None
    assert mixin.time_coverage() is None
    assert len(list(mixin.time_holes())) == 0


class HalfOrbitMixinStub(HalfOrbitMixin, PeriodMixinStub):

    filter_builders = []

    def filter_values(self, filter_name, **kwargs):
        if filter_name == "time":
            return PeriodMixinStub.filter_values(self, filter_name)
        elif filter_name == "cycle_number":
            return {1, 2, 3, 500}
        elif filter_name == "pass_number":
            return {x * kwargs["cycle_number"] for x in range(1, 4)}


def test_half_orbit_mixin():
    mixin_temporal = PeriodMixinStub()
    mixin = HalfOrbitMixinStub()
    assert mixin.cycle_range() == (1, 500)
    assert mixin.half_orbit_range() == ((1, 1), (500, 1500))
    assert mixin.time_coverage() == mixin_temporal.time_coverage()


class HalfOrbitMixinStubWithWarnings(HalfOrbitMixinStub):

    def filter_values(self, filter_name, **kwargs):
        warnings.warn("Slow listing", PerformanceWarning)
        return super().filter_values(filter_name, **kwargs)


def test_half_orbit_mixin_slow_time_coverage():
    mixin_temporal = PeriodMixinStub()
    mixin = HalfOrbitMixinStubWithWarnings()
    assert mixin.time_coverage() == mixin_temporal.time_coverage()


FILTER_BUILDER_MOCK = Mock()
FILTER_BUILDER_MOCK.parameter().name = "phase"
FILTER_BUILDER_MOCK.target_fields = Mock(return_value=("cycle_number", "pass_number"))


class HalfOrbitMixinStubWithFilterBuilders(HalfOrbitMixinStub):
    filter_builders = [FILTER_BUILDER_MOCK]

    def filter_values(self, filter_name, **kwargs):
        if filter_name != "cycle_number" and "phase" in kwargs:
            raise ValueError()
        return super().filter_values(filter_name, **kwargs)


def test_half_orbit_mixin_filter_builder_clean():
    mixin = HalfOrbitMixinStubWithFilterBuilders()
    assert mixin.time_coverage(phase="CALVAL") == mixin.time_coverage()
    assert mixin.half_orbit_range(phase="CALVAL") == mixin.half_orbit_range()


class DownloadMixinMemory(DownloadMixin):

    @property
    def fs(self):
        return fs_mem.MemoryFileSystem()


@pytest.fixture
def files_ini_memory():
    fs = fs_mem.MemoryFileSystem()
    fs.touch("/file1.txt")
    fs.touch("/file2.txt")
    fs.touch("/file3.txt")


def test_download(tmp_path_factory: pytest.TempPathFactory, files_ini_memory: None):
    path = tmp_path_factory.mktemp("output")
    assert list(path.iterdir()) == []

    mixin = DownloadMixinMemory()
    mixin.download(["file1.txt", "file3.txt"], path)
    assert sorted([x.name for x in path.iterdir()]) == ["file1.txt", "file3.txt"]


def test_download_force(
    tmp_path_factory: pytest.TempPathFactory, files_ini_memory: None
):
    path = tmp_path_factory.mktemp("output")
    mixin = DownloadMixinMemory()
    downloaded = mixin.download(["file1.txt"], path)
    assert sorted([Path(x).name for x in downloaded]) == ["file1.txt"]

    downloaded = mixin.download(["file1.txt", "file2.txt"], path)
    assert sorted([Path(x).name for x in downloaded]) == ["file2.txt"]

    downloaded = mixin.download(["file1.txt", "file3.txt"], path, force_download=True)
    assert sorted([Path(x).name for x in downloaded]) == ["file1.txt", "file3.txt"]


class DownloadMixinMock(DownloadMixin):

    @property
    def fs(self):
        mock = Mock()
        mock.get_file = Mock(side_effect=TimeoutError("foo"))
        return mock


def test_download_timeout(tmp_path_factory: pytest.TempPathFactory):
    path = tmp_path_factory.mktemp("output")
    mixin = DownloadMixinMock()
    downloaded = mixin.download(["file1.txt"], path)
    assert len(downloaded) == 0
