import typing as tp
from pathlib import Path
from unittest.mock import Mock

import fsspec.implementations.memory as fs_mem
import numpy as np
import pytest

from fcollections.core import DiscreteTimesMixin, DownloadMixin, PeriodMixin
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
