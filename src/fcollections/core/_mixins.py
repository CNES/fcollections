from __future__ import annotations

import abc
import functools
import logging
import os
import typing as tp
import warnings

import fsspec

from fcollections.time import (
    Period,
    fuse_successive_periods,
    periods_envelop,
    periods_holes,
    times_holes,
)

from ._filesdb import PerformanceWarning

if tp.TYPE_CHECKING:  # pragma: no cover
    import numpy as np

logger = logging.getLogger(__name__)


def suppress_performance_warning(func):

    @functools.wraps(func)
    def suppressed(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PerformanceWarning)
            return func(*args, **kwargs)

    return suppressed


class ITemporalMixin(abc.ABC):

    @abc.abstractmethod
    def filter_values(self, filter_name: str, **kwargs) -> set[tp.Any]:
        """The mixin relies on this method to build new functionalities."""

    @abc.abstractmethod
    def time_holes(self, **filters):
        """Find the holes in time coverage.

        Returns
        -------
        :
            A generator yielding Period representing holes in the data
        """

    @abc.abstractmethod
    def time_coverage(self, **filters) -> Period | None:
        """Find the time extent of the netcdf files.

        Returns
        -------
        :
            A Period representing the period covered by the data
        """


class PeriodMixin(ITemporalMixin):

    @suppress_performance_warning
    def time_holes(self, **filters) -> tp.Generator[Period, None, None]:
        periods = sorted(self.filter_values("time", **filters))

        if len(periods) == 0:
            logger.info("All data filtered out with %s", filters)
            return []
        reduced = fuse_successive_periods(periods)
        return periods_holes(reduced)

    @suppress_performance_warning
    def time_coverage(self, **filters) -> Period | None:
        periods = sorted(self.filter_values("time", **filters))

        if len(periods) == 0:
            logger.info("All data filtered out with %s", filters)
            return None
        return periods_envelop(periods)


class DiscreteTimesMixin(ITemporalMixin):

    def __init__(self, sampling: np.timedelta64 | None = None):
        self.sampling = sampling

    @suppress_performance_warning
    def time_holes(self, **filters) -> tp.Generator[Period, None, None]:
        if self.sampling is None:
            msg = """No sampling specified, holes detection in the time serie
            cannot proceed"""
            warnings.warn(msg)
            return []

        times = sorted(self.filter_values("time", **filters))

        if len(times) == 0:
            logger.info("All data filtered out with %s", filters)
            return []
        return times_holes(times, self.sampling)

    @suppress_performance_warning
    def time_coverage(self, **filters) -> Period | None:
        times = sorted(self.filter_values("time", **filters))

        if len(times) == 0:
            logger.info("All data filtered out with %s", filters)
            return None
        return Period(times[0], times[-1])


class HalfOrbitMixin:

    def cycle_range(self, **filters) -> tuple[int, int]:
        cycles = sorted(self.filter_values("cycle_number", **filters))
        return cycles[0], cycles[-1]

    @suppress_performance_warning
    def half_orbit_range(self, **filters) -> tuple[tuple[int, int], tuple[int, int]]:
        first_cycle, last_cycle = self.cycle_range(**filters)

        for filter_builder in self.filter_builders:
            if filter_builder.target_fields() == ("cycle_number",):
                filters.pop(filter_builder.parameter().name, None)

        filters["cycle_number"] = first_cycle
        first_pass = sorted(self.filter_values("pass_number", **filters))[0]

        filters["cycle_number"] = last_cycle
        last_pass = sorted(self.filter_values("pass_number", **filters))[-1]

        return (first_cycle, first_pass), (last_cycle, last_pass)

    def time_coverage(self, **filters) -> Period | None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", PerformanceWarning)

            try:
                cycle_range = self.cycle_range(**filters)
                for filter_builder in self.filter_builders:
                    if filter_builder.target_fields() == ("cycle_number",):
                        filters.pop(filter_builder.parameter().name, None)
                filters["cycle_number"] = list(cycle_range)
            except PerformanceWarning:
                # Don't try to accelerate if we must fall back to a slow listing
                pass

        return super().time_coverage(**filters)


class DownloadMixin(abc.ABC):

    @property
    @abc.abstractmethod
    def fs(self) -> fsspec.AbstractFileSystem:
        """The mixin relies on this attribute to build new functionalities."""

    def download(self, files: list[str], local_path: str, force_download: bool = False):
        """Retrieve files from FTP to local path.

        Parameters
        ----------
        files: str
            list of file paths to copy locally
        local_path: str
            local path to copy files to
        force_download: boolean
            force download files (True) or don't download files if already exist locally (False)

        Returns
        -------
        the list of downloaded files
        """
        downloaded = []
        for file_path in files:
            local_file = os.path.join(local_path, os.path.basename(file_path))
            if force_download or not os.path.exists(local_file):
                try:
                    logger.info("Retrieving file: %s...", file_path)
                    self.fs.get_file(file_path, local_file)
                    downloaded.append(local_file)
                except TimeoutError as exc:
                    logger.exception("An error occured retrieving file %s", exc)

        return downloaded
