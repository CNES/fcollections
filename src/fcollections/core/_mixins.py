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


def suppress_performance_warning(func: tp.Callable) -> tp.Callable:
    """Suppress PerformanceWarning when calling the input function.

    Returns
    -------
    tp.Callable
        The patched function with suppressed PerformanceWarning
    """

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
    """Mixin extending FilesDatabase with methods working on half orbits."""

    def filter_values(self, filter_name: str, **kwargs) -> set[tp.Any]:
        """The mixin relies on this method to build new functionalities."""

    def cycle_range(self, **filters) -> tuple[int, int]:
        """Extract the cycle range.

        Parameters
        ----------
        filters
            Set of filters to apply prior to extract the cycle range. This can
            be used to pass the mandatory filters for selecting a single subset,
            or to extract the cycle range for a single mission phase.

        Returns
        -------
        tuple[int, int]
            The first and last cycle matching the selection.
        """
        cycles = sorted(self.filter_values("cycle_number", **filters))
        return cycles[0], cycles[-1]

    @suppress_performance_warning
    def half_orbit_range(self, **filters) -> tuple[tuple[int, int], tuple[int, int]]:
        """Extract the half orbits range.

        Parameters
        ----------
        filters
            Set of filters to apply prior to extract the half orbits range. This
            can be used to pass the mandatory filters for selecting a single
            subset, or to extract the half orbits range for a single mission
            phase.

        Returns
        -------
        tuple[tuple[int, int], tuple[int, int]]
            Two pairs of (cycle_number, pass_number) numbering the first and
            last half orbit of the selection.
        """
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
        """Extract the time coverage.

        The mixin implementation expects that the files will be grouped by
        cycles in folders. This property can be used to first get the first and
        last cycles, before listing the times for these two cycles. This is much
        faster than getting the times for all the cycles.

        In case the hypothesis is not True (ie. folders do not contain the cycle
        number information), we fall back to the classic implementation which is
        slower.

        In addition, `cycle_number` ordering can break if multiple mission
        phases are mixed in the selection. This will usually lead to an
        inconsistent Period, which will again make the method fall back to the
        default implementation.

        Parameters
        ----------
        filters
            Set of filters to apply prior to extract the time coverage. This
            can be used to pass the mandatory filters for selecting a single
            subset, or to extract the time coverage for a single mission phase.

        Returns
        -------
        tuple[tuple[int, int], tuple[int, int]]
            Two pairs of (cycle_number, pass_number) numbering the first and
            last half orbit of the selection.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error", PerformanceWarning)

            try:
                cycle_range = self.cycle_range(**filters)

                # The input filters will probably give a range for selecting a
                # mission phase. Mission phase filters work on the cycle_number
                # variable, and giving filters on the same variable will raise
                # an error in the filter_values method. We must remove all
                # filters working on the cycle_number variable.
                edited_filters = filters.copy()
                for filter_builder in self.filter_builders:
                    if "cycle_number" in filter_builder.target_fields():
                        logger.debug(
                            "Removed filter `%s` working on the "
                            "`cycle_number` variable",
                            filter_builder.parameter().name,
                        )
                        edited_filters.pop(filter_builder.parameter().name, None)
                edited_filters["cycle_number"] = list(cycle_range)
            except PerformanceWarning:
                # Don't try to accelerate, we must fall back to a slow listing
                logger.debug(
                    "Shortcut using the `cycle_number` variable failed, "
                    "falling back listing `time` values without filters."
                )

        try:
            return super().time_coverage(**edited_filters)
        except ValueError:
            # ValueError is raised if the period start > stop. This can arise if
            # the cycle_number variable has a different order than the time
            # variable. An example is the SWOT mission where the first mission
            # phase CALVAL is numbered [400-600] whereas the second mission
            # phase SCIENCE is numbered [1-399]. This sorting break will cause
            # an inconsistent period, in which case we need to fall back to a
            # full scan
            logger.debug(
                "Shortcut using the `cycle_number` variable failed, "
                "falling back listing `time` values without filters."
            )
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
