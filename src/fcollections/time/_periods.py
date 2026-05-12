from __future__ import annotations

import dataclasses as dc

import numpy as np


@dc.dataclass(frozen=True)
class Period:
    """Period representation.

    Raises
    ------
    ValueError
        If the period start > stop.
    """

    start: np.datetime64
    """Date representing the start of the period."""
    stop: np.datetime64
    """Date representing the end of the period."""
    include_start: bool = True
    """Inclusive (True) or strict (False) start selection."""
    include_stop: bool = True
    """Inclusive (True) or strict (False) end selection."""

    def __post_init__(self):
        if self.start > self.stop:
            msg = "Cannot create a period with start > stop."
            raise ValueError(msg)

    @property
    def center(self) -> np.datetime64:
        return self.start + np.timedelta64((self.stop - self.start).item() / 2)

    def _equals(self, time: np.datetime64):
        return self.start == time or self.stop == time

    def _include(self, time: np.datetime64, include: bool):
        if self.start == time:
            return self.include_start and include
        if self.stop == time:
            return self.include_stop and include

    def intersects(
        self, times: np.datetime64 | Period, include_time: bool = True
    ) -> bool:
        """Check if the period intersects with another periods or time.

        In case a time is given, it checks if it is contained in the period.

        Parameters
        ----------
        times
            The period or time to intersect with the period
        include_time
            Set this to False to ignore the period bounds for the time/period
            intersection

        Returns
        -------
        :
            True if there is an intersection, False otherwise
        """
        if isinstance(times, np.datetime64):
            return (
                self.start <= times
                if (self.include_start and include_time)
                else self.start < times
            ) and (
                times <= self.stop
                if (self.include_stop and include_time)
                else times < self.stop
            )

        if self._equals(times.start):
            if self._include(times.start, times.include_start):
                return True
            if self._equals(times.stop):
                return True
            if self.intersects(times.stop):
                return True
            return times.intersects(self.start, self.include_start) | times.intersects(
                self.stop, self.include_stop
            )

        if self._equals(times.stop):
            if self._include(times.stop, times.include_stop):
                return True
            if self.intersects(times.start):
                return True
            return times.intersects(self.start, self.include_start) | times.intersects(
                self.stop, self.include_stop
            )

        return (
            self.intersects(times.start)
            | self.intersects(times.stop)
            | times.intersects(self.start, self.include_start)
            | times.intersects(self.stop, self.include_stop)
        )

    def union(self, other: Period) -> Period:
        """Union of two periods.

        Parameters
        ----------
        other
            Second period to combine with the current period

        Returns
        -------
        :
            A Period that encompasses both periods
        """
        if self.start == other.start:
            include_start = self.include_start | other.include_start
            start = self.start
        elif self.start < other.start:
            include_start = self.include_start
            start = self.start
        else:
            include_start = other.include_start
            start = other.start

        if self.stop == other.stop:
            include_stop = self.include_stop | other.include_stop
            stop = self.stop
        elif self.stop > other.stop:
            include_stop = self.include_stop
            stop = self.stop
        else:
            include_stop = other.include_stop
            stop = other.stop

        return Period(start, stop, include_start, include_stop)

    def intersection(
        self, other: np.datetime64 | Period
    ) -> np.datetime64 | Period | None:
        """Intersection of two periods.

        Parameters
        ----------
        other
            Second period or time to intersect with the current period. In case
            a time is given and there is an intersection, this time is returned.

        Returns
        -------
        :
            The intersecting Period or time, or None if there is no intersection
        """
        if isinstance(other, np.datetime64):
            return other if self.intersects(other) else None

        # Period intersection. Find lower bound
        if self.start < other.start:
            start = other.start
            include_start = other.include_start
        elif self.start > other.start:
            start = self.start
            include_start = self.include_start
        else:
            start = other.start
            include_start = other.include_start and self.include_start

        # Find upper bound
        if self.stop < other.stop:
            stop = self.stop
            include_stop = self.include_stop
        elif self.stop > other.stop:
            stop = other.stop
            include_stop = other.include_stop
        else:
            stop = other.stop
            include_stop = other.include_stop and self.include_stop

        # Sanity check. Singular period will be transformed to None instead
        if stop < start or (stop == start and not (include_start and include_stop)):
            return None

        return Period(start, stop, include_start, include_stop)

    def __le__(self, obj: Period) -> bool:
        return self.start <= obj.start

    def __ge__(self, obj: Period) -> bool:
        return self.stop >= obj.stop

    def __lt__(self, obj: Period) -> bool:
        return self.start < obj.start

    def __gt__(self, obj: Period) -> bool:
        return self.stop > obj.stop

    def __repr__(self):
        return f"{'[' if self.include_start else ']'}{self.start}, {self.stop}{']' if self.include_stop else '['}"
