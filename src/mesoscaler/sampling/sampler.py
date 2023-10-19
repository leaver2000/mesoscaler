# @dataclasses.dataclass(frozen=True)
from __future__ import annotations

import abc
import datetime
import itertools
from typing import Iterable

import numpy as np

from .._typing import (
    AreaExtent,
    Array,
    Iterable,
    Iterator,
    N,
    Point,
    TimeSlice,
    TimeSlicePoint,
)
from ..enums import TimeFrequency, TimeFrequencyLike
from ..utils import repr_
from .intersection import AbstractIntersection, DatasetIntersection


class TimeAndPointSampler(Iterable[TimeSlicePoint], AbstractIntersection):
    _indices: list[TimeSlicePoint] | None

    @property
    def intersection(self) -> DatasetIntersection:
        return self._intersection

    def __init__(self, intersection: DatasetIntersection) -> None:
        super().__init__()
        self._intersection = intersection
        self._indices = None

    # =================================================================================================================
    @property
    def indices(self) -> list[TimeSlicePoint]:
        if self._indices is None:
            self._indices = indices = list(self.iter_indices())
            return indices
        return self._indices

    @property
    @abc.abstractmethod
    def timedelta64(self) -> np.timedelta64:
        ...

    # =================================================================================================================
    @abc.abstractmethod
    def get_lon_lats(self) -> tuple[Array[[N], np.float_], Array[[N], np.float_]]:
        ...

    @abc.abstractmethod
    def get_time(self) -> Array[[N], np.datetime64]:
        ...

    # =================================================================================================================
    def iter_time(self) -> Iterator[TimeSlice]:
        time_indices = self.get_time()

        timedelta = self.timedelta64
        mask = np.abs(time_indices.max() - time_indices) >= timedelta

        return (np.s_[time : time + timedelta] for time in time_indices[mask])

    def iter_points(self) -> Iterator[Point]:
        lons, lats = self.get_lon_lats()
        return ((x, y) for x in lons for y in lats)

    def iter_indices(self) -> Iterator[TimeSlicePoint]:
        return itertools.product(self.iter_time(), self.iter_points())

    # =================================================================================================================
    def __iter__(self) -> Iterator[TimeSlicePoint]:
        return self.iter_indices()

    def __len__(self) -> int:
        return len(self.indices)

    def __repr__(self) -> str:
        indices = "\n".join(repr_(self.indices, map_values=True))
        return f"{type(self).__name__}[\n{indices}\n]"


class TimeSampler(TimeAndPointSampler):
    def __init__(
        self,
        intersection: DatasetIntersection,
        /,
        *,
        time_frequency: TimeFrequencyLike = TimeFrequency("hour"),
        time_step: int = 1,
    ) -> None:
        super().__init__(intersection)
        self.time_frequency = TimeFrequency(time_frequency)
        self.time_step = time_step

    @property
    def timedelta64(self) -> np.timedelta64:
        return self.time_frequency.timedelta(self.time_step)

    def get_time(self) -> Array[[N], np.datetime64]:
        return self.date_range(step=self.timedelta64)

    def date_range(
        self,
        start: datetime.datetime | np.datetime64 | str | None = None,
        stop: datetime.datetime | np.datetime64 | str | None = None,
        step: int | datetime.timedelta | np.timedelta64 | None = None,
    ) -> Array[[N], np.datetime64]:
        freq = self.time_frequency
        start = start or self.intersection.min_time
        stop = freq.datetime(stop or self.intersection.max_time)
        stop += freq.timedelta(1)  # end is exclusive
        return freq.arange(start, stop, step)


class LinearSampler(TimeSampler):
    def __init__(
        self,
        intersection: DatasetIntersection,
        /,
        *,
        lon_lat_frequency: int = 100,
        time_frequency: TimeFrequencyLike = "h",
        time_step: int = 1,
    ) -> None:
        super().__init__(intersection, time_frequency=time_frequency, time_step=time_step)
        self.lon_lat_frequency = lon_lat_frequency

    def get_lon_lats(self) -> tuple[Array[[N], np.float_], Array[[N], np.float_]]:
        frequency = self.lon_lat_frequency
        return (
            self.intersection.linspace("lon", frequency=frequency),
            self.intersection.linspace("lat", frequency=frequency),
        )


class BoundedBoxSampler(TimeSampler):
    """`x_min, y_min, x_max, y_max = area_extent`"""

    def __init__(
        self,
        intersection: DatasetIntersection,
        /,
        *,
        lon_lat_frequency: int = 100,
        time_frequency: TimeFrequencyLike = TimeFrequency("hour"),
        time_step: int = 1,
        bbox: tuple[float, float, float, float] | AreaExtent = (-120, 30.0, -70, 25.0),
    ) -> None:
        super().__init__(intersection, time_frequency=time_frequency, time_step=time_step)
        self.bbox = bbox
        self.lon_lat_frequency = lon_lat_frequency

    def get_lon_lats(self) -> tuple[Array[[N], np.float_], Array[[N], np.float_]]:
        frequency = self.lon_lat_frequency
        x_min, y_min, x_max, y_max = self.bbox
        if self.intersection.min_lat > y_min or self.intersection.max_lat < y_max:
            raise ValueError(f"area_extent latitude bounds {y_min, y_max} are outside dataset bounds")
        elif self.intersection.min_lon > x_min or self.intersection.max_lon < x_max:
            raise ValueError(f"area_extent longitude bounds {x_min, x_max} are outside dataset bounds")

        return np.linspace(x_min, x_max, frequency), np.linspace(y_min, y_max, frequency)
