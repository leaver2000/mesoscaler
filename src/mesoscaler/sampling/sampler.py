from __future__ import annotations

import abc
import datetime
import itertools

import numpy as np

from .. import time64, utils
from .._typing import (
    AreaExtent,
    Array,
    Iterator,
    N,
    Nt,
    Nx,
    Ny,
    Point,
    PointOverTime,
    TimeSlice,
)
from .domain import Domain, DomainIntersectionSampler


class TimeAndPointSampler(DomainIntersectionSampler[PointOverTime], abc.ABC):
    _indices: list[PointOverTime] | None

    @property
    def domain(self) -> Domain:
        return self._domain

    def __init__(self, domain: Domain) -> None:
        super().__init__(domain)
        self._indices = None

    # =================================================================================================================
    @property
    def indices(self) -> list[PointOverTime]:
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
    def get_lon_lats(self) -> tuple[Array[[Nx], np.float_], Array[[Ny], np.float_]]:
        ...

    @abc.abstractmethod
    def get_time(self) -> Array[[N], np.datetime64] | Array[[N, N], np.datetime64]:
        ...

    # =================================================================================================================
    def iter_time(self) -> Iterator[TimeSlice | Array[[N], np.datetime64]]:
        time_indices = self.get_time()
        if time_indices.ndim == 1:
            timedelta = self.timedelta64
            mask = np.abs(time_indices.max() - time_indices) >= timedelta

            return (np.s_[time : time + timedelta] for time in time_indices[mask])
        assert time_indices.ndim == 2
        yield from time_indices

    def iter_points(self) -> Iterator[Point]:
        lons, lats = self.get_lon_lats()
        return ((x, y) for x in lons for y in lats)

    def iter_indices(self) -> Iterator[PointOverTime]:
        return itertools.product(self.iter_points(), self.iter_time())

    # =================================================================================================================
    def __iter__(self) -> Iterator[PointOverTime]:
        return self.iter_indices()

    def __len__(self) -> int:
        return len(self.indices)

    def __repr__(self) -> str:
        indices = "\n".join(utils.repr_(self.indices, map_values=True))
        return f"{type(self).__name__}[\n{indices}\n]"


class TimeSampler(TimeAndPointSampler):
    def __init__(
        self,
        domain: Domain,
        /,
        *,
        time_frequency: time64.Time64Like = time64.Time64("hours"),
        time_step: int = 3,
    ) -> None:
        super().__init__(domain)
        self.time_frequency = time64.Time64(time_frequency)
        self.time_step = time_step

    @property
    def timedelta64(self) -> np.timedelta64:
        return self.time_frequency.delta(self.time_step)

    def get_time(self) -> Array[[N, N], np.datetime64]:
        return self.batch_time(self.time_step)

    def date_range(
        self,
        start: datetime.datetime | np.datetime64 | str | None = None,
        stop: datetime.datetime | np.datetime64 | str | None = None,
        step: int | datetime.timedelta | np.timedelta64 | None = None,
    ) -> Array[[N], np.datetime64]:
        freq = self.time_frequency
        start = start or self.min_time
        stop = freq.datetime(stop or self.max_time)
        stop += freq.delta(1)  # end is exclusive
        return freq.arange(start, stop, step)


class LinearSampler(TimeSampler):
    def __init__(
        self,
        domain: Domain,
        /,
        *,
        lon_lat_frequency: int = 100,
        time_frequency: time64.Time64Like = time64.Time64("hours"),
        time_step: int = 1,
    ) -> None:
        super().__init__(domain, time_frequency=time_frequency, time_step=time_step)
        self.lon_lat_frequency = lon_lat_frequency

    def get_lon_lats(self) -> tuple[Array[[N], np.float_], Array[[N], np.float_]]:
        frequency = self.lon_lat_frequency
        return (self.linspace("lon", frequency=frequency), self.linspace("lat", frequency=frequency))


class AreaOfInterestSampler(TimeSampler):
    """`x_min, y_min, x_max, y_max = area_extent`"""

    def __init__(
        self,
        domain: Domain,
        /,
        *,
        lon_lat_frequency: int = 5,
        time_frequency: time64.Time64Like = time64.Time64("hours"),
        time_step: int = 1,
        aoi: tuple[float, float, float, float] | AreaExtent = (-120, 30.0, -70, 25.0),
    ) -> None:
        super().__init__(domain, time_frequency=time_frequency, time_step=time_step)
        self.aoi = aoi
        self.lon_lat_frequency = lon_lat_frequency

    def get_lon_lats(self) -> tuple[Array[[N], np.float_], Array[[N], np.float_]]:
        frequency = self.lon_lat_frequency
        x_min, y_min, x_max, y_max = self.aoi
        if self.domain.min_lat > y_min or self.domain.max_lat < y_max:
            raise ValueError(f"area_extent latitude bounds {y_min, y_max} are outside dataset bounds")
        elif self.domain.min_lon > x_min or self.domain.max_lon < x_max:
            raise ValueError(f"area_extent longitude bounds {x_min, x_max} are outside dataset bounds")

        return np.linspace(x_min, x_max, frequency), np.linspace(y_min, y_max, frequency)
