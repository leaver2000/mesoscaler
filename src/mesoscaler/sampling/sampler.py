from __future__ import annotations

import abc
import functools
import itertools

import numpy as np

from .. import utils
from .._typing import (
    Any,
    AreaExtent,
    Array,
    Callable,
    Iterator,
    N,
    Nx,
    Ny,
    Point,
    PointOverTime,
    Self,
    TypeVar,
)
from ..generic import DataSampler
from .domain import AbstractDomain, Domain

_T = TypeVar("_T")


class DomainIntersectionSampler(DataSampler[_T], AbstractDomain, abc.ABC):
    @property
    def domain(self) -> Domain:
        return self._domain

    def __init__(self, domain: Domain) -> None:
        super().__init__()
        self._domain = domain

    @classmethod
    def partial(cls, *args: Any, **kwargs: Any) -> Callable[[Domain], Self]:
        return functools.partial(cls, *args, **kwargs)


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

    # =================================================================================================================
    @abc.abstractmethod
    def get_lon_lats(self) -> tuple[Array[[Nx], np.float_], Array[[Ny], np.float_]]:
        ...

    @abc.abstractmethod
    def get_time_batches(self) -> Array[[N, N], np.datetime64]:
        ...

    # =================================================================================================================
    def iter_time(self) -> Iterator[Array[[N], np.datetime64]]:
        yield from self.get_time_batches()

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
        time_batch_size: int = 1,
    ) -> None:
        super().__init__(domain)
        self.time_batch_size = time_batch_size

    def get_time_batches(self) -> Array[[N, N], np.datetime64]:
        return utils.batch(self.time, self.time_batch_size, strict=True)


class LinearSampler(TimeSampler):
    def __init__(
        self,
        domain: Domain,
        /,
        *,
        lon_lat_frequency: int = 100,
        time_batch_size: int = 1,
    ) -> None:
        super().__init__(domain, time_batch_size=time_batch_size)
        self.lon_lat_frequency = lon_lat_frequency

    def get_lon_lats(self) -> tuple[Array[[N], np.float_], Array[[N], np.float_]]:
        frequency = self.lon_lat_frequency
        lon, lat = np.linspace(self.min_lon, self.max_lon, frequency), np.linspace(
            self.min_lat, self.max_lat, frequency
        )
        return lon, lat
        # return (self.linspace("lon", frequency=frequency), self.linspace("lat", frequency=frequency))


class AreaOfInterestSampler(TimeSampler):
    """`x_min, y_min, x_max, y_max = area_extent`"""

    def __init__(
        self,
        domain: Domain,
        /,
        *,
        aoi: tuple[float, float, float, float] | AreaExtent,  #  (-120, 30.0, -70, 25.0),
        lon_lat_frequency: int = 5,
        time_batch_size: int = 1,
    ) -> None:
        super().__init__(domain, time_batch_size=time_batch_size)
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
