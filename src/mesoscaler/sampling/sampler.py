from __future__ import annotations

import abc
import dataclasses
import datetime
import itertools
from typing import Any

import numpy as np

from .._typing import (
    TYPE_CHECKING,
    AreaExtent,
    Array,
    Iterable,
    Iterator,
    Literal,
    N,
    Point,
    Self,
    TimeSlice,
    TimeSlicePoint,
)
from ..enums import TimeFrequency, TimeFrequencyLike
from ..generic import DataSampler
from ..utils import repr_

if TYPE_CHECKING:
    from ..core import DependentDataset

LiteralKey = Literal["lon", "lat", "time"]


@dataclasses.dataclass(frozen=True)
class Intersection:
    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float
    min_time: np.datetime64
    max_time: np.datetime64

    @classmethod
    def create_from_datasets(cls, __datasets: Iterable[DependentDataset]) -> Self:
        lons = []
        lats = []
        time = []

        for ds in __datasets:
            lons.append((ds.lons.to_numpy() - 180) % 360 - 180)
            lats.append(ds.lats.to_numpy())
            time.append(ds.time.to_numpy())

        chain = itertools.chain.from_iterable(map(cls._min_max, (lons, lats, time)))
        return cls(*chain)

    @staticmethod
    def _min_max(arr: list[np.ndarray]) -> tuple[Any, Any]:
        mins = []
        maxs = []
        for x in arr:
            mins.append(x.min())
            maxs.append(x.max())
        mn, mx = np.max(mins), np.min(maxs)
        return (mn, mx)

    def __getitem__(self, key: LiteralKey) -> tuple[float, float]:
        return (getattr(self, f"min_{key}"), getattr(self, f"max_{key}"))

    def linspace(self, key: Literal["lon", "lat"], *, frequency: int, round: int = 5) -> Array[[N], Any]:
        assert isinstance(frequency, int), 'frequency must be an integer when key is "lon" or "lat"'
        mn, mx = self[key]
        return np.linspace(mn, mx, frequency).round(round)


class AbstractSampler(DataSampler[TimeSlicePoint], abc.ABC):
    _indices: list[TimeSlicePoint] | None

    def __init__(self, intersection: Intersection) -> None:
        super().__init__()
        self.intersection = intersection
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


class TimeSampler(AbstractSampler):
    def __init__(
        self,
        datasets: Iterable[DependentDataset],
        /,
        *,
        time_frequency: TimeFrequencyLike = TimeFrequency("hour"),
        time_step: int = 1,
    ) -> None:
        super().__init__(Intersection.create_from_datasets(datasets))
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
        datasets: Iterable[DependentDataset],
        /,
        *,
        lon_lat_frequency: int = 100,
        time_frequency: TimeFrequencyLike = "h",
        time_step: int = 1,
    ) -> None:
        super().__init__(datasets, time_frequency=time_frequency, time_step=time_step)
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
        datasets: Iterable[DependentDataset],
        /,
        *,
        lon_lat_frequency: int = 100,
        time_frequency: TimeFrequencyLike = TimeFrequency("hour"),
        time_step: int = 1,
        bbox: tuple[float, float, float, float] | AreaExtent = (-120, 30.0, -70, 25.0),
    ) -> None:
        super().__init__(datasets, time_frequency=time_frequency, time_step=time_step)
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
