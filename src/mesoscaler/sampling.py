from __future__ import annotations

import abc
import dataclasses
import itertools
import random
from typing import Any

import numpy as np
import pandas as pd

from ._typing import (
    AreaExtent,
    Array,
    Iterator,
    Literal,
    N,
    Point,
    PointOverTime,
    Self,
    TimeSlice,
)
from .core import DependentDataset
from .generic import DataSampler

LiteralKey = Literal["lon", "lat", "time"]
TimeFrequency = Literal["h"]


@dataclasses.dataclass(frozen=True)
class DatasetIntersection:
    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float
    min_time: np.datetime64
    max_time: np.datetime64

    @classmethod
    def create(cls, __datasets: Iterator[DependentDataset] | DependentDataset, *dsets: DependentDataset):
        datasets = itertools.chain(tuple(__datasets if isinstance(__datasets, Iterator) else [__datasets]), dsets)
        return cls(*cls.chain_intersection(datasets))  # type: ignore[arg-type]

    @classmethod
    def chain_intersection(cls, __datasets: itertools.chain[DependentDataset]) -> itertools.chain[Any]:
        lons = []
        lats = []
        time = []

        for ds in __datasets:
            lons.append((ds.lons.to_numpy() - 180) % 360 - 180)
            lats.append(ds.lats.to_numpy())
            time.append(ds.time.to_numpy())

        return itertools.chain(cls._min_max(lons), cls._min_max(lats), cls._min_max(time))

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

    def linspace(
        self,
        key: LiteralKey,
        *,
        frequency: TimeFrequency | int,
        round: int = 5,
    ) -> Array[[N], Any]:
        if key == "time" and isinstance(frequency, str):
            return pd.date_range(self.min_time, self.max_time, freq=frequency).to_numpy()

        assert isinstance(frequency, int), 'frequency must be an integer when key is "lon" or "lat"'
        mn, mx = self[key]
        value = np.linspace(mn, mx, frequency).round(round)

        return value


class AbstractBaseSampler(DataSampler[PointOverTime], abc.ABC):
    _product: list[PointOverTime] | None

    def __init__(
        self,
        __datasets: Iterator[DependentDataset] | DependentDataset,
        *dsets: DependentDataset,
        lon_lat_frequency: int = 100,
        time_frequency: TimeFrequency = "h",
        time_step: int = 1,
    ) -> None:
        super().__init__()
        self.intersection = DatasetIntersection.create(__datasets, *dsets)
        self.lon_lat_frequency = lon_lat_frequency
        self.time_step = time_step
        self.time_frequency = time_frequency
        self._product = None

    @property
    def product(self) -> list[PointOverTime]:
        if self._product is None:
            self._product = product = list(self.iter_product())
            return product
        return self._product

    def shuffle(self, *, seed: int) -> Self:
        product = self.product
        random.seed(seed)
        random.shuffle(product)
        self._product = product

        return self

    @property
    def timedelta64(self) -> np.timedelta64:
        return np.timedelta64(self.time_step, self.time_frequency)

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

    def iter_product(self) -> Iterator[PointOverTime]:
        return itertools.product(self.iter_points(), self.iter_time())

    # =================================================================================================================
    def __iter__(self) -> Iterator[PointOverTime]:
        return self.iter_product()

    def __len__(self) -> int:
        return len(self.product)

    def __repr__(self) -> str:
        from .utils import repr_

        product = "\n".join(repr_(self.product, map_values=True))
        return f"{type(self).__name__}[\n{product}\n]"


class LinearSampler(AbstractBaseSampler):
    def __init__(
        self,
        __datasets: Iterator[DependentDataset] | DependentDataset,
        *dsets: DependentDataset,
        lon_lat_frequency: int = 100,
        time_frequency: TimeFrequency = "h",
        time_step: int = 1,
    ) -> None:
        super().__init__(
            __datasets, *dsets, time_frequency=time_frequency, time_step=time_step, lon_lat_frequency=lon_lat_frequency
        )

    def get_lon_lats(self) -> tuple[Array[[N], np.float_], Array[[N], np.float_]]:
        frequency = self.lon_lat_frequency
        return (
            self.intersection.linspace("lon", frequency=frequency),
            self.intersection.linspace("lat", frequency=frequency),
        )

    def get_time(self) -> Array[[N], np.datetime64]:
        frequency = self.time_frequency
        return self.intersection.linspace("time", frequency=frequency)  # type: ignore[arg-type]


class ExtentBoundLinearSampler(LinearSampler):
    """`x_min, y_min, x_max, y_max = area_extent`"""

    def __init__(
        self,
        __datasets: Iterator[DependentDataset] | DependentDataset,
        *dsets: DependentDataset,
        lon_lat_frequency: int = 100,
        time_frequency: TimeFrequency = "h",
        time_step: int = 1,
        area_extent: tuple[float, float, float, float] | AreaExtent,
    ) -> None:
        super().__init__(
            __datasets, *dsets, time_frequency=time_frequency, time_step=time_step, lon_lat_frequency=lon_lat_frequency
        )
        self.area_extent = area_extent

    def get_lon_lats(self) -> tuple[Array[[N], np.float_], Array[[N], np.float_]]:
        frequency = self.lon_lat_frequency
        x_min, y_min, x_max, y_max = self.area_extent
        if self.intersection.min_lat > y_min or self.intersection.max_lat < y_max:
            raise ValueError(f"area_extent latitude bounds {y_min, y_max} are outside dataset bounds")
        elif self.intersection.min_lon > x_min or self.intersection.max_lon < x_max:
            raise ValueError(f"area_extent longitude bounds {x_min, x_max} are outside dataset bounds")

        return np.linspace(x_min, x_max, frequency), np.linspace(y_min, y_max, frequency)
